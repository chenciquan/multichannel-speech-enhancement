#!/usr/bin/env python3
"""Run a frozen, validated small manifest through one McNet model instance.

No fitting, resampling, peak normalization, implicit alignment or ASR is done.
All cases are retained. Optional quality metrics are real backends or explicit
unavailable/errors; review prioritization never confirms a content-loss event.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import runpy
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mcse.manifest import validate_manifest
from mcse.evaluation import MetricEvaluator, summarize_cases, waveform_diagnostics


def write_json(path: Path, content: Any) -> None:
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def delta(after: Any, before: Any) -> float | None:
    return float(after - before) if after is not None and before is not None else None


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    import numpy as np
    import soundfile as sf
    import torch

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("Output directory must be absent or empty; no overwrite/resume is implicit.")
    manifest = args.manifest.resolve()
    validation = validate_manifest(manifest, check_audio=True)
    if not validation.valid:
        raise ValueError("Manifest validation failed: " + json.dumps(validation.to_dict(), ensure_ascii=False))
    for record in validation.records:
        info = sf.info(record.noisy_path)
        if info.frames < 512 or info.duration > args.max_seconds:
            raise ValueError(f"{record.sample_id}: duration must contain at least 512 samples and be <= {args.max_seconds}s.")
    adapter = runpy.run_path(str(ROOT / "scripts/mcnet_infer.py"), run_name="mcnet_batch_adapter")
    repo = args.repo.resolve()
    checkpoint = (args.checkpoint or repo / adapter["OFFLINE_CHECKPOINT"]).resolve()
    upstream = adapter["verify_upstream"](repo, checkpoint)
    input_manifest_hash = adapter["file_hashes"](manifest)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; no CPU fallback.")
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "enhanced").mkdir()
    write_json(output_dir / "manifest_validation.json", validation.to_dict())
    frozen_records = []
    for record in validation.records:
        frozen_records.append({
            "sample_id": record.sample_id, "source_id": record.source_id,
            "speaker_id": record.speaker_id, "split": record.split,
            "noisy": adapter["file_hashes"](record.noisy_path),
            "reference": adapter["file_hashes"](record.reference_path),
            "transcript": record.transcript, "noise_id": record.noise_id,
            "snr_db": record.snr_db, "extras": record.extras,
        })
    frozen = {"manifest": input_manifest_hash, "records": frozen_records,
              "max_seconds": args.max_seconds, "sample_rate": 16000,
              "reference_channel_zero_based": 4, "frozen_before_model_output": True}
    write_json(output_dir / "frozen_inputs.json", frozen)
    load_start = time.perf_counter()
    model, icirm, model_metadata = adapter["load_model"](repo, checkpoint, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_start
    model_load_peak_allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    model_load_peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
    evaluator = MetricEvaluator()
    versions = {}
    for package in ("torch", "numpy", "soundfile", "pytorch-lightning", "torchmetrics", "pystoi", "pesq"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    provenance = {
        "stage": "G1 discovery diagnostics", "upstream": upstream,
        "model": model_metadata, "model_loaded_once": True, "device": str(device),
        "versions": versions, "metric_backends": evaluator.backend_info,
        "si_sdr_definition": "Zero-mean projection SI-SDR; float64 relative machine precision floor; no delay/filter fitting.",
        "model_load_seconds": load_seconds, "training_performed": False,
        "model_load_cuda_peak_allocated_bytes": model_load_peak_allocated,
        "model_load_cuda_peak_reserved_bytes": model_load_peak_reserved,
        "asr_performed": False, "upstream_source_edited": False,
        "per_utterance_peak_normalization": False, "resampling_or_chunking": False,
        "output_format": "WAV/FLOAT", "input_manifest_sha256": input_manifest_hash["sha256"],
        "notice": "Objective signal metrics and review candidates do not confirm missing speech content.",
    }
    write_json(output_dir / "run_provenance.json", provenance)
    cases = []
    with (output_dir / "cases.jsonl").open("x", encoding="utf-8") as case_stream:
        for index, record in enumerate(validation.records):
            identifier = f"{index:04d}_{hashlib.sha256(record.sample_id.encode()).hexdigest()[:12]}"
            enhanced_path = output_dir / "enhanced" / f"{identifier}.wav"
            case: dict[str, Any] = {
                "sample_id": record.sample_id, "source_id": record.source_id,
                "speaker_id": record.speaker_id, "split": record.split,
                "noise_id": record.noise_id, "snr_db": record.snr_db,
                "transcript": record.transcript,
                "reference_path": str(record.reference_path), "noisy_path": str(record.noisy_path),
                "enhanced_path": str(enhanced_path), "reference_channel_zero_based": 4,
                "human_content_loss_judgement": "not_assessed", "review_status": "unreviewed",
            }
            x = prediction = waveform = data = None
            try:
                noisy, sample_rate = sf.read(record.noisy_path, dtype="float32", always_2d=True)
                reference, ref_rate = sf.read(record.reference_path, dtype="float64", always_2d=False)
                if sample_rate != 16000 or ref_rate != 16000 or noisy.ndim != 2 or noisy.shape[1] != 6:
                    raise ValueError("Input metadata changed after manifest validation.")
                if reference.ndim != 1 or reference.size != noisy.shape[0] or not np.isfinite(noisy).all() or not np.isfinite(reference).all():
                    raise ValueError("Inputs must be finite, aligned and equal length.")
                if adapter["file_hashes"](record.noisy_path)["sha256"] != frozen_records[index]["noisy"]["sha256"] or adapter["file_hashes"](record.reference_path)["sha256"] != frozen_records[index]["reference"]["sha256"]:
                    raise ValueError("Input bytes changed after they were frozen.")
                x = torch.from_numpy(np.ascontiguousarray(noisy.T)).unsqueeze(0).to(device)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                with torch.inference_mode():
                    prediction, data = model(x)
                    if prediction.shape != (1, 257, 1 + noisy.shape[0] // 256):
                        raise ValueError(f"Unexpected prediction shape: {tuple(prediction.shape)}")
                    waveform = adapter["reconstruct_waveform"](model, icirm, x, prediction, data["original_time_len"])
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                if waveform.shape != (1, noisy.shape[0]) or not torch.isfinite(waveform).all():
                    raise ValueError("Enhanced output has wrong length or nonfinite values.")
                enhanced = waveform[0].cpu().numpy()
                sf.write(enhanced_path, enhanced, 16000, format="WAV", subtype="FLOAT")
                noisy_metrics = evaluator.evaluate(reference, noisy[:, 4], sample_rate)
                enhanced_metrics = evaluator.evaluate(reference, enhanced, sample_rate)
                noisy_diagnostics = waveform_diagnostics(reference, noisy[:, 4], sample_rate)
                enhanced_diagnostics = waveform_diagnostics(reference, enhanced, sample_rate)
                case.update({
                    "status": "ok", "sample_rate": sample_rate,
                    "samples": int(reference.size), "duration_seconds": reference.size / sample_rate,
                    "enhanced_file": adapter["file_hashes"](enhanced_path),
                    "metrics": {"noisy": noisy_metrics, "enhanced": enhanced_metrics},
                    "improvement": {name: delta(enhanced_metrics[name], noisy_metrics[name])
                                    for name in ("si_sdr_db", "stoi", "pesq_wb")},
                    "waveform_diagnostics": {"noisy": noisy_diagnostics, "enhanced": enhanced_diagnostics},
                    "inference_seconds": elapsed, "rtf_excluding_load_io_and_metrics": elapsed / (reference.size / sample_rate),
                    "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                    "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
                })
            except Exception as error:
                case.update({"status": "error", "error_type": type(error).__name__, "error": str(error)})
            finally:
                del x, prediction, waveform, data
                if device.type == "cuda" and case.get("error_type") == "OutOfMemoryError":
                    torch.cuda.empty_cache()
            cases.append(case)
            case_stream.write(json.dumps(case, ensure_ascii=False, allow_nan=False) + "\n")
            case_stream.flush()
            print(json.dumps({"index": index + 1, "total": len(validation.records),
                              "sample_id": record.sample_id, "status": case["status"]}, ensure_ascii=False), flush=True)
    summary = summarize_cases(cases)
    summary.update({"status": "ok" if not summary["failed_cases"] else "partial_failure",
                    "input_manifest_sha256": input_manifest_hash["sha256"], "model_loaded_once": True,
                    "unique_sources": len({r.source_id for r in validation.records}),
                    "unique_speakers": len({r.speaker_id for r in validation.records}),
                    "metric_backends": evaluator.backend_info, "training_performed": False,
                    "cuda_peak_allocated_bytes": max(
                        [model_load_peak_allocated or 0] + [case.get("cuda_peak_allocated_bytes") or 0 for case in cases]) if device.type == "cuda" else None,
                    "cuda_peak_reserved_bytes": max(
                        [model_load_peak_reserved or 0] + [case.get("cuda_peak_reserved_bytes") or 0 for case in cases]) if device.type == "cuda" else None,
                    "asr_performed": False,
                    "notice": "Exploratory signal-level diagnostics only. No human content-loss event has been adjudicated."})
    write_json(output_dir / "summary.json", summary)
    fields = ["sample_id", "source_id", "speaker_id", "split", "noise_id", "snr_db", "status",
              "duration_seconds", "inference_seconds", "noisy_si_sdr_db", "enhanced_si_sdr_db", "si_sdr_improvement_db",
              "noisy_stoi", "enhanced_stoi", "noisy_stoi_status", "enhanced_stoi_status",
              "noisy_pesq_wb", "enhanced_pesq_wb", "noisy_pesq_wb_status", "enhanced_pesq_wb_status",
              "enhanced_peak", "enhanced_fraction_above_one", "low_energy_underestimated_6db_fraction",
              "human_content_loss_judgement", "error"]
    with (output_dir / "metrics.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            row = {key: case.get(key) for key in fields}
            if case["status"] == "ok":
                for system in ("noisy", "enhanced"):
                    for metric in ("si_sdr_db", "stoi", "pesq_wb"):
                        row[f"{system}_{metric}"] = case["metrics"][system][metric]
                        if metric != "si_sdr_db":
                            row[f"{system}_{metric}_status"] = case["metrics"][system]["status"][metric]["status"]
                row["si_sdr_improvement_db"] = case["improvement"]["si_sdr_db"]
                diag = case["waveform_diagnostics"]["enhanced"]
                row.update({"enhanced_peak": diag["estimate_abs_peak"],
                            "enhanced_fraction_above_one": diag["estimate_fraction_above_one"],
                            "low_energy_underestimated_6db_fraction": diag["low_reference_energy_underestimated_6db_fraction"]})
            writer.writerow(row)
    def review_key(case):
        if case["status"] != "ok":
            return (0, 0.0, case["sample_id"])
        value = case["waveform_diagnostics"]["enhanced"]["low_reference_energy_underestimated_6db_fraction"]
        return (1, -(value if value is not None else -1.0), case["sample_id"])
    with (output_dir / "review_queue.jsonl").open("x", encoding="utf-8") as stream:
        for case in sorted(cases, key=review_key):
            stream.write(json.dumps({
                "sample_id": case["sample_id"], "status": case["status"],
                "review_status": "unreviewed", "human_content_loss_judgement": "not_assessed",
                "notice": "All cases retained; energy-based ordering is only a review aid, not a missing-word decision.",
            }, ensure_ascii=False) + "\n")
    return summary, 0 if not summary["failed_cases"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=ROOT / "third_party/McNet")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-seconds", type=float, default=4.0)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 4.0:
        parser.error("The G1 runtime probe supports >0 and <=4 seconds; longer inputs need a separate resource test.")
    try:
        summary, code = run(args)
        print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
        return code
    except Exception as error:
        print(json.dumps({"status": "error", "error_type": type(error).__name__, "error": str(error),
                          "training_performed": False,
                          "notice": "Execution failure is not a speech-quality result."}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
