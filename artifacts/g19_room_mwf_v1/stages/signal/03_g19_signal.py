#!/usr/bin/env python3
"""Evaluate a fixed reference-channel MWF using G18's saved neural estimates."""
import os

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import hashlib
import io
import json
import shutil
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from g19_common import ROOT, OUT, PREV, now, sha, read, dump, status, archive
from mcse.beamforming import REFERENCE
from mcse.evaluation import MetricEvaluator, waveform_diagnostics
from mcse.wiener import mask_mwf


EVALUATION = OUT / "evaluation"
OLD_EVALUATION = PREV / "evaluation"
MANIFEST = ROOT / "data/g18_v1/manifest.jsonl"
METRICS = ("si_sdr_db", "pesq_wb", "stoi")
MASK_TOLERANCE = 1e-12
WORKERS = 2
_evaluator = None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_cases_and_systems(old_path):
    old_provenance = load_json(OLD_EVALUATION / "provenance.json")
    registry_file = PREV / "systems.json"
    require(sha(registry_file) == old_provenance["systems_sha256"], "G18 system registry hash mismatch")
    require(sha(MANIFEST) == old_provenance["manifest_sha256"], "G18 case manifest hash mismatch")
    old_filter = load_json(OLD_EVALUATION / "beam_provenance.json")["beamforming_code_sha256"]
    require(sha(ROOT / "src/mcse/beamforming.py") == old_filter and REFERENCE == 4,
            "G18 reference-channel filter implementation mismatch")
    registry = load_json(registry_file)
    frontends = [s["system"] for s in registry["direct"] if s["kind"] != "noisy"]
    beams = {s["mask_system"]: s["system"] for s in registry["beam"]}
    require(len(frontends) == len(set(frontends)) == 10 and set(frontends) == set(beams),
            "Expected ten registered frontend/MVDR pairs")
    require(all(beams[s] == s + "_mask_mvdr" for s in frontends), "Unexpected MVDR registry names")

    cases = read(MANIFEST)
    case_by_id = {c["sample_id"]: c for c in cases}
    require(len(cases) == len(case_by_id) == 504 and sum(c["condition"] == "clean" for c in cases) == 72,
            "Unexpected G18 case coverage")
    old_rows = read(old_path)
    index = {(r["system"], r["sample_id"]): r for r in old_rows}
    systems = {"noisy", *frontends, *beams.values()}
    require(len(old_rows) == len(index) == 10584 and set(index) == {
        (system, sample_id) for system in systems for sample_id in case_by_id
    }, "Unexpected G18 signal keys")
    for record in old_rows:
        case = case_by_id[record["sample_id"]]
        require(record["status"] == "ok" and all(record[k] == v for k, v in case.items()),
                f"G18 case metadata mismatch for {record['system']}/{record['sample_id']}")
    return cases, frontends, index


def worker_init():
    global _evaluator
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _evaluator = MetricEvaluator()


def verified_audio(path, expected_hash, frames, channels, subtype):
    content = Path(path).read_bytes()
    require(hashlib.sha256(content).hexdigest() == expected_hash, f"Audio hash mismatch at {path}")
    with sf.SoundFile(io.BytesIO(content)) as wav:
        require((wav.samplerate, len(wav), wav.channels, wav.subtype) == (16000, frames, channels, subtype),
                f"Unexpected WAV format or frame count at {path}")
        audio = wav.read(dtype="float32")
    require(np.isfinite(audio).all(), f"Nonfinite WAV at {path}")
    return audio


def level_diagnostics(reference, mixture_channel, output):
    ref = reference.astype(np.float64)
    mix = mixture_channel.astype(np.float64)
    est = output.astype(np.float64)
    ref_energy, mix_energy, est_energy = (float(a @ a) for a in (ref, mix, est))
    require(min(ref_energy, mix_energy) > 0, "Undefined RMS gain for a zero-energy reference or input")
    error = est - ref
    return {
        "rms_gain_ratio": float(np.sqrt(est_energy / ref_energy)),
        "input_rms_gain_ratio": float(np.sqrt(est_energy / mix_energy)),
        "reference_normalized_squared_error": float(error @ error) / ref_energy,
    }


def filter_and_score(case, direct, mvdr):
    frontend = direct["system"]
    sample_id = case["sample_id"]
    name = frontend + "_mask_mwf"
    label = f"{name}/{sample_id}"
    require(mvdr["source_system"] == frontend
            and mvdr["mask_estimate_path"] == direct["output_path"]
            and mvdr["mask_estimate_sha256"] == direct["output_sha256"],
            f"G18 MVDR estimate provenance mismatch for {label}")
    frames = case["frames"]
    noisy = verified_audio(case["noisy_path"], case["input_sha256"], frames, 6, "PCM_24")
    estimate = verified_audio(direct["output_path"], direct["output_sha256"], frames, 1, "FLOAT")

    started = time.perf_counter()
    output, info = mask_mwf(noisy, estimate, mu=1.0)
    seconds = time.perf_counter() - started
    old_mean = mvdr["beamforming_diagnostics"]["mask_mean"]
    difference = abs(info["mask_mean"] - old_mean)
    require(info["mu"] == 1.0 and info["reference_channel_zero_based"] == 4
            and difference <= MASK_TOLERANCE, f"G18 mask or reference mismatch for {label}")
    info.update(g18_mvdr_mask_mean=old_mean, mask_mean_abs_diff=difference,
                oracle_clean_reference=False, oracle_DOA=False)
    require(output.shape == (frames,) and output.dtype == np.float32 and np.isfinite(output).all(),
            f"Invalid filter output for {label}")

    path = EVALUATION / "mwf_audio" / name / (sample_id + ".wav")
    with path.open("xb") as stream:
        sf.write(stream, output, 16000, format="WAV", subtype="FLOAT")
    output_sha = sha(path)
    saved = verified_audio(path, output_sha, frames, 1, "FLOAT")
    require(saved.tobytes() == output.tobytes(), f"FLOAT WAV did not round-trip exactly for {label}")

    reference = verified_audio(case["reference_path"], case["reference_sha256"], frames, 1, "PCM_24")
    metrics = _evaluator.evaluate(reference, saved)
    require(all(metrics["status"][k]["status"] == "ok" and np.isfinite(metrics[k]) for k in METRICS),
            f"Metric failure for {label}: {metrics['status']}")
    return {
        **case, "system": name, "source_system": frontend, "mask_estimate_path": direct["output_path"],
        "mask_estimate_sha256": direct["output_sha256"], "direct_inference_seconds": direct["inference_seconds"],
        "inference_seconds": seconds, "audio_seconds": frames / 16000,
        "output_path": str(path), "output_sha256": output_sha, "beamforming_diagnostics": info,
        "metrics": metrics, "waveform_diagnostics": waveform_diagnostics(reference, saved),
        "level_diagnostics": level_diagnostics(reference, noisy[:, REFERENCE], saved), "status": "ok",
    }


def signal_report(rows, completion, provenance):
    lines = [
        "# G19B 单房间 MWF 信号评估", "",
        "复用 G18 的 10 个前端及其已保存的逐例估计。每个前端含 72 个干净条件与 432 个带噪条件。"
        "采用固定 μ=1 的多通道 Wiener 滤波器，第五个麦克风（从零计为 4）为参考。"
        "滤波只读取带噪六通道和神经估计。评分使用逐样本回读一致的单通道 FLOAT WAV 及同房间混响干净参考。", "",
        "|掩码前端|条件|例数|SI-SDR (dB)|PESQ-WB|STOI|输出/参考 RMS|输出/输入第 5 麦克风 RMS|参考归一化平方误差|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for frontend in sorted({r["source_system"] for r in rows}):
        selected = [r for r in rows if r["source_system"] == frontend]
        for clean, label in ((True, "干净"), (False, "带噪")):
            group = [r for r in selected if (r["condition"] == "clean") == clean]
            means = [np.mean([r["metrics"][k] for r in group]) for k in METRICS]
            means.extend(np.mean([r["level_diagnostics"][k] for r in group]) for k in
                         ("rms_gain_ratio", "input_rms_gain_ratio", "reference_normalized_squared_error"))
            lines.append(f"|{frontend}|{label}|{len(group)}|" + "|".join(f"{v:.4f}" for v in means) + "|")
    lines.extend([
        "", "表内数值为逐例算术平均。RMS 增益以线性比值表示。归一化平方误差为输出与干净参考之差的平方和除以参考平方和。"
        "带噪条件下这些整段比值同时受残余噪声与语音变化影响。",
        "", f"新增 {completion['mwf_rows']} 条，合并 G18 后共 {completion['combined_rows']} 条。"
        f"逐例掩码均值与 G18 MVDR 的最大绝对差为 {completion['max_mask_mean_abs_diff']:.3g}。"
        "逐例文件保留加载量、线性求解残差、协方差分解误差、波形诊断及完整质量指标。",
        "", "样本覆盖已使用的 4 位说话人、8 句语音、一个模拟房间与两段噪声录音。词错误率在后续阶段另行评估。",
        "", "UTC " + now(),
    ])
    report = ROOT / "docs/G19B_房间MWF信号报告.md"
    with report.open("x", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    archive("signal", report, [EVALUATION / "SIGNAL_COMPLETE.json", provenance, Path(__file__).resolve()])


def main():
    start = time.monotonic()
    previous_archive = OUT / "stages/preflight/archive_manifest.json"
    preflight_hash = sha(previous_archive)
    plan_path = OUT / "frozen_plan.json"
    archived = {r["source"]: r["sha256"] for r in load_json(previous_archive)["files"]}
    require(archived[plan_path.relative_to(ROOT).as_posix()] == sha(plan_path), "Frozen plan hash mismatch")
    plan = load_json(plan_path)
    require(plan["filter_sha256"] == sha(ROOT / "src/mcse/wiener.py") and plan["mu"] == 1.0
            and plan["reference_channel_zero_based"] == REFERENCE, "MWF configuration differs from preflight")
    old_path = OLD_EVALUATION / "combined_signals.jsonl"
    old_hash = load_json(OLD_EVALUATION / "BEAM_COMPLETE.json")["combined_sha256"]
    require(sha(old_path) == old_hash == plan["G18_combined_signals_sha256"], "G18 combined signals hash mismatch")
    cases, frontends, index = load_cases_and_systems(old_path)
    EVALUATION.mkdir(exist_ok=False)
    provenance = EVALUATION / "provenance.json"
    dump(provenance, {
        "started_utc": now(), "preflight_archive_sha256": preflight_hash, "frozen_plan_sha256": sha(plan_path),
        "g18_combined_sha256": old_hash, "g18_manifest_sha256": sha(MANIFEST),
        "g18_systems_sha256": sha(PREV / "systems.json"), "reference_channel_zero_based": REFERENCE,
        "mu": 1.0, "mask_mean_absolute_tolerance": MASK_TOLERANCE, "new_neural_forward": False,
        "device": "cpu", "filter_precision": "float64/complex128", "cpu_workers": WORKERS,
        "torch_and_blas_threads_per_worker": 1, "wiener_sha256": sha(ROOT / "src/mcse/wiener.py"),
        "beamforming_sha256": sha(ROOT / "src/mcse/beamforming.py"),
        "evaluator_sha256": sha(ROOT / "src/mcse/evaluation.py"), "signal_code_sha256": sha(__file__),
        "rms_gain_ratio": "RMS(output)/RMS(reverberant clean reference), uncentered",
        "input_rms_gain_ratio": "RMS(output)/RMS(noisy reference channel 4), uncentered",
        "reference_normalized_squared_error": "sum((output-reference)^2)/sum(reference^2), uncentered",
    }, True)
    new_path = EVALUATION / "mwf_signals.jsonl"
    rows = []
    with new_path.open("x", encoding="utf-8") as log, ProcessPoolExecutor(
        max_workers=WORKERS, initializer=worker_init, mp_context=get_context("spawn")
    ) as pool:
        for frontend in frontends:
            name = frontend + "_mask_mwf"
            (EVALUATION / "mwf_audio" / name).mkdir(parents=True, exist_ok=False)
            for offset in range(0, len(cases), 12):
                if time.monotonic() - start >= plan["max_signal_seconds"]:
                    raise TimeoutError("Signal evaluation exceeded the frozen time budget")
                batch = cases[offset:offset + 12]
                direct = [index[frontend, c["sample_id"]] for c in batch]
                mvdr = [index[frontend + "_mask_mvdr", c["sample_id"]] for c in batch]
                for row in pool.map(filter_and_score, batch, direct, mvdr):
                    log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    rows.append(row)
                log.flush()
                done = offset + len(batch)
                if done % 84 == 0:
                    status(stage="signal", system=name, cases_done=done, total=504, scored_rows=len(rows))
    expected = {(f + "_mask_mwf", c["sample_id"]) for f in frontends for c in cases}
    require(len(rows) == 5040 and {(r["system"], r["sample_id"]) for r in rows} == expected,
            "Incomplete MWF coverage")
    require(not expected.intersection(index) and len(rows) + len(index) == 15624, "Combined signal key conflict")
    require(sha(old_path) == old_hash, "G18 signal manifest changed during signal evaluation")
    combined = EVALUATION / "combined_signals.jsonl"
    with combined.open("xb") as target, old_path.open("rb") as old, new_path.open("rb") as new:
        old.seek(-1, os.SEEK_END)
        require(old.read(1) == b"\n", "G18 manifest has no final newline")
        old.seek(0)
        shutil.copyfileobj(old, target)
        shutil.copyfileobj(new, target)
    require(combined.stat().st_size == old_path.stat().st_size + new_path.stat().st_size,
            "Incomplete concatenation of old and new signals")
    completion = {
        "status": "ok", "completed_utc": now(), "mwf_rows": len(rows), "g18_rows": len(index),
        "combined_rows": len(index) + len(rows), "rows_by_system": dict(Counter(r["system"] for r in rows)),
        "max_mask_mean_abs_diff": max(r["beamforming_diagnostics"]["mask_mean_abs_diff"] for r in rows),
        "elapsed_seconds": time.monotonic() - start, "g18_combined_sha256": old_hash,
        "mwf_sha256": sha(new_path), "combined_sha256": sha(combined),
    }
    dump(EVALUATION / "SIGNAL_COMPLETE.json", completion, True)
    signal_report(rows, completion, provenance)


if __name__ == "__main__":
    main()
