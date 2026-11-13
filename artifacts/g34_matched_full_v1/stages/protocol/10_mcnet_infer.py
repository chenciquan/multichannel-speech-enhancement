#!/usr/bin/env python3
"""Run the pinned, unmodified upstream McNet with a Torch 2.x ISTFT adapter.

This is inference infrastructure, not a new enhancement model. Synthetic smoke
input checks loading, tensor shapes and runtime only; it is not a speech test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "9c194032659a3047a4b4c244a9f541f4d34b1682"
OFFLINE_CHECKPOINT = "model_checkpoints/offline/epoch494_criteria18.78_sdr18.78.ckpt"
OFFLINE_CHECKPOINT_BLOB = "b48a7907aaf279d17182782f765c27ac261c137d"
SOURCE_BLOBS = {
    "src/__init__.py": "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391",
    "src/model/__init__.py": "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391",
    "src/util/__init__.py": "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391",
    "src/model/mc_net.py": "4e7e87dc03b598ef01b993f49af5f3fb50ec6fd2",
    "src/util/acoustic_utils.py": "6735bd941cbefb22165f4ad50a44a62f1ee2e6ae",
    "src/util/cal_metrics.py": "e17e276f7b8a249cb4d419097f26e8c7dcf6901d",
    "src/util/git_tools.py": "ec168f73e48add63a164a137f5d1c54b24195c44",
    "src/util/my_json_encoder.py": "a12096ffc08aebc230a0682f3b73cf01b726ef1f",
    "config/mc_net_offline.yaml": "cb197b8155cf5327b01ba9ba4ab6a83ebbbef3f5",
}


def file_hashes(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha256.update(block)
            git_blob.update(block)
    return {"path": str(path.resolve()), "bytes": size,
            "sha256": sha256.hexdigest(), "git_blob_sha1": git_blob.hexdigest()}


def verify_upstream(repo: Path, checkpoint: Path) -> dict[str, Any]:
    """Verify known upstream bytes before importing code or unpickling a model."""
    verified = {}
    for relative_path, expected_blob in SOURCE_BLOBS.items():
        path = repo / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Missing upstream file: {path}")
        entry = file_hashes(path)
        if entry["git_blob_sha1"] != expected_blob:
            raise ValueError(f"Upstream file differs from pinned version: {relative_path}")
        verified[relative_path] = entry
    checkpoint_info = file_hashes(checkpoint)
    if checkpoint_info["git_blob_sha1"] != OFFLINE_CHECKPOINT_BLOB:
        raise ValueError("Checkpoint is not the byte-verified official offline checkpoint; "
                         "refusing unrestricted pickle loading of an unverified file.")
    commit = None
    if (repo / ".git").exists():
        result = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                text=True, capture_output=True, check=True)
        commit = result.stdout.strip()
        if commit != UPSTREAM_COMMIT:
            raise ValueError(f"Unexpected upstream commit: {commit}")
    return {"expected_commit": UPSTREAM_COMMIT, "checkout_commit": commit,
            "verified_source_files": verified, "checkpoint": checkpoint_info,
            "license_status": "No explicit upstream LICENSE found in prior audit; "
                              "use under the separately confirmed local permission."}


def load_model(repo: Path, checkpoint: Path, device: Any):
    import importlib
    import torch
    import yaml

    # Older torchmetrics exports these functions from audio/__init__ only when
    # optional C/metric backends are installed. Bind the actual public functions
    # from their submodules for upstream's import statements. These are NOT
    # substitute metrics: each raises ModuleNotFoundError when called without
    # its real backend. Pure inference does not call either function.
    import torchmetrics.functional.audio as audio_metrics
    import_exports = []
    for module_name, function_name in (
            ("pesq", "perceptual_evaluation_speech_quality"),
            ("stoi", "short_time_objective_intelligibility")):
        if not hasattr(audio_metrics, function_name):
            metric_module = importlib.import_module(f"torchmetrics.functional.audio.{module_name}")
            setattr(audio_metrics, function_name, getattr(metric_module, function_name))
            import_exports.append(function_name)

    # Import the original architecture and original mask decoder, not a copy.
    sys.path.insert(0, str(repo))
    from src.model.mc_net import McNet, RNN_FC, cIRMIO, stft_y_mse
    from src.util.acoustic_utils import icIRM

    config_path = repo / "config/mc_net_offline.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_config = dict(config["model"])
    io_config = model_config.pop("io")
    if io_config["class_path"] != "src.model.mc_net.cIRMIO":
        raise ValueError("Only the pinned cIRMIO configuration is supported.")
    io_args = dict(io_config["init_args"])
    if io_args.pop("loss_func") != "src.model.mc_net.stft_y_mse":
        raise ValueError("Unexpected upstream loss function in offline configuration.")
    if io_args["selected_channels"] != list(range(6)) or io_args["ref_channel"] != 4:
        raise ValueError("This adapter requires the official six-channel, reference-4 model.")
    if io_args["ft_len"] != 512 or io_args["ft_hop"] != 256:
        raise ValueError("Unexpected offline STFT dimensions.")
    io = cIRMIO(loss_func=stft_y_mse, **io_args)
    architecture = {}
    for name in ("freq", "narr", "sub", "full"):
        layer = model_config.pop(name)
        if layer["class_path"] != "src.model.mc_net.RNN_FC":
            raise ValueError(f"Unexpected upstream class for {name}.")
        architecture[name] = RNN_FC(**layer["init_args"])
    model = McNet(io=io, **architecture, **model_config)

    # Lightning's old checkpoint contains more than tensors. This explicit
    # unrestricted load is permitted ONLY after exact official blob verification.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(saved, dict) or not isinstance(saved.get("state_dict"), dict):
        raise ValueError("Expected a Lightning checkpoint containing state_dict.")
    model.load_state_dict(saved["state_dict"], strict=True)
    digest = hashlib.sha256()
    tensor_shapes = {}
    for name, tensor in sorted(saved["state_dict"].items()):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Non-tensor state_dict value: {name}")
        tensor_shapes[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii") + b"\0")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    metadata = {
        "model_config": config["model"],
        "model_config_sha256": hashlib.sha256(json.dumps(
            config["model"], sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "state_dict_shapes": tensor_shapes,
        "state_dict_sha256": digest.hexdigest(),
        "checkpoint_epoch": saved.get("epoch"),
        "checkpoint_global_step": saved.get("global_step"),
        "checkpoint_lightning_version": saved.get("pytorch-lightning_version"),
        "strict_state_dict_load": True,
        "torchmetrics_public_function_exports_added": import_exports,
        "quality_metrics_computed": False,
    }
    del saved
    model = model.to(device).eval().requires_grad_(False)
    return model, icIRM, metadata


def reconstruct_waveform(model: Any, icirm: Any, x: Any, prediction: Any,
                         original_time_len: int):
    """Use original icIRM math; only convert real pairs for Torch >= 2 ISTFT."""
    import torch

    io = model.io
    xr = x[:, io.ref_chn_idx, :]
    reference_stft = torch.stft(xr, n_fft=io.ft_len, hop_length=io.ft_hop,
                                window=io.window, win_length=io.ft_len,
                                return_complex=True)
    # This is the same real/imaginary representation expected by upstream icIRM.
    enhanced_pairs = icirm(pred_cIRM=torch.view_as_real(prediction),
                           noisy_complex_tensor=torch.view_as_real(reference_stft))
    enhanced_stft = torch.view_as_complex(enhanced_pairs.contiguous())
    # Original x is used above: do NOT multiply by XrMM a second time.
    return torch.istft(enhanced_stft, n_fft=io.ft_len, hop_length=io.ft_hop,
                       window=io.window, win_length=io.ft_len,
                       length=original_time_len)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=PROJECT_ROOT / "third_party/McNet")
    parser.add_argument("--checkpoint", type=Path,
                        help="Only the exact official offline checkpoint is accepted.")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", type=Path, help="Six-channel, 16 kHz WAV; no resampling.")
    inputs.add_argument("--smoke-seconds", type=float,
                        help="Deterministic non-speech input; not a fidelity experiment.")
    parser.add_argument("--output", type=Path, help="Output FLOAT WAV; required with --input.")
    parser.add_argument("--report", type=Path, required=True, help="JSON runtime/provenance report.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-seconds", type=float, default=4.0,
                        help="Reject longer audio; only up to 4s has been runtime-probed. No silent chunking.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.input and not args.output:
        parser.error("--output is required when using a real --input file.")
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error("--max-seconds must be finite and positive.")
    if args.smoke_seconds is not None and (
            not math.isfinite(args.smoke_seconds) or args.smoke_seconds <= 0):
        parser.error("--smoke-seconds must be finite and positive.")
    if args.smoke_seconds is not None and args.smoke_seconds > args.max_seconds:
        parser.error("--smoke-seconds must not exceed --max-seconds.")
    for destination in (args.output, args.report):
        if destination is not None and destination.exists() and not args.overwrite:
            parser.error(f"Refusing to overwrite {destination}; choose a new path or --overwrite.")
    protected_paths = {p.resolve() for p in (args.input, args.checkpoint) if p is not None}
    protected_paths.add((args.repo / OFFLINE_CHECKPOINT).resolve())
    if args.output is not None and args.output.resolve() in protected_paths:
        parser.error("Output must not overwrite input or checkpoint.")
    if args.report.resolve() in protected_paths:
        parser.error("Report must not overwrite input or checkpoint.")
    if args.output is not None and args.output.resolve() == args.report.resolve():
        parser.error("Output and report must have different paths.")
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    import importlib.metadata
    import numpy as np
    import soundfile as sf
    import torch

    repo = args.repo.resolve()
    checkpoint = (args.checkpoint or repo / OFFLINE_CHECKPOINT).resolve()
    provenance = verify_upstream(repo, checkpoint)
    sample_rate = 16000
    if args.smoke_seconds is not None:
        samples = round(args.smoke_seconds * sample_rate)
        rng = np.random.default_rng(20261105)
        audio = rng.normal(0, 0.01, size=(samples, 6)).astype(np.float32)
        input_info = {"kind": "synthetic_non_speech_smoke", "seed": 20261105,
                      "human_content_fidelity_experiment": False,
                      "notice": "Numeric/shape/runtime validation only; no speech enhancement claim."}
    else:
        info = sf.info(args.input)
        if info.channels != 6 or info.samplerate != sample_rate:
            raise ValueError(f"Expected six-channel 16000 Hz input, got "
                             f"{info.channels} channels at {info.samplerate} Hz.")
        if info.frames / info.samplerate > args.max_seconds:
            raise ValueError("Input exceeds --max-seconds; no implicit cropping or chunking.")
        audio, actual_sr = sf.read(args.input, dtype="float32", always_2d=True)
        if actual_sr != sample_rate:
            raise ValueError(f"Expected 16000 Hz, got {actual_sr}; no implicit resampling.")
        input_info = {"kind": "user_supplied_audio", "file": file_hashes(args.input),
                      "human_content_fidelity_experiment": "not_established_by_this_script"}
    if audio.ndim != 2 or audio.shape[1] != 6:
        raise ValueError(f"Expected [samples, 6] audio, got {audio.shape}.")
    if audio.shape[0] < 512:
        raise ValueError("Audio must contain at least 512 samples for the pinned STFT.")
    duration = audio.shape[0] / sample_rate
    if duration > args.max_seconds:
        raise ValueError(f"Audio is {duration:.3f}s, exceeds --max-seconds={args.max_seconds}; "
                         "no implicit cropping or chunking.")
    if not np.isfinite(audio).all():
        raise ValueError("Input contains NaN or infinity.")
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; no silent CPU fallback.")
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
    start_load = time.perf_counter()
    model, icirm, model_metadata = load_model(repo, checkpoint, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    model_load_seconds = time.perf_counter() - start_load
    x = torch.from_numpy(np.ascontiguousarray(audio.T)).unsqueeze(0).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_inference = time.perf_counter()
    with torch.inference_mode():
        prediction, data = model(x)
        if prediction.shape != (1, 257, 1 + audio.shape[0] // 256):
            raise ValueError(f"Unexpected compressed-cIRM shape: {tuple(prediction.shape)}")
        waveform = reconstruct_waveform(model, icirm, x, prediction, data["original_time_len"])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_inference
    if waveform.shape != (1, audio.shape[0]) or not torch.isfinite(waveform).all():
        raise ValueError("Output shape/finite-value validation failed.")
    result_audio = waveform[0].cpu().numpy()
    output_info = None
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(args.output, result_audio, sample_rate, format="WAV", subtype="FLOAT")
        output_info = file_hashes(args.output)
    versions = {}
    for name in ("torch", "numpy", "soundfile", "pytorch-lightning", "torchmetrics", "librosa"):
        versions[name] = importlib.metadata.version(name)
    peak = float(np.abs(result_audio).max())
    return {
        "status": "ok", "stage": "G0/G1 runtime or baseline inference only",
        "input": input_info, "output": output_info, "provenance": provenance,
        "model": model_metadata, "versions": versions, "python": sys.version,
        "device": str(device), "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_cuda_runtime": torch.version.cuda,
        "sample_rate_hz": sample_rate, "channels": 6, "reference_channel_zero_based": 4,
        "duration_seconds": duration, "samples": audio.shape[0], "dtype": str(x.dtype),
        "stft": {"n_fft": 512, "hop_length": 256, "window": "periodic Hann",
                 "center": True, "frequencies": 257},
        "compatibility_adapter": "Original McNet forward + original icIRM decoder; "
                                 "real/imag pairs converted to complex before Torch 2.x ISTFT.",
        "upstream_source_edited": False, "training_performed": False,
        "resampling_or_chunking": False, "per_utterance_peak_normalization": False,
        "output_format": "WAV/FLOAT" if output_info else None,
        "output_abs_peak": peak, "output_samples_above_one": int(np.count_nonzero(np.abs(result_audio) > 1)),
        "input_abs_peak": float(np.abs(audio).max()),
        "output_rms": float(np.sqrt(np.mean(result_audio.astype(np.float64) ** 2))),
        "model_load_seconds": model_load_seconds, "inference_seconds": elapsed,
        "rtf_including_stft_istft_excluding_load_and_disk_io": elapsed / duration,
        "cuda_peak_allocated_bytes_including_model_load": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "cuda_peak_reserved_bytes_including_model_load": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
        "timing_notice": "Single cold inference; not a warmed throughput benchmark or a streaming latency measurement.",
    }


def main() -> int:
    args = parse_args()
    try:
        report = run(args)
        exit_code = 0
    except Exception as error:
        report = {"status": "error", "error_type": type(error).__name__, "error": str(error),
                  "training_performed": False,
                  "notice": "Failure is not an experimental speech-quality result."}
        exit_code = 1
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(args.report.resolve()),
                      **({"error": report["error"]} if exit_code else {
                          "device": report["device"], "seconds": report["duration_seconds"],
                          "inference_seconds": report["inference_seconds"],
                          "parameter_count": report["model"]["parameter_count"],
                          "cuda_peak_allocated_bytes": report["cuda_peak_allocated_bytes_including_model_load"]})},
                     ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
