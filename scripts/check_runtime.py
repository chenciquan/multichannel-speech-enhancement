#!/usr/bin/env python3
"""Read runtime metadata and optionally execute a tiny CUDA identity product."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys


def inspect_runtime(require_cuda: bool) -> dict:
    report = {"status": "ok", "python": sys.version, "executable": sys.executable,
              "platform": platform.platform(), "training_performed": False,
              "speech_experiment_performed": False, "packages": {}}
    for name in ("torch", "torchaudio", "numpy", "scipy", "soundfile", "pandas",
                 "librosa", "pytorch-lightning", "torchmetrics", "PyYAML"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
    import torch

    report["torch_cuda_runtime"] = torch.version.cuda
    report["cuda_available"] = torch.cuda.is_available()
    if report["cuda_available"]:
        device = torch.device("cuda:0")
        properties = torch.cuda.get_device_properties(device)
        report["gpu"] = {"name": properties.name, "total_memory_bytes": properties.total_memory,
                         "compute_capability": list(torch.cuda.get_device_capability(device))}
        with torch.inference_mode():
            matrix = torch.arange(64, dtype=torch.float32, device=device).reshape(8, 8)
            product = matrix @ torch.eye(8, dtype=torch.float32, device=device)
            torch.cuda.synchronize(device)
            report["tiny_cuda_matrix_check"] = bool(torch.equal(product, matrix))
        if not report["tiny_cuda_matrix_check"]:
            raise RuntimeError("Tiny CUDA identity-product validation failed.")
    else:
        report["gpu"] = None
        report["tiny_cuda_matrix_check"] = None
        if require_cuda:
            raise RuntimeError("CUDA is unavailable but --require-cuda was requested.")
    nvidia_smi = Path("/usr/lib/wsl/lib/nvidia-smi")
    if nvidia_smi.is_file():
        try:
            result = subprocess.run(
                [str(nvidia_smi), "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader"],
                text=True, capture_output=True, timeout=20, check=False)
            report["nvidia_smi"] = {"returncode": result.returncode, "stdout": result.stdout.strip(),
                                    "stderr": result.stderr.strip()}
        except (OSError, subprocess.TimeoutExpired) as error:
            report["nvidia_smi"] = {"error": str(error)}
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Optional JSON output file.")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.report and args.report.exists() and not args.overwrite:
        parser.error(f"Refusing to overwrite {args.report}; use a new path or --overwrite.")
    try:
        report = inspect_runtime(args.require_cuda)
        exit_code = 0
    except Exception as error:
        report = {"status": "error", "python": sys.version, "executable": sys.executable,
                  "error_type": type(error).__name__, "error": str(error),
                  "training_performed": False, "speech_experiment_performed": False}
        exit_code = 1
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
