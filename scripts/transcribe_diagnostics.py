#!/usr/bin/env python3
"""Score all diagnostic cases with one fixed local Whisper tiny.en model.

Enhanced-vs-noisy ASR regressions are review candidates, never automatically
confirmed speech deletion. Clean-reference recognizer errors remain visible.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mcse.asr_scoring import ALIGNMENT_TIE_POLICY, NORMALIZATION, corpus_score, score_words

CHECKPOINT_SHA256 = "d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03"
CHECKPOINT_URL = (
    "https://openaipublic.azureedge.net/main/whisper/models/"
    + CHECKPOINT_SHA256 + "/tiny.en.pt"
)
WHISPER_VERSION = "20250625"
ROLES = ("reference", "noisy", "enhanced")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_cases(path: Path) -> list[dict]:
    rows, ids = [], set()
    with path.open(encoding="utf-8") as stream:
        for line_no, text in enumerate(stream, 1):
            if not text.strip():
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise ValueError(f"Cases line {line_no} must be an object")
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id or sample_id in ids:
                raise ValueError(f"Missing or duplicate sample_id at line {line_no}")
            for identifier in ("sample_id", "source_id", "speaker_id"):
                value = row.get(identifier)
                if not isinstance(value, str) or not value.strip() or value != value.strip():
                    raise ValueError(f"Invalid {identifier} at cases line {line_no}")
            ids.add(sample_id)
            row = dict(row)
            row["cases_line"] = line_no
            for role in ROLES:
                key = f"{role}_path"
                if isinstance(row.get(key), str):
                    candidate = Path(row[key])
                    row[key] = str((path.parent / candidate).resolve()
                                   if not candidate.is_absolute() else candidate.resolve())
            rows.append(row)
    if not rows:
        raise ValueError("Cases file is empty")
    return rows


def load_audio(path: Path, role: str, max_seconds: float):
    import numpy as np
    import soundfile as sf
    info = sf.info(path)
    expected_channels = (1, 6) if role == "noisy" else (1,)
    if info.samplerate != 16000 or info.channels not in expected_channels:
        raise ValueError(f"{role}: expected 16kHz and channels {expected_channels}; got {info}")
    if info.frames < 512 or info.frames / 16000 > max_seconds:
        raise ValueError(f"{role}: input length is outside 512 samples to {max_seconds}s")
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    selected_channel = 4 if info.channels == 6 else 0
    mono = np.ascontiguousarray(audio[:, selected_channel])
    if sr != 16000 or not np.isfinite(mono).all():
        raise ValueError(f"{role}: sample rate or finite-value check failed")
    return mono, {"path": str(path), "sha256": sha256(path), "sample_rate": sr,
                  "input_channels": info.channels, "selected_channel_zero_based": selected_channel,
                  "frames": info.frames, "duration_seconds": info.frames / 16000,
                  "abs_peak": float(np.abs(mono).max()),
                  "per_utterance_normalization": False, "resampling": False}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "data/models/whisper/tiny.en.pt")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-seconds", type=float, default=4.0)
    args = parser.parse_args()
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error("--max-seconds must be finite and positive")
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        parser.error("Refusing to overwrite a nonempty output directory")
    return args


def run(args) -> int:
    import numpy as np
    import soundfile
    import torch
    import whisper

    version = importlib.metadata.version("openai-whisper")
    if version != WHISPER_VERSION:
        raise ValueError(f"Expected openai-whisper=={WHISPER_VERSION}, got {version}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Download the fixed official checkpoint first: {args.checkpoint}")
    actual_sha = sha256(args.checkpoint)
    if actual_sha != CHECKPOINT_SHA256:
        raise ValueError("Local tiny.en checkpoint differs from the official fixed SHA256")
    cases_path = args.cases.resolve()
    rows = read_cases(cases_path)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    torch.manual_seed(20261106)
    np.random.seed(20261106)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    # A local path prevents load_model from downloading anything or using a
    # different cache entry. The checkpoint bytes were verified before loading.
    model = whisper.load_model(str(args.checkpoint.resolve()), device=args.device).eval()
    model.requires_grad_(False)
    if args.device == "cuda":
        torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start
    options = {"language": "en", "task": "transcribe", "temperature": 0.0,
               "condition_on_previous_text": False, "initial_prompt": None,
               "beam_size": 1, "word_timestamps": False, "without_timestamps": True,
               "compression_ratio_threshold": 2.4, "logprob_threshold": -1.0,
               "no_speech_threshold": 0.6, "fp16": args.device == "cuda", "verbose": None}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "model": "Whisper tiny.en", "package": "openai-whisper", "package_version": version,
        "model_source": CHECKPOINT_URL, "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": actual_sha, "checkpoint_bytes": args.checkpoint.stat().st_size,
        "license": "MIT (official code and model weights)",
        "license_source": "https://github.com/openai/whisper/blob/main/LICENSE",
        "model_card": "https://github.com/openai/whisper/blob/main/model-card.md",
        "cases_path": str(cases_path), "cases_sha256": sha256(cases_path),
        "device": args.device, "python": sys.version, "torch": torch.__version__,
        "numpy": np.__version__, "soundfile": soundfile.__version__,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "decoding": options, "normalization": NORMALIZATION,
        "alignment_tie_policy": ALIGNMENT_TIE_POLICY, "seed": 20261106,
        "bitwise_gpu_determinism_guaranteed": False,
        "references_used_as_decoding_prompts": False,
        "audio_loading": "16k float32 numpy via SoundFile; no ffmpeg, resampling or peak normalization",
        "model_load_seconds": load_seconds, "training_performed": False,
        "limitations": ["Single recognizer; possible hallucinations and recognizer-specific deletions",
                        "Reference/noisy/enhanced ASR errors do not establish acoustic loss",
                        "Whisper pretraining corpus overlap cannot be completely excluded",
                        "No human listening verdict is assigned by this script"],
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    completed = []
    statuses = Counter()
    with (args.output_dir / "transcriptions.jsonl").open("x", encoding="utf-8") as output:
        for index, row in enumerate(rows, 1):
            result = {k: row.get(k) for k in ("sample_id", "source_id", "speaker_id", "split", "noise_id", "snr_db")}
            result["human_speech_loss_verdict"] = "not_assessed"
            if row.get("status", "ok") != "ok":
                result.update(status="skipped", reason="Upstream case did not complete enhancement",
                              upstream_status=row.get("status"), upstream_error=row.get("error"))
            else:
                try:
                    transcript = row.get("transcript")
                    score_words(transcript, "")  # Validate before model inference, never pass text to model.
                    asr = {}
                    expected_frames = None
                    for role in ROLES:
                        if not isinstance(row.get(f"{role}_path"), str):
                            raise ValueError(f"Missing {role}_path")
                        audio, audio_info = load_audio(Path(row[f"{role}_path"]), role, args.max_seconds)
                        if expected_frames is not None and len(audio) != expected_frames:
                            raise ValueError("Reference/noisy/enhanced audio lengths differ")
                        expected_frames = len(audio)
                        began = time.perf_counter()
                        with torch.inference_mode():
                            decoded = model.transcribe(audio, **options)
                        if args.device == "cuda":
                            torch.cuda.synchronize()
                        hypothesis = decoded["text"]
                        asr[role] = {"audio": audio_info, "hypothesis_raw": hypothesis,
                                     "score": score_words(transcript, hypothesis),
                                     "inference_seconds": time.perf_counter() - began,
                                     "segments": [{k: segment.get(k) for k in
                                                   ("start", "end", "text", "avg_logprob", "no_speech_prob", "compression_ratio")}
                                                  for segment in decoded.get("segments", [])]}
                    noisy, enhanced = asr["noisy"]["score"], asr["enhanced"]["score"]
                    result.update(status="ok", transcript=transcript, asr=asr,
                                  delta_wer_enhanced_minus_noisy=enhanced["wer"] - noisy["wer"],
                                  delta_deletions_enhanced_minus_noisy=enhanced["D"] - noisy["D"],
                                  clean_reference_asr_has_errors=asr["reference"]["score"]["errors"] > 0,
                                  asr_regression_review_candidate=(enhanced["errors"] > noisy["errors"]
                                                                   or enhanced["D"] > noisy["D"]),
                                  candidate_notice="ASR-only screening; waveform/scale/alignment review required")
                    completed.append(result)
                except Exception as error:
                    result.update(status="error", error_type=type(error).__name__, error=str(error))
            statuses[result["status"]] += 1
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            print(json.dumps({"case": index, "total": len(rows), "sample_id": result["sample_id"],
                              "status": result["status"]}, ensure_ascii=False), flush=True)
    per_role = {role: corpus_score(item["asr"][role]["score"] for item in completed) for role in ROLES}
    summary = {"status": "ok" if len(completed) == len(rows) else "incomplete",
               "input_cases": len(rows), "case_status_counts": dict(statuses),
               "completed_cases": len(completed), "unique_source_utterances_input": len({r.get("source_id") for r in rows}),
               "unique_speakers_input": len({r.get("speaker_id") for r in rows}),
               "corpus": per_role,
               "wer_delta_enhanced_minus_noisy": (per_role["enhanced"]["wer"] - per_role["noisy"]["wer"]
                                                     if completed else None),
               "clean_reference_asr_error_cases": sum(r["clean_reference_asr_has_errors"] for r in completed),
               "asr_regression_review_candidates": sum(r["asr_regression_review_candidate"] for r in completed),
               "human_speech_loss_confirmed_cases": None,
               "training_performed": False,
               "denominator_notice": "Corpus editing counts use only matched cases with all three ASR passes; skipped/errors remain listed",
               "evidence_notice": "Machine-recognizer screening only; no acoustic or human-intelligibility verdict",
               "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["status"] == "ok" else 1


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except Exception as error:
        print(json.dumps({"status": "error", "error_type": type(error).__name__, "error": str(error),
                          "training_performed": False, "human_speech_loss_verdict": "not_assessed"},
                         ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
