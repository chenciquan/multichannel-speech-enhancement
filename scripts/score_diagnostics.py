#!/usr/bin/env python3
"""Score existing frozen inference outputs; never rerun or alter a model."""
from __future__ import annotations
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mcse.evaluation import MetricEvaluator, summarize_cases


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import numpy as np
    import soundfile as sf
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Refusing to overwrite existing scoring evidence")
    rows = [json.loads(t) for t in args.cases.read_text().splitlines() if t.strip()]
    evaluator = MetricEvaluator()
    args.output_dir.mkdir(parents=True)
    provenance = {"created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                  "input_cases_sha256": file_hash(args.cases), "input_cases": str(args.cases.resolve()),
                  "metric_backends": evaluator.backend_info,
                  "si_sdr": "zero-mean projection; no delay or filter fitting",
                  "audio_altered": False, "model_rerun": False, "training_performed": False}
    (args.output_dir/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    scores = []
    with (args.output_dir/"scores.jsonl").open("x") as stream:
        for row in rows:
            item = {k:row.get(k) for k in ("sample_id","source_id","speaker_id","noise_id","snr_db")}
            try:
                if row["status"] != "ok":
                    raise ValueError("Original inference case failed")
                ref,sr = sf.read(row["reference_path"],dtype="float64")
                noisy,nsr = sf.read(row["noisy_path"],dtype="float64",always_2d=True)
                enhanced,esr = sf.read(row["enhanced_path"],dtype="float64")
                if sr!=16000 or nsr!=sr or esr!=sr or noisy.shape!=(len(ref),6):
                    raise ValueError("Changed audio metadata")
                if file_hash(row["enhanced_path"])!=row["enhanced_file"]["sha256"]:
                    raise ValueError("Enhanced bytes changed since inference")
                item.update(status="ok", metrics={"noisy":evaluator.evaluate(ref,noisy[:,4],sr),
                                                 "enhanced":evaluator.evaluate(ref,enhanced,sr)},
                            audio_sha256={r:file_hash(row[r+"_path"]) for r in ("reference","noisy","enhanced")})
            except Exception as error:
                item.update(status="error",error_type=type(error).__name__,error=str(error))
            scores.append(item)
            stream.write(json.dumps(item,allow_nan=False)+"\n")
            stream.flush()
    summary=summarize_cases(scores)
    summary.update(status="ok" if summary["failed_cases"]==0 else "incomplete",
                   metric_backends=evaluator.backend_info,training_performed=False)
    (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(json.dumps(summary,indent=2))
    return int(summary["failed_cases"]>0)


if __name__=="__main__":
    raise SystemExit(main())
