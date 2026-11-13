#!/usr/bin/env python3
"""Audit the frozen G1 delivery from current bytes; never infer a human verdict."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
from urllib.parse import unquote, urlsplit
from zoneinfo import ZoneInfo

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.audio = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        for key in ("src", "href"):
            if key in attrs:
                self.links.append(attrs[key])
        if tag == "audio":
            self.audio.append(attrs["src"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite a previous audit")
    checks = []

    def check(name, passed, detail=None):
        checks.append({"requirement": name, "passed": bool(passed), "detail": detail})
        if not passed:
            raise ValueError(name)

    evidence = {}
    try:
        base = ROOT/"artifacts/g1_v1"
        dataset = ROOT/"data/diagnostic/g1_v1"
        files = {"manifest": dataset/"manifest.jsonl", "cases": base/"mcnet/cases.jsonl",
                 "asr": base/"asr/transcriptions.jsonl", "scores": base/"signal_scores/scores.jsonl"}
        tables = {name: rows(path) for name, path in files.items()}
        evidence.update({name: {"path": str(path), "sha256": sha(path)} for name, path in files.items()})
        ids = {r["sample_id"] for r in tables["manifest"]}
        check("48 unique cases shared by manifest, inference, ASR and full metrics",
              len(ids) == 48 and all(len(table) == 48 and len({r["sample_id"] for r in table}) == 48
                                    and {r["sample_id"] for r in table} == ids for table in tables.values()))
        check("all inference, ASR and metric cases succeeded",
              all(r["status"] == "ok" for name, table in tables.items() if name != "manifest" for r in table))
        check("24 unique source utterances, six speakers, discovery only",
              len({r["source_id"] for r in tables["manifest"]}) == 24
              and len({r["speaker_id"] for r in tables["manifest"]}) == 6
              and {r["split"] for r in tables["manifest"]} == {"discovery"})
        counts = Counter((r["noise_id"], r["snr_db"]) for r in tables["manifest"])
        check("four noise/SNR cells each contain 12 cases", len(counts) == 4 and set(counts.values()) == {12})
        for name in ("source_manifest", "noise_manifest", "geometry_file"):
            frozen = read(dataset/"frozen_inputs.json")[name]
            check("frozen input bytes: " + name, sha(frozen["path"]) == frozen["sha256"])
        frozen = read(dataset/"frozen_inputs.json")
        for source in frozen["sources"] + frozen["noises"]:
            check("registered source bytes: " + str(source.get("source_id", source.get("noise_id"))),
                  sha(source["path"]) == source["sha256"])
        generated = read(dataset/"summary.json")
        frozen_payload = {k: v for k, v in frozen.items() if k != "frozen_inputs_sha256"}
        frozen_hash = hashlib.sha256(json.dumps(frozen_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        check("generated manifest and canonical frozen-input hashes match current bytes",
              generated["manifest_sha256"] == sha(files["manifest"])
              and generated["frozen_inputs_sha256"] == frozen["frozen_inputs_sha256"] == frozen_hash)
        check("registered frozen protocol is byte-identical",
              (ROOT/"configs/g1_protocol_frozen.json").read_bytes() == (dataset/"protocol.json").read_bytes())
        protocol = read(dataset/"protocol.json")
        canonical = hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        check("canonical protocol matches generation hash", canonical == generated["protocol_sha256"])
        indexed = {name: {r["sample_id"]: r for r in table} for name, table in tables.items()}
        max_duration = 0
        max_peak = 0
        for sid in sorted(ids):
            m, c, a, s = (indexed[name][sid] for name in ("manifest", "cases", "asr", "scores"))
            for field in ("source_id", "speaker_id", "noise_id", "transcript"):
                check(sid + ": shared " + field, m[field] == c[field] == a[field])
            length = None
            for role, channels in (("reference", 1), ("noisy", 6), ("enhanced", 1)):
                path = Path(c[role+"_path"])
                expected = c["enhanced_file"]["sha256"] if role == "enhanced" else m[role+"_sha256"]
                digest = sha(path)
                check(sid + ": " + role + " hash chain",
                      digest == expected == a["asr"][role]["audio"]["sha256"] == s["audio_sha256"][role])
                if role != "enhanced":
                    check(sid + ": " + role + " manifest path", path.resolve() == (dataset/m[role+"_path"]).resolve())
                data, sr = sf.read(path, dtype="float32", always_2d=True)
                length = len(data) if length is None else length
                check(sid + ": " + role + " audio metadata and finite samples",
                      sr == 16000 and data.shape == (length, channels) and np.isfinite(data).all())
                max_duration = max(max_duration, length/sr)
                if role == "enhanced":
                    check(sid + ": FLOAT enhancement", sf.info(path).subtype == "FLOAT")
                    max_peak = max(max_peak, float(np.abs(data).max()))
            for metric in ("si_sdr_db", "stoi", "pesq_wb"):
                check(sid + ": complete real backend metric " + metric,
                      all(np.isfinite(s["metrics"][role][metric]) and s["metrics"][role]["status"][metric]["status"] == "ok"
                          for role in ("noisy", "enhanced")))
        check("all audio within four-second probe; enhanced outputs within full scale", max_duration <= 4 and max_peak <= 1,
              {"max_seconds": max_duration, "max_enhanced_abs_peak": max_peak})
        signal_summary = read(base/"signal_scores/summary.json")
        for metric in ("si_sdr_db", "stoi", "pesq_wb"):
            for role in ("noisy", "enhanced"):
                values = [r["metrics"][role][metric] for r in tables["scores"]]
                expected = signal_summary["metrics"][metric][role]
                check("aggregate " + role + " " + metric, expected["n"] == 48 and np.isclose(np.mean(values), expected["mean"], atol=1e-12, rtol=0))
        asr_summary = read(base/"asr/summary.json")
        for role in ("reference", "noisy", "enhanced"):
            total = {k: sum(r["asr"][role]["score"][k] for r in tables["asr"]) for k in ("N", "S", "D", "I", "errors")}
            expected = asr_summary["corpus"][role]
            check("ASR corpus edits and denominator: " + role,
                  all(total[k] == expected[k] for k in total) and total["N"] == 450
                  and np.isclose(total["errors"]/total["N"], expected["wer"], atol=1e-12, rtol=0), total)
        by_source = {}
        for r in tables["asr"]:
            by_source.setdefault(r["source_id"], []).append(r["asr"]["reference"])
        check("repeated clean-reference hypotheses identical; unique word denominator 225",
              all(len(v) == 2 and v[0]["hypothesis_raw"] == v[1]["hypothesis_raw"] for v in by_source.values())
              and sum(v[0]["score"]["N"] for v in by_source.values()) == 225)
        review = base/"review_v2"
        review_summary = read(review/"summary.json")
        with (review/"review_all_cases.csv").open(encoding="utf-8-sig", newline="") as stream:
            manual = list(csv.DictReader(stream))
        check("initial automatic-review snapshot retains all48 and its original unassessed labels",
              len(manual) == 48 and {r["sample_id"] for r in manual} == ids
              and all(r["human_verdict"] == "not_assessed" and r["listened"] == "False" for r in manual)
              and review_summary["human_judgements_completed"] == 0 and review_summary["confirmed_acoustic_deletions"] is None)
        check("seven ASR regression candidates and one added ASR-deletion candidate",
              sum(r["asr_regression_review_candidate"] for r in tables["asr"]) == len(review_summary["asr_regression_candidates"]) == 7
              and sum(r["delta_deletions_enhanced_minus_noisy"] > 0 for r in tables["asr"]) == len(review_summary["increased_ASR_deletion_candidates"]) == 1)
        links = Links()
        links.feed((review/"index.html").read_text(encoding="utf-8"))
        check("offline review has 144 local audio links, all resources present",
              len(links.audio) == 144 and all(not urlsplit(link).scheme and (review/unquote(link)).is_file() for link in links.links))
        check("all browser playback paths are mono 16k", all(sf.info(review/unquote(link)).channels == 1
              and sf.info(review/unquote(link)).samplerate == 16000 for link in links.audio))
        playback = read(review/"playback_manifest.json")
        check("48 registered reference-microphone listening copies", len(playback) == 48 and {r["sample_id"] for r in playback} == ids)
        for r in playback:
            original, sr = sf.read(r["source_path"], dtype="float32", always_2d=True)
            copy, csr = sf.read(r["path"], dtype="float32")
            check(r["sample_id"] + ": listening copy preserves mic5 exactly",
                  sr == csr == 16000 and sha(r["path"]) == r["sha256"]
                  and sha(r["source_path"]) == r["source_sha256"] and np.array_equal(original[:, 4], copy))
        check("12 selected figures present", len(list((review/"plots").glob("*.png"))) == len(review_summary["plots"]) == 12)
        provenance = read(base/"mcnet/run_provenance.json")
        for filename, metadata in provenance["upstream"]["verified_source_files"].items():
            check("official code unchanged: " + filename, sha(metadata["path"]) == metadata["sha256"])
        cp = provenance["upstream"]["checkpoint"]
        check("official checkpoint unchanged and strict load recorded", sha(cp["path"]) == cp["sha256"] and provenance["model"]["strict_state_dict_load"])
        checkout = subprocess.check_output(["git", "-C", str(ROOT/"third_party/McNet"), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(ROOT/"third_party/McNet"), "status", "--porcelain"], text=True)
        check("official checkout pinned and clean", checkout == provenance["upstream"]["expected_commit"] and not dirty.strip())
        asr_prov = read(base/"asr/provenance.json")
        check("single ASR weight matches registered SHA256", sha(asr_prov["checkpoint_path"]) == asr_prov["checkpoint_sha256"])
        check("inference, ASR, rescoring and review source hashes chained",
              provenance["input_manifest_sha256"] == sha(files["manifest"])
              and asr_prov["cases_sha256"] == sha(files["cases"])
              and read(base/"signal_scores/provenance.json")["input_cases_sha256"] == sha(files["cases"])
              and all(review_summary["evidence_sha256"][k] == sha(files[k]) for k in ("cases", "asr", "scores")))
        check("no training claimed; all human decisions pending", all(not r["training_performed"] for r in
              (provenance, asr_prov, read(base/"signal_scores/provenance.json"), review_summary))
              and all(r["human_speech_loss_verdict"] == "not_assessed" for r in tables["asr"]))
        check("original Chinese stage report remains available and registered",
              (ROOT/"docs/G1_首轮诊断报告.md").is_file()
              and read(ROOT/"configs/project_status.json")["g1_report"] == "docs/G1_首轮诊断报告.md")
        verify = base/"verification"
        test_log = (verify/"unit_tests.log").read_text(encoding="utf-8-sig")
        check("40 unit tests passed with no skips in saved execution evidence",
              read(verify/"unit_tests_result.json")["exit_code"] == 0
              and "Ran 40 tests" in test_log and "\nOK" in test_log and "skipped" not in test_log.lower())
        check("saved dependency consistency and full manifest verification passed",
              read(verify/"dependencies_result.json")["exit_code"] == 0
              and "92 packages" in (verify/"dependencies.log").read_text(encoding="utf-8-sig")
              and read(verify/"manifest_validation.json")["valid"]
              and read(verify/"manifest_validation.json")["audio_checked"]
              and read(verify/"manifest_validation.json")["row_count"] == 48)
        check("G1 dependency lock equals resolved environment snapshot",
              (ROOT/"requirements-diagnostics.lock.txt").read_bytes()
              == (ROOT/"artifacts/environment/requirements-g1-resolved.txt").read_bytes())
        result = "passed"
        error = None
    except Exception as exc:
        result = "failed"
        error = {"type": type(exc).__name__, "message": str(exc)}
    report = {"created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "status": result,
              "scope": "automated G1 evidence delivery, not full G1 problem-confirmation gate",
              "checks": checks, "passed_checks": sum(c["passed"] for c in checks), "error": error,
              "evidence": evidence,
              "not_verified": ["Human listening and word/time adjudication", "Independent confirmation set",
                               "Exact original CHiME-domain performance", "Training feasibility and method gains",
                               "Browser audio playback on a physical listening device"],
              "human_deletions_confirmed": None, "training_performed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result, "passed_checks": report["passed_checks"], "error": error, "output": str(args.output)}, ensure_ascii=False))
    return int(result != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
