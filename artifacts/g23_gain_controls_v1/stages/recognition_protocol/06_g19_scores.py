#!/usr/bin/env python3
"""Score the ten new MWF systems with the two fixed G18 recognizers."""
import os

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

from collections import Counter, defaultdict
import gc
import hashlib
from importlib.metadata import version
from io import BytesIO
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch

import g8_common
from g19_common import ROOT, OUT, PREV, now, sha, read, dump, status, archive
from mcse.asr_scoring import NORMALIZATION, score_words
from mcse.ctc import greedy_ctc_tokens


FIELDS = (
    "dataset", "system", "sample_id", "source_id", "speaker_id", "condition",
    "target_azimuth", "relative_angle", "noise_id", "snr_db", "transcript",
    "requested_rt60", "geometry_protocol", "geometry_pair", "actual_relative_azimuth",
    "noise_field",
)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def json_lines(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def file_record(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)}


def load_cache(engine, prior, directory):
    """Replay the checksum-verified chain in G18 order and append G18 last."""
    sources = [*prior["source_caches"], file_record(directory / "cache_new.jsonl")]
    cache, conflicts = {}, []
    for source in sources:
        path = ROOT / source["path"]
        assert sha(path) == source["sha256"], path
        for row in json_lines(path):
            identity = row.get("model_id", row.get("recognizer_id"))
            if engine == "CTC" or path.name == "cache_new.jsonl":
                assert identity == prior["model_id"], path
            key, hypothesis = row["sample_sha256"], row["hypothesis"]
            kept = cache.setdefault(key, hypothesis)
            if kept != hypothesis:
                assert engine == "Whisper", f"CTC cache conflict for {key}"
                conflicts.append({"sample_sha256": key, "kept": kept, "later": hypothesis})
    return cache, conflicts, sources


def fixed_decoder(engine, prior):
    model_id, config = prior["model_id"], prior["decoding"]
    assert prior["normalization"] == NORMALIZATION and prior["reference_prompt"] is False
    seed = 20261106 if engine == "Whisper" else 20261112
    torch.manual_seed(seed)
    np.random.seed(seed)
    if engine == "Whisper":
        import whisper
        original = load_json(ROOT / "artifacts/g3_overnight_v1/evaluation/asr_provenance.json")
        checkpoint = ROOT / "data/models/whisper/tiny.en.pt"
        assert version("openai-whisper") == original["version"] == "20250625"
        assert sha(checkpoint) == original["checkpoint_sha256"] == model_id
        assert config == {"model_sha256": model_id, "options": original["options"]}
        model = whisper.load_model(str(checkpoint), device="cuda").eval().requires_grad_(False)

        @torch.inference_mode()
        def decode(samples):
            return model.transcribe(samples, **config["options"])["text"]
    else:
        base = ROOT / "artifacts/g8_cross_asr_v1"
        model_identity = base / "asr_provenance.json"
        assert sha(model_identity) == model_id
        origin = load_json(model_identity)
        assert sha(base / "asset_provenance.json") == origin["asset_provenance_sha256"]
        manifest = base / "stages/asset/archive_manifest.json"
        assert sha(manifest) == origin["asset_archive_sha256"]
        files = {r["source"]: r["sha256"] for r in load_json(manifest)["files"]}
        for name in ("src/mcse/ctc.py", "scripts/g8_common.py"):
            assert sha(ROOT / name) == files[name]
        assert config == {"previous_recognizer_id": model_id, "batch": 1, "FP32": True,
                          "normalization": False, "resampling": False, "external_LM": False,
                          "greedy_CTC": True}
        model, labels = g8_common.model_load()

        @torch.inference_mode()
        def decode(samples):
            emission, _ = model(torch.from_numpy(samples)[None].cuda())
            assert torch.isfinite(emission).all() and emission.shape[-1] == 29
            return greedy_ctc_tokens(emission[0].argmax(-1).cpu().tolist(), labels)
    return decode


def baseline_references(path, cache):
    references = {}
    for row in json_lines(path):
        assert cache[row["output_samples_sha256"]] == row["hypothesis"]
        key = row["reference_samples_sha256"]
        assert cache[key] == row["reference_hypothesis"]
        if row["system"] == "noisy":
            assert row["sample_id"] not in references
            references[row["sample_id"]] = (key, row["transcript"])
    assert len(references) == 504
    return references


def waveform(path, expected_sha):
    contents = Path(path).read_bytes()
    assert hashlib.sha256(contents).hexdigest() == expected_sha, path
    samples, rate = sf.read(BytesIO(contents), dtype="float32")
    assert rate == 16000 and samples.ndim == 1 and np.isfinite(samples).all(), path
    samples = np.ascontiguousarray(samples)
    return samples, hashlib.sha256(samples.tobytes()).hexdigest()


def coverage_report(engine, folder, totals, new, conflicts):
    lines = [
        f"# G19 {engine} 识别覆盖", "",
        f"固定识别器覆盖 10 个 MWF 系统，各 504 个 G18 条件。新增解码 {new} 条，"
        f"历史缓存中同一样本的识别文本冲突 {len(conflicts)} 条。参考波形逐例匹配 G18 样本哈希，"
        "参考转写复用相同识别器的缓存。G18 基线评分文件及其 SHA-256 见 provenance.json。", "",
        "|系统|条件数|参考词数|替换|删除|插入|微平均 WER|", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, t in sorted(totals.items()):
        wer = (t["S"] + t["D"] + t["I"]) / t["N"]
        lines.append(f'|{name}|{t["rows"]}|{t["N"]}|{t["S"]}|{t["D"]}|{t["I"]}|{wer:.1%}|')
    lines.extend(["", "各行合并所有房间条件，仅供核对覆盖。逐例日志保留全部词对齐和未截断的 WER。", "", now()])
    report = ROOT / "docs" / f"G19_{engine}识别覆盖.md"
    with report.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    archive(engine.lower(), report, [folder / "provenance.json", folder / "COMPLETE.json", Path(__file__).resolve()])


def score_backend(engine, signals, signal_file, plan, deadline):
    folder = OUT / "evaluation" / engine
    previous_stage = "signal" if engine == "Whisper" else "whisper"
    preceding = OUT / "stages" / previous_stage / "archive_manifest.json"
    previous_archive_sha = sha(preceding)
    baseline = PREV / "evaluation" / engine
    baseline_scores = file_record(baseline / "asr_scores.jsonl")
    baseline_provenance = file_record(baseline / "provenance.json")
    assert baseline_scores["sha256"] == plan["G18_ASR_score_sha256"][engine]
    assert baseline_provenance["sha256"] == plan["G18_ASR_provenance_sha256"][engine]
    prior = load_json(baseline / "provenance.json")
    assert prior["engine"] == engine
    cache, conflicts, sources = load_cache(engine, prior, baseline)
    references = baseline_references(baseline / "asr_scores.jsonl", cache)
    pairs = {(r["system"], r["sample_id"]) for r in signals}
    systems = {r["system"] for r in signals}
    assert len(systems) == 10 and pairs == {(s, c) for s in systems for c in references}
    folder.mkdir(exist_ok=False)
    provenance = {
        "started_utc": now(), "engine": engine, "model_id": prior["model_id"],
        "decoding": prior["decoding"], "source_caches": sources,
        "source_precedence": "G18 source_caches in listed order, followed by G18 cache_new; first observation retained",
        "cache_hypothesis_conflicts": conflicts, "normalization": NORMALIZATION, "reference_prompt": False,
        "signal_archive_sha256": sha(OUT / "stages/signal/archive_manifest.json"),
        "previous_archive_sha256": previous_archive_sha, "mwf_signals": file_record(signal_file),
        "G18_scores": baseline_scores, "G18_provenance": baseline_provenance,
    }
    dump(folder / "provenance.json", provenance, True)
    decode = fixed_decoder(engine, prior)
    sample_keys, totals = {}, defaultdict(Counter)
    new = 0
    with (folder / "cache_new.jsonl").open("x", encoding="utf-8") as log, \
         (folder / "asr_scores.jsonl").open("x", encoding="utf-8") as scored:
        for i, row in enumerate(signals, 1):
            if time.monotonic() > deadline:
                raise TimeoutError(f"G19 ASR time budget reached during {engine}")
            ref_path, path = row["reference_path"], row["output_path"]
            if ref_path not in sample_keys:
                _, sample_keys[ref_path] = waveform(ref_path, row["reference_sha256"])
            ref_key = sample_keys[ref_path]
            assert (ref_key, row["transcript"]) == references[row["sample_id"]]
            samples, key = waveform(path, row["output_sha256"])
            sample_keys[path] = key
            if key not in cache:
                started = time.perf_counter()
                cache[key] = decode(samples)
                record = {"sample_sha256": key, "hypothesis": cache[key],
                          "seconds": time.perf_counter() - started, "model_id": prior["model_id"]}
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
                new += 1
            result = {field: row.get(field) for field in FIELDS}
            score = score_words(row["transcript"], cache[key])
            result.update(hypothesis=cache[key], score=score, reference_hypothesis=cache[ref_key],
                          reference_score=score_words(row["transcript"], cache[ref_key]),
                          output_samples_sha256=key, reference_samples_sha256=ref_key)
            scored.write(json.dumps(result, ensure_ascii=False) + "\n")
            t = totals[row["system"]]
            t.update({field: score[field] for field in ("N", "S", "D", "I")})
            t["rows"] += 1
            if i % 100 == 0:
                status(stage="asr", engine=engine, rows_done=i, rows_total=len(signals), new_decodes=new)
    dump(folder / "COMPLETE.json", {"completed_utc": now(), "rows": len(signals),
         "unique_paths": len(sample_keys), "new_decodes": new, "cache_entries": len(cache),
         "asr_scores_sha256": sha(folder / "asr_scores.jsonl"), "cache_new_sha256": sha(folder / "cache_new.jsonl")}, True)
    coverage_report(engine, folder, totals, new, conflicts)


def main():
    plan = load_json(OUT / "frozen_plan.json")
    deadline = time.monotonic() + plan["max_ASR_seconds"] - 240
    signal_file = OUT / "evaluation/mwf_signals.jsonl"
    signal_archive = load_json(OUT / "stages/signal/archive_manifest.json")
    complete = OUT / "evaluation/SIGNAL_COMPLETE.json"
    entry = next(r for r in signal_archive["files"] if Path(r["source"]).name == complete.name)
    assert sha(complete) == entry["sha256"]
    marker = load_json(complete)
    assert sha(signal_file) == marker["mwf_sha256"]
    assert marker["g18_combined_sha256"] == plan["G18_combined_signals_sha256"]
    signals = read(signal_file)
    assert len(signals) == 5040 and all(r["status"] == "ok" for r in signals)
    systems = load_json(PREV / "systems.json")["beam"]
    assert {r["system"] for r in signals} == {r["mask_system"] + "_mask_mwf" for r in systems}
    torch.set_num_threads(2)
    for engine in ("Whisper", "CTC"):
        score_backend(engine, signals, signal_file, plan, deadline)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
