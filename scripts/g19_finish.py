"""Read back G19 files, audit paired observations, and report the frozen comparisons."""
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf

from g19_common import ROOT, OUT, PREV, archive, dump, now, read, sha, status
from mcse.asr_scoring import corpus_score, score_words


EV = OUT / "evaluation"
OLD = PREV / "evaluation"
ENGINES = ("Whisper", "CTC")
QUALITY = ("si_sdr_db", "pesq_wb", "stoi")
LEVELS = ("rms_gain_ratio", "reference_normalized_squared_error")
METHODS = {"direct": "", "mvdr": "_mask_mvdr", "mwf": "_mask_mwf"}
GROUP_NAMES = {"mcnet": "原始 McNet", "fixed": "固定训练", "counter": "反向训练", "rigid": "刚性训练"}
REPORT = ROOT / "docs/G19_多通道Wiener与MVDR对照报告.md"


def check(condition, message):
    if not condition:
        raise ValueError(message)


def metadata(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_hash(path, expected):
    check(sha(path) == expected, f"SHA-256 mismatch at {path}")


def verify_archives():
    hashes = {}
    for stage in ("preflight", "signal", "whisper", "ctc"):
        manifest = OUT / "stages" / stage / "archive_manifest.json"
        record = metadata(manifest)
        for entry in record["files"]:
            check_hash(manifest.parent / entry["snapshot"], entry["sha256"])
            check_hash(ROOT / entry["source"], entry["sha256"])
        hashes[stage] = sha(manifest)
    check(metadata(EV / "provenance.json")["preflight_archive_sha256"] == hashes["preflight"],
          "Signal provenance does not refer to the preflight archive")
    return hashes


def check_concatenation(merged, parts):
    with merged.open("rb") as stream:
        for path in parts:
            with path.open("rb") as part:
                for block in iter(lambda: part.read(1024 * 1024), b""):
                    check(block == stream.read(len(block)), f"Combined signal bytes differ from {path}")
        check(not stream.read(1), "Unexpected bytes at end of combined signal file")


def read_audio(path, expected_hash, frames):
    content = Path(path).read_bytes()
    check(hashlib.sha256(content).hexdigest() == expected_hash, f"Audio file hash mismatch at {path}")
    audio, rate = sf.read(io.BytesIO(content), dtype="float32")
    check(rate == 16000 and audio.shape == (frames,) and np.isfinite(audio).all(), f"Invalid mono WAV {path}")
    return audio, hashlib.sha256(audio.tobytes()).hexdigest()


def check_signals(plan, cases, frontends):
    complete = metadata(EV / "SIGNAL_COMPLETE.json")
    baseline, added, merged = OLD / "combined_signals.jsonl", EV / "mwf_signals.jsonl", EV / "combined_signals.jsonl"
    baseline_hash = plan["G18_combined_signals_sha256"]
    check_hash(baseline, baseline_hash)
    check(complete["g18_combined_sha256"] == baseline_hash, "Historical signal manifest identity mismatch")
    check_hash(added, complete["mwf_sha256"])
    check_hash(merged, complete["combined_sha256"])
    check_concatenation(merged, (baseline, added))
    rows = read(merged)
    signals = {(r["system"], r["sample_id"]): r for r in rows}
    systems = {"noisy"} | {f + suffix for f in frontends for suffix in METHODS.values()}
    expected = {(name, sample) for name in systems for sample in cases}
    check(len(rows) == len(signals) == plan["expected_total_rows"] == 15624 and set(signals) == expected,
          "Expected 31 systems with exactly 504 signal keys each")
    check(complete["mwf_rows"] == plan["expected_new_rows"] == 5040 and complete["g18_rows"] == 10584,
          "Unexpected old or new signal counts")

    ref_keys, clean_refs = {}, {}
    for sample, case in cases.items():
        check_hash(case["noisy_path"], case["input_sha256"])
        ref, ref_keys[sample] = read_audio(case["reference_path"], case["reference_sha256"], case["frames"])
        if case["condition"] == "clean":
            clean_refs[sample] = ref.astype(np.float64)

    sample_keys, levels = {}, {}
    for key, row in signals.items():
        sample = row["sample_id"]
        check(all(row[k] == v for k, v in cases[sample].items()), f"Altered case metadata at {key}")
        scores = row["metrics"]
        check(row["status"] == "ok" and all(scores["status"][m]["status"] == "ok" and
              np.isfinite(scores[m]) for m in QUALITY), f"Missing signal score at {key}")
        output, sample_keys[key] = read_audio(row["output_path"], row["output_sha256"], row["frames"])
        if sample in clean_refs:
            ref = clean_refs[sample]
            y = output.astype(np.float64)
            energy = np.sum(ref ** 2)
            check(energy > 0, f"Zero-energy clean reference at {key}")
            levels[key] = dict(zip(LEVELS, (
                float(np.sqrt(np.sum(y ** 2) / energy)), float(np.sum((y - ref) ** 2) / energy))))
            if row["system"].endswith("_mask_mwf"):
                check(all(np.isclose(levels[key][m], row["level_diagnostics"][m], rtol=1e-10, atol=1e-12)
                          for m in LEVELS), f"Independent clean level readback mismatch at {key}")
    max_difference = 0.0
    for frontend in frontends:
        for sample in cases:
            direct = signals[frontend, sample]
            mvdr = signals[frontend + METHODS["mvdr"], sample]
            mwf = signals[frontend + METHODS["mwf"], sample]
            for filtered in (mvdr, mwf):
                diagnostics = filtered["beamforming_diagnostics"]
                check(filtered["source_system"] == frontend and filtered["mask_estimate_path"] == direct["output_path"]
                      and filtered["mask_estimate_sha256"] == direct["output_sha256"]
                      and diagnostics["oracle_clean_reference"] is False and diagnostics["oracle_DOA"] is False,
                      f"Unpaired estimate or oracle at {filtered['system']}/{sample}")
            info = mwf["beamforming_diagnostics"]
            old_mean = mvdr["beamforming_diagnostics"]["mask_mean"]
            difference = abs(info["mask_mean"] - old_mean)
            check(info["reference_channel_zero_based"] == plan["reference_channel_zero_based"] == 4
                  and info["mu"] == plan["mu"] == 1 and info["g18_mvdr_mask_mean"] == old_mean
                  and info["mask_mean_abs_diff"] == difference and difference <= 1e-12,
                  f"Unmatched MWF and MVDR mask at {frontend}/{sample}")
            max_difference = max(max_difference, difference)
    return signals, ref_keys, sample_keys, levels, {
        "combined_sha256": complete["combined_sha256"], "mwf_sha256": complete["mwf_sha256"],
        "g18_combined_sha256": baseline_hash, "matched_estimate_pairs": 5040,
        "max_mask_mean_abs_diff": max_difference, "mixture_files_checked": len(cases),
        "reference_files_checked": len(ref_keys), "output_files_checked": len(sample_keys),
        "clean_outputs_independently_measured": len(levels),
    }


def check_asr(engine, plan, archives, signals, ref_keys, sample_keys):
    folder, old_folder = EV / engine, OLD / engine
    previous = "signal" if engine == "Whisper" else "whisper"
    provenance = metadata(folder / "provenance.json")
    prior = metadata(old_folder / "provenance.json")
    check_hash(old_folder / "provenance.json", plan["G18_ASR_provenance_sha256"][engine])
    check(provenance["previous_archive_sha256"] == archives[previous] and
          all(provenance[k] == prior[k] for k in ("engine", "model_id", "decoding", "normalization", "reference_prompt")),
          f"{engine} recognizer provenance mismatch")
    combined = {}
    baseline_refs = {}
    old_hash = plan["G18_ASR_score_sha256"][engine]
    complete = metadata(folder / "COMPLETE.json")
    for new, directory, expected_hash in ((False, old_folder, old_hash), (True, folder, complete["asr_scores_sha256"])):
        path = directory / "asr_scores.jsonl"
        check_hash(path, expected_hash)
        count = 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                name, sample = key = (row["system"], row["sample_id"])
                r = signals[key]
                check(key not in combined and name.endswith("_mask_mwf") == new and
                      all(row[k] == r[k] for k in ("transcript", "source_id", "speaker_id", "condition",
                          "target_azimuth", "relative_angle", "requested_rt60", "noise_id", "noise_field"))
                      and row["output_samples_sha256"] == sample_keys[key]
                      and row["reference_samples_sha256"] == ref_keys[sample], f"{engine} score key mismatch {key}")
                words = row["transcript"]
                if not new and name == "noisy":
                    baseline_refs[sample] = row["reference_hypothesis"]
                if new:
                    check(row["score"] == score_words(words, row["hypothesis"]) and
                          row["reference_score"] == score_words(words, row["reference_hypothesis"])
                          and row["reference_hypothesis"] == baseline_refs[sample], f"{engine} rescore failed at {key}")
                combined[key] = {k: row["score"][k] for k in ("N", "S", "D", "I", "correct")}
                count += 1
        check(count == (5040 if new else 10584), f"Wrong {engine} score count in {path}")
    check(set(combined) == set(signals), f"Incomplete {engine} score coverage")
    return combined, {"rows": len(combined), "baseline_sha256": old_hash, "new_sha256": complete["asr_scores_sha256"],
                      "new_output_and_reference_alignments_recomputed": 5040}


def analyze(cases, frontends, signals, asr, levels):
    def aggregate(keys):
        rows = [signals[key] for key in keys]
        result = {"rows": len(keys), "case_conditions": len({s for _, s in keys}),
                  "speakers": len({r["speaker_id"] for r in rows}),
                  "utterances": len({r["source_id"] for r in rows}),
                  "metrics": {m: float(np.mean([r["metrics"][m] for r in rows])) for m in QUALITY}}
        result.update({engine: corpus_score(asr[engine][k] for k in keys) for engine in ENGINES})
        if all(k in levels for k in keys):
            result["levels"] = {m: float(np.mean([levels[k][m] for k in keys])) for m in LEVELS}
        return result

    def methods(pairs):
        values = {m: aggregate([(f + suffix, s) for f, s in pairs]) for m, suffix in METHODS.items()}
        mvdr, mwf = values["mvdr"], values["mwf"]
        delta = {m: mwf["metrics"][m] - mvdr["metrics"][m] for m in QUALITY}
        for e in ENGINES:
            check(mvdr[e]["N"] == mwf[e]["N"] == values["direct"][e]["N"], f"Unequal {e} paired word counts")
            delta[e + "_wer_pp"] = 100 * (mwf[e]["wer"] - mvdr[e]["wer"])
        if "levels" in mwf:
            delta.update({m: mwf["levels"][m] - mvdr["levels"][m] for m in LEVELS})
        return {**values, "mwf_minus_mvdr": delta}

    # Each entry in a bucket is one frontend and one frozen case, shared by all three methods.
    dimensions = {
        "cells": ("rt", "noise_id", "azimuth", "group"), "by_rt": ("rt", "group"),
        "per_seed": ("rt", "group", "frontend", "seed"),
        "per_speaker": ("rt", "group", "speaker"), "per_field": ("rt", "group", "field"),
        "clean": ("rt", "group"), "clean_by_seed": ("rt", "group", "frontend", "seed"),
    }
    buckets = {section: defaultdict(list) for section in dimensions}
    for frontend, system in frontends.items():
        for sample, case in cases.items():
            field = case["noise_field"]
            fields = dict(rt=case["requested_rt60"], noise_id=case["noise_id"], azimuth=case["target_azimuth"],
                          group=system["group"], frontend=frontend, seed=system.get("seed"),
                          speaker=case["speaker_id"], field=f"point_{case['relative_angle']}" if field == "point" else field)
            for section, labels in dimensions.items():
                if (field == "clean") == section.startswith("clean"):
                    buckets[section][tuple(fields[k] for k in labels)].append((frontend, sample))
    result = {section: [{**dict(zip(dimensions[section], key)), **methods(pairs)} for key, pairs in sorted(groups.items())]
              for section, groups in buckets.items()}
    check(len(result["cells"]) == 72 and len(result["clean"]) == len(result["by_rt"]) == 12,
          "Incomplete RT/noise/angle/frontend stratification")
    result["noisy_channel4"] = []
    for rt in sorted({c["requested_rt60"] for c in cases.values()}):
        for clean in (False, True):
            keys = [("noisy", sample) for sample, c in cases.items()
                    if c["requested_rt60"] == rt and (c["condition"] == "clean") == clean]
            result["noisy_channel4"].append({"rt": rt, "clean": clean, **aggregate(keys)})
    return result


def report_text(summary):
    groups = summary["by_rt"]
    lines = [
        "# G19 单房间多通道 Wiener 与 Souden MVDR 配对实验", "",
        "固定 10 个前端，在相同 504 个房间条件上比较直接输出、Souden MVDR 和 μ=1 的多通道 Wiener 滤波（MWF）。"
        "两种滤波器取同一神经估计、带噪六通道、STFT 与参考麦克风（从零计数为 4）。MWF 以共同帧数归一化两个互补掩码协方差，"
        "MVDR 分别按各掩码时间和归一化，因此差值涵盖协方差估计和滤波求解。", "",
        "8 句语音来自先前已使用的 4 位说话人。模拟覆盖一个固定房间、两段噪声录音、三个目标方向，带噪语音活动段 SNR 为 0 dB。"
        "目标是参考麦上的混响干净语音，请求 RT 表示生成器参数。原始 McNet 每格有一个前端，固定、反向、刚性训练策略各保留三个种子。", "",
        "## 带噪输入的配对结果", "",
        "每行合并两段录音、三个方向和三种空间场。三项按直接输出 / MVDR / MWF 排列，WER 由替换、删除、插入总数除以参考词总数计算。",
        "", "|请求 RT (s)|前端|配对数|SI-SDR (dB)|Whisper WER (%)|CTC WER (%)|",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in groups:
        columns = [" / ".join(f"{row[m]['metrics']['si_sdr_db']:.2f}" for m in METHODS)]
        columns.extend(" / ".join(f"{100 * row[m][e]['wer']:.1f}" for m in METHODS) for e in ENGINES)
        lines.append(f"|{row['rt']}|{GROUP_NAMES[row['group']]}|{row['mwf']['rows']}|" + "|".join(columns) + "|")
    cells = summary["cells"]
    positive = sum(r["mwf_minus_mvdr"]["si_sdr_db"] > 0 for r in cells)
    fewer = sum(all(r["mwf_minus_mvdr"][e + "_wer_pp"] < 0 for e in ENGINES) for r in cells)
    opposite = sum(r["mwf_minus_mvdr"]["Whisper_wer_pp"] * r["mwf_minus_mvdr"]["CTC_wer_pp"] < 0 for r in cells)
    lines.extend(["", f"细分为请求 RT × 录音 × 方向 × 前端类别后的 72 格中，MWF 的 SI-SDR 高于 MVDR 的有 {positive} 格，"
                  f"两种识别器 WER 同时下降的有 {fewer} 格，两个 WER 差值符号相反的有 {opposite} 格。各格共享说话人和语句。", "",
                  "下表为 MWF 减 MVDR。每格从左到右列出 SI-SDR 差（dB）及 Whisper、CTC 的 WER 差（百分点）。"
                  "SI-SDR 正值及 WER 负值分别代表相应指标的改善。", "",
                  "|请求 RT (s)|录音|前端|目标 −40°|目标 20°|目标 70°|", "|---:|---|---|---:|---:|---:|"])
    indexed = {(r["rt"], r["noise_id"], r["group"], r["azimuth"]): r for r in cells}
    for rt, noise, group in sorted({k[:3] for k in indexed}):
        columns = []
        for angle in (-40, 20, 70):
            d = indexed[rt, noise, group, angle]["mwf_minus_mvdr"]
            columns.append(f"{d['si_sdr_db']:+.2f} / {d['Whisper_wer_pp']:+.1f} / {d['CTC_wer_pp']:+.1f}")
        lines.append(f"|{rt}|{noise}|{GROUP_NAMES[group]}|" + "|".join(columns) + "|")
    lines.extend(["", "## 带噪质量及干净控制", "",
                  "每一行的两项数值按 MVDR / MWF 排列。带噪 PESQ-WB 和 STOI 是逐例算术平均。"
                  "干净列用相同参考和输出重新读取的波形计算。增益为 RMS(输出)/RMS(参考)，1 表示电平相同。"
                  "NMSE 为 Σ(输出−参考)²/Σ参考²，0 表示波形相同。两者均为逐例计算的无量纲线性比值，再在格内取平均。",
                  "", "|请求 RT (s)|前端|带噪 PESQ-WB|带噪 STOI|干净配对数|干净 RMS 增益|干净 NMSE|",
                  "|---:|---|---:|---:|---:|---:|---:|"])
    noisy = {(r["rt"], r["group"]): r for r in groups}
    for clean in summary["clean"]:
        r = noisy[clean["rt"], clean["group"]]
        columns = [" / ".join(f"{r[m]['metrics'][k]:.3f}" for m in ("mvdr", "mwf")) for k in ("pesq_wb", "stoi")]
        columns.append(str(clean["mwf"]["rows"]))
        columns.extend(" / ".join(f"{clean[m]['levels'][k]:.4f}" for m in ("mvdr", "mwf")) for k in LEVELS)
        lines.append(f"|{clean['rt']}|{GROUP_NAMES[clean['group']]}|" + "|".join(columns) + "|")
    lines.extend(["", "带噪 SI-SDR、PESQ 和 STOI 表示整段输出与目标的接近程度。干净条件可观察语音电平和波形失真。"
                  "本轮未单独输出语音分量与噪声分量，因而带噪输出的变化仍混合了目标失真与残余噪声。"
                  "逐词计数来自固定英语识别器。真实阵列、多房间、其他说话人及人耳感知需要另行验证。", "",
                  "完整分格、逐种子、逐说话人、逐场型的三种输出指标、双后端 S/D/I、参考词数和配对差值见"
                  "[机器汇总](../artifacts/g19_room_mwf_v1/evaluation/summary.json)。"
                  "旧信号清单前缀与 G18 逐字节相同，各后端保留其原始 10,584 条评分文件，并按键并入 5,040 条 MWF 评分。"
                  "逐项验收记录见[审计](../artifacts/g19_room_mwf_v1/completion_audit.json)。", "", "UTC " + now()])
    return "\n".join(lines) + "\n"


def main():
    started = time.monotonic()
    plan = metadata(OUT / "frozen_plan.json")
    archives = verify_archives()
    check_hash(ROOT / "src/mcse/wiener.py", plan["filter_sha256"])
    g18_provenance = metadata(OLD / "provenance.json")
    manifest = ROOT / "data/g18_v1/manifest.jsonl"
    check_hash(manifest, g18_provenance["manifest_sha256"])
    check_hash(PREV / "systems.json", g18_provenance["systems_sha256"])
    case_rows = read(manifest)
    cases = {r["sample_id"]: r for r in case_rows}
    check(len(case_rows) == len(cases) == 504 and sum(c["condition"] == "clean" for c in case_rows) == 72
          and len({c["speaker_id"] for c in case_rows}) == 4 and len({c["source_id"] for c in case_rows}) == 8,
          "Unexpected room corpus coverage")
    registry = metadata(PREV / "systems.json")
    groups = {"mcnet": "mcnet", "fixed": "fixed", "rotated": "counter", "rigid": "rigid"}
    frontends = {r["system"]: {**r, "group": groups[r.get("geometry", r["kind"])]}
                 for r in registry["direct"] if r["kind"] != "noisy"}
    check(Counter(r["group"] for r in frontends.values()) == {"mcnet": 1, "fixed": 3, "counter": 3, "rigid": 3}
          and {r["mask_system"] for r in registry["beam"]} == set(frontends), "Unexpected frontend registration")
    status(stage="completion", action="reading_signal_files")
    signals, refs, samples, levels, signal_audit = check_signals(plan, cases, frontends)
    asr, score_audit = {}, {}
    for engine in ENGINES:
        status(stage="completion", action="checking_scores", engine=engine)
        asr[engine], score_audit[engine] = check_asr(engine, plan, archives, signals, refs, samples)
    summary = analyze(cases, frontends, signals, asr, levels)
    coverage = {"systems": 31, "frontends": 10, "cases": 504, "clean_cases": 72, "mixed_cases": 432,
                "signal_and_each_ASR_rows": len(signals), "speakers": 4, "utterances": 8}
    summary.update(coverage=coverage, definitions={
        "delta": "Same frontend and case. MWF minus MVDR. Signal metrics use per-record means.",
        "wer": "(sum(S)+sum(D)+sum(I))/sum(N). WER deltas are percentage points.",
        "clean_gain": "RMS(output)/RMS(reverberant clean reference), linear, uncentered",
        "clean_NMSE": "sum((output-reference)^2)/sum(reference^2), linear, uncentered",
        "per_seed": "Single frontend and seed, marginal over mixed noise recordings, fields and angles.",
        "scope": "One simulated room, repeated speech, multiple training seeds for each adapter group.",
    })
    summary_file = EV / "summary.json"
    dump(summary_file, summary, True)
    audit_file = OUT / "completion_audit.json"
    dump(audit_file, {"status": "ok", "coverage": coverage, "archives_checked": archives,
                      "signals": signal_audit, "ASR": score_audit, "summary_sha256": sha(summary_file),
                      "completed_utc": now(), "elapsed_seconds": time.monotonic() - started}, True)
    with REPORT.open("x", encoding="utf-8") as report:
        report.write(report_text(summary))
    archive("completion", REPORT, [audit_file, summary_file, OUT / "frozen_plan.json", Path(__file__).resolve()])
    dump(OUT / "MAJOR_STAGE_COMPLETE.json", {"status": "ok", "stage": "G19", "completed_utc": now(),
         "report": REPORT.relative_to(ROOT).as_posix(), "report_sha256": sha(REPORT),
         "archive_sha256": sha(OUT / "stages/completion/archive_manifest.json")}, True)
    status(stage="complete", report=REPORT.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
