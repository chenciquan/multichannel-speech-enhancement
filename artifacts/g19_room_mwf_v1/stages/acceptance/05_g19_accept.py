#!/usr/bin/env python3
"""Supplement G19 completion with cache replay, file integrity checks, and paired readback."""
import json
from pathlib import Path

from g19_common import ROOT, OUT, PREV, archive, dump, now, sha, status
from mcnet_infer import OFFLINE_CHECKPOINT, verify_upstream


EVALUATION = OUT / "evaluation"
STAGES = OUT / "stages"
REPORT = ROOT / "docs/G19_多通道Wiener对照验收读回.md"
GROUPS = {"mcnet": "原始 McNet", "fixed": "固定训练", "counter": "反向训练", "rigid": "刚性训练"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def check_hash(path, expected):
    require(sha(path) == expected, f"SHA-256 mismatch at {path}")


def audit_archive(directory):
    manifest = directory / "archive_manifest.json"
    entries = load_json(manifest)["files"]
    require(entries, f"Empty stage {directory}")
    for entry in entries:
        check_hash(directory / entry["snapshot"], entry["sha256"])
        check_hash(ROOT / entry["source"], entry["sha256"])
    return {"manifest_sha256": sha(manifest), "entries": len(entries)}


def audit_cache(engine, plan, completion, expected_pairs):
    from g19_scores import json_lines, load_cache

    folder = EVALUATION / engine
    old = PREV / "evaluation" / engine
    marker = load_json(folder / "COMPLETE.json")
    check_hash(folder / "cache_new.jsonl", marker["cache_new_sha256"])
    check_hash(folder / "asr_scores.jsonl", marker["asr_scores_sha256"])
    require(marker["asr_scores_sha256"] == completion["ASR"][engine]["new_sha256"],
            f"{engine} completion refers to different scores")
    check_hash(old / "provenance.json", plan["G18_ASR_provenance_sha256"][engine])
    prior = load_json(old / "provenance.json")
    provenance = load_json(folder / "provenance.json")
    cache, conflicts, sources = load_cache(engine, prior, old)
    require(sources == provenance["source_caches"] and conflicts == provenance["cache_hypothesis_conflicts"]
            and not conflicts and prior["model_id"] == provenance["model_id"], f"{engine} cache provenance mismatch")
    new = set()
    for row in json_lines(folder / "cache_new.jsonl"):
        key = row["sample_sha256"]
        require(row["model_id"] == prior["model_id"] and key not in cache,
                f"{engine} duplicate sample or wrong recognizer at {key}")
        cache[key] = row["hypothesis"]
        new.add(key)

    pairs, outputs = set(), set()
    for row in json_lines(folder / "asr_scores.jsonl"):
        pair = (row["system"], row["sample_id"])
        require(pair in expected_pairs and pair not in pairs, f"{engine} unexpected score key {pair}")
        pairs.add(pair)
        output = row["output_samples_sha256"]
        reference = row["reference_samples_sha256"]
        require(reference not in new and cache[output] == row["hypothesis"]
                and cache[reference] == row["reference_hypothesis"], f"{engine} cache/score mismatch at {pair}")
        outputs.add(output)
    require(pairs == expected_pairs and len(pairs) == marker["rows"] == 5040
            and len(new) == marker["new_decodes"] and len(cache) == marker["cache_entries"]
            and new.issubset(outputs), f"{engine} incomplete score or cache coverage")
    return {"scored_pairs": len(pairs), "new_decodes": len(new), "unique_output_hashes": len(outputs),
            "cache_entries": len(cache), "historical_sources": len(sources), "conflicts": len(conflicts),
            "asr_scores_sha256": marker["asr_scores_sha256"], "cache_new_sha256": marker["cache_new_sha256"]}


def report_text(summary, audit):
    grouped = {(r["rt"], r["group"]): r for r in summary["by_rt"]}
    requested = (0, .15, .35)
    require(len(summary["by_rt"]) == 12 and set(grouped) == {(rt, g) for rt in requested for g in GROUPS},
            "Expected 12 mixed-condition cells for the readback")
    lines = [
        "# G19 多通道 Wiener 对照验收读回", "",
        "主报告核验了 31 个系统各 504 个条件的音频和评分。本次补充验收复核两个识别器的评分文件与新增缓存的 SHA-256，"
        "按原优先级重建缓存，按完成审计已核验的输出和参考波形样本哈希逐例匹配识别文本。", "",
    ]
    for engine, counts in audit["ASR"].items():
        lines.append(f"- {engine} 的 5,040 对输出和参考识别文本均与缓存一致，新增解码 "
                     f"{counts['new_decodes']:,} 条，缓存冲突 {counts['conflicts']} 条。")
    lines.extend([
        "", f"已复核冻结计划、{len(audit['frozen_files'])} 个冻结代码文件、完成阶段及其之前的 "
        f"{sum(r['entries'] for r in audit['verified_stages'].values())} 项归档快照和对应源文件。"
        "McNet 的 9 个上游源码或配置文件和离线权重通过既有完整性验证。临时保持唤醒已释放。", "",
        "## 带噪条件中的 MWF 减 MVDR", "",
        "同一前端及同一输入配对。SI-SDR 为逐例差值的平均值，正值表示升高。WER 按参考词数汇总，负值表示错误率下降。"
        "各 RT 的原始 McNet 含 144 对，三个适配器组各合并三个训练种子、432 对。", "",
        "|请求 RT (s)|前端组|SI-SDR 差 (dB)|Whisper WER 差（百分点）|CTC WER 差（百分点）|",
        "|---:|---|---:|---:|---:|",
    ])
    for rt in requested:
        for group, label in GROUPS.items():
            row = grouped[rt, group]
            require(row["mwf"]["rows"] == row["mvdr"]["rows"] == (144 if group == "mcnet" else 432),
                    f"Pair count mismatch at {rt}/{group}")
            d = row["mwf_minus_mvdr"]
            lines.append(f"|{rt}|{label}|{d['si_sdr_db']:+.2f}|{d['Whisper_wer_pp']:+.2f}|{d['CTC_wer_pp']:+.2f}|")
    lines.extend(["", "这些差值描述同一模拟房间、两段噪声录音及四位说话人的八句已用语音。"
                  "MWF 与 MVDR 在协方差归一化与滤波求解上均有差别。分方向、分录音的结果及干净控制见"
                  "[主报告](G19_多通道Wiener与MVDR对照报告.md)。", "",
                  "[验收记录](../artifacts/g19_room_mwf_v1/FINAL_ACCEPTANCE_AUDIT.json)", "", audit["completed_utc"]])
    return "\n".join(lines) + "\n"


def main():
    ready_path, power_path = OUT / "PIPELINE_READY.json", OUT / "power_helper.json"
    major_path = OUT / "MAJOR_STAGE_COMPLETE.json"
    ready, power, major = (load_json(p) for p in (ready_path, power_path, major_path))
    require(major["status"] == "ok" and power["status"] == "released", "Await G19 completion and power release")
    for name, digest in ready["files"].items():
        check_hash(ROOT / name, digest)
    plan_path = OUT / "frozen_plan.json"
    check_hash(plan_path, ready["frozen_plan_sha256"])
    plan = load_json(plan_path)
    stages = {p.name: audit_archive(p) for p in sorted(STAGES.iterdir()) if p.is_dir()}
    required = {"preflight", "definition_clarification", "signal", "whisper", "ctc", "completion"}
    require(required <= stages.keys(), "Missing G19 stage archive")
    require(ready["preflight_archive_sha256"] == stages["preflight"]["manifest_sha256"]
            and ready["clarification_archive_sha256"] == stages["definition_clarification"]["manifest_sha256"]
            and major["archive_sha256"] == stages["completion"]["manifest_sha256"], "Archive anchor mismatch")
    check_hash(ROOT / major["report"], major["report_sha256"])
    completion = load_json(OUT / "completion_audit.json")
    require(completion["status"] == "ok" and completion["signals"]["output_files_checked"] == 15624,
            "No successful waveform readback")
    for name, digest in completion["archives_checked"].items():
        require(digest == stages[name]["manifest_sha256"], f"Changed {name} archive")
    summary_file = EVALUATION / "summary.json"
    check_hash(summary_file, completion["summary_sha256"])
    summary = load_json(summary_file)
    from g19_scores import json_lines
    manifest = EVALUATION / "mwf_signals.jsonl"
    check_hash(manifest, completion["signals"]["mwf_sha256"])
    pairs = {(row["system"], row["sample_id"]) for row in json_lines(manifest)}
    require(len(pairs) == 5040, "Unexpected MWF signal coverage")
    scores = {engine: audit_cache(engine, plan, completion, pairs) for engine in ("Whisper", "CTC")}
    repo = ROOT / "third_party/McNet"
    upstream = verify_upstream(repo, repo / OFFLINE_CHECKPOINT)
    require(len(upstream["verified_source_files"]) == 9, "Unexpected upstream source count")
    audit = {
        "status": "ok", "completed_utc": now(), "scope": "Supplement to completion: file hashes and hypothesis-cache replay",
        "pipeline_ready_sha256": sha(ready_path), "frozen_files": ready["files"],
        "frozen_plan_sha256": ready["frozen_plan_sha256"], "verified_stages": stages,
        "completion_audit_sha256": sha(OUT / "completion_audit.json"), "summary_sha256": sha(summary_file),
        "ASR": scores, "mcnet": upstream, "power_helper": power,
    }
    report = report_text(summary, audit)
    audit_path = OUT / "FINAL_ACCEPTANCE_AUDIT.json"
    dump(audit_path, audit, True)
    with REPORT.open("x", encoding="utf-8") as stream:
        stream.write(report)
    archive("acceptance", REPORT, [audit_path, ready_path, major_path, power_path, Path(__file__).resolve()])
    audit_archive(STAGES / "acceptance")
    status(stage="accepted", report=REPORT.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
