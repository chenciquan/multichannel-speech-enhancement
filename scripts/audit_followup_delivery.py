#!/usr/bin/env python3
"""Verify the cause-audit/listening/decision milestone against actual artifacts."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit
from zoneinfo import ZoneInfo

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    return [json.loads(t) for t in Path(path).read_text().splitlines() if t.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class AudioLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "audio":
            self.links.append(dict(attrs)["src"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite previous audit evidence")
    checks = []
    evidence = {}

    def check(requirement, passed, detail=None):
        checks.append({"requirement":requirement,"passed":bool(passed),"detail":detail})
        if not passed:
            raise ValueError(requirement)

    try:
        base = ROOT/"artifacts/g1_followup_v2"
        delivery = base/"delivery"
        plan = read(base/"frozen_plan.json")
        summary = read(base/"summary.json")
        signal = read(base/"all_case_signal_analysis.json")
        probes = rows(base/"probes.jsonl")
        case_list = rows(ROOT/"artifacts/g1_v1/mcnet/cases.jsonl")
        cases = {r["sample_id"]:r for r in case_list}
        asr = {r["sample_id"]:r for r in rows(ROOT/"artifacts/g1_v1/asr/transcriptions.jsonl")}
        candidates = {sid for sid,r in asr.items() if r["asr_regression_review_candidate"]}
        normal = {sorted([r for r in case_list if r["speaker_id"] == speaker and r["sample_id"] not in candidates],
                         key=lambda r:r["sample_id"])[0]["sample_id"] for speaker in {r["speaker_id"] for r in case_list}}
        expected = candidates|normal
        check("selection frozen before probes; seven ASR candidates plus first normal case per six speakers",
              plan["frozen_before_probe_outputs"] and len(candidates) == 7 and len(normal) == 6
              and len(expected) == 13 and set(plan["selected_sample_ids"]) == expected)
        check("no experiment selection changed after outputs", sha(ROOT/"configs/g1_followup_plan.json") == plan["plan_sha256"])
        original_paths = {"cases":ROOT/"artifacts/g1_v1/mcnet/cases.jsonl",
                          "asr":ROOT/"artifacts/g1_v1/asr/transcriptions.jsonl",
                          "manifest":ROOT/"data/diagnostic/g1_v1/manifest.jsonl"}
        check("original cases, ASR and manifest hash chain intact", all(sha(path) == plan["input_hashes"][name]
                                                                       for name,path in original_paths.items()))
        check("full original evidence re-audit passed", read(base/"verification/original_evidence_reaudit.json")["status"] == "passed")
        check("full 48-case signal analysis retained", len(signal) == 48 and {r["sample_id"] for r in signal} == set(cases))
        for key in ("correlation","projection_gain","rms_gain_db","difference_energy_relative_db"):
            values = [r["output_vs_input"][key] for r in signal]
            computed = {"min":np.min(values),"median":np.median(values),"mean":np.mean(values),"max":np.max(values)}
            check("reported aggregate: "+key, summary["all_case_output_input"][key]["n"] == 48
                  and all(np.isclose(v,summary["all_case_output_input"][key][k],rtol=0,atol=1e-12) for k,v in computed.items()))
        check("all48 best global lags zero; no fitted delay used to change scores", all(
              r["output_vs_input"]["best_lag_within_16ms_samples"] == 0
              and not r["output_vs_input"]["lag_search_used_to_modify_metrics_or_audio"] for r in signal))
        conditions = set(plan["plan"]["forward_conditions"])
        counts = Counter((r["sample_id"],r["condition"]) for r in probes)
        check("exactly52 probes; one successful output per frozen case and four conditions",
              len(probes) == 52 and set(counts) == {(sid,c) for sid in expected for c in conditions}
              and set(counts.values()) == {1} and all(r["status"] == "ok" for r in probes))
        for r in probes:
            sid = r["sample_id"]
            y,sr = sf.read(r["output_path"], dtype="float32")
            check(sid+": "+r["condition"]+" output hash/metadata",
                  sha(r["output_path"]) == r["output_sha256"] and y.ndim == 1 and sr == 16000
                  and np.isfinite(y).all() and len(y) == cases[sid]["samples"]
                  and sf.info(r["output_path"]).subtype == "FLOAT")
            check(sid+": "+r["condition"]+" parity checks recorded within bounds",
                  r["feature_max_abs_error"] == 0 and r["original_decoder_max_abs_error"] == 0
                  and r["independent_cirm_formula_max_abs_error"] < 1e-6
                  and r["clean_field_ref_quantization_max_abs_error"] <= 1.3e-7)
            if r["condition"] == "original":
                old,_ = sf.read(cases[sid]["enhanced_path"], dtype="float32")
                check(sid+": original waveform repeats exactly", np.array_equal(old,y))
        config = read(base/"checkpoint_config_comparison.json")
        check("checkpoint sequence values agree; tuple/list distinction explicitly recorded",
              len(config) == 10 and all(v["equal"] for v in config.values())
              and config["sub_freqs"]["checkpoint_type"] == "tuple"
              and config["sub_freqs"]["constructed_type"] == "list")
        provenance = read(base/"provenance.json")
        check("pinned model remains unmodified; no probe training", not provenance["training"]
              and not provenance["upstream_modified"] and not subprocess.check_output(
                  ["git","-C",str(ROOT/"third_party/McNet"),"status","--porcelain"],text=True).strip())
        feedback = read(base/"listening_feedback.json")
        control = read(base/"control_feedback.json")
        frows = [r for event in feedback["events"] for r in event["reviews"]]
        check("two explicit user replies cover all seven candidates exactly",
              len(frows) == 7 and {r["sample_id"] for r in frows} == candidates
              and feedback["events"][0]["answer_verbatim"] == "两者都能听清，没有明显退步"
              and feedback["events"][1]["answer_verbatim"] == "已逐条试听，其余6条也没有明显退步")
        check("third explicit reply covers frozen six normal controls",
              len(control["reviews"]) == 6 and {r["sample_id"] for r in control["reviews"]} == normal
              and control["answer_verbatim"] == "这6条正常对照也没有明显退步"
              and not control["blinded"] and not control["formal_word_time_annotation"])
        check("all listening records say no obvious regression, with limited scope",
              all(r["judgement"] == "no obvious regression reported" for r in frows+control["reviews"]))
        links = AudioLinks()
        page = base/"decision/control_listening.html"
        links.feed(page.read_text(encoding="utf-8"))
        check("control listening page has18 local mono16k audio resources",
              len(links.links) == 18 and all(not urlsplit(link).scheme
              and sf.info(page.parent/unquote(link)).channels == 1
              and sf.info(page.parent/unquote(link)).samplerate == 16000 for link in links.links))
        decision = read(delivery/"decision.json")
        check("decision denominator:13 mixes from12 sources; no full48 or blinded-listening claim",
              decision["user_feedback_cases"] == 13 and decision["unique_source_utterances_reviewed"] == 12
              and len({cases[sid]["source_id"] for sid in expected}) == 12
              and not decision["whole_discovery_set_listened"] and decision["formal_word_time_annotations"] == 0)
        check("unsupported stable deletion does not start a recovery module or confirmation batch",
              not decision["stable_acoustic_deletion_supported"] and not decision["absence_of_all_possible_deletions_proven"]
              and not decision["start_recovery_module"] and decision["method_design_gate"] == "not_passed"
              and not decision["new_utterance_confirmation_run"] and not decision["training_performed"])
        expected_hashes = {"probe_summary":base/"summary.json","probes":base/"probes.jsonl",
                           "signal_analysis":base/"all_case_signal_analysis.json","feedback":base/"listening_feedback.json",
                           "control_feedback":base/"control_feedback.json"}
        check("final decision traces all numerical and user-feedback evidence",
              all(decision["evidence_sha256"][name] == sha(path) for name,path in expected_hashes.items()))
        reviews = read(delivery/"candidate_reviews.json")
        check("candidate review keeps original ASR text and original denominators",
              len(reviews) == 7 and {r["sample_id"] for r in reviews} == candidates
              and all(r["noisy_hypothesis"] == asr[r["sample_id"]]["asr"]["noisy"]["hypothesis_raw"]
                      and r["enhanced_hypothesis"] == asr[r["sample_id"]]["asr"]["enhanced"]["hypothesis_raw"]
                      and not r["blinded_listener_test"] and not r["formal_word_time_annotation_completed"] for r in reviews))
        verify = base/"verification"
        testlog = (verify/"unit_tests.log").read_text(encoding="utf-8-sig")
        check("44 tests with analytic gain/lag/projection checks passed without skips",
              read(verify/"unit_tests_result.json")["exit_code"] == 0 and "Ran 44 tests" in testlog
              and "\nOK" in testlog and "skipped" not in testlog.lower()
              and "test_projection_recovers_known_components" in testlog)
        check("same92 installed packages remain compatible", read(verify/"dependencies_result.json")["exit_code"] == 0
              and "92 packages" in (verify/"dependencies.log").read_text(encoding="utf-8-sig"))
        freeze = subprocess.check_output([str(ROOT/".tools/uv-x86_64-unknown-linux-gnu/uv"),"pip","freeze","--python",str(ROOT/".venv/bin/python")],text=True)
        check("installed package list still equals G1 lock; no new dependencies",
              freeze.splitlines() == (ROOT/"requirements-diagnostics.lock.txt").read_text().splitlines())
        report = ROOT/"docs/G1_问题核查与阶段决策.md"
        text = report.read_text(encoding="utf-8")
        missing = []
        for link in re.findall(r"\]\(([^)]+)\)",text):
            if urlsplit(link).scheme or link.startswith("#"):
                continue
            target = (report.parent/unquote(link.split("#")[0])).resolve()
            if target != args.output.resolve() and not target.is_file():
                missing.append(link)
        check("Chinese decision report, figures and all local reference targets exist", report.is_file() and not missing, missing)
        state = read(ROOT/"configs/project_status.json")
        check("current project state matches reviewed negative decision",
              state["current_milestone"] == "G1_followup_problem_decision" and state["human_reviews_completed"] == 13
              and state["unit_tests_passed"] == 44 and state["method_design_gate"] == "not_passed"
              and not state["training_performed"] and not state["method_selected"]
              and not state["whole_discovery_set_listened"])
        evidence = {str(path.relative_to(ROOT)):{"sha256":sha(path)} for path in
                    [report, ROOT/"configs/project_status.json", base/"frozen_plan.json", base/"summary.json",
                     base/"probes.jsonl", delivery/"decision.json", base/"listening_feedback.json",base/"control_feedback.json",
                     ROOT/"scripts/probe_baseline_behavior.py",ROOT/"scripts/summarize_followup.py",
                     ROOT/"scripts/audit_followup_delivery.py",ROOT/"tests/test_behavior_probe.py"]}
        status,error = "passed",None
    except Exception as exc:
        status,error = "failed",{"type":type(exc).__name__,"message":str(exc)}
    report = {"created_at":datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),"status":status,
              "scope":"follow-up cause audit, candidate/control feedback and method-entry decision; not validation of a new method",
              "checks":checks,"passed_checks":sum(c["passed"] for c in checks),"error":error,"evidence":evidence,
              "remaining_research_unknowns":["exact cause of weak suppression", "word-time and blinded human evaluation",
                                             "recorded spatial-noise performance", "independent failure confirmation if a failure is later supported"],
              "next_stage_executed":False,"training_performed":False}
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status":status,"passed_checks":report["passed_checks"],"error":error,"output":str(args.output)},ensure_ascii=False))
    return int(status != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
