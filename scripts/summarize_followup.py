#!/usr/bin/env python3
"""Combine frozen numerical probes and explicit user feedback, without inventing listening labels."""
from __future__ import annotations
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--control-feedback", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Refusing to replace an existing decision bundle")
    base = args.probe_dir.resolve()
    summary = read(base/"summary.json")
    signal = read(base/"all_case_signal_analysis.json")
    feedback = read(args.feedback)
    control_feedback = read(args.control_feedback)
    asr = [json.loads(t) for t in (ROOT/"artifacts/g1_v1/asr/transcriptions.jsonl").read_text().splitlines()]
    candidates = [r for r in asr if r["asr_regression_review_candidate"]]
    if summary["status"] != "ok" or len(signal) != 48 or len(candidates) != 7:
        raise ValueError("Expected complete original discovery evidence")
    reported = [review["sample_id"] for event in feedback["events"] for review in event["reviews"]]
    if len(reported) != 7 or set(reported) != {r["sample_id"] for r in candidates}:
        raise ValueError("Explicit user feedback must cover exactly seven candidates")
    selected = read(base/"frozen_plan.json")["selected_sample_ids"]
    expected_controls = set(selected)-set(reported)
    control_ids = [r["sample_id"] for r in control_feedback["reviews"]]
    if len(control_ids) != 6 or set(control_ids) != expected_controls:
        raise ValueError("Normal-control feedback does not match the frozen six controls")
    if any(r["judgement"] != "no obvious regression reported" for event in feedback["events"] for r in event["reviews"]) or any(
            r["judgement"] != "no obvious regression reported" for r in control_feedback["reviews"]):
        raise ValueError("This negative-decision summary requires the recorded feedback; handle contrary evidence explicitly")
    args.output_dir.mkdir(parents=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13,4), layout="constrained")
    ordered = sorted(signal,key=lambda r:r["sample_id"])
    colors = ["#cb4b16" if r["asr_candidate"] else "#26728c" for r in ordered]
    axes[0].scatter(range(1,49),[r["output_vs_input"]["correlation"] for r in ordered], c=colors, s=22)
    axes[0].set(xlabel="Mixture index (same source may repeat)", ylabel="Output vs input correlation", ylim=(.99,1.0002))
    speech = [r["linear_projection"]["speech_coefficient"] for r in ordered]
    noise = [r["linear_projection"]["noise_coefficient"] for r in ordered]
    axes[1].scatter(speech, noise, c=colors,s=22)
    axes[1].axhline(1,color="gray",linestyle="--",linewidth=.8)
    axes[1].axvline(1,color="gray",linestyle="--",linewidth=.8)
    axes[1].set(xlabel="Speech projection coefficient",ylabel="Noise projection coefficient",xlim=(.89,1.01),ylim=(.89,1.01))
    conditions = ["original", "common_gain_0.25", "nonreference_channels_reversed_ref4_fixed", "reconstructed_clean_field"]
    probes = [json.loads(t) for t in (base/"probes.jsonl").read_text().splitlines()]
    original = [r["stored_output_repeat"]["max_abs_difference"] for r in probes if r["condition"] == conditions[0]]
    normalized = [r["gain_restored_vs_original_output"]["difference_energy_relative_db"] for r in probes if r["condition"] == conditions[1]]
    permutation = [r["vs_original_output"]["difference_energy_relative_db"] for r in probes if r["condition"] == conditions[2]]
    clean = [r["vs_clean_reference"]["difference_energy_relative_db"] for r in probes if r["condition"] == conditions[3]]
    axes[2].boxplot([np.maximum(normalized,-110),permutation,clean],labels=["Gain restored", "Channel permuted", "Clean input"],showfliers=True)
    axes[2].set(ylabel="Difference energy / comparison energy (dB)",ylim=(-115,5))
    axes[2].text(.04,.03,"Below -110 dB displayed at -110",transform=axes[2].transAxes,fontsize=8)
    axes[2].tick_params(axis="x",labelsize=8)
    fig.suptitle("G1 follow-up: engineering diagnostics, not word-loss labels\nOrange: ASR candidates; coefficients are descriptive projections only",fontsize=11)
    fig.savefig(args.output_dir/"behavior_summary.png",dpi=150)
    plt.close(fig)
    candidate_rows = []
    for r in sorted(candidates,key=lambda x:x["sample_id"]):
        candidate_rows.append({"sample_id":r["sample_id"],"source_id":r["source_id"],
              "reference_text":r["transcript"],
              "noisy_hypothesis":r["asr"]["noisy"]["hypothesis_raw"],
              "enhanced_hypothesis":r["asr"]["enhanced"]["hypothesis_raw"],
              "delta_wer":r["delta_wer_enhanced_minus_noisy"],
              "delta_asr_deletions":r["delta_deletions_enhanced_minus_noisy"],
              "noisy_edit_alignment":r["asr"]["noisy"]["score"]["alignment"],
              "enhanced_edit_alignment":r["asr"]["enhanced"]["score"]["alignment"],
              "user_listening_feedback":"no obvious regression reported",
              "blinded_listener_test":False,
              "formal_word_time_annotation_completed":False,
              "confirmed_acoustic_deletion":None})
    decision = {"created_at":datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
          "status":"complete_for_followup_decision", "discovery_cases":48,
          "asr_candidates":7,"user_feedback_cases":13,"candidate_feedback_cases":7,"normal_control_feedback_cases":6,
          "user_reported_obvious_regression_cases":0,
          "unique_source_utterances_reviewed":len({r["source_id"] for r in asr if r["sample_id"] in selected}),
          "whole_discovery_set_listened":False,"formal_word_time_annotations":0,
          "stable_acoustic_deletion_supported":False,"absence_of_all_possible_deletions_proven":False,
          "method_design_gate":"not_passed", "start_recovery_module":False,
          "new_utterance_confirmation_run":False,
          "confirmation_skip_reason":"No supported operational acoustic failure to confirm; do not manufacture a failure-confirmation batch",
          "verified_facts":["Original decoder agrees with adapter on 52 probes", "13 stored outputs repeat exactly", "No best global input-output lag in ±16ms for48 cases", "Input/output high correlation and weak noise-projection attenuation", "Seven ASR candidates and six normal controls have no obvious user-reported listening regression"],
          "possible_cause":"Synthetic single-direction mono-noise rendering differs from original recorded multichannel noise; causal contribution not isolated",
          "unknowns":["Exact cause of weak suppression", "Robustness in recorded spatial noise", "Phoneme-level and blinded human intelligibility", "Unseen-utterance generalization"],
          "next_recommended_stage":"pre-register a small noise-spatial-condition diagnostic before selecting any method; no training yet",
          "evidence_sha256":{"probe_summary":sha(base/"summary.json"), "probes":sha(base/"probes.jsonl"),
                             "signal_analysis":sha(base/"all_case_signal_analysis.json"),"feedback":sha(args.feedback),
                             "control_feedback":sha(args.control_feedback)},
          "training_performed":False,"upstream_model_modified":False}
    for filename, data in (("decision.json",decision),("candidate_reviews.json",candidate_rows),
                           ("control_reviews.json",control_feedback)):
        (args.output_dir/filename).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"decision":decision["method_design_gate"],"reviewed_candidates":len(candidate_rows),"repeat_max_error":max(original)},ensure_ascii=False))


if __name__ == "__main__":
    main()
