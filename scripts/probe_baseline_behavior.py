#!/usr/bin/env python3
"""Read-only model/data audit plus bounded diagnostic input interventions; no training."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import numpy as np
import soundfile as sf
from scipy.signal import correlate, correlation_lags

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from mcse.simulation import render_plane_wave


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(t) for t in Path(path).read_text(encoding="utf-8").splitlines() if t.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def db_ratio(numerator, denominator):
    return float(10*np.log10(max(float(numerator), 1e-30)/max(float(denominator), 1e-30)))


def comparison(reference, estimate):
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(estimate, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Comparison requires finite, equal-length mono arrays")
    ac, bc = a-a.mean(), b-b.mean()
    power = float(ac@ac)
    gain = float(ac@bc/max(power, 1e-30))
    correlation = float(ac@bc/max(float(np.linalg.norm(ac)*np.linalg.norm(bc)), 1e-30))
    cc = correlate(bc, ac, mode="full", method="fft")
    lags = correlation_lags(len(bc), len(ac), mode="full")
    region = np.abs(lags) <= 256
    return {"correlation": correlation, "projection_gain": gain,
            "rms_gain_db": db_ratio(b@b, a@a),
            "difference_energy_relative_db": db_ratio((b-a)@(b-a), a@a),
            "max_abs_difference": float(np.max(np.abs(b-a))),
            "best_lag_within_16ms_samples": int(lags[region][np.argmax(cc[region])]),
            "lag_search_used_to_modify_metrics_or_audio": False}


def projection(speech, noise, output):
    columns = np.column_stack([speech-speech.mean(), noise-noise.mean()])
    target = output-output.mean()
    weights, _, rank, singular = np.linalg.lstsq(columns, target, rcond=None)
    residual = target-columns@weights
    return {"speech_coefficient": float(weights[0]), "noise_coefficient": float(weights[1]),
            "rank": int(rank), "condition_number": float(singular[0]/singular[-1]),
            "unexplained_energy_fraction": float(residual@residual/max(float(target@target), 1e-30)),
            "notice": "Whole-utterance regression on known clean/noise waveforms; not true nonlinear source separation or a word-loss test"}


def describe(values):
    return {"n": len(values), "min": float(np.min(values)), "median": float(np.median(values)),
            "mean": float(np.mean(values)), "max": float(np.max(values))}


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        parser.error("Refusing to replace prior evidence")
    plan_path = ROOT/"configs/g1_followup_plan.json"
    plan = read(plan_path)
    cases_path = ROOT/"artifacts/g1_v1/mcnet/cases.jsonl"
    asr_path = ROOT/"artifacts/g1_v1/asr/transcriptions.jsonl"
    manifest_path = ROOT/"data/diagnostic/g1_v1/manifest.jsonl"
    cases = rows(cases_path)
    asr = {r["sample_id"]: r for r in rows(asr_path)}
    manifest = {r["sample_id"]: r for r in rows(manifest_path)}
    source_frozen = read(ROOT/"data/diagnostic/g1_v1/frozen_inputs.json")
    source_map = {r["source_id"]: r for r in source_frozen["sources"]}
    expected_ids = {c["sample_id"] for c in cases}
    if len(cases) != 48 or len(expected_ids) != 48 or expected_ids != set(asr) or expected_ids != set(manifest):
        raise ValueError("Frozen evidence IDs do not match")
    selected = [c for c in cases if asr[c["sample_id"]]["asr_regression_review_candidate"]]
    for speaker in sorted({c["speaker_id"] for c in cases}):
        selected.append(sorted([c for c in cases if c["speaker_id"] == speaker
                                and not asr[c["sample_id"]]["asr_regression_review_candidate"]], key=lambda c:c["sample_id"])[0])
    selected = sorted({c["sample_id"]: c for c in selected}.values(), key=lambda c:c["sample_id"])
    if len(selected)*len(plan["forward_conditions"]) > plan["maximum_forward_probes"]:
        raise ValueError("Probe budget exceeded")
    output.mkdir(parents=True)
    (output/"probe_audio").mkdir()
    write(output/"frozen_plan.json", {"plan": plan, "plan_sha256": sha(plan_path),
          "selected_sample_ids": [c["sample_id"] for c in selected],
          "input_hashes": {"cases": sha(cases_path), "asr": sha(asr_path), "manifest": sha(manifest_path)},
          "frozen_before_probe_outputs": True})
    write(output/"human_feedback.json", {"recorded_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
          "source": "explicit user reply in this Codex conversation", "reviews": plan["human_feedback_received"],
          "scope": "one identified word in one case only; remaining cases unassessed"})
    all_case = []
    for c in cases:
        sid = c["sample_id"]
        m = manifest[sid]
        signals = {}
        for role in ("reference", "noisy", "enhanced"):
            path = Path(c[role+"_path"])
            wanted = c["enhanced_file"]["sha256"] if role == "enhanced" else m[role+"_sha256"]
            if sha(path) != wanted:
                raise ValueError("Frozen input/output changed: " + sid)
            audio, sr = sf.read(path, dtype="float64", always_2d=True)
            if sr != 16000:
                raise ValueError("Unexpected sampling rate")
            signals[role] = audio[:,4] if role == "noisy" else audio[:,0]
        s, x, y = (signals[k] for k in ("reference", "noisy", "enhanced"))
        all_case.append({"sample_id": sid, "source_id": c["source_id"], "speaker_id": c["speaker_id"],
                         "asr_candidate": asr[sid]["asr_regression_review_candidate"],
                         "output_vs_input": comparison(x,y), "output_vs_clean": comparison(s,y),
                         "linear_projection": projection(s, x-s, y)})
    write(output/"all_case_signal_analysis.json", all_case)
    adapter = runpy.run_path(str(ROOT/"scripts/mcnet_infer.py"), run_name="g1_followup_adapter")
    repo = ROOT/"third_party/McNet"
    checkpoint = repo/adapter["OFFLINE_CHECKPOINT"]
    upstream = adapter["verify_upstream"](repo, checkpoint)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("This frozen probe is intended for the same CUDA device; no silent fallback")
    torch.manual_seed(20261111)
    model, icirm, model_info = adapter["load_model"](repo, checkpoint, device)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hparams = saved.get("hyper_parameters", {})
    config_agreement = {key: {"checkpoint": hparams.get(key), "constructed": value,
                             "checkpoint_type": type(hparams.get(key)).__name__,
                             "constructed_type": type(value).__name__,
                             "python_equal": hparams.get(key) == value,
                             "equal": json.dumps(hparams.get(key), sort_keys=True) == json.dumps(value, sort_keys=True)}
                        for key, value in model.hparams.items() if key in hparams}
    del saved
    write(output/"checkpoint_config_comparison.json", config_agreement)
    if not config_agreement or not all(item["equal"] for item in config_agreement.values()):
        raise ValueError("Checkpoint hyperparameters disagree with constructed model")
    write(output/"provenance.json", {"upstream": upstream, "model": model_info, "device": str(device),
          "checkpoint_hparams_comparison": config_agreement, "training": False, "upstream_modified": False,
          "forward_conditions": plan["forward_conditions"], "nonreference_permutation": [5,3,2,1,4,0]})
    actual_istft = torch.istft

    def legacy_istft(spectrum, *args, **kwargs):
        if not spectrum.is_complex():
            spectrum = torch.view_as_complex(spectrum.contiguous())
        return actual_istft(spectrum, *args, **kwargs)

    probes = []
    with (output/"probes.jsonl").open("x", encoding="utf-8") as stream, torch.inference_mode():
        for index, c in enumerate(selected):
            sid = c["sample_id"]
            m = manifest[sid]
            noisy, _ = sf.read(c["noisy_path"], dtype="float32", always_2d=True)
            clean_ref, _ = sf.read(c["reference_path"], dtype="float32")
            old_output, _ = sf.read(c["enhanced_path"], dtype="float32")
            source = source_map[c["source_id"]]
            if sha(source["path"]) != source["sha256"]:
                raise ValueError("Clean source changed")
            mono, _ = sf.read(source["path"], dtype="float64")
            clean_field = (render_plane_wave(mono, m["target_delays_samples"])*m["common_scale"]).astype(np.float32)
            clean_reference_max_error = float(np.max(np.abs(clean_field[:,4]-clean_ref)))
            if clean_reference_max_error > 1.3e-7:
                raise ValueError("Reconstructed clean field differs from PCM24 reference beyond quantization")
            conditions = {"original": noisy, "common_gain_0.25": noisy*.25,
                          "nonreference_channels_reversed_ref4_fixed": noisy[:,[5,3,2,1,4,0]],
                          "reconstructed_clean_field": clean_field}
            baseline = None
            for label, audio in conditions.items():
                x = torch.from_numpy(np.ascontiguousarray(audio.T)).unsqueeze(0).to(device)
                prediction, data = model(x)
                wav = adapter["reconstruct_waveform"](model, icirm, x, prediction, data["original_time_len"])
                # Independent normalization/order calculation from raw channel STFTs.
                raw = torch.stft(x.reshape(6,-1), 512, 256, window=model.io.window, return_complex=True).reshape(1,6,257,-1)
                raw_tf = raw.permute(0,3,2,1)
                mu = raw[:,4].abs().mean().reshape(1,1,1,1)
                features = torch.view_as_real(raw_tf/(mu+1e-8)).reshape(1,raw.shape[-1],257,12)
                feature_max_error = float((features-data["input"]).abs().max())
                # Independent identity: decompressed cIRM is 20*atanh(compressed/10).
                decoded = 20*torch.atanh(torch.view_as_real(prediction).clamp(-9.9,9.9)/10)
                direct = torch.view_as_complex(decoded.contiguous())*raw[:,4]
                independent_wav = actual_istft(direct, 512, 256, window=model.io.window, length=len(audio))
                formula_max_error = float((independent_wav-wav).abs().max())
                # Run the actual original decoder, adapting only Torch2 ISTFT input type.
                with patch("torch.istft", side_effect=legacy_istft):
                    legacy_wav = model.io.prepare_time_domain(x=x, prediction=prediction, **data)
                upstream_decoder_max_error = float((legacy_wav-wav).abs().max())
                if feature_max_error > 2e-5 or formula_max_error > 2e-5 or upstream_decoder_max_error > 2e-6:
                    raise ValueError("Feature/cIRM/original-decoder parity failed")
                result = wav[0].cpu().numpy()
                filename = f"{index:02d}_{label}.wav"
                sf.write(output/"probe_audio"/filename, result, 16000, subtype="FLOAT")
                item = {"sample_id": sid, "source_id": c["source_id"], "condition": label, "status": "ok",
                        "feature_max_abs_error": feature_max_error,
                        "independent_cirm_formula_max_abs_error": formula_max_error,
                        "original_decoder_max_abs_error": upstream_decoder_max_error,
                        "clean_field_ref_quantization_max_abs_error": clean_reference_max_error,
                        "output_path": str(output/"probe_audio"/filename), "output_sha256": sha(output/"probe_audio"/filename),
                        "compressed_mask_real_median": float(prediction.real.median()),
                        "compressed_mask_imag_abs_median": float(prediction.imag.abs().median()),
                        "output_vs_condition_reference_input": comparison(audio[:,4], result)}
                if label == "original":
                    baseline = result.copy()
                    item["stored_output_repeat"] = comparison(old_output, result)
                elif label == "common_gain_0.25":
                    item["gain_restored_vs_original_output"] = comparison(baseline, result/.25)
                elif label == "nonreference_channels_reversed_ref4_fixed":
                    item["vs_original_output"] = comparison(baseline, result)
                elif label == "reconstructed_clean_field":
                    item["vs_clean_reference"] = comparison(clean_ref, result)
                probes.append(item)
                stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False)+"\n")
                stream.flush()
            print(json.dumps({"case": index+1,"total": len(selected),"sample_id": sid}), flush=True)
    summary = {"status": "ok", "all_cases": len(all_case), "selected_cases": len(selected), "forward_calls": len(probes),
               "selection": [c["sample_id"] for c in selected],
               "all_case_output_input": {key: describe([r["output_vs_input"][key] for r in all_case])
                                         for key in ("correlation","projection_gain","rms_gain_db","difference_energy_relative_db")},
               "all_case_best_lag_samples": dict(Counter(r["output_vs_input"]["best_lag_within_16ms_samples"] for r in all_case)),
               "all_case_projection": {key: describe([r["linear_projection"][key] for r in all_case])
                                       for key in ("speech_coefficient","noise_coefficient","unexplained_energy_fraction")},
               "max_feature_error": max(r["feature_max_abs_error"] for r in probes),
               "max_independent_decoder_error": max(r["independent_cirm_formula_max_abs_error"] for r in probes),
               "max_original_decoder_error": max(r["original_decoder_max_abs_error"] for r in probes),
               "max_stored_repeat_error": max(r["stored_output_repeat"]["max_abs_difference"] for r in probes if "stored_output_repeat" in r),
               "max_gain_restored_difference_db": max(r["gain_restored_vs_original_output"]["difference_energy_relative_db"] for r in probes if "gain_restored_vs_original_output" in r),
               "permuted_relative_difference_db": describe([r["vs_original_output"]["difference_energy_relative_db"] for r in probes if "vs_original_output" in r]),
               "clean_input_relative_difference_db": describe([r["vs_clean_reference"]["difference_energy_relative_db"] for r in probes if "vs_clean_reference" in r]),
               "human_feedback_cases": 1, "confirmed_word_deletion": None, "training": False,
               "notice": "Descriptive engineering probes, not independent method validation or evidence of human word loss"}
    write(output/"summary.json", summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
