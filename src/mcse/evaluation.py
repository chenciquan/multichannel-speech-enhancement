"""Signal-level diagnostic metrics; none establishes linguistic preservation.

SI-SDR is zero-mean, with projection/scaling inside the metric only. Audio written
by the inference runner is never normalized by these functions. Optional PESQ and
STOI use their real backends or return an explicit unavailable/error status.
"""
from __future__ import annotations

import importlib
import importlib.metadata
from typing import Any

import numpy as np


class AudioValidationError(ValueError):
    pass


def validate_pair(reference: Any, estimate: Any, sample_rate: int = 16000) -> tuple[np.ndarray, np.ndarray]:
    if type(sample_rate) is not int or sample_rate != 16000:
        raise AudioValidationError("This diagnostic protocol requires integer sample_rate=16000.")
    ref = np.asarray(reference, dtype=np.float64)
    est = np.asarray(estimate, dtype=np.float64)
    if ref.ndim != 1 or est.ndim != 1:
        raise AudioValidationError("Reference and estimate must both be one-dimensional mono arrays.")
    if ref.shape != est.shape:
        raise AudioValidationError(f"No implicit alignment/cropping: lengths differ ({ref.size}, {est.size}).")
    if ref.size < 2:
        raise AudioValidationError("Audio must contain at least two samples.")
    if not np.isfinite(ref).all() or not np.isfinite(est).all():
        raise AudioValidationError("Audio contains NaN or infinity.")
    return ref, est


def si_sdr(reference: Any, estimate: Any, sample_rate: int = 16000) -> float:
    """Zero-mean SI-SDR in dB, using no delay/filter fitting.

    Silence/constant references and zero estimates are undefined and raise,
    rather than returning a misleading successful finite score. A float64
    machine-precision floor relative to estimate energy handles exact equality
    while retaining global estimate-gain invariance.
    """
    ref, est = validate_pair(reference, estimate, sample_rate)
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    ref_energy = float(np.dot(ref, ref))
    est_energy = float(np.dot(est, est))
    if ref_energy <= np.finfo(np.float64).tiny:
        raise AudioValidationError("SI-SDR is undefined for a silent/constant zero-mean reference.")
    if est_energy <= np.finfo(np.float64).tiny:
        raise AudioValidationError("SI-SDR is undefined for a silent/constant zero-mean estimate.")
    target = ref * (float(np.dot(est, ref)) / ref_energy)
    error = est - target
    floor = np.finfo(np.float64).eps * est_energy
    numerator = max(float(np.dot(target, target)), floor)
    denominator = max(float(np.dot(error, error)), floor)
    return float(10.0 * np.log10(numerator / denominator))


class MetricEvaluator:
    """Probe optional metric implementations once and report their provenance."""
    def __init__(self) -> None:
        self.backends: dict[str, Any] = {}
        self.backend_info: dict[str, dict[str, Any]] = {}
        for metric, package, symbol in (
            ("stoi", "pystoi", "stoi"), ("pesq_wb", "pesq", "pesq")):
            try:
                module = importlib.import_module(package)
                function = getattr(module, symbol)
                self.backends[metric] = function
                try:
                    version = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    version = getattr(module, "__version__", "unknown")
                self.backend_info[metric] = {"status": "available", "package": package, "version": version}
            except (ImportError, AttributeError, OSError) as error:
                self.backend_info[metric] = {
                    "status": "unavailable", "package": package,
                    "error_type": type(error).__name__, "error": str(error),
                }

    def evaluate(self, reference: Any, estimate: Any, sample_rate: int = 16000) -> dict[str, Any]:
        ref, est = validate_pair(reference, estimate, sample_rate)
        scores: dict[str, Any] = {
            "si_sdr_db": None, "stoi": None, "pesq_wb": None, "status": {},
        }
        try:
            scores["si_sdr_db"] = si_sdr(ref, est, sample_rate)
            scores["status"]["si_sdr_db"] = {"status": "ok"}
        except AudioValidationError as error:
            scores["status"]["si_sdr_db"] = {"status": "undefined", "error": str(error)}
        for metric in ("stoi", "pesq_wb"):
            if metric not in self.backends:
                scores["status"][metric] = dict(self.backend_info[metric])
                continue
            try:
                if metric == "stoi":
                    value = self.backends[metric](ref, est, sample_rate, extended=False)
                else:
                    value = self.backends[metric](sample_rate, ref, est, "wb")
                value = float(value)
                if not np.isfinite(value):
                    raise ValueError("Metric backend returned a nonfinite score.")
                scores[metric] = value
                scores["status"][metric] = {"status": "ok"}
            except Exception as error:
                scores["status"][metric] = {
                    "status": "error", "error_type": type(error).__name__, "error": str(error),
                }
        return scores


def waveform_diagnostics(reference: Any, estimate: Any, sample_rate: int = 16000) -> dict[str, Any]:
    """Energy/peak diagnostics only; do not label frame energy as missing words.

    20-ms frames, 10-ms hop. Speech-active frames are a reproducible reference-RMS
    threshold (-40 dB relative to the maximum reference-frame RMS), not a learned
    VAD. Remainder frames are included with their actual lengths.
    """
    ref, est = validate_pair(reference, estimate, sample_rate)
    window, hop = 320, 160
    starts = range(0, ref.size, hop)
    ref_rms = np.array([np.sqrt(np.mean(ref[start:start + window] ** 2)) for start in starts])
    est_rms = np.array([np.sqrt(np.mean(est[start:start + window] ** 2)) for start in range(0, est.size, hop)])
    peak_rms = float(np.max(ref_rms))
    active = ref_rms > max(peak_rms * 0.01, 1e-12)
    inactive = ~active
    low = np.zeros(active.shape, dtype=bool)
    if active.any():
        low = active & (ref_rms <= np.quantile(ref_rms[active], 0.25))
    amplitude_floor = max(peak_rms * 1e-8, 1e-15)
    ratio_db = 20.0 * np.log10((est_rms + amplitude_floor) / (ref_rms + amplitude_floor))
    return {
        "estimate_abs_peak": float(np.max(np.abs(est))),
        "estimate_samples_above_one": int(np.count_nonzero(np.abs(est) > 1.0)),
        "estimate_fraction_above_one": float(np.mean(np.abs(est) > 1.0)),
        "estimate_rms": float(np.sqrt(np.mean(est ** 2))),
        "reference_active_frame_fraction": float(np.mean(active)),
        "inactive_frame_estimate_mean_square": float(np.mean(est_rms[inactive] ** 2)) if inactive.any() else None,
        "low_reference_energy_underestimated_6db_fraction": float(np.mean(ratio_db[low] < -6.0)) if low.any() else None,
        "active_reference_underestimated_6db_fraction": float(np.mean(ratio_db[active] < -6.0)) if active.any() else None,
        "definition": "Reference RMS > peak_frame_RMS * 0.01; 320-sample frames/160 hop; low energy=bottom active quartile.",
        "notice": "Energy underestimation and noise leakage proxies only; not phoneme or linguistic-loss labels.",
    }


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize all cases with each metric's explicit successful denominator."""
    successful = [case for case in cases if case.get("status") == "ok"]
    summary: dict[str, Any] = {
        "total_cases": len(cases), "successful_cases": len(successful),
        "failed_cases": len(cases) - len(successful), "metrics": {},
        "human_content_loss_judgement": "not_assessed",
    }
    for metric in ("si_sdr_db", "stoi", "pesq_wb"):
        entry = {}
        for system in ("noisy", "enhanced"):
            values = [case["metrics"][system][metric] for case in successful
                      if case["metrics"][system].get(metric) is not None]
            entry[system] = {"n": len(values), "mean": float(np.mean(values)) if values else None,
                             "median": float(np.median(values)) if values else None}
        pairs = [(case["metrics"]["enhanced"][metric] - case["metrics"]["noisy"][metric])
                 for case in successful if case["metrics"]["enhanced"].get(metric) is not None
                 and case["metrics"]["noisy"].get(metric) is not None]
        entry["paired_improvement"] = {"n": len(pairs), "mean": float(np.mean(pairs)) if pairs else None,
                                       "median": float(np.median(pairs)) if pairs else None}
        summary["metrics"][metric] = entry
    return summary
