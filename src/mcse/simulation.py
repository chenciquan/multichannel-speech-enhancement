"""Small, deterministic directional-noise diagnostics, not a real array corpus.

The same mono source is propagated to every microphone using its geometric
far-field delay. A mono DEMAND channel is *re-rendered* as a point source; its
original multichannel/diffuse spatial field is not retained. No reverberation,
ASR, enhancement model, or training is performed here.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf


SAMPLE_RATE = 16000
REFERENCE_CHANNEL = 4
FIR_HALF_WIDTH = 32
GUARD_SAMPLES = 64
MAX_SOURCE_SECONDS = 3.8
MAX_OUTPUT_SECONDS = 4.0
ACTIVITY_FRAME_SAMPLES = 320
ACTIVITY_HOP_SAMPLES = 160
ACTIVITY_THRESHOLD_DB = -40.0
CLIPPING_PEAK = 0.95


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a nonempty string without surrounding whitespace")
    return value


def _safe_id(value: Any, name: str) -> str:
    result = _nonempty_string(value, name)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", result) or result in {".", ".."}:
        raise ValueError(f"{name} must use only ASCII letters, digits, _, - or .")
    return result


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    manifest = Path(path).resolve()
    rows = []
    with manifest.open(encoding="utf-8-sig") as stream:
        for line, text in enumerate(stream, 1):
            if not text.strip():
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError as error:
                raise ValueError(f"{manifest}:{line}: {error}") from error
            if not isinstance(item, dict):
                raise ValueError(f"{manifest}:{line}: expected an object")
            rows.append(item)
    if not rows:
        raise ValueError(f"No records in {manifest}")
    return rows


def _unit_vector(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite 3-vector")
    if not np.isclose(np.linalg.norm(result), 1.0, atol=1e-5):
        raise ValueError(f"{name} must have unit length; no implicit normalization")
    return result


def load_geometry(path: str | Path) -> dict[str, Any]:
    """Require provenance and a domain label; never infer a model-matched array."""
    geometry_path = Path(path).resolve()
    raw = json.loads(geometry_path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        raise ValueError("Geometry must be a JSON object")
    for key in ("geometry_id", "coordinate_convention", "array_domain_label"):
        _nonempty_string(raw.get(key), key)
    positions = np.asarray(raw.get("microphone_positions_m"), dtype=np.float64)
    if positions.shape != (6, 3) or not np.isfinite(positions).all():
        raise ValueError("microphone_positions_m must be finite 6x3 metre coordinates")
    if np.unique(positions, axis=0).shape[0] != 6:
        raise ValueError("All six microphone coordinates must be distinct")
    if raw.get("reference_channel_zero_based") != REFERENCE_CHANNEL:
        raise ValueError("Geometry reference_channel_zero_based must be 4")
    _unit_vector(raw.get("target_direction_unit"), "target_direction_unit")
    noises = raw.get("noise_directions_unit")
    if not isinstance(noises, list) or len(noises) != 2:
        raise ValueError("noise_directions_unit must contain one vector for each of two noise records")
    for index, direction in enumerate(noises):
        _unit_vector(direction, f"noise_directions_unit[{index}]")
    speed = raw.get("speed_of_sound_m_s", 343.0)
    if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not math.isfinite(speed) or speed <= 0:
        raise ValueError("speed_of_sound_m_s must be finite and positive")
    # Validate the chosen FIR/guard budget for every configured direction.
    for direction in [raw["target_direction_unit"], *noises]:
        delay = geometric_delays(positions, direction, speed_of_sound=float(speed))
        if FIR_HALF_WIDTH + math.ceil(float(np.max(np.abs(delay)))) + 1 > GUARD_SAMPLES:
            raise ValueError("Array delays exceed fixed FIR/guard budget; do not crop silently")
    return raw | {"speed_of_sound_m_s": float(speed)}


def geometric_delays(positions_m: Any, direction_to_source: Any, *,
                     sample_rate: int = SAMPLE_RATE, reference_channel: int = REFERENCE_CHANNEL,
                     speed_of_sound: float = 343.0) -> np.ndarray:
    """Arrival samples relative to the reference, negative means earlier.

    direction_to_source points *toward the source*, not its propagation direction.
    A microphone closer to that source receives the plane wave earlier.
    """
    positions = np.asarray(positions_m, dtype=np.float64)
    direction = _unit_vector(direction_to_source, "direction_to_source")
    if positions.shape != (6, 3) or not np.isfinite(positions).all():
        raise ValueError("Expected six finite 3D microphone coordinates")
    if sample_rate != SAMPLE_RATE or reference_channel != REFERENCE_CHANNEL:
        raise ValueError("Only the fixed 16kHz/reference-4 protocol is supported")
    if not math.isfinite(speed_of_sound) or speed_of_sound <= 0:
        raise ValueError("Invalid speed of sound")
    return -(positions - positions[reference_channel]) @ direction * sample_rate / speed_of_sound


def render_plane_wave(mono: Any, delays_samples: Any, *, guard_samples: int = GUARD_SAMPLES,
                      half_width: int = FIR_HALF_WIDTH) -> np.ndarray:
    """Render one waveform into six delayed channels without cropping its edges."""
    signal = np.asarray(mono, dtype=np.float64)
    delays = np.asarray(delays_samples, dtype=np.float64)
    if signal.ndim != 1 or signal.size == 0 or not np.isfinite(signal).all():
        raise ValueError("Expected a nonempty finite mono signal")
    if delays.shape != (6,) or not np.isfinite(delays).all():
        raise ValueError("Expected six finite delays")
    if type(half_width) is not int or half_width < 4 or type(guard_samples) is not int:
        raise ValueError("Invalid FIR/guard size")
    if guard_samples < half_width + math.ceil(float(np.max(np.abs(delays)))) + 1:
        raise ValueError("Guard cannot contain every interpolation tail")
    padded = np.pad(signal, (guard_samples, guard_samples))
    taps = np.arange(-half_width, half_width + 1, dtype=np.float64)
    window = np.kaiser(len(taps), 8.6)
    rendered = []
    for delay in delays:
        kernel = np.sinc(taps - delay) * window
        kernel /= kernel.sum()
        rendered.append(fftconvolve(padded, kernel, mode="same"))
    result = np.stack(rendered, axis=1)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite rendered audio")
    return result


def speech_activity_mask(reference: Any) -> np.ndarray:
    """20ms RMS frames/10ms hop, >= -40dB relative to max frame RMS.

    Frame decisions are expanded to all samples covered by an active frame.
    This is a declared SNR convention, not a phoneme/word/VAD ground truth.
    """
    signal = np.asarray(reference, dtype=np.float64)
    if signal.ndim != 1 or not signal.size or not np.isfinite(signal).all():
        raise ValueError("Expected a finite mono reference")
    starts = list(range(0, len(signal), ACTIVITY_HOP_SAMPLES))
    rms = np.array([np.sqrt(np.mean(signal[start:start + ACTIVITY_FRAME_SAMPLES] ** 2))
                    for start in starts])
    maximum = float(rms.max())
    if maximum <= 1e-8:
        raise ValueError("Reference speech is silent or too small for reliable PCM24/SNR measurement")
    active = rms >= maximum * 10.0 ** (ACTIVITY_THRESHOLD_DB / 20.0)
    mask = np.zeros(len(signal), dtype=bool)
    for start, decision in zip(starts, active):
        if decision:
            mask[start:start + ACTIVITY_FRAME_SAMPLES] = True
    return mask


def snr_db(reference: Any, noise: Any, activity: Any) -> float:
    speech = np.asarray(reference, dtype=np.float64)
    interference = np.asarray(noise, dtype=np.float64)
    mask = np.asarray(activity, dtype=bool)
    if speech.ndim != 1 or interference.shape != speech.shape or mask.shape != speech.shape or not mask.any():
        raise ValueError("SNR requires aligned mono signals and a nonempty activity mask")
    speech_power = float(np.mean(speech[mask] ** 2))
    noise_power = float(np.mean(interference[mask] ** 2))
    if speech_power <= 0 or noise_power <= 0 or not math.isfinite(speech_power + noise_power):
        raise ValueError("SNR signal/noise power must be finite and nonzero")
    return float(10 * np.log10(speech_power / noise_power))


def mix_at_snr(speech_field: Any, noise_field: Any, requested_snr_db: float,
               activity: Any) -> tuple[np.ndarray, np.ndarray, float]:
    """Return mixture, scaled-noise field and one noise gain shared by all mics."""
    speech = np.asarray(speech_field, dtype=np.float64)
    noise = np.asarray(noise_field, dtype=np.float64)
    if speech.ndim != 2 or speech.shape[1] != 6 or noise.shape != speech.shape:
        raise ValueError("Speech/noise fields must have the same [samples,6] shape")
    if not np.isfinite(speech).all() or not np.isfinite(noise).all() or not math.isfinite(requested_snr_db):
        raise ValueError("Mixture inputs and SNR must be finite")
    before = snr_db(speech[:, REFERENCE_CHANNEL], noise[:, REFERENCE_CHANNEL], activity)
    gain = float(10 ** ((before - requested_snr_db) / 20))
    scaled_noise = noise * gain
    return speech + scaled_noise, scaled_noise, gain


def condition_plan(sources: list[dict[str, Any]], noises: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each source gets both noises once; SNR assignment crosses by source index."""
    if len(noises) != 2:
        raise ValueError("The initial diagnostic protocol requires exactly two noise recordings")
    plan = []
    for index, source in enumerate(sorted(sources, key=lambda row: row["source_id"])):
        for noise_index, noise in enumerate(noises):
            level = [0, 5][(index + noise_index) % 2]
            plan.append({"sample_id": f"{source['source_id']}__{noise['noise_id']}__snr{level}",
                         "source_id": source["source_id"], "speaker_id": source["speaker_id"],
                         "noise_id": noise["noise_id"], "noise_index": noise_index, "snr_db": level})
    return plan


def _read_audio(path: Path, *, maximum_seconds: float | None = None) -> np.ndarray:
    info = sf.info(path)
    if info.channels != 1 or info.samplerate != SAMPLE_RATE or info.frames <= 0:
        raise ValueError(f"Expected nonempty mono 16000Hz audio: {path}")
    if maximum_seconds is not None and info.frames > maximum_seconds * SAMPLE_RATE:
        raise ValueError(f"Source exceeds {maximum_seconds}s; no automatic cropping: {path}")
    samples, rate = sf.read(path, dtype="float64", always_2d=False)
    if rate != SAMPLE_RATE or samples.ndim != 1 or not np.isfinite(samples).all():
        raise ValueError(f"Invalid audio samples: {path}")
    return samples


def _input_records(manifest: Path, *, noise: bool) -> list[dict[str, Any]]:
    raw_rows = load_jsonl(manifest)
    rows = []
    seen = set()
    for item in raw_rows:
        row = dict(item)
        key = "noise_id" if noise else "source_id"
        if noise and key not in row and "id" in row:
            row[key] = row["id"]
        identifier = _safe_id(row.get(key), key)
        if identifier in seen:
            raise ValueError(f"Repeated {key}: {identifier}")
        seen.add(identifier)
        audio_path = (manifest.parent / _nonempty_string(row.get("path"), "path")).resolve()
        if not audio_path.is_file():
            raise ValueError(f"Audio file does not exist: {audio_path}")
        if noise:
            _nonempty_string(row.get("noise_type"), "noise_type")
        else:
            _nonempty_string(row.get("speaker_id"), "speaker_id")
            _nonempty_string(row.get("transcript"), "transcript")
        if "sample_rate" in row and row["sample_rate"] != SAMPLE_RATE:
            raise ValueError(f"Incorrect declared sample rate: {identifier}")
        samples = _read_audio(audio_path, maximum_seconds=None if noise else MAX_SOURCE_SECONDS)
        duration = len(samples) / SAMPLE_RATE
        if "duration_seconds" in row and (
                not isinstance(row["duration_seconds"], (int, float))
                or not math.isfinite(row["duration_seconds"])
                or abs(duration - row["duration_seconds"]) > 1.01 / SAMPLE_RATE):
            raise ValueError(f"Incorrect declared duration: {identifier}")
        rows.append(row | {"path": str(audio_path), "sha256": sha256_file(audio_path),
                           "samples": len(samples), "duration_seconds": duration})
    return rows


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _stable_seed(seed: int, sample_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}|{sample_id}".encode()).digest()[:8], "big")


def build_diagnostics(sources_path: str | Path, noises_path: str | Path,
                      geometry_path: str | Path, output_dir: str | Path, *,
                      seed: int = 20261106) -> dict[str, Any]:
    """Freeze selected inputs/protocol, then write 40--60 physical mixtures.

    The destination must not exist. On failure, partial evidence remains for
    inspection; this function never removes or overwrites an existing tree.
    """
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing diagnostic directory: {destination}")
    source_manifest, noise_manifest, geometry_file = map(
        lambda p: Path(p).resolve(), (sources_path, noises_path, geometry_path))
    sources = _input_records(source_manifest, noise=False)
    noises = _input_records(noise_manifest, noise=True)
    if not 20 <= len(sources) <= 30 or not 4 <= len({r["speaker_id"] for r in sources}) <= 6:
        raise ValueError("Initial protocol requires 20--30 source utterances and 4--6 speakers")
    if len(noises) != 2 or len({row["noise_type"] for row in noises}) != 2:
        raise ValueError("Initial protocol requires two distinct noise recordings/types")
    if any(row["samples"] < max(source["samples"] for source in sources) for row in noises):
        raise ValueError("Each noise recording must be at least as long as the longest source; no implicit looping")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    geometry = load_geometry(geometry_file)
    plan = condition_plan(sources, noises)
    protocol = {
        "protocol_version": 1, "stage": "G1_discovery", "split": "discovery",
        "sample_rate": SAMPLE_RATE, "reference_channel_zero_based": REFERENCE_CHANNEL,
        "geometry": geometry, "propagation": "far_field_fractional_delay_plane_wave",
        "direction_convention": "unit vectors point from array toward source; arrival=-relative_position.dot(direction)/c",
        "noise_spatial_model": "synthetic_directional_noise; original recording spatial field is not retained",
        "reverberation": "none", "target": "reference_channel_speech_image",
        "fir_half_width": FIR_HALF_WIDTH, "fir_window": "kaiser_beta_8.6",
        "guard_samples_each_side": GUARD_SAMPLES, "source_samples_cropped": 0,
        "max_source_seconds": MAX_SOURCE_SECONDS, "max_output_seconds": MAX_OUTPUT_SECONDS,
        "snr_definition": {"channel": REFERENCE_CHANNEL, "frame_samples": ACTIVITY_FRAME_SAMPLES,
                           "hop_samples": ACTIVITY_HOP_SAMPLES, "frame_rms_threshold_db_below_max": ACTIVITY_THRESHOLD_DB,
                           "active_frame_expansion": "union of all samples covered by active frames"},
        "snr_db": [0, 5], "assignment": "two noises/source; crossed 0/5 dB by sorted source index",
        "clipping_policy": "one scale<=1 shared by both conditions and every speech/noise channel per source",
        "clipping_peak": CLIPPING_PEAK, "output_encoding": "WAV/PCM_24",
        "seed": seed, "training_performed": False, "model_outputs_inspected_for_selection": False,
        "is_final_test_set": False,
    }
    frozen = {"source_manifest": {"path": str(source_manifest), "sha256": sha256_file(source_manifest)},
              "noise_manifest": {"path": str(noise_manifest), "sha256": sha256_file(noise_manifest)},
              "geometry_file": {"path": str(geometry_file), "sha256": sha256_file(geometry_file)},
              "sources": sources, "noises": noises, "condition_plan": plan,
              "protocol_sha256": canonical_hash(protocol)}
    frozen["frozen_inputs_sha256"] = canonical_hash(frozen)
    destination.mkdir(parents=True, exist_ok=False)
    _write_json(destination / "protocol.json", protocol)
    _write_json(destination / "frozen_inputs.json", frozen)
    _write_json(destination / "BUILD_STARTED.json", {"status": "building", "frozen_inputs_sha256": frozen["frozen_inputs_sha256"]})
    (destination / "noisy").mkdir()
    (destination / "reference").mkdir()
    source_by_id = {row["source_id"]: row for row in sources}
    noise_by_id = {row["noise_id"]: row for row in noises}
    noise_audio = {key: _read_audio(Path(row["path"])) for key, row in noise_by_id.items()}
    delays_target = geometric_delays(geometry["microphone_positions_m"], geometry["target_direction_unit"],
                                      speed_of_sound=geometry["speed_of_sound_m_s"])
    delays_noise = [geometric_delays(geometry["microphone_positions_m"], direction,
                                    speed_of_sound=geometry["speed_of_sound_m_s"])
                    for direction in geometry["noise_directions_unit"]]
    output_records = []
    for source_id in sorted(source_by_id):
        source = source_by_id[source_id]
        mono = _read_audio(Path(source["path"]), maximum_seconds=MAX_SOURCE_SECONDS)
        speech = render_plane_wave(mono, delays_target)
        if speech.shape[0] > MAX_OUTPUT_SECONDS * SAMPLE_RATE:
            raise ValueError("Padded output exceeds 4s; no cropping applied")
        reference = speech[:, REFERENCE_CHANNEL]
        activity = speech_activity_mask(reference)
        conditions = []
        for row in [item for item in plan if item["source_id"] == source_id]:
            sample_seed = _stable_seed(seed, row["sample_id"])
            rng = np.random.default_rng(sample_seed)
            recording = noise_audio[row["noise_id"]]
            start = int(rng.integers(0, len(recording) - len(mono) + 1))
            field = render_plane_wave(recording[start:start + len(mono)], delays_noise[row["noise_index"]])
            mixture, scaled_noise, noise_gain = mix_at_snr(speech, field, row["snr_db"], activity)
            conditions.append((row, mixture, scaled_noise, noise_gain, start, sample_seed))
        maximum = max(float(np.max(np.abs(reference))),
                      *(float(np.max(np.abs(item[1]))) for item in conditions))
        common_scale = min(1.0, CLIPPING_PEAK / maximum)
        reference_relative = f"reference/{source_id}.wav"
        sf.write(destination / reference_relative, reference * common_scale,
                 SAMPLE_RATE, format="WAV", subtype="PCM_24")
        reference_written, _ = sf.read(destination / reference_relative, dtype="float64")
        reference_hash = sha256_file(destination / reference_relative)
        for row, mixture, scaled_noise, noise_gain, start, sample_seed in conditions:
            noisy_relative = f"noisy/{row['sample_id']}.wav"
            sf.write(destination / noisy_relative, mixture * common_scale,
                     SAMPLE_RATE, format="WAV", subtype="PCM_24")
            written, _ = sf.read(destination / noisy_relative, dtype="float64", always_2d=True)
            measured = snr_db(reference_written, written[:, REFERENCE_CHANNEL] - reference_written, activity)
            if abs(measured - row["snr_db"]) > 0.05:
                raise ValueError(f"PCM24 SNR mismatch for {row['sample_id']}: {measured}")
            output_records.append({
                "sample_id": row["sample_id"], "source_id": source_id, "speaker_id": source["speaker_id"],
                "split": "discovery", "noisy_path": noisy_relative, "reference_path": reference_relative,
                "transcript": source["transcript"], "noise_id": row["noise_id"], "snr_db": row["snr_db"],
                "sample_rate": SAMPLE_RATE, "ref_channel": REFERENCE_CHANNEL,
                "noise_type": noise_by_id[row["noise_id"]]["noise_type"], "noise_segment_start_sample": start,
                "noise_segment_samples": len(mono), "noise_gain": noise_gain, "common_scale": common_scale,
                "achieved_snr_db_pcm24": measured, "speech_active_samples": int(activity.sum()),
                "source_duration_seconds": len(mono) / SAMPLE_RATE,
                "duration_seconds": len(reference) / SAMPLE_RATE, "mix_seed": sample_seed,
                "target_delays_samples": delays_target.tolist(),
                "noise_delays_samples": delays_noise[row["noise_index"]].tolist(),
                "array_domain_label": geometry["array_domain_label"],
                "noise_spatial_model": protocol["noise_spatial_model"],
                "source_sha256": source["sha256"], "noise_source_sha256": noise_by_id[row["noise_id"]]["sha256"],
                "noisy_sha256": sha256_file(destination / noisy_relative), "reference_sha256": reference_hash,
                "protocol_sha256": frozen["protocol_sha256"], "frozen_inputs_sha256": frozen["frozen_inputs_sha256"],
            })
    with (destination / "manifest.jsonl").open("x", encoding="utf-8") as stream:
        for row in output_records:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    summary = {
        "status": "complete", "source_utterances": len(sources),
        "speakers": len({row["speaker_id"] for row in sources}), "mixtures": len(output_records),
        "conditions": dict(sorted(Counter(f"{row['noise_type']}|{row['snr_db']}dB" for row in output_records).items())),
        "max_duration_seconds": max(row["duration_seconds"] for row in output_records),
        "max_snr_error_db_pcm24": max(abs(row["achieved_snr_db_pcm24"] - row["snr_db"]) for row in output_records),
        "array_domain_label": geometry["array_domain_label"],
        "noise_spatial_model": protocol["noise_spatial_model"],
        "manifest_sha256": sha256_file(destination / "manifest.jsonl"),
        "protocol_sha256": frozen["protocol_sha256"], "frozen_inputs_sha256": frozen["frozen_inputs_sha256"],
        "enhancement_or_asr_executed": False, "training_performed": False,
        "notice": "Synthetic directional-noise discovery set; no intrinsic McNet error or human intelligibility conclusion follows from generation.",
    }
    _write_json(destination / "summary.json", summary)
    _write_json(destination / "BUILD_COMPLETE.json", {"status": "complete", "manifest_sha256": summary["manifest_sha256"]})
    return summary
