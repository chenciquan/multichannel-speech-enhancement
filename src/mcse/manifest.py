"""Validate the small diagnostic JSONL manifest without third-party packages.

Paths are resolved relative to the manifest, not the caller's working directory.
Header/channel checks cannot establish microphone geometry, source independence,
copyright permission, or the absence of noise/RIR leakage. Those need separate
provenance review. Integer PCM samples are finite by construction; IEEE-float WAV
is deliberately unsupported by this standard-library-only checker.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import wave


SPLITS = frozenset({"discovery", "confirmation", "train", "validation", "test"})
STRING_FIELDS = (
    "sample_id", "source_id", "speaker_id", "split", "noisy_path",
    "reference_path", "transcript", "noise_id",
)
REQUIRED_FIELDS = frozenset((*STRING_FIELDS, "snr_db", "sample_rate", "ref_channel"))


@dataclass(frozen=True)
class ManifestRecord:
    sample_id: str
    source_id: str
    speaker_id: str
    split: str
    noisy_path: Path
    reference_path: Path
    transcript: str
    noise_id: str
    snr_db: float
    sample_rate: int
    ref_channel: int
    line: int
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    line: int | None = None
    sample_id: str | None = None


@dataclass
class ValidationReport:
    manifest: Path
    audio_checked: bool
    row_count: int = 0
    records: list[ManifestRecord] = field(default_factory=list)
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest": str(self.manifest),
            "valid": self.valid,
            "audio_checked": self.audio_checked,
            "row_count": self.row_count,
            "schema_valid_records": len(self.records),
            "unique_sources": len({r.source_id for r in self.records}),
            "unique_speakers": len({r.speaker_id for r in self.records}),
            "by_split": dict(sorted(Counter(r.split for r in self.records).items())),
            "errors": [asdict(issue) for issue in self.errors],
            "warnings": [asdict(issue) for issue in self.warnings],
        }


def _parse_record(raw: Any, line: int, base: Path) -> ManifestRecord:
    if not isinstance(raw, dict):
        raise ValueError("Each nonblank JSONL line must be an object.")
    missing = REQUIRED_FIELDS - raw.keys()
    if missing:
        raise ValueError(f"Missing fields: {', '.join(sorted(missing))}.")
    for name in STRING_FIELDS:
        if not isinstance(raw[name], str) or not raw[name].strip():
            raise ValueError(f"{name} must be a nonempty string.")
    for name in ("sample_id", "source_id", "speaker_id", "noise_id", "split"):
        if raw[name] != raw[name].strip():
            raise ValueError(f"{name} must not have leading or trailing whitespace.")
    if raw["split"] not in SPLITS:
        raise ValueError(f"split must be one of {', '.join(sorted(SPLITS))}.")
    if type(raw["sample_rate"]) is not int or raw["sample_rate"] != 16000:
        raise ValueError("sample_rate must be the integer 16000.")
    if type(raw["ref_channel"]) is not int or raw["ref_channel"] != 4:
        raise ValueError("ref_channel must be the zero-based integer 4 (fifth microphone).")
    snr = raw["snr_db"]
    try:
        valid_snr = not isinstance(snr, bool) and isinstance(snr, (int, float)) and math.isfinite(snr)
    except OverflowError:
        valid_snr = False
    if not valid_snr:
        raise ValueError("snr_db must be a finite number.")
    values = {name: raw[name] for name in STRING_FIELDS}
    for name in ("noisy_path", "reference_path"):
        values[name] = (base / raw[name]).resolve()
    return ManifestRecord(
        **values, snr_db=float(snr), sample_rate=16000, ref_channel=4, line=line,
        extras={name: value for name, value in raw.items() if name not in REQUIRED_FIELDS},
    )


def _read_pcm_header_and_check_data(path: Path) -> tuple[int, int, int]:
    """Check integer PCM and detect truncated payloads without loading whole files."""
    with wave.open(str(path), "rb") as audio:
        channels, width, rate, frames, compression, _ = audio.getparams()
        if compression != "NONE" or width not in (1, 2, 3, 4):
            raise ValueError("Only uncompressed 8/16/24/32-bit integer PCM WAV is supported.")
        if frames <= 0:
            raise ValueError("Audio has no frames.")
        expected = channels * width * frames
        received = 0
        while True:
            block = audio.readframes(65536)
            if not block:
                break
            received += len(block)
        if received != expected:
            raise ValueError(f"Truncated PCM payload: expected {expected} bytes, read {received}.")
        return channels, rate, frames


def _check_audio(record: ManifestRecord, report: ValidationReport) -> None:
    headers: dict[str, tuple[int, int, int]] = {}
    for name, channels in (("noisy_path", 6), ("reference_path", 1)):
        path = getattr(record, name)
        try:
            header = _read_pcm_header_and_check_data(path)
            headers[name] = header
        except (OSError, ValueError, wave.Error, EOFError) as error:
            report.errors.append(ValidationIssue(
                "AUDIO_READ_ERROR", f"{name}: {path}: {error}", record.line, record.sample_id,
            ))
            continue
        if header[0] != channels:
            report.errors.append(ValidationIssue(
                "AUDIO_CHANNELS", f"{name} requires {channels} channel(s), got {header[0]}.",
                record.line, record.sample_id,
            ))
        if header[1] != record.sample_rate:
            report.errors.append(ValidationIssue(
                "AUDIO_SAMPLE_RATE", f"{name} requires {record.sample_rate} Hz, got {header[1]}.",
                record.line, record.sample_id,
            ))
    if len(headers) == 2 and headers["noisy_path"][2] != headers["reference_path"][2]:
        report.errors.append(ValidationIssue(
            "AUDIO_LENGTH", "Noisy and reference WAV files must have the same frame count.",
            record.line, record.sample_id,
        ))


def validate_manifest(path: str | Path, *, check_audio: bool = False) -> ValidationReport:
    """Return all discovered validation issues; never create or modify input files.

    Source IDs cannot cross any split. Training speakers cannot also occur in
    validation/test. Discovery and confirmation may share speakers; their source
    utterances must still be different. Multiple conditions of one source within
    one split are allowed, but every sample_id must be unique.
    """
    manifest = Path(path).resolve()
    report = ValidationReport(manifest=manifest, audio_checked=check_audio)
    try:
        stream = manifest.open(encoding="utf-8-sig")
    except OSError as error:
        report.errors.append(ValidationIssue("MANIFEST_READ_ERROR", str(error)))
        return report
    try:
        with stream:
            for line, text in enumerate(stream, 1):
                if not text.strip():
                    continue
                report.row_count += 1
                try:
                    record = _parse_record(json.loads(text), line, manifest.parent)
                except (json.JSONDecodeError, ValueError) as error:
                    report.errors.append(ValidationIssue("SCHEMA_ERROR", str(error), line))
                    continue
                report.records.append(record)
    except (OSError, UnicodeError) as error:
        report.errors.append(ValidationIssue("MANIFEST_READ_ERROR", str(error)))
    if report.row_count == 0:
        report.errors.append(ValidationIssue("EMPTY_MANIFEST", "Manifest contains no records."))

    seen_samples: dict[str, ManifestRecord] = {}
    seen_sources: dict[str, ManifestRecord] = {}
    speaker_splits: dict[str, set[str]] = {}
    for record in report.records:
        if record.sample_id in seen_samples:
            report.errors.append(ValidationIssue(
                "DUPLICATE_SAMPLE_ID", f"sample_id also appears on line {seen_samples[record.sample_id].line}.",
                record.line, record.sample_id,
            ))
        else:
            seen_samples[record.sample_id] = record
        previous = seen_sources.setdefault(record.source_id, record)
        if previous.split != record.split:
            report.errors.append(ValidationIssue(
                "SOURCE_SPLIT_LEAKAGE", f"source_id occurs in {previous.split} and {record.split}.",
                record.line, record.sample_id,
            ))
        if previous.speaker_id != record.speaker_id:
            report.errors.append(ValidationIssue(
                "SOURCE_SPEAKER_MISMATCH", "One source_id is assigned to multiple speakers.",
                record.line, record.sample_id,
            ))
        speaker_splits.setdefault(record.speaker_id, set()).add(record.split)
        if check_audio:
            _check_audio(record, report)
    for speaker, splits in sorted(speaker_splits.items()):
        overlap = splits & {"validation", "test"}
        if "train" in splits and overlap:
            report.errors.append(ValidationIssue(
                "SPEAKER_SPLIT_LEAKAGE", f"Training speaker {speaker!r} also occurs in {', '.join(sorted(overlap))}.",
            ))
    report.warnings.append(ValidationIssue(
        "PROVENANCE_REVIEW_REQUIRED",
        "Channel count does not verify a real array. Review geometry, timing, noise/RIR splits and permissions separately.",
    ))
    return report
