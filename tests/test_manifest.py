"""Small synthetic PCM fixtures test validation only, never enhancement quality."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcse.manifest import validate_manifest


class ManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.manifest = self.folder / "samples.jsonl"

    def record(self, **overrides):
        record = {
            "sample_id": "utterance-a_noise-a_snr0",
            "source_id": "utterance-a", "speaker_id": "speaker-a",
            "split": "discovery", "noisy_path": "audio/noisy.wav",
            "reference_path": "audio/reference.wav", "transcript": "A short test.",
            "noise_id": "noise-a", "snr_db": 0, "sample_rate": 16000, "ref_channel": 4,
        }
        return record | overrides

    def write_manifest(self, *records):
        self.manifest.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    def write_wav(self, filename, channels, *, frames=320, sample_rate=16000):
        path = self.folder / "audio" / filename
        path.parent.mkdir(exist_ok=True)
        with wave.open(str(path), "wb") as output:
            output.setnchannels(channels)
            output.setsampwidth(2)
            output.setframerate(sample_rate)
            output.writeframes(b"\x01\x00" * channels * frames)
        return path

    def codes(self, report):
        return {issue.code for issue in report.errors}

    def test_paths_resolve_from_manifest_without_requiring_audio(self):
        self.write_manifest(self.record())
        report = validate_manifest(self.manifest)
        self.assertTrue(report.valid)
        self.assertFalse(report.audio_checked)
        self.assertEqual(report.records[0].noisy_path, (self.folder / "audio/noisy.wav").resolve())
        self.assertEqual(report.to_dict()["unique_sources"], 1)

    def test_repeated_source_conditions_allowed_but_duplicate_samples_rejected(self):
        first = self.record()
        second = self.record(sample_id="utterance-a_noise-b_snr5", noise_id="noise-b", snr_db=5)
        self.write_manifest(first, second)
        self.assertTrue(validate_manifest(self.manifest).valid)
        self.write_manifest(first, first)
        self.assertIn("DUPLICATE_SAMPLE_ID", self.codes(validate_manifest(self.manifest)))

    def test_source_utterance_cannot_cross_discovery_confirmation(self):
        self.write_manifest(self.record(), self.record(sample_id="confirmation-a", split="confirmation"))
        self.assertIn("SOURCE_SPLIT_LEAKAGE", self.codes(validate_manifest(self.manifest)))

    def test_discovery_confirmation_may_share_speaker_with_distinct_sources(self):
        self.write_manifest(self.record(), self.record(sample_id="confirmation-b", source_id="utterance-b", split="confirmation"))
        self.assertTrue(validate_manifest(self.manifest).valid)

    def test_train_speaker_cannot_leak_into_validation_or_test(self):
        for split in ("validation", "test"):
            with self.subTest(split=split):
                self.write_manifest(self.record(split="train"), self.record(sample_id="eval-b", source_id="utterance-b", split=split))
                self.assertIn("SPEAKER_SPLIT_LEAKAGE", self.codes(validate_manifest(self.manifest)))

    def test_nonfinite_snr_and_wrong_reference_index_rejected(self):
        for overrides in (
            {"snr_db": float("nan")}, {"snr_db": float("inf")}, {"snr_db": True},
            {"snr_db": 10 ** 400}, {"ref_channel": 5}, {"sample_rate": 8000},
            {"source_id": "utterance-a "},
        ):
            with self.subTest(overrides=overrides):
                self.write_manifest(self.record(**overrides))
                self.assertIn("SCHEMA_ERROR", self.codes(validate_manifest(self.manifest)))

    def test_valid_integer_pcm_and_relative_paths(self):
        self.write_wav("noisy.wav", 6)
        self.write_wav("reference.wav", 1)
        self.write_manifest(self.record())
        self.assertTrue(validate_manifest(self.manifest, check_audio=True).valid)

    def test_incorrect_channel_count_and_rate_reported(self):
        self.write_wav("noisy.wav", 1, sample_rate=8000)
        self.write_wav("reference.wav", 2)
        self.write_manifest(self.record())
        codes = self.codes(validate_manifest(self.manifest, check_audio=True))
        self.assertIn("AUDIO_CHANNELS", codes)
        self.assertIn("AUDIO_SAMPLE_RATE", codes)

    def test_mismatched_lengths_rejected(self):
        self.write_wav("noisy.wav", 6, frames=320)
        self.write_wav("reference.wav", 1, frames=319)
        self.write_manifest(self.record())
        self.assertIn("AUDIO_LENGTH", self.codes(validate_manifest(self.manifest, check_audio=True)))

    def test_truncated_pcm_rejected(self):
        noisy = self.write_wav("noisy.wav", 6)
        noisy.write_bytes(noisy.read_bytes()[:-12])
        self.write_wav("reference.wav", 1)
        self.write_manifest(self.record())
        self.assertIn("AUDIO_READ_ERROR", self.codes(validate_manifest(self.manifest, check_audio=True)))

    def test_missing_and_empty_manifests_fail_without_creating_files(self):
        self.assertIn("MANIFEST_READ_ERROR", self.codes(validate_manifest(self.manifest)))
        self.assertFalse(self.manifest.exists())
        self.manifest.write_text("\n", encoding="utf-8")
        self.assertIn("EMPTY_MANIFEST", self.codes(validate_manifest(self.manifest)))

    def test_malformed_json_does_not_hide_other_records(self):
        self.manifest.write_text("{bad json}\n" + json.dumps(self.record()) + "\n", encoding="utf-8")
        report = validate_manifest(self.manifest)
        self.assertEqual(report.row_count, 2)
        self.assertEqual(len(report.records), 1)
        self.assertIn("SCHEMA_ERROR", self.codes(report))


if __name__ == "__main__":
    unittest.main()
