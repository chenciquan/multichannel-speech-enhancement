"""Numerical/temporary fixtures test propagation and data contracts, not speech.

No downloaded recordings or enhancement outputs are read by this test module.
"""
from __future__ import annotations

from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
HAS_DEPENDENCIES = all(importlib.util.find_spec(name) is not None
                       for name in ("numpy", "scipy", "soundfile"))


@unittest.skipUnless(HAS_DEPENDENCIES, "Requires installed numerical/audio inference dependencies")
class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import soundfile as sf
        from mcse import simulation
        cls.np, cls.sf, cls.sim = np, sf, simulation

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)

    def geometry(self):
        return {
            "geometry_id": "synthetic_fixture_array_not_a_real_corpus",
            "microphone_positions_m": [[index * .02, 0, 0] for index in range(6)],
            "reference_channel_zero_based": 4,
            "coordinate_convention": "+x along array; +y broadside; vectors toward source",
            "array_domain_label": "custom_array_domain_shift",
            "target_direction_unit": [0, 1, 0],
            "noise_directions_unit": [[1, 0, 0], [-1, 0, 0]],
            "speed_of_sound_m_s": 343.0,
        }

    def geometry_file(self, value=None):
        path = self.folder / "geometry.json"
        path.write_text(json.dumps(value or self.geometry()), encoding="utf-8")
        return path

    def fixtures(self, count=20):
        """Short synthetic tones/noise remain confined to this temporary folder."""
        np, sf = self.np, self.sf
        root = self.folder / "inputs"
        (root / "speech").mkdir(parents=True)
        (root / "noise").mkdir()
        n = 1600
        time = np.arange(n) / 16000
        sources = []
        for index in range(count):
            source_id = f"fixture-{index:02d}"
            # Deliberately large amplitude exercises one shared anti-clipping gain.
            audio = .8 * np.sin(2 * np.pi * (160 + 23 * index) * time)
            audio[:32] = 0
            audio[-32:] = 0
            relative = f"speech/{source_id}.flac"
            sf.write(root / relative, audio, 16000, format="FLAC", subtype="PCM_24")
            sources.append({"source_id": source_id, "speaker_id": f"fixture-speaker-{index % 4}",
                            "path": relative, "transcript": "SYNTHETIC NUMERICAL TEST ONLY",
                            "sample_rate": 16000, "duration_seconds": n / 16000,
                            "license": "temporary synthetic unit-test fixture; not a speech dataset"})
        rng = np.random.default_rng(983)
        noises = []
        for index in range(2):
            relative = f"noise/noise-{index}.wav"
            sf.write(root / relative, rng.normal(0, .2, 8000), 16000, format="WAV", subtype="PCM_24")
            noises.append({"noise_id": f"fixture-noise-{index}", "path": relative,
                           "noise_type": f"synthetic-fixture-type-{index}"})
        source_path, noise_path = root / "sources.jsonl", root / "noises.jsonl"
        for path, rows in ((source_path, sources), (noise_path, noises)):
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        return source_path, noise_path

    def test_geometric_arrival_sign_and_exact_integer_delays(self):
        np = self.np
        # One microphone pitch equals one sample of acoustic travel.
        pitch = 343.0 / 16000
        positions = [[index * pitch, 0, 0] for index in range(6)]
        delays = self.sim.geometric_delays(positions, [1, 0, 0])
        np.testing.assert_allclose(delays, [4, 3, 2, 1, 0, -1], atol=1e-12)
        mono = np.zeros(256)
        mono[64] = 1
        field = self.sim.render_plane_wave(mono, delays)
        # A microphone nearer +x must receive this +x source earlier.
        np.testing.assert_array_equal(np.argmax(field, axis=0), [132, 131, 130, 129, 128, 127])
        self.assertEqual(field.shape, (384, 6))

    def test_fractional_delay_follows_independent_sinusoidal_phase(self):
        np = self.np
        samples = np.arange(4096)
        omega = 2 * np.pi * 650 / 16000
        mono = np.sin(omega * samples)
        delays = np.array([0, .25, -.5, .8, 1.3, -1.7])
        field = self.sim.render_plane_wave(mono, delays)
        central = np.arange(164, 4000)
        expected = np.sin(omega * (central[:, None] - 64 - delays[None, :]))
        np.testing.assert_allclose(field[central], expected, atol=2e-4, rtol=0)

    def test_guard_retains_first_and_last_sample_impulses(self):
        np = self.np
        mono = np.zeros(300)
        mono[0], mono[-1] = .7, .3
        delays = np.array([-5.2, -3, -.4, .2, 0, 4.8])
        field = self.sim.render_plane_wave(mono, delays)
        self.assertEqual(field.shape, (428, 6))
        np.testing.assert_allclose(field.sum(axis=0), 1.0, atol=1e-12)
        np.testing.assert_allclose(field[:16], 0, atol=1e-14)
        np.testing.assert_allclose(field[-16:], 0, atol=1e-14)
        with self.assertRaisesRegex(ValueError, "Guard"):
            self.sim.render_plane_wave(mono, delays, guard_samples=20)

    def test_snr_uses_reference_activity_and_one_noise_gain_for_all_mics(self):
        np = self.np
        rng = np.random.default_rng(12)
        speech = rng.normal(0, .2, (2048, 6))
        noise = rng.normal(0, .6, (2048, 6))
        active = np.zeros(2048, dtype=bool)
        active[200:1600] = True
        mixed, scaled_noise, gain = self.sim.mix_at_snr(speech, noise, 5, active)
        independent_ratio = np.mean(speech[active, 4] ** 2) / np.mean((mixed - speech)[active, 4] ** 2)
        self.assertAlmostEqual(independent_ratio, 10 ** .5, places=12)
        np.testing.assert_allclose(scaled_noise, noise * gain, atol=1e-14)
        np.testing.assert_allclose(mixed - scaled_noise, speech, atol=1e-14)

    def test_activity_mask_excludes_distant_silence_and_rejects_empty_speech(self):
        np = self.np
        speech = np.zeros(4000)
        speech[1400:2000] = .1
        activity = self.sim.speech_activity_mask(speech)
        self.assertFalse(activity[:1000].any())
        self.assertTrue(activity[1500:1900].all())
        self.assertFalse(activity[2400:].any())
        with self.assertRaisesRegex(ValueError, "silent"):
            self.sim.speech_activity_mask(np.zeros(1000))

    def test_assignment_is_balanced_and_independent_of_source_manifest_order(self):
        sources = [{"source_id": f"s-{index:02d}", "speaker_id": f"p-{index % 4}"} for index in range(24)]
        noises = [{"noise_id": "washing"}, {"noise_id": "kitchen"}]
        plan = self.sim.condition_plan(sources, noises)
        self.assertEqual(plan, self.sim.condition_plan(list(reversed(sources)), noises))
        self.assertEqual(len(plan), 48)
        self.assertEqual(set(Counter((r["noise_id"], r["snr_db"]) for r in plan).values()), {12})
        for source in sources:
            selected = [row for row in plan if row["source_id"] == source["source_id"]]
            self.assertEqual({row["snr_db"] for row in selected}, {0, 5})
            self.assertEqual({row["noise_id"] for row in selected}, {"washing", "kitchen"})

    def test_geometry_requires_domain_label_unit_directions_and_distinct_mics(self):
        loaded = self.sim.load_geometry(self.geometry_file())
        self.assertEqual(loaded["array_domain_label"], "custom_array_domain_shift")
        for bad in (
                self.geometry() | {"array_domain_label": ""},
                self.geometry() | {"target_direction_unit": [0, 2, 0]},
                self.geometry() | {"microphone_positions_m": [[0, 0, 0]] * 6}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.sim.load_geometry(self.geometry_file(bad))

    def test_complete_build_is_deterministic_pcm24_and_manifest_valid(self):
        from mcse.manifest import validate_manifest
        source_path, noise_path = self.fixtures()
        geometry = self.geometry_file()
        first, second = self.folder / "out-a", self.folder / "out-b"
        summary = self.sim.build_diagnostics(source_path, noise_path, geometry, first)
        repeated = self.sim.build_diagnostics(source_path, noise_path, geometry, second)
        self.assertEqual(summary["mixtures"], 40)
        self.assertEqual(summary["speakers"], 4)
        self.assertLess(summary["max_snr_error_db_pcm24"], .001)
        self.assertEqual(summary["manifest_sha256"], repeated["manifest_sha256"])
        self.assertEqual(summary["frozen_inputs_sha256"], repeated["frozen_inputs_sha256"])
        self.assertTrue(validate_manifest(first / "manifest.jsonl", check_audio=True).valid)
        records = self.sim.load_jsonl(first / "manifest.jsonl")
        self.assertTrue((first / "BUILD_COMPLETE.json").is_file())
        self.assertTrue((first / "frozen_inputs.json").is_file())
        self.assertTrue(any(row["common_scale"] < 1 for row in records))
        for row in records:
            info = self.sf.info(first / row["noisy_path"])
            self.assertEqual((info.channels, info.samplerate, info.subtype), (6, 16000, "PCM_24"))
            audio, _ = self.sf.read(first / row["noisy_path"], always_2d=True)
            self.assertLessEqual(float(self.np.max(self.np.abs(audio))), .9500001)
            self.assertEqual(row["noise_spatial_model"].split(";")[0], "synthetic_directional_noise")
            self.assertEqual(row["array_domain_label"], "custom_array_domain_shift")
        for index in range(0, len(records), 2):
            self.assertEqual(records[index]["reference_path"], records[index + 1]["reference_path"])
            self.assertEqual(records[index]["common_scale"], records[index + 1]["common_scale"])

    def test_existing_destination_is_never_overwritten(self):
        destination = self.folder / "existing"
        destination.mkdir()
        marker = destination / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.sim.build_diagnostics("missing-sources", "missing-noises", "missing-geometry", destination)
        self.assertEqual(marker.read_text(), "keep")

    def test_incomplete_source_selection_fails_before_creating_destination(self):
        source_path, noise_path = self.fixtures(count=19)
        destination = self.folder / "not-created"
        with self.assertRaisesRegex(ValueError, "20--30"):
            self.sim.build_diagnostics(source_path, noise_path, self.geometry_file(), destination)
        self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
