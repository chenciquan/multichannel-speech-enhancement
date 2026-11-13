"""Analytic checks for numerical quantities used in the follow-up decision."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("behavior_probe", ROOT/"scripts/probe_baseline_behavior.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class BehaviorProbeTests(unittest.TestCase):
    def test_known_gain_and_unfitted_difference(self):
        x = np.random.default_rng(73).normal(size=8192)
        result = probe.comparison(x, x*.25)
        self.assertAlmostEqual(result["projection_gain"], .25, places=12)
        self.assertAlmostEqual(result["correlation"], 1, places=12)
        self.assertAlmostEqual(result["rms_gain_db"], 20*np.log10(.25), places=12)
        self.assertAlmostEqual(result["difference_energy_relative_db"], 20*np.log10(.75), places=12)
        self.assertEqual(result["best_lag_within_16ms_samples"], 0)
        self.assertFalse(result["lag_search_used_to_modify_metrics_or_audio"])

    def test_lag_sign_and_no_automatic_alignment(self):
        x = np.zeros(4096)
        x[700:1700] = np.random.default_rng(31).normal(size=1000)
        y = np.concatenate([np.zeros(7), x[:-7]])
        result = probe.comparison(x, y)
        self.assertEqual(result["best_lag_within_16ms_samples"], 7)
        self.assertGreater(result["max_abs_difference"], 1)
        self.assertLess(result["correlation"], .2)

    def test_projection_recovers_known_components_and_keeps_unexplained_term(self):
        t = np.arange(4096)
        s = np.sin(2*np.pi*13*t/4096)
        n = np.cos(2*np.pi*17*t/4096)
        extra = np.sin(2*np.pi*23*t/4096)
        result = probe.projection(s, n, .8*s+.3*n+.1*extra)
        self.assertEqual(result["rank"], 2)
        self.assertAlmostEqual(result["speech_coefficient"], .8, places=12)
        self.assertAlmostEqual(result["noise_coefficient"], .3, places=12)
        self.assertAlmostEqual(result["unexplained_energy_fraction"], .01/(.64+.09+.01), places=12)

    def test_comparison_rejects_unaligned_or_nonfinite_audio(self):
        for x, y in ((np.ones(30), np.ones(31)), (np.ones((30,2)), np.ones((30,2))),
                     (np.ones(30), np.full(30,np.nan))):
            with self.subTest(shape=np.shape(y)), self.assertRaises(ValueError):
                probe.comparison(x, y)
