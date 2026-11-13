from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mcse.evaluation import (AudioValidationError, MetricEvaluator, si_sdr,
                             summarize_cases, validate_pair, waveform_diagnostics)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        n = np.arange(16000)
        self.reference = np.sin(2 * np.pi * 137 * n / 16000)
        self.noise = np.cos(2 * np.pi * 521 * n / 16000)

    def test_si_sdr_known_orthogonal_error_has_analytic_value(self):
        value = si_sdr(self.reference, self.reference + self.noise / 3)
        self.assertAlmostEqual(value, 10 * np.log10(9), places=9)

    def test_si_sdr_is_global_estimate_gain_invariant(self):
        estimate = self.reference + self.noise / 3
        self.assertAlmostEqual(si_sdr(self.reference, estimate), si_sdr(self.reference, estimate * 0.007), places=9)
        self.assertAlmostEqual(si_sdr(self.reference, self.reference), si_sdr(self.reference, self.reference * 7), places=9)

    def test_si_sdr_is_zero_mean_but_does_not_fit_delay(self):
        estimate = self.reference + self.noise / 3
        self.assertAlmostEqual(si_sdr(self.reference + 0.6, estimate + 1.2), si_sdr(self.reference, estimate), places=9)
        self.assertLess(si_sdr(self.reference, np.roll(self.reference, 17)), 0)

    def test_zero_reference_and_zero_estimate_are_explicitly_undefined(self):
        with self.assertRaises(AudioValidationError):
            si_sdr(np.zeros(16000), self.reference)
        with self.assertRaises(AudioValidationError):
            si_sdr(self.reference, np.zeros(16000))

    def test_validation_rejects_shape_length_rate_and_nonfinite(self):
        for ref, est, sr in (
            (self.reference[:-1], self.reference, 16000),
            (self.reference[:, None], self.reference[:, None], 16000),
            (self.reference, self.reference, 8000),
            (self.reference, self.reference * np.nan, 16000),
        ):
            with self.subTest(rate=sr, shape=np.shape(est)):
                with self.assertRaises(AudioValidationError):
                    validate_pair(ref, est, sr)

    def test_optional_backends_missing_return_none_and_unavailable_not_fake_zero(self):
        with mock.patch("mcse.evaluation.importlib.import_module", side_effect=ModuleNotFoundError("not installed")):
            evaluator = MetricEvaluator()
        scores = evaluator.evaluate(self.reference, self.reference + self.noise / 3)
        self.assertIsNone(scores["stoi"])
        self.assertIsNone(scores["pesq_wb"])
        self.assertEqual(scores["status"]["stoi"]["status"], "unavailable")
        self.assertIsNotNone(scores["si_sdr_db"])

    def test_optional_backend_failure_is_distinct_from_absence(self):
        with mock.patch("mcse.evaluation.importlib.import_module", side_effect=ModuleNotFoundError("not installed")):
            evaluator = MetricEvaluator()
        evaluator.backends["stoi"] = mock.Mock(side_effect=ValueError("too short"))
        scores = evaluator.evaluate(self.reference, self.reference)
        self.assertIsNone(scores["stoi"])
        self.assertEqual(scores["status"]["stoi"]["status"], "error")

    def test_energy_diagnostic_keeps_physical_gain_and_marks_local_underestimation(self):
        ref = np.concatenate((np.zeros(3200), self.reference))
        diag = waveform_diagnostics(ref, ref * 0.25)
        self.assertAlmostEqual(diag["estimate_abs_peak"], 0.25)
        self.assertEqual(diag["active_reference_underestimated_6db_fraction"], 1)
        self.assertEqual(diag["inactive_frame_estimate_mean_square"], 0)
        self.assertIn("not phoneme", diag["notice"])

    def test_summary_uses_per_metric_denominators_and_preserves_failures(self):
        cases = [
            {"status": "ok", "metrics": {
                "noisy": {"si_sdr_db": 2, "stoi": None, "pesq_wb": None},
                "enhanced": {"si_sdr_db": 5, "stoi": 0.7, "pesq_wb": None}}},
            {"status": "error", "error": "failed"},
        ]
        result = summarize_cases(cases)
        self.assertEqual(result["total_cases"], 2)
        self.assertEqual(result["failed_cases"], 1)
        self.assertEqual(result["metrics"]["si_sdr_db"]["paired_improvement"]["mean"], 3)
        self.assertEqual(result["metrics"]["stoi"]["enhanced"]["n"], 1)
        self.assertEqual(result["metrics"]["stoi"]["paired_improvement"]["n"], 0)
        self.assertIsNone(result["metrics"]["pesq_wb"]["enhanced"]["mean"])
        self.assertEqual(result["human_content_loss_judgement"], "not_assessed")


if __name__ == "__main__":
    unittest.main()
