"""
Unit tests for Calibration and Uncertainty Module (Phase 3.4).
"""

import unittest
import numpy as np
from research_upgrade.src.calibration import (
    UncalibratedBaseline,
    PlattScaler,
    IsotonicCalibrator,
    TemperatureScaler,
    compute_calibration_metrics,
    compute_predictive_entropy,
    compute_confidence_margin,
    compute_maximum_class_probability,
    evaluate_selective_prediction,
    TextGroupConformalPredictor,
)


class TestCalibrationModule(unittest.TestCase):
    """Test suite for probability calibration, metrics, uncertainty, and conformal prediction."""

    def setUp(self):
        np.random.seed(42)
        self.N = 200
        # Synthetic scores with separation
        y = np.random.binomial(1, 0.4, size=self.N)
        scores = np.where(y == 1, np.random.beta(5, 2, size=self.N), np.random.beta(2, 5, size=self.N))
        self.y = y
        self.scores = np.clip(scores, 0.01, 0.99)

    def test_uncalibrated_baseline(self):
        cal = UncalibratedBaseline()
        cal.fit(self.scores, self.y)
        proba = cal.predict_proba(self.scores)
        self.assertEqual(proba.shape, (self.N, 2))
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(self.N), atol=1e-5)

    def test_platt_scaler(self):
        cal = PlattScaler(C=1.0)
        cal.fit(self.scores, self.y)
        proba = cal.predict_proba(self.scores)
        self.assertEqual(proba.shape, (self.N, 2))
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(self.N), atol=1e-5)
        p1 = cal.predict_positive_proba(self.scores)
        self.assertTrue(np.all((p1 >= 0.0) & (p1 <= 1.0)))

    def test_isotonic_calibrator(self):
        cal = IsotonicCalibrator()
        cal.fit(self.scores, self.y)
        proba = cal.predict_proba(self.scores)
        self.assertEqual(proba.shape, (self.N, 2))
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(self.N), atol=1e-5)

    def test_temperature_scaler(self):
        cal = TemperatureScaler()
        cal.fit(self.scores, self.y)
        proba = cal.predict_proba(self.scores)
        self.assertEqual(proba.shape, (self.N, 2))
        self.assertGreater(cal.temperature, 0.0)

    def test_calibration_metrics(self):
        metrics = compute_calibration_metrics(self.y, self.scores, n_bins=5)
        self.assertIn("brier_score", metrics)
        self.assertIn("expected_calibration_error", metrics)
        self.assertIn("adaptive_ece", metrics)
        self.assertIn("maximum_calibration_error", metrics)
        self.assertIn("reliability_bins", metrics)
        self.assertGreaterEqual(metrics["brier_score"], 0.0)
        self.assertGreaterEqual(metrics["expected_calibration_error"], 0.0)

    def test_uncertainty_estimators(self):
        entropy = compute_predictive_entropy(np.array([0.5, 0.99, 0.01]))
        self.assertAlmostEqual(entropy[0], 1.0, places=2)
        self.assertLess(entropy[1], 0.1)

        margin = compute_confidence_margin(np.array([0.5, 0.9, 0.1]))
        self.assertAlmostEqual(margin[0], 0.0, places=2)
        self.assertAlmostEqual(margin[1], 0.8, places=2)

        conf = compute_maximum_class_probability(np.array([0.2, 0.8]))
        np.testing.assert_allclose(conf, np.array([0.8, 0.8]), atol=1e-5)

    def test_selective_prediction(self):
        res = evaluate_selective_prediction(self.y, self.scores, n_thresholds=10)
        self.assertIn("area_under_risk_coverage_curve", res)
        self.assertIn("risk_coverage_curve", res)
        self.assertGreaterEqual(res["area_under_risk_coverage_curve"], 0.0)

    def test_group_conformal_prediction(self):
        group_ids = np.repeat(np.arange(20), 10)  # 20 groups of 10 pairs each
        cp = TextGroupConformalPredictor(alpha=0.10)
        cp.calibrate(group_ids, self.y, self.scores)
        self.assertTrue(cp.is_calibrated)
        cov = cp.evaluate_test_coverage(group_ids, self.y, self.scores)
        self.assertIn("empirical_group_coverage", cov)
        self.assertIn("empirical_pair_coverage", cov)
        self.assertGreater(cov["empirical_group_coverage"], 0.5)


if __name__ == "__main__":
    unittest.main()

