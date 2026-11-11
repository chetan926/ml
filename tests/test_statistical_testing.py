"""
Unit tests for Comprehensive Statistical Evaluation Module.
"""

import unittest
import numpy as np
from research_upgrade.src.evaluation.statistical_testing import (
    compute_comprehensive_classification_metrics,
    compute_paired_bootstrap_confidence_intervals,
    compute_mcnemar_test,
)


class TestStatisticalTesting(unittest.TestCase):
    """Test suite for 20 classification metrics, bootstrap CIs, and McNemar testing."""

    def setUp(self):
        np.random.seed(42)
        self.y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0] * 10)
        self.y_pred = np.array([1, 1, 0, 0, 0, 1, 1, 0, 1, 0] * 10)
        self.y_prob = np.where(self.y_pred == 1, 0.85, 0.15)

    def test_comprehensive_metrics_keys(self):
        res = compute_comprehensive_classification_metrics(self.y_true, self.y_pred, self.y_prob)
        required_keys = [
            "accuracy", "precision", "recall", "f1_score", "f0_5_score", "f2_score",
            "specificity", "negative_predictive_value", "false_positive_rate",
            "false_negative_rate", "false_discovery_rate", "false_omission_rate",
            "balanced_accuracy", "matthews_corrcoef", "cohens_kappa", "youdens_j",
            "geometric_mean", "jaccard_score", "threat_score", "diagnostic_odds_ratio",
            "roc_auc", "pr_auc", "average_precision"
        ]
        for k in required_keys:
            self.assertIn(k, res, f"Missing key: {k}")
        self.assertGreater(res["accuracy"], 0.7)
        self.assertGreater(res["f1_score"], 0.7)

    def test_paired_bootstrap(self):
        cis = compute_paired_bootstrap_confidence_intervals(self.y_true, self.y_pred, n_resamples=50)
        for metric in ["mcc", "f1_score", "recall", "precision"]:
            self.assertIn(metric, cis)
            self.assertLessEqual(cis[metric]["ci_lower"], cis[metric]["ci_upper"])

    def test_mcnemar_test(self):
        y_baseline = np.zeros_like(self.y_true)
        mcnemar = compute_mcnemar_test(self.y_true, self.y_pred, y_baseline)
        self.assertIn("chi2_statistic", mcnemar)
        self.assertIn("p_value", mcnemar)
        self.assertTrue(mcnemar["significant_at_05"])


if __name__ == "__main__":
    unittest.main()

