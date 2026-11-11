"""
Unit Tests for Candidate Generation Reproducibility and Determinism (Phase 3.3)
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection
"""

import unittest
import numpy as np
from research_upgrade.src.candidate_generation.score_fusion import (
    LogisticRegressionCandidateScorer,
    RandomForestCandidateScorer
)
from research_upgrade.src.candidate_generation.threshold_selection import ThresholdSelector


class TestCandidateReproducibility(unittest.TestCase):
    def setUp(self):
        self.feature_names = ["f1", "f2", "f3", "f4"]
        np.random.seed(42)
        self.X_tr = np.random.uniform(0.0, 1.0, size=(100, 4))
        self.y_tr = (self.X_tr[:, 0] + self.X_tr[:, 1] > 0.9).astype(np.int32)
        self.X_te = np.random.uniform(0.0, 1.0, size=(30, 4))

    def test_logistic_regression_determinism(self):
        s1 = LogisticRegressionCandidateScorer(self.feature_names, random_state=42).fit(self.X_tr, self.y_tr)
        preds1 = s1.predict_score(self.X_te)

        s2 = LogisticRegressionCandidateScorer(self.feature_names, random_state=42).fit(self.X_tr, self.y_tr)
        preds2 = s2.predict_score(self.X_te)

        np.testing.assert_allclose(preds1, preds2, atol=1e-7, err_msg="Logistic Regression scores diverged!")

    def test_random_forest_determinism(self):
        s1 = RandomForestCandidateScorer(self.feature_names, n_estimators=50, random_state=42).fit(self.X_tr, self.y_tr)
        preds1 = s1.predict_score(self.X_te)

        s2 = RandomForestCandidateScorer(self.feature_names, n_estimators=50, random_state=42).fit(self.X_tr, self.y_tr)
        preds2 = s2.predict_score(self.X_te)

        np.testing.assert_allclose(preds1, preds2, atol=1e-7, err_msg="Random Forest scores diverged!")

    def test_threshold_selection_determinism(self):
        selector = ThresholdSelector(num_grid_points=200)
        s = LogisticRegressionCandidateScorer(self.feature_names, random_state=42).fit(self.X_tr, self.y_tr)
        scores = s.predict_score(self.X_tr)

        t1 = selector.select_recall_constrained_threshold(self.y_tr, scores, target_recall=0.98)
        t2 = selector.select_recall_constrained_threshold(self.y_tr, scores, target_recall=0.98)

        self.assertEqual(t1["selected_threshold"], t2["selected_threshold"])
        self.assertEqual(t1["validation_metrics"]["candidate_recall"], t2["validation_metrics"]["candidate_recall"])
        self.assertEqual(t1["validation_metrics"]["workload_reduction"], t2["validation_metrics"]["workload_reduction"])


if __name__ == "__main__":
    unittest.main()

