"""
Unit Tests for Score Fusion Models (Phase 3.3)
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection
"""

import unittest
import numpy as np
from research_upgrade.src.candidate_generation.score_fusion import (
    SingleViewThresholdScorer,
    WeightedMultiViewScorer,
    LogisticRegressionCandidateScorer,
    RandomForestCandidateScorer,
    GradientBoostingCandidateScorer
)


class TestScoreFusionModels(unittest.TestCase):
    def setUp(self):
        self.feature_names = ["feat_a", "feat_b", "feat_c"]
        np.random.seed(42)
        self.X_train = np.random.uniform(0.0, 1.0, size=(100, 3))
        # Label correlated with feat_a and feat_b
        self.y_train = (self.X_train[:, 0] + self.X_train[:, 1] > 1.0).astype(np.int32)
        self.X_test = np.random.uniform(0.0, 1.0, size=(20, 3))

    def test_single_view_scorer(self):
        scorer = SingleViewThresholdScorer("feat_b", self.feature_names)
        scorer.fit(self.X_train, self.y_train)
        scores = scorer.predict_score(self.X_test)
        self.assertEqual(len(scores), 20)
        np.testing.assert_allclose(scores, self.X_test[:, 1])

    def test_weighted_multiview_scorer(self):
        scorer = WeightedMultiViewScorer(self.feature_names)
        scorer.fit(self.X_train, self.y_train)
        scores = scorer.predict_score(self.X_test)
        self.assertEqual(len(scores), 20)
        self.assertTrue(np.all(scores >= 0.0) and np.all(scores <= 1.0))

    def test_logistic_regression_scorer(self):
        scorer = LogisticRegressionCandidateScorer(self.feature_names, random_state=42)
        scorer.fit(self.X_train, self.y_train)
        scores = scorer.predict_score(self.X_test)
        self.assertEqual(len(scores), 20)
        self.assertTrue(np.all(scores >= 0.0) and np.all(scores <= 1.0))

    def test_random_forest_scorer(self):
        scorer = RandomForestCandidateScorer(self.feature_names, n_estimators=20, random_state=42)
        scorer.fit(self.X_train, self.y_train)
        scores = scorer.predict_score(self.X_test)
        self.assertEqual(len(scores), 20)
        self.assertTrue(np.all(scores >= 0.0) and np.all(scores <= 1.0))

    def test_gradient_boosting_scorer(self):
        scorer = GradientBoostingCandidateScorer(self.feature_names, max_iter=20, random_state=42)
        scorer.fit(self.X_train, self.y_train)
        scores = scorer.predict_score(self.X_test)
        self.assertEqual(len(scores), 20)
        self.assertTrue(np.all(scores >= 0.0) and np.all(scores <= 1.0))

    def test_unfitted_raises_error(self):
        scorer = LogisticRegressionCandidateScorer(self.feature_names)
        with self.assertRaises(RuntimeError):
            scorer.predict_score(self.X_test)


if __name__ == "__main__":
    unittest.main()

