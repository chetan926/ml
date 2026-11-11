"""
Unit Tests for Threshold Selection (Phase 3.3)
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection
"""

import unittest
import numpy as np
from research_upgrade.src.candidate_generation.threshold_selection import ThresholdSelector


class TestThresholdSelection(unittest.TestCase):
    def setUp(self):
        self.selector = ThresholdSelector(num_grid_points=100)
        # Synthetic validation set
        self.y_val = np.array([1] * 50 + [0] * 50, dtype=np.int32)
        # Positives have higher scores on average with some overlap
        np.random.seed(42)
        pos_scores = np.random.normal(loc=0.75, scale=0.15, size=50)
        neg_scores = np.random.normal(loc=0.25, scale=0.15, size=50)
        self.scores_val = np.clip(np.concatenate([pos_scores, neg_scores]), 0.0, 1.0)

    def test_recall_constrained_selection(self):
        res = self.selector.select_recall_constrained_threshold(
            self.y_val, self.scores_val, target_recall=0.95
        )
        self.assertIn("selected_threshold", res)
        self.assertIn("target_recall_constraint", res)
        self.assertIn("validation_metrics", res)
        vm = res["validation_metrics"]
        self.assertGreaterEqual(vm["candidate_recall"], 0.95)
        self.assertGreater(vm["workload_reduction"], 0.0)

    def test_workload_constrained_selection(self):
        res = self.selector.select_workload_constrained_threshold(
            self.y_val, self.scores_val, target_workload_reduction=0.40
        )
        self.assertIn("selected_threshold", res)
        vm = res["validation_metrics"]
        self.assertGreaterEqual(vm["workload_reduction"], 0.40)
        self.assertGreater(vm["candidate_recall"], 0.0)

    def test_operating_points_profile(self):
        profile = self.selector.generate_operating_points_profile(self.y_val, self.scores_val)
        self.assertIn("recall_constrained_operating_points", profile)
        self.assertIn("workload_constrained_operating_points", profile)
        self.assertEqual(len(profile["recall_constrained_operating_points"]), 5)
        self.assertEqual(len(profile["workload_constrained_operating_points"]), 5)


if __name__ == "__main__":
    unittest.main()

