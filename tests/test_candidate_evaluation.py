"""
Unit Tests for Candidate Evaluation and Metrics (Phase 3.3)
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection
"""

import unittest
import numpy as np
from research_upgrade.src.candidate_generation.candidate_evaluation import (
    evaluate_candidate_threshold,
    compute_operating_curve,
    CandidateMetrics
)


class TestCandidateEvaluation(unittest.TestCase):
    def setUp(self):
        # 10 pairs: 4 positive, 6 negative
        self.y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=np.int32)
        self.scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0], dtype=np.float64)

    def test_threshold_separation(self):
        # At threshold 0.6: 4 positive pairs escalated (scores 0.9, 0.8, 0.7, 0.6)
        # 0 negative pairs escalated (highest negative is 0.5)
        m = evaluate_candidate_threshold(self.y_true, self.scores, threshold=0.6)
        self.assertEqual(m.total_pairs, 10)
        self.assertEqual(m.total_positives, 4)
        self.assertEqual(m.total_negatives, 6)
        self.assertEqual(m.escalated_pairs, 4)
        self.assertEqual(m.filtered_pairs, 6)
        self.assertEqual(m.true_positives, 4)
        self.assertEqual(m.false_positives, 0)
        self.assertEqual(m.true_negatives, 6)
        self.assertEqual(m.false_negatives, 0)
        self.assertEqual(m.candidate_recall, 1.0)
        self.assertEqual(m.candidate_precision, 1.0)
        self.assertEqual(m.workload_reduction, 0.6)
        self.assertEqual(m.retention_rate, 0.4)

    def test_conservative_threshold(self):
        # At threshold 0.0: all pairs escalated
        m = evaluate_candidate_threshold(self.y_true, self.scores, threshold=0.0)
        self.assertEqual(m.candidate_recall, 1.0)
        self.assertEqual(m.workload_reduction, 0.0)
        self.assertEqual(m.escalated_pairs, 10)
        self.assertEqual(m.filtered_pairs, 0)

    def test_aggressive_threshold(self):
        # At threshold 1.0: no pairs escalated
        m = evaluate_candidate_threshold(self.y_true, self.scores, threshold=1.0)
        self.assertEqual(m.candidate_recall, 0.0)
        self.assertEqual(m.workload_reduction, 1.0)
        self.assertEqual(m.escalated_pairs, 0)
        self.assertEqual(m.filtered_pairs, 10)

    def test_operating_curve(self):
        curve = compute_operating_curve(self.y_true, self.scores, num_thresholds=20)
        self.assertGreater(len(curve), 0)
        # Recall must be non-increasing with increasing threshold
        recalls = [m.candidate_recall for m in curve]
        for i in range(len(recalls) - 1):
            self.assertGreaterEqual(recalls[i], recalls[i + 1] - 1e-9)


if __name__ == "__main__":
    unittest.main()

