"""
Unit Tests for Text Candidate Generation Pipeline (Phase 3.3)
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection
"""

import unittest
import pandas as pd
import numpy as np
from research_upgrade.src.candidate_generation.text_candidate_scorer import TextCandidateScorerPipeline
from research_upgrade.src.features.feature_pipeline import TEXT_FEATURE_COLUMNS


class TestTextCandidateGeneration(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n_train, n_val, n_test = 60, 30, 30

        def make_df(n, part):
            records = []
            for i in range(n):
                label = int(i % 2 == 0)
                rec = {
                    "pair_id": f"text_{part}_{i}",
                    "modality": "text",
                    "ground_truth_label": label,
                    "partition": part,
                    "extraction_status": "SUCCESS"
                }
                for col in TEXT_FEATURE_COLUMNS:
                    # Positive pairs have higher mean similarity
                    base = 0.7 if label == 1 else 0.2
                    rec[col] = float(np.clip(np.random.normal(base, 0.15), 0.0, 1.0))
                records.append(rec)
            return pd.DataFrame(records)

        self.pipeline = TextCandidateScorerPipeline(random_seed=42)
        self.pipeline.df_train_feats = make_df(n_train, "train")
        self.pipeline.df_val_feats = make_df(n_val, "val")
        self.pipeline.df_test_feats = make_df(n_test, "test")
        self.pipeline._is_prepared = True

    def test_train_models(self):
        models = self.pipeline.train_models()
        self.assertEqual(len(models), 7)
        self.assertIn("single_view_word_tfidf", models)
        self.assertIn("single_view_dense_semantic", models)
        self.assertIn("logistic_regression", models)
        self.assertIn("random_forest", models)

    def test_validation_threshold_selection(self):
        self.pipeline.train_models()
        val_res = self.pipeline.evaluate_validation_and_select_thresholds(target_recall=0.95)
        self.assertEqual(len(val_res), 7)
        for name, res in val_res.items():
            self.assertIn("selected_threshold", res)
            self.assertIn("validation_metrics", res)
            vm = res["validation_metrics"]
            self.assertGreaterEqual(vm["candidate_recall"], 0.90)

    def test_test_set_evaluation(self):
        self.pipeline.train_models()
        self.pipeline.evaluate_validation_and_select_thresholds(target_recall=0.95)
        test_res = self.pipeline.evaluate_test_set()
        self.assertEqual(len(test_res), 7)
        for name, res in test_res.items():
            self.assertIn("test_recall", res)
            self.assertIn("test_workload_reduction", res)
            self.assertIn("generalization_gap_recall", res)
            self.assertGreater(res["test_recall"], 0.0)


if __name__ == "__main__":
    unittest.main()

