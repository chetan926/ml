"""
Unit tests for Integrated Proposed Framework (Phase 3.6+).
"""

import unittest
import numpy as np
import pandas as pd
from research_upgrade.src.pipeline import IntegratedTextPipeline, IntegratedCodePipeline


class TestIntegratedPipeline(unittest.TestCase):
    """Test suite for full end-to-end integrated pipelines for text and code."""

    def setUp(self):
        np.random.seed(42)
        # Mock text dataframes
        n_train = 80
        n_val = 30
        n_test = 40

        cols = [
            "text_word_tfidf_sim", "text_word_jaccard", "text_word_overlap_ratio",
            "text_char_tfidf_sim", "text_char_jaccard_3gram", "text_char_lcs_ratio",
            "text_semantic_minilm_sim", "text_length_ratio", "text_token_count_delta",
            "text_sentence_delta", "text_ttr_sim"
        ]

        def make_df(n, is_test=False):
            data = {c: np.random.uniform(0.0, 1.0, size=n) for c in cols}
            data["ground_truth_label"] = np.random.binomial(1, 0.4, size=n)
            data["pair_id"] = [f"pair_{i}" for i in range(n)]
            data["premise_group_id"] = [f"group_{i % 10}" for i in range(n)]
            return pd.DataFrame(data)

        self.df_train = make_df(n_train)
        self.df_val = make_df(n_val)
        self.df_test = make_df(n_test, is_test=True)

        # Mock code dataframe
        code_cols = [
            "code_token_jaccard", "code_token_frequency_overlap", "code_rkrgst_sim",
            "code_norm_token_jaccard", "code_norm_freq_overlap", "code_norm_rkrgst_sim",
            "code_nesting_depth_sim", "code_control_block_sim", "code_method_structure_sim",
            "code_operator_dist_sim", "code_control_flow_sim", "code_cyclomatic_sim",
            "code_sloc_ratio"
        ]
        code_data = {c: np.random.uniform(0.0, 1.0, size=60) for c in code_cols}
        code_data["ground_truth_label"] = np.random.binomial(1, 0.5, size=60)
        code_data["pair_id"] = [f"code_pair_{i}" for i in range(60)]
        self.df_code = pd.DataFrame(code_data)

    def test_text_pipeline(self):
        pipe = IntegratedTextPipeline(random_state=42)
        pipe.fit(self.df_train, self.df_val, budget_fraction=0.8)
        self.assertTrue(pipe.is_fitted)

        results = pipe.predict_test(self.df_test)
        self.assertEqual(results["modality"], "text")
        self.assertIn("classification_metrics", results)
        self.assertIn("calibration_metrics", results)
        self.assertIn("routing_cost_summary", results)
        self.assertIn("group_conformal_prediction", results)
        self.assertIn("explainability_metrics", results)
        self.assertGreaterEqual(results["classification_metrics"]["accuracy"], 0.0)

    def test_code_pipeline(self):
        pipe = IntegratedCodePipeline(random_state=42)
        results = pipe.fit_and_evaluate_cross_validation(self.df_code, n_splits=3, budget_fraction=0.7)
        self.assertEqual(results["modality"], "code")
        self.assertIn("classification_metrics", results)
        self.assertIn("calibration_metrics", results)
        self.assertIn("routing_cost_summary", results)
        self.assertGreaterEqual(results["classification_metrics"]["accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()

