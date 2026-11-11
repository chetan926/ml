"""
Unit Tests for Code Candidate Generation Pipeline (Phase 3.3)
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection
"""

import unittest
import pandas as pd
import numpy as np
from research_upgrade.src.candidate_generation.code_candidate_scorer import CodeCandidateScorerPipeline
from research_upgrade.src.features.feature_pipeline import CODE_FEATURE_COLUMNS


class TestCodeCandidateGeneration(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n_pairs = 80

        other_cases = ["case-01", "case-02", "case-03", "case-04", "case-06", "case-07"]
        records = []
        for i in range(n_pairs):
            # Case assignment: first 20 pairs case-05 (all negative), rest mixed from other cases
            if i < 20:
                case_id = "case-05"
                label = 0
            else:
                case_id = other_cases[i % len(other_cases)]
                label = int(i % 2 == 0)

            rec = {
                "pair_id": f"code_{i:04d}",
                "dataset_id": "ir_plag",
                "modality": "code",
                "case_id": case_id,
                "ref_type": "original",
                "sus_type": "plagiarized" if label == 1 else "non-plagiarized",
                "obfuscation_level": "L1",
                "ground_truth_label": label,
                "policy": "policy_a",
                "extraction_time_ms": 4.5,
                "extraction_status": "SUCCESS"
            }
            for col in CODE_FEATURE_COLUMNS:
                base = 0.7 if label == 1 else 0.25
                rec[col] = float(np.clip(np.random.normal(base, 0.15), 0.0, 1.0))
            records.append(rec)

        self.pipeline = CodeCandidateScorerPipeline(random_seed=42)
        self.pipeline.df_policy_a_feats = pd.DataFrame(records)
        self.pipeline.df_policy_b_feats = pd.DataFrame(records)
        self.pipeline._is_prepared = True

    def test_cross_validation_evaluation(self):
        cv_res = self.pipeline.evaluate_cross_validation(policy="policy_a", n_splits=3, target_recall=0.95)
        self.assertGreater(len(cv_res), 0)
        for name, res in cv_res.items():
            self.assertIn("mean_recall", res)
            self.assertIn("mean_workload_reduction", res)
            self.assertGreater(res["mean_recall"], 0.0)

    def test_case05_cross_fitting_evaluation(self):
        c5_res = self.pipeline.evaluate_case05_cross_fitting()
        self.assertEqual(c5_res["case"], "case-05")
        self.assertIn("limitation", c5_res)
        self.assertIn("results", c5_res)
        for name, r in c5_res["results"].items():
            self.assertEqual(r["positive_pairs"], 0)
            self.assertEqual(r["negative_pairs"], 20)
            self.assertIn("workload_reduction", r)


if __name__ == "__main__":
    unittest.main()
