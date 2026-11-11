"""
Integration Tests for Unified Multi-View Feature Pipelines (Phase 3.2)
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection
"""

import unittest
import pandas as pd
from research_upgrade.src.features.feature_pipeline import (
    TextMultiViewPipeline,
    CodeMultiViewPipeline,
    TEXT_FEATURE_COLUMNS,
    CODE_FEATURE_COLUMNS
)
from research_upgrade.src.evaluation.feature_quality import validate_feature_matrix_quality


SAMPLE_TEXT_DATA = pd.DataFrame([
    {
        "id": "text_sample_01",
        "reference_text": "Machine learning algorithms optimize mathematical loss functions over high dimensional parameter spaces.",
        "suspicious_text": "Machine learning models optimize loss functions over high dimensional spaces through gradient descent.",
        "is_plagiarized": 1,
        "split": "train",
        "premise_group_id": "group_001"
    },
    {
        "id": "text_sample_02",
        "reference_text": "Photosynthesis converts solar irradiance into chemical energy stored within carbohydrates.",
        "suspicious_text": "The Federal Reserve raised benchmark interest rates to combat inflationary pressure.",
        "is_plagiarized": 0,
        "split": "train",
        "premise_group_id": "group_002"
    },
    {
        "id": "text_sample_03",
        "reference_text": "Conformal prediction guarantees distribution-free marginal coverage for finite samples.",
        "suspicious_text": "Distribution-free marginal coverage is rigorously guaranteed in finite samples via conformal prediction.",
        "is_plagiarized": 1,
        "split": "val",
        "premise_group_id": "group_003"
    }
])

SAMPLE_CODE_DATA = pd.DataFrame([
    {
        "id": "code_sample_01",
        "reference_code": "public class A { public int f(int x) { return x * 2; } }",
        "suspicious_code": "public class B { public int f(int y) { return y * 2; } }",
        "case": "case-01",
        "ref_type": "original",
        "sus_type": "plagiarized",
        "obfuscation_level": "L1",
        "is_plagiarized": 1
    },
    {
        "id": "code_sample_02",
        "reference_code": "public class Sorting { public void sort(int[] arr) { java.util.Arrays.sort(arr); } }",
        "suspicious_code": "public class Graph { public void traverse() { System.out.println(\"BFS\"); } }",
        "case": "case-02",
        "ref_type": "original",
        "sus_type": "non-plagiarized",
        "obfuscation_level": "NONE",
        "is_plagiarized": 0
    }
])


class TestTextMultiViewPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = TextMultiViewPipeline()
        train_df = SAMPLE_TEXT_DATA[SAMPLE_TEXT_DATA["split"] == "train"]
        cls.pipeline.fit(train_df)

    def test_schema(self):
        schema = self.pipeline.get_feature_schema()
        self.assertEqual(schema["modality"], "text")
        self.assertEqual(schema["total_features"], 11)
        self.assertEqual(len(self.pipeline.feature_names), 11)
        for col in TEXT_FEATURE_COLUMNS:
            self.assertIn(col, self.pipeline.feature_names)

    def test_transform_dataframe(self):
        feat_df = self.pipeline.transform_dataframe(SAMPLE_TEXT_DATA, partition_name="test_subset")
        self.assertEqual(len(feat_df), len(SAMPLE_TEXT_DATA))

        # Check metadata preservation
        self.assertIn("pair_id", feat_df.columns)
        self.assertIn("dataset_id", feat_df.columns)
        self.assertIn("modality", feat_df.columns)
        self.assertIn("ground_truth_label", feat_df.columns)
        self.assertIn("partition", feat_df.columns)
        self.assertIn("premise_group_id", feat_df.columns)
        self.assertIn("extraction_time_ms", feat_df.columns)
        self.assertIn("extraction_status", feat_df.columns)

        # Check feature columns
        for col in TEXT_FEATURE_COLUMNS:
            self.assertIn(col, feat_df.columns)

        # Quality and bound audit
        quality_report = validate_feature_matrix_quality(
            feat_df,
            expected_feature_cols=TEXT_FEATURE_COLUMNS,
            modality="text"
        )
        self.assertEqual(quality_report["status"], "PASSED", f"Quality check failed: {quality_report['errors']}")
        self.assertEqual(len(quality_report["errors"]), 0)


class TestCodeMultiViewPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = CodeMultiViewPipeline()

    def test_schema(self):
        schema = self.pipeline.get_feature_schema()
        self.assertEqual(schema["modality"], "code")
        self.assertEqual(schema["total_features"], 13)
        self.assertEqual(len(self.pipeline.feature_names), 13)
        for col in CODE_FEATURE_COLUMNS:
            self.assertIn(col, self.pipeline.feature_names)

    def test_transform_dataframe(self):
        feat_df = self.pipeline.transform_dataframe(SAMPLE_CODE_DATA, policy="policy_a")
        self.assertEqual(len(feat_df), len(SAMPLE_CODE_DATA))

        # Check metadata preservation
        self.assertIn("pair_id", feat_df.columns)
        self.assertIn("dataset_id", feat_df.columns)
        self.assertIn("modality", feat_df.columns)
        self.assertIn("case_id", feat_df.columns)
        self.assertIn("ref_type", feat_df.columns)
        self.assertIn("sus_type", feat_df.columns)
        self.assertIn("obfuscation_level", feat_df.columns)
        self.assertIn("ground_truth_label", feat_df.columns)
        self.assertIn("policy", feat_df.columns)
        self.assertIn("extraction_time_ms", feat_df.columns)
        self.assertIn("extraction_status", feat_df.columns)

        # Check feature columns
        for col in CODE_FEATURE_COLUMNS:
            self.assertIn(col, feat_df.columns)

        # Quality and bound audit
        quality_report = validate_feature_matrix_quality(
            feat_df,
            expected_feature_cols=CODE_FEATURE_COLUMNS,
            modality="code"
        )
        self.assertEqual(quality_report["status"], "PASSED", f"Quality check failed: {quality_report['errors']}")
        self.assertEqual(len(quality_report["errors"]), 0)


if __name__ == "__main__":
    unittest.main()
