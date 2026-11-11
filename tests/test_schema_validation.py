"""
Schema and Record Integrity Automated Test Suite (Phase 3.2)
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection
"""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from research_upgrade.src.data.schema_validation import (
    validate_text_schema,
    validate_code_schema
)
from research_upgrade.src.config import (
    TEXT_DATASET_PATH,
    CODE_DATASET_PATH
)


class TestSchemaValidation(unittest.TestCase):
    """Automated tests for dataset schema, data types, and record validation."""

    def test_text_benchmark_raw_schema(self):
        """Validates that raw text CSV satisfies the required schema."""
        self.assertTrue(TEXT_DATASET_PATH.exists(), f"Text dataset missing: {TEXT_DATASET_PATH}")
        res = validate_text_schema(TEXT_DATASET_PATH)
        self.assertEqual(res["status"], "PASSED", f"Schema validation failed: {res['errors']}")
        self.assertEqual(res["total_rows"], 19666)
        self.assertEqual(res["unique_ids"], 19666)
        self.assertEqual(res["empty_reference_texts"], 0)
        self.assertEqual(res["empty_suspicious_texts"], 0)
        self.assertIn(0, res["label_distribution"])
        self.assertIn(1, res["label_distribution"])

    def test_code_benchmark_raw_schema(self):
        """Validates that raw code CSV satisfies the required schema."""
        self.assertTrue(CODE_DATASET_PATH.exists(), f"Code dataset missing: {CODE_DATASET_PATH}")
        res = validate_code_schema(CODE_DATASET_PATH)
        self.assertEqual(res["status"], "PASSED", f"Schema validation failed: {res['errors']}")
        self.assertEqual(res["total_rows"], 733)
        self.assertEqual(res["unique_ids"], 733)
        self.assertEqual(res["empty_reference_code"], 0)
        self.assertEqual(res["empty_suspicious_code"], 0)
        self.assertEqual(len(res["case_distribution"]), 7)

    def test_synthetic_malformed_text_detection(self):
        """Asserts that malformed text records trigger validation errors."""
        # Missing column
        bad_df_1 = pd.DataFrame({"id": ["TXT_01"], "reference_text": ["abc"]})
        res_1 = validate_text_schema(bad_df_1)
        self.assertEqual(res_1["status"], "FAILED")

        # Invalid label (non-binary)
        bad_df_2 = pd.DataFrame({
            "id": ["TXT_01"],
            "reference_text": ["abc"],
            "suspicious_text": ["def"],
            "label": [2],
            "is_plagiarized": [2],
            "split": ["val"]
        })
        res_2 = validate_text_schema(bad_df_2)
        self.assertEqual(res_2["status"], "FAILED")

        # Empty reference text
        bad_df_3 = pd.DataFrame({
            "id": ["TXT_01"],
            "reference_text": ["   "],
            "suspicious_text": ["def"],
            "label": [1],
            "is_plagiarized": [1],
            "split": ["val"]
        })
        res_3 = validate_text_schema(bad_df_3)
        self.assertEqual(res_3["status"], "FAILED")


if __name__ == "__main__":
    unittest.main()

