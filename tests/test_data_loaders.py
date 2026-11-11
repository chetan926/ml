"""
Data Loaders Automated Test Suite (Phase 3.2)
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection
"""

import unittest
from pathlib import Path
import pandas as pd

# Ensure project root is in sys.path
import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from research_upgrade.src.data.text_loader import TextBenchmarkDataLoader
from research_upgrade.src.data.code_loader import CodeBenchmarkDataLoader
from research_upgrade.src.config import (
    N_RAW_TEXT_ROWS,
    N_TOTAL_CLEAN_ROWS,
    N_TOTAL_PURGED_ROWS,
    N_RAW_CODE_PAIRS,
    N_CODE_POLICY_A_RETAINED,
    N_CODE_POLICY_B_RETAINED,
    N_CODE_DUPLICATE_GROUPS,
    N_IR_PLAG_PHYSICAL_JAVA_FILES
)


class TestDataLoaders(unittest.TestCase):
    """Automated integration tests for Text and Code Benchmark Data Loaders."""

    @classmethod
    def setUpClass(cls):
        cls.text_loader = TextBenchmarkDataLoader()
        cls.code_loader = CodeBenchmarkDataLoader()

    def test_text_loader_partitions_and_counts(self):
        """Validates exact integer partition allocations for text benchmark."""
        meta = self.text_loader.get_metadata()

        expected = {
            "train": 4000,
            "val": 1500,
            "cal_prob": 1000,
            "cal_conf": 1000,
            "dev_reserve": 2337,
            "test": 3000,
            "test_reserve": 6821
        }
        for part, exp_size in expected.items():
            self.assertIn(part, meta)
            self.assertEqual(meta[part]["total_pairs"], exp_size, f"Size mismatch for {part}")

        clean_sum = sum(meta[p]["total_pairs"] for p in expected)
        self.assertEqual(clean_sum, N_TOTAL_CLEAN_ROWS)

    def test_text_loader_purged_rows(self):
        """Validates that exactly 8 purged rows are isolated."""
        purged_df = self.text_loader.get_purged_records()
        self.assertEqual(len(purged_df), N_TOTAL_PURGED_ROWS)

    def test_text_loader_premise_groups(self):
        """Validates premise-group extraction for conformal prediction."""
        cal_conf_groups = self.text_loader.get_premise_groups("cal_conf")
        # 334 unique premises in cal_conf
        self.assertEqual(len(cal_conf_groups), 334)
        total_grouped_pairs = sum(len(pairs) for pairs in cal_conf_groups.values())
        self.assertEqual(total_grouped_pairs, 1000)

    def test_code_loader_policies(self):
        """Validates Policy A (720 pairs) and Policy B (709 pairs) retention."""
        raw_df = self.code_loader.get_dataset(policy="raw")
        self.assertEqual(len(raw_df), N_RAW_CODE_PAIRS)

        policy_a = self.code_loader.get_dataset(policy="policy_a")
        self.assertEqual(len(policy_a), N_CODE_POLICY_A_RETAINED)

        policy_b = self.code_loader.get_dataset(policy="policy_b")
        self.assertEqual(len(policy_b), N_CODE_POLICY_B_RETAINED)

    def test_code_loader_duplicate_groups(self):
        """Validates 11 duplicate groups in IR-Plag."""
        meta = self.code_loader.get_metadata()
        self.assertEqual(meta["duplicate_groups"], N_CODE_DUPLICATE_GROUPS)

    def test_code_loader_case_05_folds(self):
        """Validates audited 5-fold cross-fitting proof on Case-05."""
        folds = self.code_loader.get_case_05_folds()
        self.assertEqual(len(folds), 5)

        # Fold 1: train positives = 0
        self.assertEqual(folds[0]["pos_train_count"], 0)
        self.assertEqual(folds[0]["pos_test_count"], 53)
        self.assertFalse(folds[0]["platt_scaling_feasible_on_train"])

        # Folds 2-5: test positives = 0
        for f in folds[1:]:
            self.assertEqual(f["pos_train_count"], 53)
            self.assertEqual(f["pos_test_count"], 0)
            self.assertTrue(f["platt_scaling_feasible_on_train"])

    def test_code_loader_physical_files(self):
        """Validates 467 physical Java files in IR-Plag."""
        meta = self.code_loader.get_metadata()
        self.assertEqual(meta["physical_java_files"], N_IR_PLAG_PHYSICAL_JAVA_FILES)


if __name__ == "__main__":
    unittest.main()

