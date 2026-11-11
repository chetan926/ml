"""
Dataset Partition and Leakage Protection Automated Test Suite (Phase 3.1)
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection
"""

import unittest
from pathlib import Path
import json
import csv

# Ensure project root is importable
import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from research_upgrade.src.config import (
    PHASE2_AUDIT_DIR,
    PHASE2_TEXT_LEAKAGE_PATH,
    PHASE2_TEXT_EXCLUSION_PATH,
    PHASE2_CODE_DUPLICATE_PATH,
    PHASE2_CASE05_FEASIBILITY_PATH,
    N_RAW_TEXT_ROWS,
    N_PURGED_CROSS_CONTAMINATION,
    N_PURGED_REDUNDANT_DUPLICATES,
    N_TOTAL_PURGED_ROWS,
    N_TOTAL_CLEAN_ROWS,
    N_RAW_CODE_PAIRS,
    N_CODE_DUPLICATE_ROWS,
    N_CODE_DUPLICATE_GROUPS,
    N_CODE_POLICY_A_RETAINED,
    N_CODE_POLICY_B_RETAINED,
    CASE_05_PHYSICAL_JAVA_FILES,
    CASE_05_TOTAL_PAIRS,
    CASE_05_POSITIVE_PAIRS,
    CASE_05_NEGATIVE_PAIRS,
    CASE_05_BASE_PROGRAMS_COUNT
)


class TestDatasetLeakageProtection(unittest.TestCase):
    """Verifies that all Phase 2 dataset integrity, anti-leakage, and cross-fitting proofs hold."""

    def test_text_leakage_report_zero_cross_overlap(self):
        """Asserts that the 7x7 partition overlap matrix has strictly 0 cross-split premise leaks."""
        self.assertTrue(
            PHASE2_TEXT_LEAKAGE_PATH.exists(),
            f"Missing text leakage report at: {PHASE2_TEXT_LEAKAGE_PATH}"
        )
        with open(PHASE2_TEXT_LEAKAGE_PATH, "r", encoding="utf-8") as f:
            leakage_data = json.load(f)

        matrix = leakage_data["partition_overlap_matrix"]
        splits = list(matrix.keys())
        self.assertEqual(len(splits), 7, f"Expected 7 partitions, found {len(splits)}")

        for s1 in splits:
            for s2 in splits:
                overlap = matrix[s1][s2]
                if s1 == s2:
                    self.assertGreater(
                        overlap, 0,
                        f"Diagonal entry for {s1} must be positive, got {overlap}"
                    )
                else:
                    self.assertEqual(
                        overlap, 0,
                        f"Cross-split leakage detected between {s1} and {s2}: {overlap} overlapping premises!"
                    )

        # Verify summary status
        self.assertEqual(leakage_data["leakage_status"], "ZERO_LEAKAGE_CONFIRMED")
        self.assertEqual(leakage_data["crossing_premises_count"], 1)
        self.assertEqual(leakage_data["crossing_rows_count"], 6)

    def test_text_exclusion_ledger_exact_reconciliation(self):
        """Asserts that exactly 8 purged rows are accounted for with verified exclusion reasons."""
        self.assertTrue(
            PHASE2_TEXT_EXCLUSION_PATH.exists(),
            f"Missing exclusion ledger at: {PHASE2_TEXT_EXCLUSION_PATH}"
        )
        rows = []
        with open(PHASE2_TEXT_EXCLUSION_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)

        self.assertEqual(
            len(rows), N_TOTAL_PURGED_ROWS,
            f"Expected {N_TOTAL_PURGED_ROWS} purged rows, found {len(rows)}"
        )

        cross_contamination_count = sum(
            1 for r in rows if r["exclusion_reason"] == "CROSS_SPLIT_PREMISE_CONTAMINATION"
        )
        redundant_duplicate_count = sum(
            1 for r in rows if r["exclusion_reason"] == "REDUNDANT_EXACT_DUPLICATE_ROW"
        )

        self.assertEqual(cross_contamination_count, N_PURGED_CROSS_CONTAMINATION)
        self.assertEqual(redundant_duplicate_count, N_PURGED_REDUNDANT_DUPLICATES)
        self.assertTrue(all(r["final_disposition"] == "PURGED_PRE_SPLIT" for r in rows))

    def test_code_duplicate_accounting_reconciliation(self):
        """Asserts that IR-Plag code duplicates match Policy A (720) and Policy B (709)."""
        self.assertTrue(
            PHASE2_CODE_DUPLICATE_PATH.exists(),
            f"Missing code duplicate report at: {PHASE2_CODE_DUPLICATE_PATH}"
        )
        with open(PHASE2_CODE_DUPLICATE_PATH, "r", encoding="utf-8") as f:
            code_dupe_data = json.load(f)

        self.assertEqual(code_dupe_data["total_raw_pairs"], N_RAW_CODE_PAIRS)
        self.assertEqual(code_dupe_data["participating_duplicate_rows"], N_CODE_DUPLICATE_ROWS)
        self.assertEqual(code_dupe_data["duplicate_groups_count"], N_CODE_DUPLICATE_GROUPS)
        self.assertEqual(code_dupe_data["policy_a_total_retained"], N_CODE_POLICY_A_RETAINED)
        self.assertEqual(code_dupe_data["policy_b_total_retained"], N_CODE_POLICY_B_RETAINED)
        self.assertEqual(code_dupe_data["reversed_duplicate_pairs"], 0)

    def test_case_05_cross_fitting_proof_and_physical_files(self):
        """Asserts that Case-05 single-class failure proof and file inventory are preserved."""
        self.assertTrue(
            PHASE2_CASE05_FEASIBILITY_PATH.exists(),
            f"Missing Case-05 feasibility report at: {PHASE2_CASE05_FEASIBILITY_PATH}"
        )
        with open(PHASE2_CASE05_FEASIBILITY_PATH, "r", encoding="utf-8") as f:
            c5_data = json.load(f)

        self.assertEqual(c5_data["physical_java_files"], CASE_05_PHYSICAL_JAVA_FILES)
        self.assertEqual(c5_data["total_pairs"], CASE_05_TOTAL_PAIRS)
        self.assertEqual(c5_data["positive_pairs"], CASE_05_POSITIVE_PAIRS)
        self.assertEqual(c5_data["negative_pairs"], CASE_05_NEGATIVE_PAIRS)
        self.assertEqual(c5_data["base_program_identities_count"], CASE_05_BASE_PROGRAMS_COUNT)
        self.assertEqual(c5_data["structural_degeneracy"], "ALL_53_POSITIVES_SHARE_PROG_ORIGINAL")
        self.assertEqual(c5_data["methodological_consequence"], "CALIBRATION_IMPOSSIBLE_RAW_MARGIN_ENFORCED")

        folds = c5_data["fold_cross_fitting_proof"]
        self.assertEqual(len(folds), 5)

        # Fold 1: prog_original held out in test
        f1 = folds[0]
        self.assertEqual(f1["fold_index"], 1)
        self.assertEqual(f1["pos_train_count"], 0)
        self.assertEqual(f1["neg_train_count"], 33)
        self.assertEqual(f1["pos_test_count"], 53)
        self.assertEqual(f1["neg_test_count"], 21)
        self.assertFalse(f1["platt_scaling_feasible_on_train"])
        self.assertEqual(f1["mathematical_status"], "DIVERGES_ZERO_TRAIN_POSITIVES")

        # Folds 2-5: prog_original in train
        for f in folds[1:]:
            self.assertEqual(f["pos_train_count"], 53)
            self.assertIn(f["neg_train_count"], [36, 37, 38])
            self.assertEqual(f["pos_test_count"], 0)
            self.assertIn(f["neg_test_count"], [16, 17, 18])
            self.assertTrue(f["platt_scaling_feasible_on_train"])
            self.assertEqual(f["mathematical_status"], "UNDEFINED_ZERO_TEST_POSITIVES")

    def test_zero_overlap_lexical_positives_accounting(self):
        """Verifies separation between active test (33 pairs, 3.22%) and historical benchmark (119 pairs, 11.19%)."""
        text_accounting_path = PHASE2_AUDIT_DIR / "text_dataset_accounting.json"
        self.assertTrue(text_accounting_path.exists())
        with open(text_accounting_path, "r", encoding="utf-8") as f:
            text_acc = json.load(f)

        stats = text_acc.get("test_partition_stats", {})
        self.assertEqual(stats.get("test_positive_pairs"), 1026)
        self.assertEqual(stats.get("test_zero_jaccard_positives"), 33)
        pct_33 = (33 / 1026) * 100
        self.assertAlmostEqual(pct_33, 3.22, places=1)

        self.assertEqual(stats.get("historical_stress_test_positives"), 1063)
        self.assertEqual(stats.get("historical_zero_tfidf_positives"), 119)
        self.assertAlmostEqual(stats.get("historical_zero_tfidf_pct", 0.0), 11.19, places=2)
        self.assertAlmostEqual(stats.get("historical_nonzero_lexical_pct", 0.0), 88.81, places=2)


if __name__ == "__main__":
    unittest.main()
