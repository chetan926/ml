"""
Governance and Baseline Freezing Automated Test Suite (Phase 3.1)
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection
"""

import unittest
from pathlib import Path
import random
import numpy as np

# Ensure project root is importable
import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from research_upgrade.src.config import (
    PROJECT_ROOT,
    BASELINE_SRC_DIR,
    PHASE2_MANIFEST_PATH,
    RANDOM_SEED,
    set_global_seed
)
from research_upgrade.src.governance import (
    compute_file_sha256,
    load_registered_manifest,
    verify_baseline_integrity,
    assert_baseline_intact
)


class TestBaselineGovernance(unittest.TestCase):
    """Verifies that the legacy baseline in src/ is 100% frozen and intact."""

    def test_baseline_manifest_exists(self):
        """Verifies that repository_sha256_manifest.csv exists on disk."""
        self.assertTrue(
            PHASE2_MANIFEST_PATH.exists(),
            f"Baseline manifest missing at: {PHASE2_MANIFEST_PATH}"
        )

    def test_registered_file_count_is_exactly_28(self):
        """Asserts that exactly 28 files are registered in the baseline manifest."""
        records = load_registered_manifest()
        self.assertEqual(
            len(records), 28,
            f"Expected exactly 28 baseline files in manifest, found {len(records)}"
        )

    def test_all_28_files_exist_on_disk(self):
        """Asserts that every file listed in the manifest physically exists."""
        records = load_registered_manifest()
        for r in records:
            p = PROJECT_ROOT / r["relative_path"]
            self.assertTrue(
                p.exists() and p.is_file(),
                f"Baseline file missing from disk: {p}"
            )

    def test_all_28_sha256_hashes_match_manifest(self):
        """Computes SHA-256 for all 28 files and asserts byte-exact match."""
        records = load_registered_manifest()
        for r in records:
            p = PROJECT_ROOT / r["relative_path"]
            computed_hash = compute_file_sha256(p)
            self.assertEqual(
                computed_hash, r["sha256_checksum"],
                f"SHA-256 mismatch for {r['relative_path']}: expected {r['sha256_checksum']}, got {computed_hash}"
            )
            self.assertEqual(
                p.stat().st_size, r["file_size_bytes"],
                f"File size mismatch for {r['relative_path']}: expected {r['file_size_bytes']}, got {p.stat().st_size}"
            )

    def test_no_untracked_files_in_src(self):
        """Asserts that no unauthorized files have been added to src/."""
        records = load_registered_manifest()
        expected_paths = {r["relative_path"] for r in records}
        actual_paths = {
            p.relative_to(PROJECT_ROOT).as_posix()
            for p in BASELINE_SRC_DIR.glob("**/*.py")
        }
        untracked = actual_paths - expected_paths
        self.assertEqual(
            len(untracked), 0,
            f"Found unauthorized untracked files in src/: {untracked}"
        )

    def test_governance_verification_function(self):
        """Asserts that verify_baseline_integrity() returns PASSED."""
        report = verify_baseline_integrity()
        self.assertEqual(report["status"], "PASSED")
        self.assertEqual(report["matched_count"], 28)
        self.assertEqual(report["mismatched_count"], 0)
        self.assertEqual(report["missing_count"], 0)
        self.assertEqual(report["untracked_count"], 0)

    def test_assert_baseline_intact_does_not_raise(self):
        """Asserts that assert_baseline_intact() completes without error."""
        try:
            assert_baseline_intact()
        except RuntimeError as e:
            self.fail(f"assert_baseline_intact() raised an unexpected exception: {e}")


class TestReproducibilityConfig(unittest.TestCase):
    """Verifies deterministic random seed configuration."""

    def test_deterministic_seed_initialization(self):
        """Verifies that set_global_seed() produces identical random sequences."""
        set_global_seed(RANDOM_SEED)
        py_rand_1 = [random.random() for _ in range(5)]
        np_rand_1 = np.random.rand(5).tolist()

        set_global_seed(RANDOM_SEED)
        py_rand_2 = [random.random() for _ in range(5)]
        np_rand_2 = np.random.rand(5).tolist()

        self.assertEqual(py_rand_1, py_rand_2, "Python random sequence is not deterministic")
        self.assertEqual(np_rand_1, np_rand_2, "NumPy random sequence is not deterministic")


if __name__ == "__main__":
    unittest.main()

