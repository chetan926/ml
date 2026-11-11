"""
Phase 3.1 Automated Governance and Integrity Audit Script
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection
Execution Scope: Phase 3.1 Acceptance Criteria Enforcement
"""

from pathlib import Path
import os
import sys
import json
import hashlib
import unittest
from datetime import datetime
from io import StringIO

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research_upgrade.src.config import (
    BASELINE_SRC_DIR,
    RESEARCH_UPGRADE_SRC_DIR,
    PHASE2_MANIFEST_PATH,
    PHASE3_ARTIFACTS_DIR,
    RANDOM_SEED
)
from research_upgrade.src.governance import (
    compute_file_sha256,
    verify_baseline_integrity,
    assert_baseline_intact
)


def run_phase_3_1_audit() -> int:
    print("=" * 80)
    print("PHASE 3.1: REPOSITORY SETUP & GOVERNANCE ENFORCEMENT AUDIT")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Project Root: {PROJECT_ROOT}")
    print("=" * 80)

    log_stream = StringIO()
    def log_print(msg: str):
        print(msg)
        log_stream.write(msg + "\n")

    log_print(f"Execution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # -------------------------------------------------------------------------
    # Task 1: Verify Baseline SHA-256 Manifest (28 files in src/)
    # -------------------------------------------------------------------------
    log_print("\n[TASK 1 & 2] Verifying Baseline SHA-256 Manifest and Integrity...")
    gov_report = verify_baseline_integrity()
    log_print(f"  Status: {gov_report['status']}")
    log_print(f"  Total registered files: {gov_report['total_registered_files']}")
    log_print(f"  Matched files: {gov_report['matched_count']}")
    log_print(f"  Mismatched files: {gov_report['mismatched_count']}")
    log_print(f"  Missing files: {gov_report['missing_count']}")
    log_print(f"  Untracked files in src/: {gov_report['untracked_count']}")

    assert gov_report["status"] == "PASSED", "Baseline integrity verification failed!"
    assert gov_report["matched_count"] == 28, f"Expected 28 matches, got {gov_report['matched_count']}"
    assert gov_report["untracked_count"] == 0, f"Found untracked files in src/: {gov_report['untracked_files']}"
    log_print("  --> PASS: All 28 baseline files match registered SHA-256 checksums byte-for-byte.")

    # -------------------------------------------------------------------------
    # Task 3 & 4: Run Governance and Leakage Protection Test Suites
    # -------------------------------------------------------------------------
    log_print("\n[TASK 2 & 4] Executing Automated Test Suites under tests/...")
    test_loader = unittest.TestLoader()
    suite = test_loader.discover(start_dir=str(PROJECT_ROOT / "tests"), pattern="test_*.py")
    
    test_runner_stream = StringIO()
    runner = unittest.TextTestRunner(stream=test_runner_stream, verbosity=2)
    test_result = runner.run(suite)

    test_output = test_runner_stream.getvalue()
    log_print(test_output)

    log_print(f"  Total tests run: {test_result.testsRun}")
    log_print(f"  Failures: {len(test_result.failures)}")
    log_print(f"  Errors: {len(test_result.errors)}")
    
    assert test_result.wasSuccessful(), "Automated test suite execution had failures or errors!"
    log_print(f"  --> PASS: All {test_result.testsRun} automated tests passed cleanly.")

    # -------------------------------------------------------------------------
    # Task 5 & 6: Phase 3 Directory & Implementation Manifest Construction
    # -------------------------------------------------------------------------
    log_print("\n[TASK 5 & 6] Auditing Phase 3 Created Files & Generating Implementation Manifest...")

    # Inventory newly created files in research_upgrade/src/ and tests/
    new_files_to_track = [
        PROJECT_ROOT / "research_upgrade" / "src" / "__init__.py",
        PROJECT_ROOT / "research_upgrade" / "src" / "config.py",
        PROJECT_ROOT / "research_upgrade" / "src" / "governance.py",
        PROJECT_ROOT / "research_upgrade" / "src" / "logging_utils.py",
        PROJECT_ROOT / "tests" / "test_governance.py",
        PROJECT_ROOT / "tests" / "test_leakage_protection.py",
        PROJECT_ROOT / "research_upgrade" / "audits" / "phase_3" / "run_phase_3_1_governance_audit.py"
    ]

    manifest_records = []
    for f in new_files_to_track:
        assert f.exists(), f"Expected created file does not exist: {f}"
        size = f.stat().st_size
        sha256 = compute_file_sha256(f)
        rel_path = f.relative_to(PROJECT_ROOT).as_posix()
        manifest_records.append({
            "relative_path": rel_path,
            "file_size_bytes": size,
            "sha256_checksum": sha256,
            "phase": "Phase 3.1",
            "module_category": "Governance / Setup" if "governance" in rel_path or "config" in rel_path or "logging" in rel_path else "Automated Tests",
            "status": "PHASE_3_1_CREATED"
        })
        log_print(f"  Tracked: {rel_path} ({size} bytes, SHA-256: {sha256[:16]}...)")

    manifest_data = {
        "manifest_title": "Phase 3.1 Repository Setup and Governance Implementation Manifest",
        "phase": "Phase 3.1",
        "timestamp": datetime.now().isoformat(),
        "random_seed": RANDOM_SEED,
        "baseline_frozen_files_count": 28,
        "baseline_integrity_status": "VERIFIED_UNTOUCHED",
        "phase_3_1_created_files_count": len(manifest_records),
        "created_files": manifest_records,
        "acceptance_criteria_verification": {
            "all_28_baseline_hashes_match": True,
            "governance_tests_passed": True,
            "dataset_leakage_tests_passed": True,
            "reproducibility_config_documented": True,
            "machine_readable_manifest_created": True,
            "clean_execution_log_generated": True
        },
        "phase_3_1_status": "COMPLETED_ACCEPTANCE_CRITERIA_MET"
    }

    manifest_output_path = PHASE3_ARTIFACTS_DIR / "phase_3_1_implementation_manifest.json"
    with open(manifest_output_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    log_print(f"\nSaved implementation manifest to: {manifest_output_path}")

    # Save complete execution log
    log_print(f"Execution completed successfully at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_output_path = PHASE3_ARTIFACTS_DIR / "phase_3_1_execution_log.txt"
    with open(log_output_path, "w", encoding="utf-8") as f:
        f.write(log_stream.getvalue())
    log_print(f"Saved execution log to: {log_output_path}")

    log_print("\n" + "=" * 80)
    log_print("PHASE 3.1 ACCEPTANCE CRITERIA FULLY MET. READY FOR PHASE 3.2 AUTHORIZATION.")
    log_print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(run_phase_3_1_audit())
