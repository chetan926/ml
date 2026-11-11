"""
Automated Baseline Governance and Integrity Verification Module (Phase 3.1)
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection
"""

from pathlib import Path
import os
import sys
import csv
import hashlib
from typing import Optional, Dict, Any, List

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from research_upgrade.src.config import (
    PROJECT_ROOT,
    BASELINE_SRC_DIR,
    PHASE2_MANIFEST_PATH,
)


def compute_file_sha256(file_path: Path) -> str:
    """Computes hexadecimal SHA-256 digest of a file reading in 64KB chunks."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_registered_manifest(manifest_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Loads registered baseline SHA-256 records from manifest CSV."""
    if manifest_path is None:
        manifest_path = PHASE2_MANIFEST_PATH

    if not manifest_path.exists():
        raise FileNotFoundError(f"Baseline manifest not found at: {manifest_path}")

    records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "relative_path": row["relative_path"],
                "file_size_bytes": int(row["file_size_bytes"]),
                "sha256_checksum": row["sha256_checksum"],
                "status": row.get("status", "CURRENT_SNAPSHOT_HASHED")
            })
    return records


def verify_baseline_integrity(
    manifest_path: Optional[Path] = None,
    src_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Exhaustively audits all registered baseline files in `src/`.
    Verifies existence, file size, SHA-256 checksum, and checks for untracked files.
    """
    if manifest_path is None:
        manifest_path = PHASE2_MANIFEST_PATH
    if src_dir is None:
        src_dir = BASELINE_SRC_DIR

    manifest_records = load_registered_manifest(manifest_path)
    registered_paths = {r["relative_path"]: r for r in manifest_records}

    matched_records = []
    mismatched_records = []
    missing_files = []

    for rel_path, expected in registered_paths.items():
        disk_path = PROJECT_ROOT / rel_path
        if not disk_path.exists():
            missing_files.append(rel_path)
            continue

        actual_size = disk_path.stat().st_size
        actual_hash = compute_file_sha256(disk_path)

        if actual_hash == expected["sha256_checksum"] and actual_size == expected["file_size_bytes"]:
            matched_records.append({
                "relative_path": rel_path,
                "sha256": actual_hash,
                "size_bytes": actual_size,
                "status": "VERIFIED_MATCH"
            })
        else:
            mismatched_records.append({
                "relative_path": rel_path,
                "expected_sha256": expected["sha256_checksum"],
                "actual_sha256": actual_hash,
                "expected_size": expected["file_size_bytes"],
                "actual_size": actual_size,
                "status": "CHECKSUM_MISMATCH"
            })

    # Check for unauthorized file additions in `src/`
    actual_py_files = set()
    for p in src_dir.glob("**/*.py"):
        rel = p.relative_to(PROJECT_ROOT).as_posix()
        actual_py_files.add(rel)

    expected_py_files = set(registered_paths.keys())
    untracked_files = sorted(list(actual_py_files - expected_py_files))

    is_intact = (
        len(missing_files) == 0
        and len(mismatched_records) == 0
        and len(untracked_files) == 0
        and len(matched_records) == 28
    )

    return {
        "status": "PASSED" if is_intact else "FAILED",
        "total_registered_files": len(registered_paths),
        "matched_count": len(matched_records),
        "mismatched_count": len(mismatched_records),
        "missing_count": len(missing_files),
        "untracked_count": len(untracked_files),
        "missing_files": missing_files,
        "mismatched_files": mismatched_records,
        "untracked_files": untracked_files,
        "manifest_path": str(manifest_path),
        "baseline_src_dir": str(src_dir)
    }


def assert_baseline_intact(manifest_path: Optional[Path] = None) -> None:
    """Raises RuntimeError if baseline repository files are modified, missing, or injected."""
    res = verify_baseline_integrity(manifest_path)
    if res["status"] != "PASSED":
        error_msg = [
            "GOVERNANCE VIOLATION: Baseline 'src/' code integrity check failed!",
            f"Matched: {res['matched_count']}/{res['total_registered_files']}",
        ]
        if res["missing_files"]:
            error_msg.append(f"Missing files: {res['missing_files']}")
        if res["mismatched_files"]:
            error_msg.append(f"Mismatched files: {res['mismatched_files']}")
        if res["untracked_files"]:
            error_msg.append(f"Untracked files injected into src/: {res['untracked_files']}")
        raise RuntimeError("\n".join(error_msg))


if __name__ == "__main__":
    report = verify_baseline_integrity()
    print("Baseline Integrity Check Result:")
    print(f"Status: {report['status']}")
    print(f"Matched: {report['matched_count']} / {report['total_registered_files']}")
    if report["status"] != "PASSED":
        print(f"Details: {report}")
        exit(1)
    else:
        print("All 28 baseline files verified intact with zero deviations.")
        exit(0)
