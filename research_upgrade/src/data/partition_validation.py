"""
Dataset Partition and Anti-Leakage Validation Module (Phase 3.2)
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection
"""

from typing import Dict, Any, List, Set
import pandas as pd

EXPECTED_TEXT_PARTITION_SIZES = {
    "train": 4000,
    "val": 1500,
    "cal_prob": 1000,
    "cal_conf": 1000,
    "dev_reserve": 2337,
    "test": 3000,
    "test_reserve": 6821
}

EXPECTED_TEXT_PREMISE_COUNTS = {
    "train": 1349,
    "val": 508,
    "cal_prob": 336,
    "cal_conf": 334,
    "dev_reserve": 791,
    "test": 1015,
    "test_reserve": 2307
}


def validate_text_partitions(partitions_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Validates that:
    1. All 7 text partitions exist.
    2. Every partition matches its exact integer row count.
    3. Every partition matches its exact unique premise count.
    4. Pairwise premise intersection across all 7 partitions is strictly 0 (Zero Leakage).
    5. Clean partition sum matches 19,658.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # 1. Presence check
    missing_partitions = [
        name for name in EXPECTED_TEXT_PARTITION_SIZES if name not in partitions_dict
    ]
    if missing_partitions:
        errors.append(f"Missing text partitions: {missing_partitions}")
        return {
            "status": "FAILED",
            "errors": errors
        }

    # 2. Row count and premise count checks
    actual_sizes = {}
    actual_premises = {}
    premise_sets: Dict[str, Set[str]] = {}

    for name, expected_size in EXPECTED_TEXT_PARTITION_SIZES.items():
        df_p = partitions_dict[name]
        actual_size = len(df_p)
        actual_sizes[name] = actual_size
        if actual_size != expected_size:
            errors.append(
                f"Partition '{name}' size mismatch: expected {expected_size}, got {actual_size}"
            )

        prems = set(df_p["reference_text"].dropna().unique())
        premise_sets[name] = prems
        actual_premises[name] = len(prems)

        expected_prems = EXPECTED_TEXT_PREMISE_COUNTS.get(name)
        if expected_prems and len(prems) != expected_prems:
            warnings.append(
                f"Partition '{name}' unique premise count mismatch: expected {expected_prems}, got {len(prems)}"
            )

    # 3. Pairwise Cross-Partition Overlap Check (7x7 Matrix)
    overlap_matrix: Dict[str, Dict[str, int]] = {}
    cross_leaks = 0

    partition_names = list(EXPECTED_TEXT_PARTITION_SIZES.keys())
    for s1 in partition_names:
        overlap_matrix[s1] = {}
        for s2 in partition_names:
            overlap = len(premise_sets[s1].intersection(premise_sets[s2]))
            overlap_matrix[s1][s2] = overlap
            if s1 != s2 and overlap > 0:
                cross_leaks += overlap
                errors.append(
                    f"DATA LEAKAGE DETECTED: {overlap} premises overlap between '{s1}' and '{s2}'!"
                )

    total_clean_rows = sum(actual_sizes.values())
    if total_clean_rows != 19658:
        errors.append(f"Total clean rows mismatch: expected 19658, got {total_clean_rows}")

    status = "PASSED" if len(errors) == 0 else "FAILED"

    return {
        "status": status,
        "total_clean_rows": total_clean_rows,
        "partition_sizes": actual_sizes,
        "partition_premise_counts": actual_premises,
        "cross_partition_premise_leaks": cross_leaks,
        "partition_overlap_matrix": overlap_matrix,
        "errors": errors,
        "warnings": warnings
    }


def validate_code_partitions(
    df_code: pd.DataFrame,
    policy_a_df: pd.DataFrame,
    policy_b_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Validates IR-Plag code dataset policies:
    1. Raw pair count == 733.
    2. Policy A retained == 720 (24 duplicate rows -> 11 retained).
    3. Policy B retained == 709 (24 duplicate rows -> 0 retained).
    4. Reversed duplicate pairs == 0.
    """
    errors: List[str] = []

    if len(df_code) != 733:
        errors.append(f"Raw code pair count mismatch: expected 733, got {len(df_code)}")

    if len(policy_a_df) != 720:
        errors.append(f"Policy A pair count mismatch: expected 720, got {len(policy_a_df)}")

    if len(policy_b_df) != 709:
        errors.append(f"Policy B pair count mismatch: expected 709, got {len(policy_b_df)}")

    # Check reversed duplicate pairs via canonical keys
    canon_keys = set()
    n_reversed_conflicts = 0
    for _, r in df_code.iterrows():
        c1, c2 = str(r["reference_code"]), str(r["suspicious_code"])
        key_fwd = (c1, c2)
        key_rev = (c2, c1)
        if key_rev in canon_keys and key_fwd != key_rev:
            n_reversed_conflicts += 1
        canon_keys.add(key_fwd)

    if n_reversed_conflicts > 0:
        errors.append(f"Detected {n_reversed_conflicts} reversed duplicate pairs in code dataset!")

    status = "PASSED" if len(errors) == 0 else "FAILED"

    return {
        "status": status,
        "raw_pairs_count": len(df_code),
        "policy_a_count": len(policy_a_df),
        "policy_b_count": len(policy_b_df),
        "reversed_duplicate_conflicts": n_reversed_conflicts,
        "errors": errors
    }

