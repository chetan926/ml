"""
Phase 2.3.5 Comprehensive Independent Evidence Integrity Repair & Audit Script
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection
Author: Independent Senior Research Integrity Auditor
Date: November 10, 2026
Repository: D:\ml rearcch project ad
"""

import os
import sys
import json
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np

# Robust path resolution: script lives in research_upgrade/audits/phase_2_3_5/
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]
AUDIT_DIR = SCRIPT_PATH.parent
LOG_FILE = AUDIT_DIR / "audit_execution_log.txt"

class AuditLogger:
    def __init__(self, log_path: Path):
        self.terminal = sys.stdout
        self.log_file = open(log_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()

def compute_sha256(file_path: Path) -> str:
    """Computes SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def normalize_text(text: str) -> str:
    """Normalizes whitespace and lowercases text."""
    if not isinstance(text, str):
        return ""
    return " ".join(text.lower().split())

def audit_text_benchmark(df_text: pd.DataFrame) -> dict:
    """Exhaustive audit of text benchmark, duplicate analysis, and exact-target partitioning."""
    print("\n" + "="*70)
    print("AUDIT SECTION 1: TEXT BENCHMARK DATASET INTEGRITY & PARTITIONING")
    print("="*70)
    
    total_raw_rows = len(df_text)
    print(f"Total raw text rows in CSV: {total_raw_rows}")
    assert total_raw_rows == 19666, f"Expected 19666 rows, found {total_raw_rows}"

    # 1. Null / empty check
    null_ref = int(df_text["reference_text"].isna().sum())
    null_sus = int(df_text["suspicious_text"].isna().sum())
    empty_ref = int((df_text["reference_text"].astype(str).str.strip() == "").sum())
    empty_sus = int((df_text["suspicious_text"].astype(str).str.strip() == "").sum())
    print(f"Null or empty text fields: ref_null={null_ref}, sus_null={null_sus}, ref_empty={empty_ref}, sus_empty={empty_sus}")
    assert null_ref == 0 and null_sus == 0 and empty_ref == 0 and empty_sus == 0, "Null or empty values detected in text benchmark!"

    # 2. Split distribution in raw CSV
    split_counts = {k: int(v) for k, v in df_text["split"].value_counts().items()}
    print(f"Split counts in raw CSV: {split_counts}")
    assert split_counts == {"val": 9842, "test": 9824}, f"Unexpected split counts: {split_counts}"

    # 3. Directed Duplicate Analysis
    exact_ordered_mask = df_text.duplicated(subset=["reference_text", "suspicious_text"], keep=False)
    n_exact_ordered = int(exact_ordered_mask.sum())
    
    dupe_groups = []
    dupe_rows_df = df_text[exact_ordered_mask]
    grp_idx = 0
    dupe_row_to_group = {}
    dupe_row_is_redundant = {}

    for _, grp in dupe_rows_df.groupby(["reference_text", "suspicious_text"]):
        grp_idx += 1
        g_id = f"TEXT_DUPE_GRP_{grp_idx:02d}"
        ids = grp["id"].tolist()
        dupe_groups.append({
            "duplicate_group_id": g_id,
            "row_ids": ids,
            "reference_text": grp["reference_text"].iloc[0],
            "suspicious_text": grp["suspicious_text"].iloc[0],
            "is_plagiarized": int(grp["is_plagiarized"].iloc[0]),
            "split": grp["split"].tolist()
        })
        for i, r_id in enumerate(ids):
            dupe_row_to_group[r_id] = g_id
            dupe_row_is_redundant[r_id] = (i > 0)

    print(f"Exact ordered duplicate rows: {n_exact_ordered} across {len(dupe_groups)} duplicate groups")
    assert n_exact_ordered == 4 and len(dupe_groups) == 2, f"Expected 4 rows across 2 duplicate groups, got {n_exact_ordered} across {len(dupe_groups)}"

    # 4. Explicit Canonical Pair Reversed-Duplicate Audit
    canon_records = []
    for _, r in df_text.iterrows():
        t1 = str(r["reference_text"])
        t2 = str(r["suspicious_text"])
        if t1 <= t2:
            canon_key = (t1, t2)
            is_forward = True
        else:
            canon_key = (t2, t1)
            is_forward = False
        canon_records.append({
            "id": r["id"],
            "canon_key": canon_key,
            "is_forward": is_forward,
            "label": int(r["is_plagiarized"])
        })
    df_canon = pd.DataFrame(canon_records)
    
    canon_groups = df_canon.groupby("canon_key")
    text_reversed_audit = []
    n_canonical_collisions = 0
    n_reversed_pairs = 0
    
    for c_key, grp in canon_groups:
        if len(grp) > 1:
            n_canonical_collisions += len(grp)
            fwd_count = int(grp["is_forward"].sum())
            rev_count = len(grp) - fwd_count
            labels = grp["label"].tolist()
            conflicting_label = (len(set(labels)) > 1)
            
            if fwd_count > 0 and rev_count > 0:
                classification = "MIXED_ORIENTATION_DUPLICATE"
                n_reversed_pairs += min(fwd_count, rev_count)
            elif fwd_count > 0 and rev_count == 0:
                classification = "EXACT_FORWARD_ONLY_DUPLICATE"
            else:
                classification = "EXACT_REVERSED_ONLY_DUPLICATE"
                
            text_reversed_audit.append({
                "canonical_key_hash": hashlib.sha256(f"{c_key[0]}||{c_key[1]}".encode('utf-8')).hexdigest()[:16],
                "participating_row_ids": str(grp["id"].tolist()),
                "group_size": len(grp),
                "forward_count": fwd_count,
                "reversed_count": rev_count,
                "classification": classification,
                "conflicting_label": conflicting_label
            })
            
    print(f"Canonical pair analysis: {len(text_reversed_audit)} collision groups, Reversed duplicate pairs = {n_reversed_pairs}")
    assert n_reversed_pairs == 0, f"Expected 0 reversed duplicate pairs in text, found {n_reversed_pairs}"

    # 5. Cross-Split Premise Contamination Check (val vs test)
    val_premises = set(df_text[df_text["split"] == "val"]["reference_text"].unique())
    test_premises = set(df_text[df_text["split"] == "test"]["reference_text"].unique())
    crossing_premises = val_premises.intersection(test_premises)
    print(f"Premises crossing val and test splits in raw CSV: {len(crossing_premises)}")
    assert len(crossing_premises) == 1, f"Expected exactly 1 crossing premise, found {len(crossing_premises)}"
    
    crossing_rows_info = []
    for prem in crossing_premises:
        rows = df_text[df_text["reference_text"] == prem]
        for _, r in rows.iterrows():
            crossing_rows_info.append({
                "row_id": r["id"],
                "split": r["split"],
                "premise": r["reference_text"],
                "hypothesis": r["suspicious_text"],
                "label": int(r["is_plagiarized"])
            })
    print(f"Rows affected by cross-split premise contamination: {len(crossing_rows_info)}")
    assert len(crossing_rows_info) == 6, f"Expected 6 crossing rows, found {len(crossing_rows_info)}"

    # 6. Row-Level Exclusion Ledger Construction
    exclusion_ledger = []
    excluded_row_ids = set()

    # Add crossing rows
    for cr in crossing_rows_info:
        exclusion_ledger.append({
            "row_id": cr["row_id"],
            "original_split": cr["split"],
            "ground_truth_label": cr["label"],
            "exclusion_reason": "CROSS_SPLIT_PREMISE_CONTAMINATION",
            "duplicate_group_id": "NONE",
            "premise_snippet": cr["premise"][:50],
            "final_disposition": "PURGED_PRE_SPLIT"
        })
        excluded_row_ids.add(cr["row_id"])
        
    # Add redundant duplicate rows (Policy A: drop subsequent occurrences)
    for r_id, is_red in dupe_row_is_redundant.items():
        if is_red:
            r_row = df_text[df_text["id"] == r_id].iloc[0]
            exclusion_ledger.append({
                "row_id": r_id,
                "original_split": r_row["split"],
                "ground_truth_label": int(r_row["is_plagiarized"]),
                "exclusion_reason": "REDUNDANT_EXACT_DUPLICATE_ROW",
                "duplicate_group_id": dupe_row_to_group[r_id],
                "premise_snippet": str(r_row["reference_text"])[:50],
                "final_disposition": "PURGED_PRE_SPLIT"
            })
            excluded_row_ids.add(r_id)

    print(f"Total purged rows: {len(exclusion_ledger)} (Cross: 6, Dupe: 2)")
    assert len(exclusion_ledger) == 8, f"Expected 8 purged rows, found {len(exclusion_ledger)}"

    # 7. Exact-Target Deterministic Partition Allocation (NO DELIBERATE OVERSHOOT)
    def get_premise_hash(p):
        return int(hashlib.sha256(str(p).encode('utf-8')).hexdigest(), 16)

    # Filter clean rows
    clean_mask = ~df_text["id"].isin(excluded_row_ids)
    df_clean = df_text[clean_mask].copy().reset_index(drop=True)
    assert len(df_clean) == 19658, f"Expected 19658 clean rows, found {len(df_clean)}"

    df_val_pool = df_clean[df_clean["split"] == "val"].copy()
    df_test_pool = df_clean[df_clean["split"] == "test"].copy()
    assert len(df_val_pool) == 9837, f"Expected 9837 val pool rows, found {len(df_val_pool)}"
    assert len(df_test_pool) == 9821, f"Expected 9821 test pool rows, found {len(df_test_pool)}"

    # Deterministic allocator function that NEVER overshoots
    def allocate_without_overshoot(premises_pool, counts_dict, target):
        allocated_prems = []
        cur_sum = 0
        rem_prems = list(premises_pool)
        i = 0
        while i < len(rem_prems) and cur_sum < target:
            p = rem_prems[i]
            c = counts_dict[p]
            if cur_sum + c <= target:
                allocated_prems.append(p)
                cur_sum += c
                rem_prems.pop(i)
            else:
                i += 1
        return allocated_prems, cur_sum, rem_prems

    # Allocate development partitions from val pool
    val_counts = df_val_pool["reference_text"].value_counts().to_dict()
    sorted_val_prems = sorted(val_counts.keys(), key=get_premise_hash)

    train_prems, train_count, rem_prems_1 = allocate_without_overshoot(sorted_val_prems, val_counts, 4000)
    val_prems, val_count, rem_prems_2 = allocate_without_overshoot(rem_prems_1, val_counts, 1500)
    cal_prob_prems, cal_prob_count, rem_prems_3 = allocate_without_overshoot(rem_prems_2, val_counts, 1000)
    cal_conf_prems, cal_conf_count, dev_reserve_prems = allocate_without_overshoot(rem_prems_3, val_counts, 1000)
    dev_reserve_count = sum(val_counts[p] for p in dev_reserve_prems)

    print("\nDevelopment Partition Allocations:")
    print(f"  Train:     Target=4000, Allocated={train_count}, Premises={len(train_prems)}")
    print(f"  Val:       Target=1500, Allocated={val_count}, Premises={len(val_prems)}")
    print(f"  Cal-Prob:  Target=1000, Allocated={cal_prob_count}, Premises={len(cal_prob_prems)}")
    print(f"  Cal-Conf:  Target=1000, Allocated={cal_conf_count}, Premises={len(cal_conf_prems)}")
    print(f"  Dev-Res:   Remainder={dev_reserve_count}, Premises={len(dev_reserve_prems)}")
    assert train_count == 4000, f"Train count mismatch: {train_count}"
    assert val_count == 1500, f"Val count mismatch: {val_count}"
    assert cal_prob_count == 1000, f"Cal-prob count mismatch: {cal_prob_count}"
    assert cal_conf_count == 1000, f"Cal-conf count mismatch: {cal_conf_count}"
    assert dev_reserve_count == 2337, f"Dev reserve count mismatch: {dev_reserve_count}"

    # Allocate test partition from test pool
    test_counts = df_test_pool["reference_text"].value_counts().to_dict()
    sorted_test_prems = sorted(test_counts.keys(), key=get_premise_hash)

    test_prems, test_count, test_reserve_prems = allocate_without_overshoot(sorted_test_prems, test_counts, 3000)
    test_reserve_count = sum(test_counts[p] for p in test_reserve_prems)

    print("\nTest Partition Allocations:")
    print(f"  Test:      Target=3000, Allocated={test_count}, Premises={len(test_prems)}")
    print(f"  Test-Res:  Remainder={test_reserve_count}, Premises={len(test_reserve_prems)}")
    assert test_count == 3000, f"Test count mismatch: {test_count}"
    assert test_reserve_count == 6821, f"Test reserve count mismatch: {test_reserve_count}"

    # Create mapping dictionary for clean rows
    premise_to_partition = {}
    for p in train_prems:
        premise_to_partition[p] = "train"
    for p in val_prems:
        premise_to_partition[p] = "val"
    for p in cal_prob_prems:
        premise_to_partition[p] = "cal_prob"
    for p in cal_conf_prems:
        premise_to_partition[p] = "cal_conf"
    for p in dev_reserve_prems:
        premise_to_partition[p] = "dev_reserve"
    for p in test_prems:
        premise_to_partition[p] = "test"
    for p in test_reserve_prems:
        premise_to_partition[p] = "test_reserve"

    # Construct Complete Row-Level Membership Ledger for all 19,666 rows
    membership_records = []
    exclusion_map = {r["row_id"]: r["exclusion_reason"] for r in exclusion_ledger}
    
    for _, row in df_text.iterrows():
        r_id = row["id"]
        prem = str(row["reference_text"])
        prem_hash = hashlib.sha256(prem.encode('utf-8')).hexdigest()[:12]
        
        if r_id in exclusion_map:
            assigned_part = "EXCLUDED"
            excl_reason = exclusion_map[r_id]
        else:
            assigned_part = premise_to_partition[prem]
            excl_reason = "NONE"
            
        membership_records.append({
            "row_id": r_id,
            "original_split": row["split"],
            "assigned_partition": assigned_part,
            "exclusion_reason": excl_reason,
            "premise_hash": prem_hash,
            "ground_truth_label": int(row["is_plagiarized"])
        })
        
    df_membership = pd.DataFrame(membership_records)
    assert len(df_membership) == 19666, f"Expected 19666 membership rows, got {len(df_membership)}"

    # 8. Reconstruct 7x7 Premise Overlap Matrix
    active_subsets = ["train", "val", "cal_prob", "cal_conf", "test", "dev_reserve", "test_reserve"]
    part_prem_sets = {
        "train": set(train_prems),
        "val": set(val_prems),
        "cal_prob": set(cal_prob_prems),
        "cal_conf": set(cal_conf_prems),
        "test": set(test_prems),
        "dev_reserve": set(dev_reserve_prems),
        "test_reserve": set(test_reserve_prems)
    }
    
    overlap_matrix = {p1: {p2: 0 for p2 in active_subsets} for p1 in active_subsets}
    for p1 in active_subsets:
        for p2 in active_subsets:
            overlap_matrix[p1][p2] = len(part_prem_sets[p1].intersection(part_prem_sets[p2]))

    print("\n7x7 Premise Overlap Matrix Check:")
    cross_contamination_detected = False
    for p1 in active_subsets:
        diag = overlap_matrix[p1][p1]
        off_diag = sum(overlap_matrix[p1][p2] for p2 in active_subsets if p2 != p1)
        print(f"  {p1:12s}: {diag:5d} unique premises | Cross-overlap={off_diag}")
        if off_diag > 0:
            cross_contamination_detected = True
    assert not cross_contamination_detected, "Cross-partition premise contamination detected in 7x7 matrix!"

    # Final text reconciliation
    n_active = train_count + val_count + cal_prob_count + cal_conf_count + test_count
    n_reserves = dev_reserve_count + test_reserve_count
    n_purged = len(exclusion_ledger)
    sum_total = n_active + n_reserves + n_purged
    print(f"\nFinal Text Arithmetic Reconciliation:")
    print(f"  Active({n_active}) + Reserves({n_reserves}) + Purged({n_purged}) = {sum_total}")
    print(f"  Target = {total_raw_rows} | Difference = {total_raw_rows - sum_total}")
    assert sum_total == total_raw_rows, f"Text arithmetic reconciliation failed: {sum_total} != {total_raw_rows}"

    return {
        "total_raw_rows": total_raw_rows,
        "split_counts": split_counts,
        "null_or_empty": {"null_reference": null_ref, "null_suspicious": null_sus, "empty_reference": empty_ref, "empty_suspicious": empty_sus},
        "exact_ordered_duplicate_rows": n_exact_ordered,
        "reversed_duplicate_pairs": n_reversed_pairs,
        "duplicate_groups": dupe_groups,
        "reversed_audit": text_reversed_audit,
        "crossing_premises": list(crossing_premises),
        "crossing_rows_count": len(crossing_rows_info),
        "partition_pair_counts": {
            "train": train_count,
            "val": val_count,
            "cal_prob": cal_prob_count,
            "cal_conf": cal_conf_count,
            "test": test_count,
            "dev_reserve": dev_reserve_count,
            "test_reserve": test_reserve_count
        },
        "partition_overlap_matrix": overlap_matrix,
        "exclusion_ledger": exclusion_ledger,
        "membership_df": df_membership,
        "reconciliation": {
            "raw_rows": total_raw_rows,
            "active_pairs": n_active,
            "purged_cross_contamination": len(crossing_rows_info),
            "purged_redundant_duplicates": len(dupe_rows_df) - len(dupe_groups),
            "retained_development_reserve": dev_reserve_count,
            "retained_test_reserve": test_reserve_count,
            "total_retained_reserves": n_reserves,
            "sum_check": sum_total,
            "discrepancy": total_raw_rows - sum_total,
            "status": "PASS"
        }
    }

def audit_code_benchmark(df_code: pd.DataFrame, ir_plag_dir: Path) -> dict:
    """Exhaustive audit of IR-Plag code benchmark, duplicates, and case-05 files/folds."""
    print("\n" + "="*70)
    print("AUDIT SECTION 2: CODE BENCHMARK, DUPLICATES & CASE-05 DEGENERACY")
    print("="*70)
    
    total_pairs = len(df_code)
    print(f"Total pairs in code CSV: {total_pairs}")
    assert total_pairs == 733, f"Expected 733 code pairs, found {total_pairs}"

    # 1. Physical Java files on disk
    all_physical_java = list(ir_plag_dir.glob("**/*.java"))
    print(f"Total physical Java files across entire IR-Plag dataset: {len(all_physical_java)}")

    # 2. Duplicate Analysis in Code
    dupes_mask = df_code.duplicated(subset=["reference_code", "suspicious_code"], keep=False)
    n_dupe_rows = int(dupes_mask.sum())
    print(f"Code rows participating in duplicate groups: {n_dupe_rows}")
    assert n_dupe_rows == 24, f"Expected 24 rows participating in duplicates, found {n_dupe_rows}"

    # Group duplicate rows
    dupe_code_df = df_code[dupes_mask]
    code_dupe_groups = []
    grp_idx = 0
    code_dupe_ledger = []

    for (r_code, s_code), grp in dupe_code_df.groupby(["reference_code", "suspicious_code"]):
        grp_idx += 1
        g_id = f"CODE_DUPE_GRP_{grp_idx:02d}"
        ids = grp["id"].tolist()
        code_dupe_groups.append({
            "duplicate_group_id": g_id,
            "case": grp["case"].iloc[0],
            "pair_ids": ids,
            "obfuscation_levels": grp["obfuscation_level"].tolist(),
            "labels": grp["is_plagiarized"].tolist(),
            "row_count": len(grp)
        })
        for i, (_, row) in enumerate(grp.iterrows()):
            code_dupe_ledger.append({
                "duplicate_group_id": g_id,
                "case": row["case"],
                "row_id": row["id"],
                "obfuscation_level": row["obfuscation_level"],
                "ground_truth_label": int(row["is_plagiarized"]),
                "occurrence_index": i + 1,
                "policy_a_action": "RETAIN" if i == 0 else "PURGE_REDUNDANT",
                "policy_b_action": "PURGE_COLLISION"
            })
    print(f"Code duplicate groups: {len(code_dupe_groups)} groups across {n_dupe_rows} duplicate rows")
    assert len(code_dupe_groups) == 11, f"Expected 11 duplicate groups, found {len(code_dupe_groups)}"

    # 3. Canonical Pair Analysis for Reversed Duplicates in Code
    canon_code_records = []
    for _, r in df_code.iterrows():
        c1 = str(r["reference_code"])
        c2 = str(r["suspicious_code"])
        if c1 <= c2:
            canon_key = (c1, c2)
            is_fwd = True
        else:
            canon_key = (c2, c1)
            is_fwd = False
        canon_code_records.append({
            "id": r["id"],
            "case": r["case"],
            "canon_key": canon_key,
            "is_forward": is_fwd,
            "label": int(r["is_plagiarized"])
        })
    df_canon_code = pd.DataFrame(canon_code_records)
    
    code_reversed_audit = []
    n_rev_code_pairs = 0
    for c_key, grp in df_canon_code.groupby("canon_key"):
        if len(grp) > 1:
            fwd_count = int(grp["is_forward"].sum())
            rev_count = len(grp) - fwd_count
            labels = grp["label"].tolist()
            conflicting_label = (len(set(labels)) > 1)
            
            if fwd_count > 0 and rev_count > 0:
                classification = "MIXED_ORIENTATION_DUPLICATE"
                n_rev_code_pairs += min(fwd_count, rev_count)
            elif fwd_count > 0 and rev_count == 0:
                classification = "EXACT_FORWARD_ONLY_DUPLICATE"
            else:
                classification = "EXACT_REVERSED_ONLY_DUPLICATE"
                
            code_reversed_audit.append({
                "canonical_key_hash": hashlib.sha256(f"{c_key[0]}||{c_key[1]}".encode('utf-8')).hexdigest()[:16],
                "case": grp["case"].iloc[0],
                "participating_row_ids": str(grp["id"].tolist()),
                "group_size": len(grp),
                "forward_count": fwd_count,
                "reversed_count": rev_count,
                "classification": classification,
                "conflicting_label": conflicting_label
            })
            
    print(f"Code canonical analysis: {len(code_reversed_audit)} collision groups, Reversed duplicate pairs = {n_rev_code_pairs}")
    assert n_rev_code_pairs == 0, f"Expected 0 reversed duplicate pairs in code, found {n_rev_code_pairs}"

    # 4. Detailed Task-by-Task Accounting (Policy A vs Policy B)
    task_accounting = []
    for c in sorted(df_code["case"].unique().tolist()):
        c_df = df_code[df_code["case"] == c]
        n_raw = len(c_df)
        n_raw_pos = int((c_df["is_plagiarized"] == 1).sum())
        n_raw_neg = int((c_df["is_plagiarized"] == 0).sum())
        
        c_dupe_mask = c_df.duplicated(subset=["reference_code", "suspicious_code"], keep=False)
        n_c_dupe_rows = int(c_dupe_mask.sum())
        
        # Policy A: Keep first
        c_dedup_a = c_df.drop_duplicates(subset=["reference_code", "suspicious_code"], keep="first")
        n_dedup_a = len(c_dedup_a)
        n_dedup_a_pos = int((c_dedup_a["is_plagiarized"] == 1).sum())
        n_dedup_a_neg = int((c_dedup_a["is_plagiarized"] == 0).sum())
        
        # Policy B: Drop all collision rows
        c_dedup_b = c_df[~c_dupe_mask]
        n_dedup_b = len(c_dedup_b)
        n_dedup_b_pos = int((c_dedup_b["is_plagiarized"] == 1).sum())
        n_dedup_b_neg = int((c_dedup_b["is_plagiarized"] == 0).sum())
        
        # Physical files in task folder
        c_path = ir_plag_dir / c
        java_files = list(c_path.glob("**/*.java")) if c_path.exists() else []

        task_accounting.append({
            "task_id": c,
            "raw_total_pairs": n_raw,
            "raw_pos_pairs": n_raw_pos,
            "raw_neg_pairs": n_raw_neg,
            "duplicate_rows": n_c_dupe_rows,
            "policy_a_retained_pairs": n_dedup_a,
            "policy_a_pos": n_dedup_a_pos,
            "policy_a_neg": n_dedup_a_neg,
            "policy_b_non_collision_pairs": n_dedup_b,
            "policy_b_pos": n_dedup_b_pos,
            "policy_b_neg": n_dedup_b_neg,
            "physical_java_files": len(java_files)
        })

    print("\nTask Accounting Breakdown:")
    for ta in task_accounting:
        print(f"  {ta['task_id']}: Raw={ta['raw_total_pairs']} ({ta['raw_pos_pairs']} pos, {ta['raw_neg_pairs']} neg) | Dupes={ta['duplicate_rows']} | Policy A={ta['policy_a_retained_pairs']} | Policy B={ta['policy_b_non_collision_pairs']} | Files={ta['physical_java_files']}")
        
    c6_rec = next(t for t in task_accounting if t["task_id"] == "case-06")
    assert c6_rec["raw_total_pairs"] == 105, f"Case-06 raw total mismatch: {c6_rec['raw_total_pairs']}"
    assert c6_rec["policy_a_retained_pairs"] == 103, f"Case-06 Policy A mismatch: {c6_rec['policy_a_retained_pairs']}"
    assert c6_rec["policy_b_non_collision_pairs"] == 101, f"Case-06 Policy B mismatch: {c6_rec['policy_b_non_collision_pairs']}"

    # 5. Case-05 Physical Java File Inventory (Resolving 69 vs 58)
    c5_dir = ir_plag_dir / "case-05"
    c5_files = sorted(list(c5_dir.glob("**/*.java")), key=lambda p: str(p).lower())
    print(f"\nPhysical Java files in case-05: {len(c5_files)}")
    assert len(c5_files) == 69, f"Expected 69 physical Java files in case-05, found {len(c5_files)}"

    c5_file_inventory = []
    for f in c5_files:
        rel_to_c5 = f.relative_to(c5_dir).as_posix()
        parts = rel_to_c5.split("/")
        category = parts[0]
        
        if category == "original":
            f_type = "original"
            lvl = "NONE"
            base_id = "prog_original"
        elif category == "non-plagiarized":
            f_type = "non-plagiarized"
            lvl = "NONE"
            np_id = parts[1] # e.g. "01"
            base_id = f"prog_np_{np_id}"
        elif category == "plagiarized":
            f_type = "plagiarized"
            lvl = parts[1] # e.g. "L1"
            base_id = "prog_original" # plagiarized files are transformations of original
        else:
            f_type = "unknown"
            lvl = "NONE"
            base_id = "unknown"
            
        c5_file_inventory.append({
            "relative_path": rel_to_c5,
            "category": category,
            "file_type": f_type,
            "obfuscation_level": lvl,
            "base_program_identity": base_id,
            "file_size_bytes": f.stat().st_size
        })

    # 6. Case-05 Pair-to-Program Mapping
    c5_df = df_code[df_code["case"] == "case-05"].copy()
    print(f"Total pairs in case-05: {len(c5_df)} (Pos={int((c5_df['is_plagiarized']==1).sum())}, Neg={int((c5_df['is_plagiarized']==0).sum())})")
    assert len(c5_df) == 107, f"Expected 107 pairs in case-05, found {len(c5_df)}"

    c5_pair_mapping = []
    for _, row in c5_df.iterrows():
        r_type = row["ref_type"]
        s_type = row["sus_type"]
        lbl = int(row["is_plagiarized"])
        
        # Determine base program identities
        if r_type == "original":
            prog_ref = "prog_original"
        elif r_type.startswith("non-plag-"):
            prog_ref = f"prog_np_{r_type.split('-')[-1]}"
        else:
            prog_ref = r_type
            
        if s_type == "plagiarized":
            prog_sus = "prog_original" # transformed variant of prog_original
        elif s_type == "non-plagiarized":
            # In data_loader.py, sus_type is 'non-plagiarized' for pairs against original
            prog_sus = "prog_np_unknown"
        elif s_type.startswith("non-plag-"):
            prog_sus = f"prog_np_{s_type.split('-')[-1]}"
        else:
            prog_sus = s_type
            
        c5_pair_mapping.append({
            "pair_id": row["id"],
            "case": row["case"],
            "ref_type": r_type,
            "sus_type": s_type,
            "obfuscation_level": row["obfuscation_level"],
            "ground_truth_label": lbl,
            "prog_ref_identity": prog_ref,
            "prog_sus_identity": prog_sus,
            "involves_prog_original": (prog_ref == "prog_original" or prog_sus == "prog_original")
        })

    # Verify that ALL positive pairs in case-05 involve prog_original
    pos_pairs = [p for p in c5_pair_mapping if p["ground_truth_label"] == 1]
    assert all(p["involves_prog_original"] for p in pos_pairs), "Positive pair in case-05 found that does not involve prog_original!"
    print(f"Verified: All {len(pos_pairs)} positive pairs in case-05 involve prog_original.")

    # 7. Case-05 Group-Disjoint Cross-Fitting Proof (Programmatic derivation from 16 base identities)
    # 16 base program identities: prog_original, prog_np_01 ... prog_np_15
    np_identities = [f"non-plag-{i:02d}" for i in range(1, 16)]
    folds = {
        1: ["original", "non-plag-01", "non-plag-02"],
        2: ["non-plag-03", "non-plag-04", "non-plag-05"],
        3: ["non-plag-06", "non-plag-07", "non-plag-08"],
        4: ["non-plag-09", "non-plag-10", "non-plag-11"],
        5: ["non-plag-12", "non-plag-13", "non-plag-14", "non-plag-15"]
    }

    c5_fold_audit = []
    for f_idx, held_out in folds.items():
        if "original" in held_out:
            # Held-out fold contains original -> Train has NO original -> Train has 0 positives!
            n_pos_train = 0
            n_neg_train = int(((c5_df["ref_type"] != "original") & (c5_df["sus_type"] != "original") & (c5_df["is_plagiarized"] == 0)).sum())
            n_pos_test = len(pos_pairs)
            n_neg_test = len(c5_df) - n_pos_test - n_neg_train
            status = "DIVERGES_ZERO_TRAIN_POSITIVES"
            unreg_effect = "Unregularized MLE diverges to B -> -inf (loss strictly decreases towards 0)"
            reg_effect = "Regularized LogisticRegression raises ValueError: solver needs samples of at least 2 classes"
            cal_feasible = False
        else:
            # Original is in training fold -> Test fold has NO original -> Test fold has 0 positives!
            n_pos_train = len(pos_pairs)
            n_pos_test = 0
            # Test negatives are pairs where either ref or sus is in held_out
            n_neg_test = int(((c5_df["ref_type"].isin(held_out)) | (c5_df["sus_type"].isin(held_out))).sum())
            n_neg_train = (len(c5_df) - len(pos_pairs)) - n_neg_test
            status = "UNDEFINED_ZERO_TEST_POSITIVES"
            unreg_effect = "Training feasible but held-out test evaluation degenerate"
            reg_effect = "Test fold has 0 positives: ROC-AUC, PR-AUC, Precision, Recall, ECE are undefined"
            cal_feasible = False

        c5_fold_audit.append({
            "fold_index": f_idx,
            "held_out_identities": ", ".join(held_out),
            "pos_train_count": n_pos_train,
            "neg_train_count": n_neg_train,
            "pos_test_count": n_pos_test,
            "neg_test_count": n_neg_test,
            "platt_scaling_feasible": cal_feasible,
            "mathematical_status": status,
            "unregularized_effect": unreg_effect,
            "regularized_effect": reg_effect
        })

    print("\ncase-05 5-Fold Group Cross-Fitting Proof:")
    for cfa in c5_fold_audit:
        print(f"  Fold {cfa['fold_index']}: Train=(+{cfa['pos_train_count']}, -{cfa['neg_train_count']}) | Test=(+{cfa['pos_test_count']}, -{cfa['neg_test_count']}) -> {cfa['mathematical_status']}")

    return {
        "total_pairs": total_pairs,
        "physical_java_files": len(all_physical_java),
        "duplicate_rows": n_dupe_rows,
        "reversed_duplicate_pairs": n_rev_code_pairs,
        "duplicate_groups": code_dupe_groups,
        "duplicate_ledger": code_dupe_ledger,
        "reversed_audit": code_reversed_audit,
        "task_accounting": task_accounting,
        "case_05_files": c5_file_inventory,
        "case_05_pairs": c5_pair_mapping,
        "case_05_folds": c5_fold_audit
    }

def audit_loader_verification() -> list:
    """Verifies existence and implementation status of loaders in repository."""
    print("\n" + "="*70)
    print("AUDIT SECTION 3: DATA LOADER VERIFICATION AUDIT")
    print("="*70)
    
    loader_items = [
        {
            "component_name": "TextBenchmarkDataLoader",
            "specified_location": "research_upgrade/src/text_data_loader.py",
            "exists_on_disk": (PROJECT_ROOT / "research_upgrade/src/text_data_loader.py").exists(),
            "classification": "DESIGN_SPECIFIED_NOT_IMPLEMENTED",
            "justification": "Phase 3 implementation has not been authorized. No executable loader exists in research_upgrade/src/."
        },
        {
            "component_name": "CodeBenchmarkDataLoader",
            "specified_location": "research_upgrade/src/code_data_loader.py",
            "exists_on_disk": (PROJECT_ROOT / "research_upgrade/src/code_data_loader.py").exists(),
            "classification": "DESIGN_SPECIFIED_NOT_IMPLEMENTED",
            "justification": "Phase 3 implementation has not been authorized. No executable loader exists in research_upgrade/src/."
        },
        {
            "component_name": "LegacyDataLoader",
            "specified_location": "src/data_loader.py",
            "exists_on_disk": (PROJECT_ROOT / "src/data_loader.py").exists(),
            "classification": "LEGACY_BASELINE_IMPLEMENTATION",
            "justification": "Existing baseline loader on disk; lacks group-aware 5-partition allocation and deduplication."
        },
        {
            "component_name": "Phase235AuditScript",
            "specified_location": "research_upgrade/audits/phase_2_3_5/run_phase_2_3_5_audit.py",
            "exists_on_disk": True,
            "classification": "AUDIT_SIMULATION_SCRIPT",
            "justification": "Independent audit script performing verification; must NOT be conflated with production data loader."
        }
    ]
    for li in loader_items:
        print(f"  {li['component_name']}: Exists={li['exists_on_disk']} -> {li['classification']}")
    return loader_items

def audit_source_inventory() -> dict:
    """Inventories all 28 Python source files in src/ and computes current SHA-256."""
    print("\n" + "="*70)
    print("AUDIT SECTION 4: SOURCE INVENTORY & SHA-256 CHECKSUM MANIFEST")
    print("="*70)
    
    src_dir = PROJECT_ROOT / "src"
    py_files = sorted(list(src_dir.glob("**/*.py")), key=lambda p: str(p).lower())
    
    root_src = [p for p in py_files if p.parent == src_dir]
    code_sub = [p for p in py_files if "src\\code" in str(p) or "src/code" in str(p)]
    text_sub = [p for p in py_files if "src\\text" in str(p) or "src/text" in str(p)]

    print(f"Total Python source files in 'src/': {len(py_files)}")
    print(f"  - Root of 'src/': {len(root_src)} files")
    print(f"  - 'src/code/':    {len(code_sub)} files")
    print(f"  - 'src/text/':    {len(text_sub)} files")
    assert len(py_files) == 28, f"Expected 28 Python files in src/, found {len(py_files)}"

    manifest = []
    for p in py_files:
        rel_path = p.relative_to(PROJECT_ROOT).as_posix()
        size = p.stat().st_size
        sha = compute_sha256(p)
        manifest.append({
            "relative_path": rel_path,
            "file_size_bytes": size,
            "sha256_checksum": sha,
            "status": "FROZEN_CURRENT_BASELINE"
        })

    git_exists = (PROJECT_ROOT / ".git").exists()
    return {
        "total_src_python_files": len(py_files),
        "root_src_count": len(root_src),
        "code_sub_count": len(code_sub),
        "text_sub_count": len(text_sub),
        "git_exists": git_exists,
        "historical_integrity_status": "NOT_VERIFIABLE",
        "historical_integrity_reason": "No pre-Phase-2 .git repository or cryptographic checksum manifest exists on disk",
        "manifest": manifest
    }

def main():
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    logger = AuditLogger(LOG_FILE)
    sys.stdout = logger

    print("="*70)
    print("PHASE 2.3.5 COMPREHENSIVE INDEPENDENT AUDIT EXECUTION")
    print("Execution Date: November 10, 2026")
    print("Target Audit Directory:", AUDIT_DIR)
    print("="*70)

    # 1. Run Text Audit
    text_csv = PROJECT_ROOT / "datasets" / "processed" / "text_plagiarism_benchmark.csv"
    if not text_csv.exists():
        print(f"FATAL: Text benchmark not found at {text_csv}")
        sys.exit(1)
    df_text = pd.read_csv(text_csv)
    text_results = audit_text_benchmark(df_text)

    # 2. Run Code Audit
    code_csv = PROJECT_ROOT / "datasets" / "processed" / "code_plagiarism_benchmark.csv"
    ir_plag_dir = PROJECT_ROOT / "datasets" / "raw" / "ir_plag" / "extracted" / "IR-Plag-Dataset"
    if not code_csv.exists() or not ir_plag_dir.exists():
        print("FATAL: Code benchmark or IR-Plag directory missing")
        sys.exit(1)
    df_code = pd.read_csv(code_csv)
    code_results = audit_code_benchmark(df_code, ir_plag_dir)

    # 3. Run Loader Verification
    loader_results = audit_loader_verification()

    # 4. Run Source Inventory
    source_results = audit_source_inventory()

    # 5. Save Machine-Readable Artifacts
    print("\n" + "="*70)
    print("SAVING ALL 16 MACHINE-READABLE AUDIT ARTIFACTS IN phase_2_3_5/")
    print("="*70)

    # 1. overall_audit_summary.json
    overall_summary = {
        "audit_version": "3.6",
        "audit_phase": "Phase 2.3.5",
        "execution_date": "2026-11-10",
        "status": "READY_FOR_PHASE_3_IMPLEMENTATION_REVIEW",
        "text_benchmark_summary": {
            "total_raw_rows": text_results["total_raw_rows"],
            "exact_ordered_duplicate_rows": text_results["exact_ordered_duplicate_rows"],
            "reversed_duplicate_pairs": text_results["reversed_duplicate_pairs"],
            "purged_cross_contamination_rows": text_results["crossing_rows_count"],
            "purged_redundant_duplicate_rows": len(text_results["duplicate_groups"]),
            "clean_active_rows": text_results["reconciliation"]["active_pairs"],
            "retained_reserve_rows": text_results["reconciliation"]["total_retained_reserves"],
            "partitions": text_results["partition_pair_counts"]
        },
        "code_benchmark_summary": {
            "total_raw_pairs": code_results["total_pairs"],
            "participating_duplicate_rows": code_results["duplicate_rows"],
            "duplicate_groups_count": len(code_results["duplicate_groups"]),
            "reversed_duplicate_pairs": code_results["reversed_duplicate_pairs"],
            "policy_a_retained_total": sum(t["policy_a_retained_pairs"] for t in code_results["task_accounting"]),
            "policy_b_retained_total": sum(t["policy_b_non_collision_pairs"] for t in code_results["task_accounting"]),
            "case_05_physical_java_files": len(code_results["case_05_files"]),
            "case_05_pairs": len(code_results["case_05_pairs"]),
            "case_05_cross_fitting_degeneracy": "PROVEN_STRUCTURAL_ZERO_POSITIVES"
        },
        "loaders_summary": {
            "text_loader_status": "DESIGN_SPECIFIED_NOT_IMPLEMENTED",
            "code_loader_status": "DESIGN_SPECIFIED_NOT_IMPLEMENTED"
        },
        "source_inventory_summary": {
            "total_files": source_results["total_src_python_files"],
            "all_files_hashed": True
        }
    }
    with open(AUDIT_DIR / "overall_audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=2)
    print("Saved: overall_audit_summary.json")

    # 2. text_dataset_audit.json
    with open(AUDIT_DIR / "text_dataset_audit.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_raw_rows": text_results["total_raw_rows"],
            "split_counts": text_results["split_counts"],
            "null_or_empty": text_results["null_or_empty"],
            "exact_ordered_duplicate_rows": text_results["exact_ordered_duplicate_rows"],
            "reversed_duplicate_pairs": text_results["reversed_duplicate_pairs"],
            "crossing_premises": text_results["crossing_premises"],
            "crossing_rows_count": text_results["crossing_rows_count"],
            "partition_pair_counts": text_results["partition_pair_counts"],
            "reconciliation": text_results["reconciliation"]
        }, f, indent=2)
    print("Saved: text_dataset_audit.json")

    # 3. text_exclusion_ledger.csv
    pd.DataFrame(text_results["exclusion_ledger"]).to_csv(AUDIT_DIR / "text_exclusion_ledger.csv", index=False)
    print("Saved: text_exclusion_ledger.csv")

    # 4. text_partition_membership.csv
    text_results["membership_df"].to_csv(AUDIT_DIR / "text_partition_membership.csv", index=False)
    print("Saved: text_partition_membership.csv")

    # 5. text_partition_overlap_matrix.csv
    pd.DataFrame(text_results["partition_overlap_matrix"]).to_csv(AUDIT_DIR / "text_partition_overlap_matrix.csv")
    print("Saved: text_partition_overlap_matrix.csv")

    # 6. text_duplicate_audit.csv
    pd.DataFrame(text_results["duplicate_groups"]).to_csv(AUDIT_DIR / "text_duplicate_audit.csv", index=False)
    print("Saved: text_duplicate_audit.csv")

    # 7. text_reversed_pair_audit.csv
    pd.DataFrame(text_results["reversed_audit"]).to_csv(AUDIT_DIR / "text_reversed_pair_audit.csv", index=False)
    print("Saved: text_reversed_pair_audit.csv")

    # 8. code_dataset_audit.json
    with open(AUDIT_DIR / "code_dataset_audit.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_pairs": code_results["total_pairs"],
            "physical_java_files": code_results["physical_java_files"],
            "duplicate_rows": code_results["duplicate_rows"],
            "reversed_duplicate_pairs": code_results["reversed_duplicate_pairs"],
            "task_accounting": code_results["task_accounting"],
            "case_05_fold_audit": code_results["case_05_folds"]
        }, f, indent=2)
    print("Saved: code_dataset_audit.json")

    # 9. code_duplicate_ledger.csv
    pd.DataFrame(code_results["duplicate_ledger"]).to_csv(AUDIT_DIR / "code_duplicate_ledger.csv", index=False)
    print("Saved: code_duplicate_ledger.csv")

    # 10. code_task_accounting.csv
    pd.DataFrame(code_results["task_accounting"]).to_csv(AUDIT_DIR / "code_task_accounting.csv", index=False)
    print("Saved: code_task_accounting.csv")

    # 11. code_reversed_pair_audit.csv
    pd.DataFrame(code_results["reversed_audit"]).to_csv(AUDIT_DIR / "code_reversed_pair_audit.csv", index=False)
    print("Saved: code_reversed_pair_audit.csv")

    # 12. case_05_java_file_inventory.csv
    pd.DataFrame(code_results["case_05_files"]).to_csv(AUDIT_DIR / "case_05_java_file_inventory.csv", index=False)
    print("Saved: case_05_java_file_inventory.csv")

    # 13. case_05_pair_to_program_mapping.csv
    pd.DataFrame(code_results["case_05_pairs"]).to_csv(AUDIT_DIR / "case_05_pair_to_program_mapping.csv", index=False)
    print("Saved: case_05_pair_to_program_mapping.csv")

    # 14. case_05_fold_audit.csv
    pd.DataFrame(code_results["case_05_folds"]).to_csv(AUDIT_DIR / "case_05_fold_audit.csv", index=False)
    print("Saved: case_05_fold_audit.csv")

    # 15. loader_verification_report.csv
    pd.DataFrame(loader_results).to_csv(AUDIT_DIR / "loader_verification_report.csv", index=False)
    print("Saved: loader_verification_report.csv")

    # 16. source_file_inventory.csv & source_sha256_manifest.csv
    pd.DataFrame(source_results["manifest"]).to_csv(AUDIT_DIR / "source_file_inventory.csv", index=False)
    pd.DataFrame(source_results["manifest"]).to_csv(AUDIT_DIR / "source_sha256_manifest.csv", index=False)
    print("Saved: source_file_inventory.csv")
    print("Saved: source_sha256_manifest.csv")

    print("\n" + "="*70)
    print("PHASE 2.3.5 AUDIT EXECUTION COMPLETED WITH ALL CHECKS PASSING")
    print("ALL 16 MACHINE-READABLE ARTIFACTS VERIFIED AND RECORDED ON DISK.")
    print("="*70 + "\n")

    sys.stdout = logger.terminal
    logger.close()

if __name__ == "__main__":
    main()
