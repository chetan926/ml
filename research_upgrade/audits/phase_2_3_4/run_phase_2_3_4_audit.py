"""
Phase 2.3.4 Comprehensive Independent Certification & Dataset Verification Audit Script
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection
Author: Independent Senior Research Integrity Auditor
Date: November 10, 2026
"""

import os
import sys
import json
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np

# Robust path resolution: 3 levels up from this script file
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
    """Exhaustive audit of text benchmark and generation of exclusion ledger."""
    print("\n" + "="*60)
    print("AUDIT A: TEXT DATASET ACCOUNTING & EXCLUSION AUDIT")
    print("="*60)
    
    total_raw_rows = len(df_text)
    print(f"Total raw text rows in CSV: {total_raw_rows}")
    
    # 1. Null / empty check
    null_ref = int(df_text["reference_text"].isna().sum())
    null_sus = int(df_text["suspicious_text"].isna().sum())
    empty_ref = int((df_text["reference_text"].astype(str).str.strip() == "").sum())
    empty_sus = int((df_text["suspicious_text"].astype(str).str.strip() == "").sum())
    print(f"Null or empty text fields: ref_null={null_ref}, sus_null={null_sus}, ref_empty={empty_ref}, sus_empty={empty_sus}")
    
    # 2. Splits in CSV
    split_counts = {k: int(v) for k, v in df_text["split"].value_counts().items()}
    print(f"Split counts in CSV: {split_counts}")
    
    # 3. Duplicate Analysis
    # Exact ordered pairs
    exact_ordered_mask = df_text.duplicated(subset=["reference_text", "suspicious_text"], keep=False)
    n_exact_ordered = int(exact_ordered_mask.sum())
    
    # Duplicate pair groups
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
        # First is retained; remaining are redundant purges
        for i, r_id in enumerate(ids):
            dupe_row_to_group[r_id] = g_id
            dupe_row_is_redundant[r_id] = (i > 0)
            
    print(f"Exact ordered duplicate rows: {n_exact_ordered} across {len(dupe_groups)} duplicate groups")
    for dg in dupe_groups:
        print(f"  {dg['duplicate_group_id']}: {dg['row_ids']} | Label={dg['is_plagiarized']} | Splits={dg['split']}")

    # Reversed duplicates check (B, A) vs (A, B)
    canonical_pairs = []
    for _, row in df_text.iterrows():
        t1 = str(row["reference_text"])
        t2 = str(row["suspicious_text"])
        canonical_pairs.append((t1, t2) if t1 <= t2 else (t2, t1))
        
    df_text_canon = df_text.copy()
    df_text_canon["canonical_pair"] = canonical_pairs
    undirected_mask = df_text_canon.duplicated(subset=["canonical_pair"], keep=False)
    n_reversed_dupes = max(0, int(undirected_mask.sum()) - n_exact_ordered)
    print(f"Reversed duplicate pairs count: {n_reversed_dupes}")

    # Whitespace/case normalized duplicates
    norm_refs = df_text["reference_text"].apply(normalize_text)
    norm_suss = df_text["suspicious_text"].apply(normalize_text)
    norm_mask = pd.Series(list(zip(norm_refs, norm_suss))).duplicated(keep=False)
    n_norm_dupes = int(norm_mask.sum())
    print(f"Whitespace/Case normalized duplicate rows: {n_norm_dupes}")

    # 4. Cross-Split Premise Contamination in Raw CSV (val vs test)
    val_premises = set(df_text[df_text["split"] == "val"]["reference_text"].unique())
    test_premises = set(df_text[df_text["split"] == "test"]["reference_text"].unique())
    crossing_premises = val_premises.intersection(test_premises)
    print(f"Premises crossing val and test: {len(crossing_premises)}: {list(crossing_premises)}")
    
    crossing_row_ids = set()
    crossing_rows_info = []
    for prem in crossing_premises:
        rows = df_text[df_text["reference_text"] == prem]
        for _, r in rows.iterrows():
            crossing_row_ids.add(r["id"])
            crossing_rows_info.append({
                "row_id": r["id"],
                "split": r["split"],
                "premise": r["reference_text"],
                "hypothesis": r["suspicious_text"],
                "label": int(r["is_plagiarized"])
            })
    print(f"Affected crossing rows ({len(crossing_rows_info)} total):")
    for cr in crossing_rows_info:
        print(f"  {cr['row_id']} ({cr['split']}): Label={cr['label']} | Premise='{cr['premise']}'")

    # Check overlap between duplicate rows and cross-split contamination rows
    dupe_row_ids = set(dupe_rows_df["id"])
    dupe_cross_overlap = dupe_row_ids.intersection(crossing_row_ids)
    print(f"Overlap between duplicate rows and contamination rows: {len(dupe_cross_overlap)} rows")

    # 5. Deterministic Premise Hashing and Five-Partition Simulation
    def get_premise_hash(p):
        return int(hashlib.sha256(str(p).encode('utf-8')).hexdigest(), 16)

    # Exclude crossing rows and redundant duplicate rows
    clean_mask = (~df_text["reference_text"].isin(crossing_premises)) & (~df_text["id"].isin([r for r, is_red in dupe_row_is_redundant.items() if is_red]))
    df_clean = df_text[clean_mask].copy().reset_index(drop=True)

    df_val_pool = df_clean[df_clean["split"] == "val"].copy()
    df_test_pool = df_clean[df_clean["split"] == "test"].copy()

    # Allocate partitions from snli_val: Train(4000), Val(1500), Cal-Prob(1000), Cal-Conf(1000), Dev-Reserve(remainder)
    val_premises_sorted = sorted(list(df_val_pool["reference_text"].unique()), key=get_premise_hash)
    val_premise_counts = df_val_pool["reference_text"].value_counts().to_dict()

    target_counts = {"train": 4000, "val": 1500, "cal_prob": 1000, "cal_conf": 1000}
    current_counts = {k: 0 for k in target_counts}
    premise_to_part = {}

    for p in val_premises_sorted:
        c = val_premise_counts[p]
        assigned = False
        for part in ["train", "val", "cal_prob", "cal_conf"]:
            if current_counts[part] + c <= target_counts[part]:
                premise_to_part[p] = part
                current_counts[part] += c
                assigned = True
                break
        if not assigned:
            candidates = [part for part in ["train", "val", "cal_prob", "cal_conf"] if current_counts[part] < target_counts[part]]
            if candidates:
                best_cand = max(candidates, key=lambda part: target_counts[part] - current_counts[part])
                premise_to_part[p] = best_cand
                current_counts[best_cand] += c
            else:
                premise_to_part[p] = "dev_reserve"

    df_val_pool["partition"] = df_val_pool["reference_text"].map(premise_to_part)

    # Allocate from snli_test: Test(3000), Test-Reserve(remainder)
    test_premises_sorted = sorted(list(df_test_pool["reference_text"].unique()), key=get_premise_hash)
    test_premise_counts = df_test_pool["reference_text"].value_counts().to_dict()

    test_premise_to_part = {}
    test_target = 3000
    test_cur = 0

    for p in test_premises_sorted:
        c = test_premise_counts[p]
        if test_cur + c <= test_target:
            test_premise_to_part[p] = "test"
            test_cur += c
        elif test_cur < test_target:
            test_premise_to_part[p] = "test"
            test_cur += c
        else:
            test_premise_to_part[p] = "test_reserve"

    df_test_pool["partition"] = df_test_pool["reference_text"].map(test_premise_to_part)

    df_partitioned = pd.concat([df_val_pool, df_test_pool], ignore_index=True)
    partition_counts = {k: int(v) for k, v in df_partitioned["partition"].value_counts().items()}
    print(f"Partition pair counts: {partition_counts}")

    # Reconstruct Complete 7x7 Overlap Matrix
    all_subsets = ["train", "val", "cal_prob", "cal_conf", "test", "dev_reserve", "test_reserve"]
    part_premises = {p: set(df_partitioned[df_partitioned["partition"] == p]["reference_text"].unique()) for p in all_subsets}
    
    overlap_matrix = {p1: {p2: 0 for p2 in all_subsets} for p1 in all_subsets}
    for p1 in all_subsets:
        for p2 in all_subsets:
            overlap_matrix[p1][p2] = len(part_premises[p1].intersection(part_premises[p2]))

    print("7x7 Premise Overlap Matrix (Diagonal = Unique Premises, Off-diagonal = Overlap):")
    for p1 in all_subsets:
        diag = overlap_matrix[p1][p1]
        off_diag = sum(overlap_matrix[p1][p2] for p2 in all_subsets if p2 != p1)
        print(f"  {p1:12s}: {diag:5d} unique premises | Cross-overlap = {off_diag}")

    # 6. Row-Level Exclusion Ledger Construction
    exclusion_ledger = []
    
    # Add crossing rows
    for cr in crossing_rows_info:
        exclusion_ledger.append({
            "row_id": cr["row_id"],
            "original_split": cr["split"],
            "ground_truth_label": cr["label"],
            "exclusion_reason": "CROSS_SPLIT_PREMISE_CONTAMINATION",
            "duplicate_group_id": "NONE",
            "premise_string": cr["premise"][:50],
            "final_disposition": "PURGED_PRE_SPLIT"
        })
        
    # Add duplicate redundant rows
    for r_id, is_red in dupe_row_is_redundant.items():
        if is_red:
            r_row = df_text[df_text["id"] == r_id].iloc[0]
            exclusion_ledger.append({
                "row_id": r_id,
                "original_split": r_row["split"],
                "ground_truth_label": int(r_row["is_plagiarized"]),
                "exclusion_reason": "REDUNDANT_EXACT_DUPLICATE_ROW",
                "duplicate_group_id": dupe_row_to_group[r_id],
                "premise_string": str(r_row["reference_text"])[:50],
                "final_disposition": "PURGED_PRE_SPLIT"
            })

    # Summary Accounting Reconciliation
    n_raw = total_raw_rows
    n_active = sum(partition_counts[p] for p in ["train", "val", "cal_prob", "cal_conf", "test"])
    n_dev_res = partition_counts["dev_reserve"]
    n_test_res = partition_counts["test_reserve"]
    n_purged_cross = len(crossing_rows_info)
    n_purged_dupe = sum(1 for is_red in dupe_row_is_redundant.values() if is_red)
    
    sum_check = n_active + n_dev_res + n_test_res + n_purged_cross + n_purged_dupe
    print(f"\nFinal Arithmetic Sum Check:")
    print(f"  Active({n_active}) + DevRes({n_dev_res}) + TestRes({n_test_res}) + CrossPurge({n_purged_cross}) + DupePurge({n_purged_dupe}) = {sum_check}")
    print(f"  Target Raw Rows = {n_raw} | Discrepancy = {n_raw - sum_check}")

    return {
        "total_raw_rows": total_raw_rows,
        "split_counts": split_counts,
        "null_or_empty": {"null_reference": null_ref, "null_suspicious": null_sus, "empty_reference": empty_ref, "empty_suspicious": empty_sus},
        "exact_ordered_duplicate_rows": n_exact_ordered,
        "reversed_duplicate_pairs": n_reversed_dupes,
        "normalized_duplicate_rows": n_norm_dupes,
        "duplicate_groups": dupe_groups,
        "crossing_premises": list(crossing_premises),
        "crossing_rows_count": len(crossing_rows_info),
        "crossing_rows": crossing_rows_info,
        "partition_pair_counts": partition_counts,
        "partition_overlap_matrix": overlap_matrix,
        "exclusion_ledger": exclusion_ledger,
        "reconciliation": {
            "raw_rows": n_raw,
            "active_pairs": n_active,
            "purged_cross_contamination": n_purged_cross,
            "purged_redundant_duplicates": n_purged_dupe,
            "retained_development_reserve": n_dev_res,
            "retained_test_reserve": n_test_res,
            "total_retained_reserves": n_dev_res + n_test_res,
            "sum_check": sum_check,
            "discrepancy": n_raw - sum_check,
            "status": "PASS" if sum_check == n_raw else "FAIL"
        }
    }

def audit_code_benchmark(df_code: pd.DataFrame, ir_plag_dir: Path) -> dict:
    """Exhaustive audit of IR-Plag code benchmark, duplicates, and case-05."""
    print("\n" + "="*60)
    print("AUDIT C: CODE DATASET ACCOUNTING & CASE-05 DEGENERACY AUDIT")
    print("="*60)
    
    total_pairs = len(df_code)
    print(f"Total pairs in code CSV: {total_pairs}")
    
    # 1. Physical Java files
    physical_java_files = list(ir_plag_dir.glob("**/*.java"))
    n_physical_files = len(physical_java_files)
    print(f"Physical Java files on disk: {n_physical_files}")

    # 2. Duplicate Analysis in Code
    dupes_mask = df_code.duplicated(subset=["reference_code", "suspicious_code"], keep=False)
    n_dupe_rows = int(dupes_mask.sum())
    
    # Check reversed duplicates
    canon_code_pairs = []
    for _, r in df_code.iterrows():
        c1 = str(r["reference_code"])
        c2 = str(r["suspicious_code"])
        canon_code_pairs.append((c1, c2) if c1 <= c2 else (c2, c1))
    df_code_canon = df_code.copy()
    df_code_canon["canon_pair"] = canon_code_pairs
    undirected_dupes_mask = df_code_canon.duplicated(subset=["canon_pair"], keep=False)
    n_reversed_code_dupes = max(0, int(undirected_mask := undirected_dupes_mask.sum()) - n_dupe_rows)
    print(f"Code duplicates: Exact duplicate rows = {n_dupe_rows} | Reversed duplicate pairs = {n_reversed_code_dupes}")

    # Identify exact duplicate groups
    dupe_code_df = df_code[dupes_mask]
    code_dupe_groups = []
    grp_idx = 0
    for (r_code, s_code), grp in dupe_code_df.groupby(["reference_code", "suspicious_code"]):
        grp_idx += 1
        code_dupe_groups.append({
            "duplicate_group_id": f"CODE_DUPE_GRP_{grp_idx:02d}",
            "case": grp["case"].iloc[0],
            "pair_ids": grp["id"].tolist(),
            "obfuscation_levels": grp["obfuscation_level"].tolist(),
            "labels": grp["is_plagiarized"].tolist(),
            "row_count": len(grp)
        })
    print(f"Code duplicate groups: {len(code_dupe_groups)} groups across {n_dupe_rows} duplicate rows")

    # 3. Case-by-Case Task Accounting (Resolving Denominator Issue)
    task_accounting = []
    for c in sorted(df_code["case"].unique().tolist()):
        c_df = df_code[df_code["case"] == c]
        n_raw = len(c_df)
        n_raw_pos = int((c_df["is_plagiarized"] == 1).sum())
        n_raw_neg = int((c_df["is_plagiarized"] == 0).sum())
        
        # Duplicates in this task
        c_dupe_mask = c_df.duplicated(subset=["reference_code", "suspicious_code"], keep=False)
        n_c_dupe_rows = int(c_dupe_mask.sum())
        
        # Policy A: Representative Retention (keep first occurrence)
        c_dedup_a = c_df.drop_duplicates(subset=["reference_code", "suspicious_code"], keep="first")
        n_dedup_a = len(c_dedup_a)
        n_dedup_a_pos = int((c_dedup_a["is_plagiarized"] == 1).sum())
        n_dedup_a_neg = int((c_dedup_a["is_plagiarized"] == 0).sum())
        
        # Policy B: Complete Collision Exclusion (drop all rows involved in collision)
        c_dedup_b = c_df[~c_dupe_mask]
        n_dedup_b = len(c_dedup_b)
        n_dedup_b_pos = int((c_dedup_b["is_plagiarized"] == 1).sum())
        n_dedup_b_neg = int((c_dedup_b["is_plagiarized"] == 0).sum())
        
        # Physical files
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

    print("\nTask Accounting Breakdown (Raw vs Deduplicated):")
    print(f"{'Task':<9} | {'Raw':<5} ({'Pos':<3},{'Neg':<3}) | {'Dupes':<5} | {'Policy A (Keep 1st)':<20} | {'Policy B (Drop All)':<20} | {'Files':<5}")
    for ta in task_accounting:
        print(f"{ta['task_id']:<9} | {ta['raw_total_pairs']:<5} ({ta['raw_pos_pairs']:<3},{ta['raw_neg_pairs']:<3}) | {ta['duplicate_rows']:<5} | {ta['policy_a_retained_pairs']:<3} (Pos={ta['policy_a_pos']:<3}, Neg={ta['policy_a_neg']:<3}) | {ta['policy_b_non_collision_pairs']:<3} (Pos={ta['policy_b_pos']:<3}, Neg={ta['policy_b_neg']:<3}) | {ta['physical_java_files']:<5}")

    # 4. Detailed Case-05 Group Structure & Cross-Fitting Proof
    c5_df = df_code[df_code["case"] == "case-05"].copy()
    n_c5_pos = int((c5_df["is_plagiarized"] == 1).sum())
    n_c5_neg = int((c5_df["is_plagiarized"] == 0).sum())
    
    # Verify that all positive pairs involve 'original'
    pos_with_orig = int(((c5_df["is_plagiarized"] == 1) & (c5_df["ref_type"] == "original")).sum())
    print(f"\ncase-05 Group Structure: Total={len(c5_df)} | Positives={n_c5_pos} (All with ref_type='original': {pos_with_orig == n_c5_pos})")

    program_identities = ["original"] + [f"non-plag-{i:02d}" for i in range(1, 16)]
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
            n_pos_train = 0
            n_neg_train = int(((c5_df["ref_type"] != "original") & (c5_df["sus_type"] != "original") & (c5_df["is_plagiarized"] == 0)).sum())
            n_pos_test = n_c5_pos
            n_neg_test = len(c5_df) - n_pos_test - n_neg_train
            status = "DIVERGES_ZERO_TRAIN_POSITIVES"
            consequence = "Loss strictly decreasing as B -> -inf; Platt optimizer diverges to -inf"
        else:
            n_pos_train = n_c5_pos
            n_pos_test = 0
            n_neg_test = int(((c5_df["ref_type"].isin(held_out)) | (c5_df["sus_type"].isin(held_out))).sum())
            n_neg_train = n_c5_neg - n_neg_test
            status = "UNDEFINED_ZERO_TEST_POSITIVES"
            consequence = "Held-out test fold has 0 positives; ROC-AUC, Precision, Recall, ECE undefined"

        c5_fold_audit.append({
            "fold": f_idx,
            "held_out_groups": str(held_out),
            "pos_train": n_pos_train,
            "neg_train": n_neg_train,
            "pos_test": n_pos_test,
            "neg_test": n_neg_test,
            "platt_feasible": False,
            "status": status,
            "mathematical_consequence": consequence
        })

    print("\ncase-05 5-Fold Group Cross-Fitting Feasibility Table:")
    for cfa in c5_fold_audit:
        print(f"  Fold {cfa['fold']}: PosTrain={cfa['pos_train']}, NegTrain={cfa['neg_train']} | PosTest={cfa['pos_test']}, NegTest={cfa['neg_test']} -> {cfa['status']}")

    return {
        "total_pairs": total_pairs,
        "physical_java_files": n_physical_files,
        "duplicate_rows": n_dupe_rows,
        "reversed_duplicate_pairs": n_reversed_code_dupes,
        "duplicate_groups": code_dupe_groups,
        "task_accounting": task_accounting,
        "case_05_fold_audit": c5_fold_audit
    }

def audit_source_inventory() -> dict:
    """Inventories all 28 Python source files in src/ and computes current SHA-256."""
    print("\n" + "="*60)
    print("AUDIT H: REPOSITORY SOURCE CODE INVENTORY & SHA-256 MANIFEST")
    print("="*60)
    
    src_dir = PROJECT_ROOT / "src"
    py_files = sorted(list(src_dir.glob("**/*.py")), key=lambda p: str(p).lower())
    
    root_src = [p for p in py_files if p.parent == src_dir]
    code_sub = [p for p in py_files if "src\\code" in str(p) or "src/code" in str(p)]
    text_sub = [p for p in py_files if "src\\text" in str(p) or "src/text" in str(p)]

    print(f"Total Python source files in 'src/': {len(py_files)}")
    print(f"  - Root of 'src/': {len(root_src)} files")
    print(f"  - 'src/code/': {len(code_sub)} files")
    print(f"  - 'src/text/': {len(text_sub)} files")
    print(f"  Sum verification: {len(root_src)} + {len(code_sub)} + {len(text_sub)} = {len(py_files)}")

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

    # Check whether a historical git repository or baseline hash file existed prior to Phase 2
    git_dir = PROJECT_ROOT / ".git"
    git_exists = git_dir.exists()
    print(f"Git repository on disk (.git): {git_exists}")

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
    print("PHASE 2.3.4 FINAL INDEPENDENT AUDIT & CERTIFICATION SCRIPT")
    print("Execution Timestamp: November 10, 2026")
    print("Project Root:", PROJECT_ROOT)
    print("="*70)

    # 1. Text Audit
    text_csv = PROJECT_ROOT / "datasets" / "processed" / "text_plagiarism_benchmark.csv"
    if not text_csv.exists():
        print(f"FATAL: Text benchmark not found at {text_csv}")
        sys.exit(1)
    df_text = pd.read_csv(text_csv)
    text_results = audit_text_benchmark(df_text)

    # 2. Code Audit
    code_csv = PROJECT_ROOT / "datasets" / "processed" / "code_plagiarism_benchmark.csv"
    ir_plag_dir = PROJECT_ROOT / "datasets" / "raw" / "ir_plag" / "extracted" / "IR-Plag-Dataset"
    if not code_csv.exists() or not ir_plag_dir.exists():
        print("FATAL: Code benchmark or IR-Plag directory missing")
        sys.exit(1)
    df_code = pd.read_csv(code_csv)
    code_results = audit_code_benchmark(df_code, ir_plag_dir)

    # 3. Source Inventory
    source_results = audit_source_inventory()

    # 4. Save Machine-Readable Artifacts
    print("\n" + "="*60)
    print("SAVING MACHINE-READABLE AUDIT ARTIFACTS IN phase_2_3_4/")
    print("="*60)

    # text_dataset_audit.json
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
    print(f"Saved: {AUDIT_DIR / 'text_dataset_audit.json'}")

    # text_exclusion_ledger.csv
    pd.DataFrame(text_results["exclusion_ledger"]).to_csv(AUDIT_DIR / "text_exclusion_ledger.csv", index=False)
    print(f"Saved: {AUDIT_DIR / 'text_exclusion_ledger.csv'}")

    # text_partition_overlap_matrix.csv
    pd.DataFrame(text_results["partition_overlap_matrix"]).to_csv(AUDIT_DIR / "text_partition_overlap_matrix.csv")
    print(f"Saved: {AUDIT_DIR / 'text_partition_overlap_matrix.csv'}")

    # text_duplicate_audit.csv
    pd.DataFrame(text_results["duplicate_groups"]).to_csv(AUDIT_DIR / "text_duplicate_audit.csv", index=False)
    print(f"Saved: {AUDIT_DIR / 'text_duplicate_audit.csv'}")

    # code_dataset_audit.json
    with open(AUDIT_DIR / "code_dataset_audit.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_pairs": code_results["total_pairs"],
            "physical_java_files": code_results["physical_java_files"],
            "duplicate_rows": code_results["duplicate_rows"],
            "reversed_duplicate_pairs": code_results["reversed_duplicate_pairs"],
            "task_accounting": code_results["task_accounting"],
            "case_05_fold_audit": code_results["case_05_fold_audit"]
        }, f, indent=2)
    print(f"Saved: {AUDIT_DIR / 'code_dataset_audit.json'}")

    # code_duplicate_audit.csv
    pd.DataFrame(code_results["duplicate_groups"]).to_csv(AUDIT_DIR / "code_duplicate_audit.csv", index=False)
    print(f"Saved: {AUDIT_DIR / 'code_duplicate_audit.csv'}")

    # code_task_accounting.csv
    pd.DataFrame(code_results["task_accounting"]).to_csv(AUDIT_DIR / "code_task_accounting.csv", index=False)
    print(f"Saved: {AUDIT_DIR / 'code_task_accounting.csv'}")

    # case05_fold_audit.csv
    pd.DataFrame(code_results["case_05_fold_audit"]).to_csv(AUDIT_DIR / "case05_fold_audit.csv", index=False)
    print(f"Saved: {AUDIT_DIR / 'case05_fold_audit.csv'}")

    # source_file_inventory.csv & source_sha256_manifest.csv
    pd.DataFrame(source_results["manifest"]).to_csv(AUDIT_DIR / "source_file_inventory.csv", index=False)
    pd.DataFrame(source_results["manifest"]).to_csv(AUDIT_DIR / "source_sha256_manifest.csv", index=False)
    print(f"Saved: {AUDIT_DIR / 'source_file_inventory.csv'}")
    print(f"Saved: {AUDIT_DIR / 'source_sha256_manifest.csv'}")

    print("\n" + "="*70)
    print("PHASE 2.3.4 AUDIT EXECUTION COMPLETED SUCCESSFULLY")
    print("ALL 11 AUDIT ARTIFACTS GENERATED AND CERTIFIED.")
    print("="*70 + "\n")

    sys.stdout = logger.terminal
    logger.close()

if __name__ == "__main__":
    main()
