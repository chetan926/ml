"""
Reproducible IR-Plag Code Benchmark Data Loader (Phase 3.2)
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection
"""

from pathlib import Path
import hashlib
from typing import Dict, Any, List, Optional
import pandas as pd

from research_upgrade.src.config import (
    CODE_DATASET_PATH,
    IR_PLAG_DIR,
    N_RAW_CODE_PAIRS,
    N_CODE_DUPLICATE_ROWS,
    N_CODE_DUPLICATE_GROUPS,
    N_CODE_POLICY_A_RETAINED,
    N_CODE_POLICY_B_RETAINED,
    CASE_05_PHYSICAL_JAVA_FILES,
    N_IR_PLAG_PHYSICAL_JAVA_FILES
)
from research_upgrade.src.data.schema_validation import validate_code_schema
from research_upgrade.src.data.partition_validation import validate_code_partitions


class CodeBenchmarkDataLoader:
    """
    Reproducible loader for the IR-Plag Java source code benchmark.
    Preserves exact Policy A (720 pairs) and Policy B (709 pairs) accounting,
    verifies physical Java file counts, and generates the audited Case-05 cross-fitting folds.
    """

    def __init__(
        self,
        csv_path: Optional[Path] = None,
        ir_plag_dir: Optional[Path] = None
    ):
        self.csv_path = Path(csv_path) if csv_path else CODE_DATASET_PATH
        self.ir_plag_dir = Path(ir_plag_dir) if ir_plag_dir else IR_PLAG_DIR

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Code benchmark CSV not found at: {self.csv_path}")

        self._raw_df: Optional[pd.DataFrame] = None
        self._policy_a_df: Optional[pd.DataFrame] = None
        self._policy_b_df: Optional[pd.DataFrame] = None
        self._dupe_groups: List[Dict[str, Any]] = []
        self._physical_files: List[Dict[str, Any]] = []
        self._is_loaded = False

        self._load_and_process()

    def _load_and_process(self) -> None:
        """Loads code CSV, performs duplicate accounting, and parses physical files."""
        self._raw_df = pd.read_csv(self.csv_path)
        schema_res = validate_code_schema(self._raw_df)
        if schema_res["status"] != "PASSED":
            raise ValueError(f"Schema validation failed on raw code dataset: {schema_res['errors']}")

        assert len(self._raw_df) == N_RAW_CODE_PAIRS, (
            f"Expected {N_RAW_CODE_PAIRS} raw pairs, got {len(self._raw_df)}"
        )

        # 1. Duplicate Accounting
        dupes_mask = self._raw_df.duplicated(
            subset=["reference_code", "suspicious_code"], keep=False
        )
        participating_dupes = int(dupes_mask.sum())
        assert participating_dupes == N_CODE_DUPLICATE_ROWS, (
            f"Expected {N_CODE_DUPLICATE_ROWS} participating duplicate rows, got {participating_dupes}"
        )

        # Group duplicate rows
        dupe_df = self._raw_df[dupes_mask]
        grp_idx = 0
        self._dupe_groups = []
        for (r_code, s_code), grp in dupe_df.groupby(["reference_code", "suspicious_code"]):
            grp_idx += 1
            self._dupe_groups.append({
                "duplicate_group_id": f"CODE_DUPE_GRP_{grp_idx:02d}",
                "case": grp["case"].iloc[0],
                "pair_ids": grp["id"].tolist(),
                "obfuscation_levels": grp["obfuscation_level"].tolist(),
                "labels": grp["is_plagiarized"].tolist(),
                "row_count": len(grp)
            })

        assert len(self._dupe_groups) == N_CODE_DUPLICATE_GROUPS, (
            f"Expected {N_CODE_DUPLICATE_GROUPS} duplicate groups, got {len(self._dupe_groups)}"
        )

        # Policy A: Drop subsequent occurrences (keep first) -> 720 pairs
        policy_a_mask = ~self._raw_df.duplicated(
            subset=["reference_code", "suspicious_code"], keep="first"
        )
        self._policy_a_df = self._raw_df[policy_a_mask].copy().reset_index(drop=True)
        assert len(self._policy_a_df) == N_CODE_POLICY_A_RETAINED, (
            f"Expected {N_CODE_POLICY_A_RETAINED} pairs under Policy A, got {len(self._policy_a_df)}"
        )

        # Policy B: Drop entire collision groups (keep neither) -> 709 pairs
        policy_b_mask = ~dupes_mask
        self._policy_b_df = self._raw_df[policy_b_mask].copy().reset_index(drop=True)
        assert len(self._policy_b_df) == N_CODE_POLICY_B_RETAINED, (
            f"Expected {N_CODE_POLICY_B_RETAINED} pairs under Policy B, got {len(self._policy_b_df)}"
        )

        # Validate partitions
        val_res = validate_code_partitions(self._raw_df, self._policy_a_df, self._policy_b_df)
        if val_res["status"] != "PASSED":
            raise RuntimeError(f"Code partition validation failed: {val_res['errors']}")

        # 2. Inventory physical Java files if IR-Plag directory exists
        if self.ir_plag_dir.exists():
            java_files = sorted(list(self.ir_plag_dir.glob("**/*.java")), key=lambda p: str(p).lower())
            self._physical_files = []
            for jf in java_files:
                rel = jf.relative_to(self.ir_plag_dir).as_posix()
                parts = rel.split("/")
                task_id = parts[0]
                category = parts[1] if len(parts) > 1 else "root"
                lvl = parts[2] if len(parts) > 2 and parts[2].startswith("L") else "NONE"
                self._physical_files.append({
                    "relative_path": rel,
                    "task_id": task_id,
                    "category": category,
                    "obfuscation_level": lvl,
                    "file_size_bytes": jf.stat().st_size
                })
            assert len(self._physical_files) == N_IR_PLAG_PHYSICAL_JAVA_FILES, (
                f"Expected {N_IR_PLAG_PHYSICAL_JAVA_FILES} Java files, found {len(self._physical_files)}"
            )

        self._is_loaded = True

    def get_dataset(self, policy: str = "policy_a") -> pd.DataFrame:
        """
        Returns dataset under specified duplicate handling policy.
        Options: 'policy_a' (720 pairs), 'policy_b' (709 pairs), 'raw' (733 pairs).
        """
        if not self._is_loaded:
            self._load_and_process()
        p = policy.lower()
        if p == "policy_a":
            return self._policy_a_df.copy()
        elif p == "policy_b":
            return self._policy_b_df.copy()
        elif p == "raw":
            return self._raw_df.copy()
        else:
            raise ValueError(f"Unknown policy '{policy}'. Choose 'policy_a', 'policy_b', or 'raw'.")

    def get_case(self, case_name: str, policy: str = "policy_a") -> pd.DataFrame:
        """Returns rows for a specific case (e.g. 'case-05') under the selected policy."""
        df = self.get_dataset(policy=policy)
        case_df = df[df["case"] == case_name].copy().reset_index(drop=True)
        if len(case_df) == 0:
            raise KeyError(f"Case '{case_name}' not found in code dataset.")
        return case_df

    def get_case_05_folds(self) -> List[Dict[str, Any]]:
        """
        Generates the audited 5-fold group-disjoint cross-validation split for Case-05.
        Base program identities: prog_original, prog_np_01 to prog_np_15.
        Fold 1: prog_original held out -> Train=(+0, -33), Test=(+53, -21).
        Folds 2-5: prog_original in train -> Train=(+53, -36..38), Test=(+0, -16..18).
        """
        c5_df = self.get_case("case-05", policy="raw")
        assert len(c5_df) == 107, f"Expected 107 pairs in Case-05, got {len(c5_df)}"

        def get_ref_base(r):
            if r["ref_type"] == "original":
                return "prog_original"
            return r["ref_type"].replace("non-plag-", "prog_np_")

        def get_sus_base(r, i):
            if r["sus_type"] == "plagiarized":
                return "prog_original"
            return r["sus_type"].replace("non-plag-", "prog_np_")

        records = []
        for i, r in c5_df.iterrows():
            records.append({
                "id": r["id"],
                "ref_base": get_ref_base(r),
                "sus_base": get_sus_base(r, i),
                "label": int(r["is_plagiarized"])
            })
        ep_df = pd.DataFrame(records)

        folds_spec = {
            1: ["prog_original", "prog_np_01", "prog_np_02"],
            2: ["prog_np_03", "prog_np_04", "prog_np_05"],
            3: ["prog_np_06", "prog_np_07", "prog_np_08"],
            4: ["prog_np_09", "prog_np_10", "prog_np_11"],
            5: ["prog_np_12", "prog_np_13", "prog_np_14", "prog_np_15"]
        }

        fold_results = []
        for f_idx, held_out in folds_spec.items():
            held_set = set(held_out)
            train_mask = (~ep_df["ref_base"].isin(held_set)) & (~ep_df["sus_base"].isin(held_set))
            train_p = ep_df[train_mask]
            test_p = ep_df[~train_mask]

            pos_train = int((train_p["label"] == 1).sum())
            neg_train = int((train_p["label"] == 0).sum())
            pos_test = int((test_p["label"] == 1).sum())
            neg_test = int((test_p["label"] == 0).sum())

            fold_results.append({
                "fold_index": f_idx,
                "held_out_identities": held_out,
                "train_pair_ids": train_p["id"].tolist(),
                "test_pair_ids": test_p["id"].tolist(),
                "pos_train_count": pos_train,
                "neg_train_count": neg_train,
                "pos_test_count": pos_test,
                "neg_test_count": neg_test,
                "platt_scaling_feasible_on_train": (pos_train > 0 and neg_train > 0)
            })

        return fold_results

    def get_physical_files(self) -> List[Dict[str, Any]]:
        """Returns inventory records of all 467 physical Java files."""
        if not self._is_loaded:
            self._load_and_process()
        return list(self._physical_files)

    def get_metadata(self) -> Dict[str, Any]:
        """Returns summary metadata for IR-Plag code benchmark."""
        if not self._is_loaded:
            self._load_and_process()
        task_meta = {}
        for c, grp in self._raw_df.groupby("case"):
            pos = int((grp["is_plagiarized"] == 1).sum())
            neg = int((grp["is_plagiarized"] == 0).sum())
            task_meta[c] = {
                "raw_pairs": len(grp),
                "positive_pairs": pos,
                "negative_pairs": neg,
                "obfuscated_files": int(grp["obfuscation_level"].dropna().count())
            }
        return {
            "total_raw_pairs": len(self._raw_df),
            "policy_a_pairs": len(self._policy_a_df),
            "policy_b_pairs": len(self._policy_b_df),
            "duplicate_groups": len(self._dupe_groups),
            "physical_java_files": len(self._physical_files),
            "task_accounting": task_meta
        }

