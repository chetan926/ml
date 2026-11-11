"""
Reproducible Text Benchmark Data Loader (Phase 3.2)
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection
"""

from pathlib import Path
import hashlib
from typing import Dict, Any, List, Optional
import pandas as pd

from research_upgrade.src.config import (
    TEXT_DATASET_PATH,
    PHASE2_TEXT_EXCLUSION_PATH,
    N_RAW_TEXT_ROWS,
    N_TOTAL_CLEAN_ROWS,
    N_TOTAL_PURGED_ROWS
)
from research_upgrade.src.data.schema_validation import validate_text_schema
from research_upgrade.src.data.partition_validation import validate_text_partitions


class TextBenchmarkDataLoader:
    """
    Reproducible loader for the SNLI/Pereira-derived text plagiarism benchmark.
    Enforces deterministic premise-group allocations and strict zero cross-split leakage.
    """

    def __init__(self, csv_path: Optional[Path] = None):
        self.csv_path = Path(csv_path) if csv_path else TEXT_DATASET_PATH
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Text benchmark CSV not found at: {self.csv_path}")

        self._raw_df: Optional[pd.DataFrame] = None
        self._clean_df: Optional[pd.DataFrame] = None
        self._purged_df: Optional[pd.DataFrame] = None
        self._partitions: Dict[str, pd.DataFrame] = {}
        self._is_loaded = False

        self._load_and_partition()

    @staticmethod
    def _get_premise_hash(premise: str) -> int:
        """Deterministic integer hash for premise string."""
        return int(hashlib.sha256(str(premise).encode("utf-8")).hexdigest(), 16)

    def _load_and_partition(self) -> None:
        """Loads raw CSV, purges cross-split and redundant duplicate rows, and partitions."""
        self._raw_df = pd.read_csv(self.csv_path)
        schema_res = validate_text_schema(self._raw_df)
        if schema_res["status"] != "PASSED":
            raise ValueError(f"Schema validation failed on raw text dataset: {schema_res['errors']}")

        assert len(self._raw_df) == N_RAW_TEXT_ROWS, (
            f"Expected {N_RAW_TEXT_ROWS} rows, found {len(self._raw_df)}"
        )

        # 1. Identify rows to purge
        # A: Cross-split premise contamination
        val_prems = set(self._raw_df[self._raw_df["split"] == "val"]["reference_text"].unique())
        test_prems = set(self._raw_df[self._raw_df["split"] == "test"]["reference_text"].unique())
        crossing_prems = val_prems.intersection(test_prems)

        crossing_mask = self._raw_df["reference_text"].isin(crossing_prems)
        crossing_ids = set(self._raw_df[crossing_mask]["id"].tolist())

        # B: Redundant ordered duplicate rows (Policy A: drop subsequent occurrences)
        dupe_mask = self._raw_df.duplicated(subset=["reference_text", "suspicious_text"], keep="first")
        redundant_dupe_ids = set(self._raw_df[dupe_mask]["id"].tolist())

        purged_ids = crossing_ids.union(redundant_dupe_ids)
        assert len(purged_ids) == N_TOTAL_PURGED_ROWS, (
            f"Expected {N_TOTAL_PURGED_ROWS} purged IDs, found {len(purged_ids)}"
        )

        self._purged_df = self._raw_df[self._raw_df["id"].isin(purged_ids)].copy().reset_index(drop=True)
        self._clean_df = self._raw_df[~self._raw_df["id"].isin(purged_ids)].copy().reset_index(drop=True)

        assert len(self._clean_df) == N_TOTAL_CLEAN_ROWS, (
            f"Expected {N_TOTAL_CLEAN_ROWS} clean rows, found {len(self._clean_df)}"
        )

        # 2. Split into Development Pool (from 'val') and Test Pool (from 'test')
        df_val_pool = self._clean_df[self._clean_df["split"] == "val"].copy()
        df_test_pool = self._clean_df[self._clean_df["split"] == "test"].copy()

        assert len(df_val_pool) == 9837, f"Expected 9837 val pool rows, got {len(df_val_pool)}"
        assert len(df_test_pool) == 9821, f"Expected 9821 test pool rows, got {len(df_test_pool)}"

        # 3. Deterministic Non-Overshooting Allocator
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

        # Development Partitions
        val_counts = df_val_pool["reference_text"].value_counts().to_dict()
        sorted_val_prems = sorted(val_counts.keys(), key=self._get_premise_hash)

        train_prems, train_cnt, rem_1 = allocate_without_overshoot(sorted_val_prems, val_counts, 4000)
        val_prems, val_cnt, rem_2 = allocate_without_overshoot(rem_1, val_counts, 1500)
        cal_prob_prems, cal_prob_cnt, rem_3 = allocate_without_overshoot(rem_2, val_counts, 1000)
        cal_conf_prems, cal_conf_cnt, dev_res_prems = allocate_without_overshoot(rem_3, val_counts, 1000)

        # Test Partitions
        test_counts = df_test_pool["reference_text"].value_counts().to_dict()
        sorted_test_prems = sorted(test_counts.keys(), key=self._get_premise_hash)

        test_prems, test_cnt, test_res_prems = allocate_without_overshoot(sorted_test_prems, test_counts, 3000)

        # Construct Slice DataFrames with explicit partition assignment
        premise_to_partition = {}
        for p in train_prems: premise_to_partition[p] = "train"
        for p in val_prems: premise_to_partition[p] = "val"
        for p in cal_prob_prems: premise_to_partition[p] = "cal_prob"
        for p in cal_conf_prems: premise_to_partition[p] = "cal_conf"
        for p in dev_res_prems: premise_to_partition[p] = "dev_reserve"
        for p in test_prems: premise_to_partition[p] = "test"
        for p in test_res_prems: premise_to_partition[p] = "test_reserve"

        self._clean_df["partition"] = self._clean_df["reference_text"].map(premise_to_partition)

        for part_name in ["train", "val", "cal_prob", "cal_conf", "dev_reserve", "test", "test_reserve"]:
            df_part = self._clean_df[self._clean_df["partition"] == part_name].copy().reset_index(drop=True)
            self._partitions[part_name] = df_part

        # Run automated anti-leakage and size validation
        part_res = validate_text_partitions(self._partitions)
        if part_res["status"] != "PASSED":
            raise RuntimeError(f"Text partition validation failed: {part_res['errors']}")

        self._is_loaded = True

    def get_partition(self, name: str) -> pd.DataFrame:
        """Returns the specified partition DataFrame."""
        if not self._is_loaded:
            self._load_and_partition()
        if name not in self._partitions:
            raise KeyError(
                f"Unknown partition '{name}'. Valid partitions: {list(self._partitions.keys())}"
            )
        return self._partitions[name].copy()

    def get_all_partitions(self) -> Dict[str, pd.DataFrame]:
        """Returns all 7 partitions as a dictionary."""
        if not self._is_loaded:
            self._load_and_partition()
        return {k: v.copy() for k, v in self._partitions.items()}

    def get_purged_records(self) -> pd.DataFrame:
        """Returns the 8 purged records."""
        if not self._is_loaded:
            self._load_and_partition()
        return self._purged_df.copy()

    def get_premise_groups(self, partition_name: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Returns candidate pairs grouped by unique reference premise text for group conformal analysis.
        Maps reference_text -> list of pair record dicts.
        """
        df_part = self.get_partition(partition_name)
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for _, row in df_part.iterrows():
            premise = row["reference_text"]
            if premise not in groups:
                groups[premise] = []
            groups[premise].append({
                "id": row["id"],
                "suspicious_text": row["suspicious_text"],
                "is_plagiarized": int(row["is_plagiarized"]),
                "label": row["label"]
            })
        return groups

    def get_metadata(self) -> Dict[str, Any]:
        """Returns partition sizes, class distributions, and premise counts."""
        if not self._is_loaded:
            self._load_and_partition()
        meta = {}
        for name, df_p in self._partitions.items():
            pos = int((df_p["is_plagiarized"] == 1).sum())
            neg = int((df_p["is_plagiarized"] == 0).sum())
            meta[name] = {
                "total_pairs": len(df_p),
                "positive_pairs": pos,
                "negative_pairs": neg,
                "positive_prevalence_pct": round((pos / len(df_p)) * 100, 2),
                "unique_premises": int(df_p["reference_text"].nunique())
            }
        return meta

