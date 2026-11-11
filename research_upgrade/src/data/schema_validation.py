"""
Dataset Schema and Record Integrity Validation Module (Phase 3.2)
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection
"""

from pathlib import Path
from typing import Dict, Any, Union, List
import pandas as pd
import numpy as np

TEXT_REQUIRED_COLUMNS = [
    "id",
    "reference_text",
    "suspicious_text",
    "label",
    "is_plagiarized",
    "split"
]

CODE_REQUIRED_COLUMNS = [
    "id",
    "case",
    "reference_code",
    "suspicious_code",
    "ref_type",
    "sus_type",
    "obfuscation_level",
    "is_plagiarized"
]

CODE_ALLOWED_CASES = {
    "case-01", "case-02", "case-03", "case-04", "case-05", "case-06", "case-07"
}


def validate_text_schema(data: Union[pd.DataFrame, Path, str]) -> Dict[str, Any]:
    """
    Validates text benchmark schema, column existence, data types, missing values,
    label integrity, and ID uniqueness.
    """
    if isinstance(data, (str, Path)):
        p = Path(data)
        if not p.exists():
            return {
                "status": "FAILED",
                "error": f"Text dataset file not found: {p}",
                "total_rows": 0
            }
        df = pd.read_csv(p)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise TypeError(f"Unsupported data type for validation: {type(data)}")

    errors: List[str] = []
    warnings: List[str] = []

    # 1. Column presence check
    missing_cols = [c for c in TEXT_REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required text columns: {missing_cols}")

    total_rows = len(df)
    if total_rows == 0:
        errors.append("Dataset contains 0 rows.")
        return {
            "status": "FAILED",
            "errors": errors,
            "total_rows": 0
        }

    # 2. ID uniqueness
    unique_ids = df["id"].nunique()
    if unique_ids != total_rows:
        errors.append(f"Duplicate IDs detected: {total_rows - unique_ids} duplicates found.")

    # 3. Missing values check
    null_counts = {
        col: int(df[col].isnull().sum())
        for col in TEXT_REQUIRED_COLUMNS if col in df.columns
    }
    for col, count in null_counts.items():
        if count > 0:
            errors.append(f"Column '{col}' has {count} null values.")

    # 4. Empty strings in text fields
    empty_ref = int((df["reference_text"].astype(str).str.strip() == "").sum()) if "reference_text" in df.columns else 0
    empty_sus = int((df["suspicious_text"].astype(str).str.strip() == "").sum()) if "suspicious_text" in df.columns else 0
    if empty_ref > 0:
        errors.append(f"{empty_ref} empty reference texts detected.")
    if empty_sus > 0:
        errors.append(f"{empty_sus} empty suspicious texts detected.")

    # 5. Label validity
    if "is_plagiarized" in df.columns:
        invalid_labels = df[~df["is_plagiarized"].isin([0, 1])]
        if len(invalid_labels) > 0:
            errors.append(f"Found {len(invalid_labels)} non-binary plagiarism labels.")
        label_dist = df["is_plagiarized"].value_counts().to_dict()
    else:
        label_dist = {}

    # 6. Split validity
    if "split" in df.columns:
        split_dist = df["split"].value_counts().to_dict()
        invalid_splits = set(split_dist.keys()) - {"val", "test"}
        if invalid_splits:
            warnings.append(f"Unrecognized splits found: {invalid_splits}")
    else:
        split_dist = {}

    status = "PASSED" if len(errors) == 0 else "FAILED"

    return {
        "status": status,
        "modality": "text",
        "total_rows": total_rows,
        "unique_ids": unique_ids,
        "null_counts": null_counts,
        "empty_reference_texts": empty_ref,
        "empty_suspicious_texts": empty_sus,
        "label_distribution": {int(k): int(v) for k, v in label_dist.items()},
        "split_distribution": {str(k): int(v) for k, v in split_dist.items()},
        "errors": errors,
        "warnings": warnings
    }


def validate_code_schema(data: Union[pd.DataFrame, Path, str]) -> Dict[str, Any]:
    """
    Validates code benchmark schema, case identifiers, data types, missing code,
    label integrity, and duplicate records.
    """
    if isinstance(data, (str, Path)):
        p = Path(data)
        if not p.exists():
            return {
                "status": "FAILED",
                "error": f"Code dataset file not found: {p}",
                "total_rows": 0
            }
        df = pd.read_csv(p)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise TypeError(f"Unsupported data type for validation: {type(data)}")

    errors: List[str] = []
    warnings: List[str] = []

    # 1. Column presence check
    missing_cols = [c for c in CODE_REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required code columns: {missing_cols}")

    total_rows = len(df)
    if total_rows == 0:
        errors.append("Dataset contains 0 rows.")
        return {
            "status": "FAILED",
            "errors": errors,
            "total_rows": 0
        }

    # 2. ID uniqueness
    unique_ids = df["id"].nunique()
    if unique_ids != total_rows:
        errors.append(f"Duplicate IDs detected: {total_rows - unique_ids} duplicates found.")

    # 3. Missing values check (obfuscation_level is expected NaN for non-plagiarized)
    null_counts = {}
    for col in CODE_REQUIRED_COLUMNS:
        if col in df.columns:
            cnt = int(df[col].isnull().sum())
            null_counts[col] = cnt
            if col != "obfuscation_level" and cnt > 0:
                errors.append(f"Column '{col}' has {cnt} unexpected null values.")

    # 4. Case identifier check
    if "case" in df.columns:
        cases_found = set(df["case"].unique())
        invalid_cases = cases_found - CODE_ALLOWED_CASES
        if invalid_cases:
            errors.append(f"Unrecognized cases found: {invalid_cases}")
        case_dist = df["case"].value_counts().to_dict()
    else:
        case_dist = {}

    # 5. Empty code check
    empty_ref = int((df["reference_code"].astype(str).str.strip() == "").sum()) if "reference_code" in df.columns else 0
    empty_sus = int((df["suspicious_code"].astype(str).str.strip() == "").sum()) if "suspicious_code" in df.columns else 0
    if empty_ref > 0:
        errors.append(f"{empty_ref} empty reference code snippets detected.")
    if empty_sus > 0:
        errors.append(f"{empty_sus} empty suspicious code snippets detected.")

    # 6. Label validity
    if "is_plagiarized" in df.columns:
        invalid_labels = df[~df["is_plagiarized"].isin([0, 1])]
        if len(invalid_labels) > 0:
            errors.append(f"Found {len(invalid_labels)} non-binary plagiarism labels.")
        label_dist = df["is_plagiarized"].value_counts().to_dict()
    else:
        label_dist = {}

    # 7. Duplicate row analysis
    if "reference_code" in df.columns and "suspicious_code" in df.columns:
        dupe_mask = df.duplicated(subset=["reference_code", "suspicious_code"], keep=False)
        dupe_rows_count = int(dupe_mask.sum())
    else:
        dupe_rows_count = 0

    status = "PASSED" if len(errors) == 0 else "FAILED"

    return {
        "status": status,
        "modality": "code",
        "total_rows": total_rows,
        "unique_ids": unique_ids,
        "null_counts": null_counts,
        "empty_reference_code": empty_ref,
        "empty_suspicious_code": empty_sus,
        "case_distribution": {str(k): int(v) for k, v in case_dist.items()},
        "label_distribution": {int(k): int(v) for k, v in label_dist.items()},
        "participating_duplicate_rows": dupe_rows_count,
        "errors": errors,
        "warnings": warnings
    }

