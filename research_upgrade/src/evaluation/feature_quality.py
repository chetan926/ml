"""
Feature Matrix Quality and Numerical Integrity Validation Module (Phase 3.2)
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


def validate_feature_matrix_quality(
    df_features: pd.DataFrame,
    expected_feature_cols: List[str],
    modality: str
) -> Dict[str, Any]:
    """
    Exhaustively audits an extracted feature DataFrame for:
    1. Presence of all expected feature columns.
    2. Zero NaN values.
    3. Zero infinite values.
    4. Strict numerical range boundedness in [0.0, 1.0].
    5. Identifier and label column preservation.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # 1. Feature columns existence
    missing_cols = [c for c in expected_feature_cols if c not in df_features.columns]
    if missing_cols:
        errors.append(f"Missing expected {modality} feature columns: {missing_cols}")

    # 2. Metadata columns presence
    required_meta = ["pair_id", "modality", "ground_truth_label", "extraction_status"]
    missing_meta = [c for c in required_meta if c not in df_features.columns]
    if missing_meta:
        errors.append(f"Missing required metadata columns: {missing_meta}")

    total_rows = len(df_features)
    if total_rows == 0:
        errors.append("Feature DataFrame contains 0 rows.")
        return {
            "status": "FAILED",
            "modality": modality,
            "total_rows": 0,
            "errors": errors
        }

    # 3. Numerical checks for each feature column
    feature_stats: Dict[str, Dict[str, float]] = {}
    for col in expected_feature_cols:
        if col not in df_features.columns:
            continue

        series = df_features[col]
        n_nans = int(series.isnull().sum())
        n_infs = int(np.isinf(series).sum())

        if n_nans > 0:
            errors.append(f"Column '{col}' contains {n_nans} NaN values!")
        if n_infs > 0:
            errors.append(f"Column '{col}' contains {n_infs} infinite values!")

        min_val = float(series.min()) if total_rows > 0 else 0.0
        max_val = float(series.max()) if total_rows > 0 else 0.0
        mean_val = float(series.mean()) if total_rows > 0 else 0.0
        std_val = float(series.std()) if total_rows > 1 else 0.0

        if min_val < -1e-6:
            errors.append(f"Column '{col}' has value below 0.0: min={min_val}")
        if max_val > 1.0 + 1e-6:
            errors.append(f"Column '{col}' has value above 1.0: max={max_val}")

        feature_stats[col] = {
            "min": round(min_val, 6),
            "max": round(max_val, 6),
            "mean": round(mean_val, 6),
            "std": round(std_val, 6),
            "nan_count": n_nans,
            "inf_count": n_infs
        }

    status = "PASSED" if len(errors) == 0 else "FAILED"

    return {
        "status": status,
        "modality": modality,
        "total_evaluated_rows": total_rows,
        "feature_count": len(expected_feature_cols),
        "feature_statistics": feature_stats,
        "errors": errors,
        "warnings": warnings
    }

