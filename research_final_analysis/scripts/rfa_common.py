"""research_final_analysis shared library: loaders, metrics, io helpers.

Every number emitted by this package is recomputed from repository bytes:
raw datasets, phase-3 feature parquets, or stored phase-5/6 prediction files.
No value is copied by hand from narrative documents.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

ROOT = Path("/home/user/ml")
sys.path.insert(0, str(ROOT))

RFA = ROOT / "research_final_analysis"
FEATS_DIR = ROOT / "research_upgrade" / "artifacts" / "phase_3" / "phase_3_3" / "features"
P5 = ROOT / "research_upgrade" / "phase_5_q1_upgrade" / "artifacts"
P6 = ROOT / "research_upgrade" / "phase_6_q1_hardening" / "artifacts"

SEED = 42


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def wj(d: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(d, open(path, "w"), indent=2, cls=NumpyEncoder)


def wc(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def full_metrics(y_true, y_pred, scores: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Complete classification metric set; NR when a score vector is absent."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out: Dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "sensitivity": round(float(tp / (tp + fn) if (tp + fn) else 0), 4),
        "specificity": round(float(tn / (tn + fp) if (tn + fp) else 0), 4),
        "fpr": round(float(fp / (fp + tn) if (fp + tn) else 0), 4),
        "fnr": round(float(fn / (fn + tp) if (fn + tp) else 0), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "mcc": round(float(matthews_corrcoef(y_true, y_pred)), 4),
        "confusion": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
        "n": int(len(y_true)), "positives": int(y_true.sum()),
    }
    if scores is not None and len(np.unique(y_true)) > 1:
        out["roc_auc"] = round(float(roc_auc_score(y_true, scores)), 4)
        out["pr_auc"] = round(float(average_precision_score(y_true, scores)), 4)
    else:
        out["roc_auc"] = "NR (score vector unavailable)"
        out["pr_auc"] = "NR (score vector unavailable)"
    return out


def wilson(k: int, n: int, z: float = 1.959963984540054):
    import math
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round((c - r) / d, 4), round((c + r) / d, 4)


def group_bootstrap_ci(groups, y_true, y_pred, n_resamples=1000, seed=SEED):
    from research_upgrade.phase_4_1_corrections.src.evaluation.statistical_testing import (
        compute_group_aware_bootstrap_ci,
    )
    return compute_group_aware_bootstrap_ci(
        np.asarray(groups), np.asarray(y_true).astype(int),
        np.asarray(y_pred).astype(int), n_resamples=n_resamples, random_state=seed)


def load_text_raw():
    from research_upgrade.src.data.text_loader import TextBenchmarkDataLoader
    loader = TextBenchmarkDataLoader()
    return {s: loader.get_partition(s) for s in ("train", "val", "test")}


def load_code_raw():
    from research_upgrade.src.data.code_loader import CodeBenchmarkDataLoader
    loader = CodeBenchmarkDataLoader()
    return {"policy_a": loader.get_dataset("policy_a"),
            "policy_b": loader.get_dataset("policy_b")}


def load_features():
    return {
        "text_train": pd.read_parquet(FEATS_DIR / "text_feats_train.parquet"),
        "text_val": pd.read_parquet(FEATS_DIR / "text_feats_val.parquet"),
        "text_test": pd.read_parquet(FEATS_DIR / "text_feats_test.parquet"),
        "code_a": pd.read_parquet(FEATS_DIR / "code_feats_policy_a.parquet"),
        "code_b": pd.read_parquet(FEATS_DIR / "code_feats_policy_b.parquet"),
    }


def load_phase5_predictions():
    return {
        "text": pd.read_csv(P5 / "PHASE_5_PREDICTIONS_TEXT_TEST.csv"),
        "code": pd.read_csv(P5 / "PHASE_5_PREDICTIONS_CODE_OOF.csv"),
    }
