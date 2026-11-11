"""
Phase 4 Corrected Statistical Testing & Group-Aware Evaluation Module.
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection.

Methodological Corrections Applied:
1. Clustered / Group-Aware Bootstrap: Resamples sampling units at the premise/task group level with replacement.
2. Paired Confidence Intervals: Computes empirical bootstrap percentiles for (Proposed - Baseline) metric differences.
3. Complete 20+ Classification Metrics including confusion matrix counts and curve integrals.
4. Corrects p-value underflow: Numerical zero is replaced with exact floating-point scientific representation (e.g. '< 1e-16').
"""

import math
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    cohen_kappa_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    average_precision_score,
    brier_score_loss,
    log_loss,
)
from scipy.stats import chi2


def format_p_value(p: float, eps: float = 1e-16) -> str:
    """Formats p-values safely to avoid reporting numerical underflow as exact zero."""
    if p is None or math.isnan(p):
        return "N/A"
    if p < eps:
        return f"< {eps:.0e}"
    if p < 0.001:
        return f"{p:.4e}"
    return f"{p:.6f}"


def compute_comprehensive_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Calculates all mandatory classification metrics and confusion matrix elements."""
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    n = len(y_true)
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    f0_5 = float(fbeta_score(y_true, y_pred, beta=0.5, zero_division=0))
    f2 = float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0))

    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    fdr = float(fp / (fp + tp)) if (fp + tp) > 0 else 0.0
    for_rate = float(fn / (fn + tn)) if (fn + tn) > 0 else 0.0
    bal_acc = float((rec + spec) / 2.0)
    mcc = float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0
    kappa = float(cohen_kappa_score(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0
    youdens_j = float(rec + spec - 1.0)
    gmean = float(np.sqrt(rec * spec))
    threat_score = float(tp / (tp + fn + fp)) if (tp + fn + fp) > 0 else 0.0
    dor = float((tp * tn) / (fp * fn)) if (fp * fn) > 0 else (float("inf") if (tp * tn) > 0 else 0.0)

    # Probability curve metrics
    roc_auc = 0.5
    pr_auc = 0.0
    avg_prec = 0.0
    prec_at_98_rec = 0.0

    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=np.float64)
        if len(np.unique(y_true)) > 1:
            try:
                roc_auc = float(roc_auc_score(y_true, y_prob))
                p_curve, r_curve, _ = precision_recall_curve(y_true, y_prob)
                pr_auc = float(auc(r_curve, p_curve))
                avg_prec = float(average_precision_score(y_true, y_prob))

                # Precision @ 98% recall
                valid_idx = np.where(r_curve >= 0.98)[0]
                prec_at_98_rec = float(np.max(p_curve[valid_idx])) if len(valid_idx) > 0 else 0.0
            except Exception:
                pass

    return {
        "total_samples": n,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "accuracy": round(acc, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1_score": round(f1, 6),
        "f0_5_score": round(f0_5, 6),
        "f2_score": round(f2, 6),
        "specificity": round(spec, 6),
        "negative_predictive_value": round(npv, 6),
        "false_positive_rate": round(fpr, 6),
        "false_negative_rate": round(fnr, 6),
        "false_discovery_rate": round(fdr, 6),
        "false_omission_rate": round(for_rate, 6),
        "balanced_accuracy": round(bal_acc, 6),
        "matthews_corrcoef": round(mcc, 6),
        "cohens_kappa": round(kappa, 6),
        "youdens_j": round(youdens_j, 6),
        "geometric_mean": round(gmean, 6),
        "threat_score": round(threat_score, 6),
        "diagnostic_odds_ratio": round(dor, 4) if np.isfinite(dor) else "inf",
        "roc_auc": round(roc_auc, 6),
        "pr_auc": round(pr_auc, 6),
        "average_precision": round(avg_prec, 6),
        "precision_at_98_recall": round(prec_at_98_rec, 6),
    }


def compute_group_aware_bootstrap_ci(
    group_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Performs clustered bootstrap resampling at the group unit level.
    """
    rng = np.random.RandomState(random_state)
    df = pd.DataFrame({
        "group_id": np.asarray(group_ids),
        "y_true": np.asarray(y_true, dtype=np.int32),
        "y_pred": np.asarray(y_pred, dtype=np.int32),
    })

    unique_groups = df["group_id"].unique()
    n_groups = len(unique_groups)

    # Pre-aggregate group indices for speed
    group_indices = df.groupby("group_id").indices

    metrics_samples: Dict[str, List[float]] = {
        "mcc": [],
        "f1_score": [],
        "recall": [],
        "precision": [],
        "accuracy": [],
    }

    for _ in range(n_resamples):
        sampled_groups = rng.choice(unique_groups, size=n_groups, replace=True)
        sample_row_idx = []
        for g in sampled_groups:
            sample_row_idx.extend(group_indices[g])

        sub_df = df.iloc[sample_row_idx]
        y_t = sub_df["y_true"].values
        y_p = sub_df["y_pred"].values

        if len(np.unique(y_t)) < 2:
            continue

        acc = accuracy_score(y_t, y_p)
        rec = recall_score(y_t, y_p, zero_division=0)
        prec = precision_score(y_t, y_p, zero_division=0)
        f1 = f1_score(y_t, y_p, zero_division=0)
        mcc = matthews_corrcoef(y_t, y_p) if len(np.unique(y_p)) > 1 else 0.0

        metrics_samples["accuracy"].append(float(acc))
        metrics_samples["recall"].append(float(rec))
        metrics_samples["precision"].append(float(prec))
        metrics_samples["f1_score"].append(float(f1))
        metrics_samples["mcc"].append(float(mcc))

    ci_results: Dict[str, Dict[str, float]] = {}
    lower_p = (alpha / 2.0) * 100.0
    upper_p = (1.0 - alpha / 2.0) * 100.0

    for m_name, vals in metrics_samples.items():
        arr = np.asarray(vals)
        ci_results[m_name] = {
            "mean": round(float(np.mean(arr)), 4),
            "std_error": round(float(np.std(arr, ddof=1)), 4),
            "ci_lower": round(float(np.percentile(arr, lower_p)), 4),
            "ci_upper": round(float(np.percentile(arr, upper_p)), 4),
        }

    return ci_results


def compute_paired_difference_bootstrap_ci(
    group_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred_proposed: np.ndarray,
    y_pred_baseline: np.ndarray,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Computes clustered bootstrap confidence intervals for the paired metric difference:
    Delta = Proposed - Baseline.
    """
    rng = np.random.RandomState(random_state)
    df = pd.DataFrame({
        "group_id": np.asarray(group_ids),
        "y_true": np.asarray(y_true, dtype=np.int32),
        "y_prop": np.asarray(y_pred_proposed, dtype=np.int32),
        "y_base": np.asarray(y_pred_baseline, dtype=np.int32),
    })

    unique_groups = df["group_id"].unique()
    n_groups = len(unique_groups)
    group_indices = df.groupby("group_id").indices

    delta_samples: Dict[str, List[float]] = {
        "delta_mcc": [],
        "delta_f1": [],
        "delta_accuracy": [],
        "delta_recall": [],
        "delta_precision": [],
    }

    for _ in range(n_resamples):
        sampled_groups = rng.choice(unique_groups, size=n_groups, replace=True)
        idx_list = []
        for g in sampled_groups:
            idx_list.extend(group_indices[g])

        sub = df.iloc[idx_list]
        yt = sub["y_true"].values
        yp = sub["y_prop"].values
        yb = sub["y_base"].values

        if len(np.unique(yt)) < 2:
            continue

        d_acc = accuracy_score(yt, yp) - accuracy_score(yt, yb)
        d_f1 = f1_score(yt, yp, zero_division=0) - f1_score(yt, yb, zero_division=0)
        d_rec = recall_score(yt, yp, zero_division=0) - recall_score(yt, yb, zero_division=0)
        d_prec = precision_score(yt, yp, zero_division=0) - precision_score(yt, yb, zero_division=0)

        mcc_p = matthews_corrcoef(yt, yp) if len(np.unique(yp)) > 1 else 0.0
        mcc_b = matthews_corrcoef(yt, yb) if len(np.unique(yb)) > 1 else 0.0
        d_mcc = mcc_p - mcc_b

        delta_samples["delta_accuracy"].append(float(d_acc))
        delta_samples["delta_f1"].append(float(d_f1))
        delta_samples["delta_recall"].append(float(d_rec))
        delta_samples["delta_precision"].append(float(d_prec))
        delta_samples["delta_mcc"].append(float(d_mcc))

    res: Dict[str, Dict[str, float]] = {}
    lower_p = (alpha / 2.0) * 100.0
    upper_p = (1.0 - alpha / 2.0) * 100.0

    for d_name, vals in delta_samples.items():
        arr = np.asarray(vals)
        res[d_name] = {
            "mean_difference": round(float(np.mean(arr)), 4),
            "std_error": round(float(np.std(arr, ddof=1)), 4),
            "ci_lower": round(float(np.percentile(arr, lower_p)), 4),
            "ci_upper": round(float(np.percentile(arr, upper_p)), 4),
            "is_statistically_significant": bool(
                np.percentile(arr, lower_p) > 0.0 or np.percentile(arr, upper_p) < 0.0
            ),
        }
    return res


def compute_mcnemar_significance(
    y_true: np.ndarray,
    y_pred_proposed: np.ndarray,
    y_pred_baseline: np.ndarray,
) -> Dict[str, Any]:
    """
    Computes McNemar's paired chi-squared test with Edwards continuity correction
    and formats underflowed p-values safely.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    yp = np.asarray(y_pred_proposed, dtype=np.int32)
    yb = np.asarray(y_pred_baseline, dtype=np.int32)

    corr_prop = yp == y_true
    corr_base = yb == y_true

    b = int(np.sum(corr_prop & ~corr_base))  # Prop correct, Base incorrect
    c = int(np.sum(~corr_prop & corr_base))  # Base correct, Prop incorrect

    # Edwards continuity correction
    chi2_stat = float((abs(b - c) - 1.0) ** 2 / (b + c)) if (b + c) > 0 else 0.0
    raw_p = float(chi2.sf(chi2_stat, df=1)) if chi2_stat > 0 else 1.0

    odds_ratio = float(b / c) if c > 0 else (float("inf") if b > 0 else 1.0)

    return {
        "contingency_table": {
            "both_correct": int(np.sum(corr_prop & corr_base)),
            "proposed_only_correct (b)": b,
            "baseline_only_correct (c)": c,
            "both_incorrect": int(np.sum(~corr_prop & ~corr_base)),
        },
        "chi2_statistic": round(chi2_stat, 4),
        "raw_p_value": raw_p,
        "formatted_p_value": format_p_value(raw_p),
        "is_significant_at_05": bool(raw_p < 0.05),
        "is_significant_at_001": bool(raw_p < 0.001),
        "odds_ratio": round(odds_ratio, 4) if np.isfinite(odds_ratio) else "inf",
    }

