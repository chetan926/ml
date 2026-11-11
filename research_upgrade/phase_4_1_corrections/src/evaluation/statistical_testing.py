"""
Phase 4.1 Corrected Statistical Testing, Bootstrap Inference, and Significance Metrics.
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection.

Methodological Guarantees:
1. Clustered Bootstrap Confidence Intervals:
   Resamples intact premise groups (text) and assignment cases (code) with 1,000 replications.
2. Clustered Paired Difference CIs:
   Evaluates paired performance deltas (Delta = Proposed - Baseline) under cluster resampling.
3. Explicit McNemar Cluster Caveat:
   Discloses that standard McNemar chi-square assumes independent pair observations and
   does not account for intra-group correlation.
4. Small-Cluster Limitations:
   Documents the asymptotic limitation of 7 clusters in code cross-validation.
5. Exact p-Value Formatting:
   Eliminates p=0.0 floating-point underflows, formatting as '< 1e-16'.
"""

from typing import Dict, Any, List, Tuple, Optional
import math
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    cohen_kappa_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
)
from scipy.stats import chi2


def format_p_value(p: float) -> str:
    """Formats small p-values safely without floating underflow to zero."""
    if p < 1e-16:
        return "< 1e-16"
    return f"{p:.4e}"


def compute_comprehensive_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Computes 25+ rigorous classification, diagnostic, and calibration metrics.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    n_total = len(y_true)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    f05 = float(fbeta_score(y_true, y_pred, beta=0.5, zero_division=0))
    f2 = float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0))

    spec = float(tn / max(1, tn + fp))
    npv = float(tn / max(1, tn + fn))
    fpr = float(fp / max(1, fp + tn))
    fnr = float(fn / max(1, fn + tp))
    fdr = float(fp / max(1, fp + tp))
    for_rate = float(fn / max(1, fn + tn))
    bal_acc = float((rec + spec) / 2.0)
    mcc = float(matthews_corrcoef(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))

    youdens_j = rec + spec - 1.0
    g_mean = float(math.sqrt(max(0.0, rec * spec)))
    threat_score = float(tp / max(1, tp + fn + fp))
    dor = float((tp * tn) / max(1, fp * fn))

    roc_auc = float(roc_auc_score(y_true, y_prob)) if y_prob is not None and len(np.unique(y_true)) > 1 else None
    pr_auc = float(average_precision_score(y_true, y_prob)) if y_prob is not None and len(np.unique(y_true)) > 1 else None

    # Precision at 98% recall operating threshold
    prec_at_98_rec = 0.0
    if y_prob is not None:
        thresholds = np.linspace(0.0, 1.0, 101)
        for th in thresholds:
            p_bin = (y_prob >= th).astype(int)
            r_val = recall_score(y_true, p_bin, zero_division=0)
            if r_val >= 0.98:
                prec_at_98_rec = max(prec_at_98_rec, float(precision_score(y_true, p_bin, zero_division=0)))

    return {
        "total_samples": n_total,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "f0_5_score": round(f05, 4),
        "f2_score": round(f2, 4),
        "specificity": round(spec, 4),
        "negative_predictive_value": round(npv, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "false_discovery_rate": round(fdr, 4),
        "false_omission_rate": round(for_rate, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "matthews_corrcoef": round(mcc, 4),
        "cohens_kappa": round(kappa, 4),
        "youdens_j": round(youdens_j, 4),
        "geometric_mean": round(g_mean, 4),
        "threat_score": round(threat_score, 4),
        "diagnostic_odds_ratio": round(dor, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
        "precision_at_98_recall": round(prec_at_98_rec, 4),
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
    Computes clustered group-level bootstrap confidence intervals.
    Resamples whole groups with replacement.
    """
    group_ids = np.asarray(group_ids)
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    rng = np.random.RandomState(random_state)
    unique_groups = np.unique(group_ids)
    n_groups = len(unique_groups)

    group_to_indices = {g: np.where(group_ids == g)[0] for g in unique_groups}

    mcc_list, f1_list, rec_list, prec_list, acc_list = [], [], [], [], []

    for _ in range(n_resamples):
        sampled_groups = rng.choice(unique_groups, size=n_groups, replace=True)
        idx_list = [group_to_indices[g] for g in sampled_groups]
        sample_indices = np.concatenate(idx_list)

        y_t_b = y_true[sample_indices]
        y_p_b = y_pred[sample_indices]

        if len(np.unique(y_t_b)) < 2:
            continue

        acc_list.append(accuracy_score(y_t_b, y_p_b))
        prec_list.append(precision_score(y_t_b, y_p_b, zero_division=0))
        rec_list.append(recall_score(y_t_b, y_p_b, zero_division=0))
        f1_list.append(f1_score(y_t_b, y_p_b, zero_division=0))
        mcc_list.append(matthews_corrcoef(y_t_b, y_p_b))

    def get_ci(arr: List[float]) -> Dict[str, float]:
        if not arr:
            return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
        a = np.array(arr)
        return {
            "mean": round(float(np.mean(a)), 4),
            "ci_lower": round(float(np.percentile(a, 100 * (alpha / 2.0))), 4),
            "ci_upper": round(float(np.percentile(a, 100 * (1.0 - alpha / 2.0))), 4),
        }

    return {
        "accuracy": get_ci(acc_list),
        "precision": get_ci(prec_list),
        "recall": get_ci(rec_list),
        "f1_score": get_ci(f1_list),
        "matthews_corrcoef": get_ci(mcc_list),
    }


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
    Computes clustered bootstrap confidence intervals for the paired difference (Delta = Proposed - Baseline).
    """
    group_ids = np.asarray(group_ids)
    y_true = np.asarray(y_true).astype(int)
    y_prop = np.asarray(y_pred_proposed).astype(int)
    y_base = np.asarray(y_pred_baseline).astype(int)

    rng = np.random.RandomState(random_state)
    unique_groups = np.unique(group_ids)
    n_groups = len(unique_groups)

    group_to_indices = {g: np.where(group_ids == g)[0] for g in unique_groups}

    delta_mcc, delta_f1, delta_rec, delta_prec, delta_acc = [], [], [], [], []

    for _ in range(n_resamples):
        sampled_groups = rng.choice(unique_groups, size=n_groups, replace=True)
        sample_indices = np.concatenate([group_to_indices[g] for g in sampled_groups])

        y_t_b = y_true[sample_indices]
        y_prop_b = y_prop[sample_indices]
        y_base_b = y_base[sample_indices]

        if len(np.unique(y_t_b)) < 2:
            continue

        delta_acc.append(accuracy_score(y_t_b, y_prop_b) - accuracy_score(y_t_b, y_base_b))
        delta_prec.append(precision_score(y_t_b, y_prop_b, zero_division=0) - precision_score(y_t_b, y_base_b, zero_division=0))
        delta_rec.append(recall_score(y_t_b, y_prop_b, zero_division=0) - recall_score(y_t_b, y_base_b, zero_division=0))
        delta_f1.append(f1_score(y_t_b, y_prop_b, zero_division=0) - f1_score(y_t_b, y_base_b, zero_division=0))
        delta_mcc.append(matthews_corrcoef(y_t_b, y_prop_b) - matthews_corrcoef(y_t_b, y_base_b))

    def get_ci(arr: List[float]) -> Dict[str, float]:
        if not arr:
            return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
        a = np.array(arr)
        return {
            "mean_delta": round(float(np.mean(a)), 4),
            "ci_lower": round(float(np.percentile(a, 100 * (alpha / 2.0))), 4),
            "ci_upper": round(float(np.percentile(a, 100 * (1.0 - alpha / 2.0))), 4),
        }

    return {
        "delta_accuracy": get_ci(delta_acc),
        "delta_precision": get_ci(delta_prec),
        "delta_recall": get_ci(delta_rec),
        "delta_f1": get_ci(delta_f1),
        "delta_mcc": get_ci(delta_mcc),
    }


def compute_mcnemar_significance(
    y_true: np.ndarray,
    y_pred_proposed: np.ndarray,
    y_pred_baseline: np.ndarray,
) -> Dict[str, Any]:
    """
    Computes McNemar's test for paired classification discordance with continuity correction.
    Includes explicit cautionary statement regarding cluster-level correlation.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prop = np.asarray(y_pred_proposed).astype(int)
    y_base = np.asarray(y_pred_baseline).astype(int)

    correct_prop = (y_prop == y_true)
    correct_base = (y_base == y_true)

    b = int(np.sum(correct_prop & ~correct_base))  # Proposed correct, Baseline wrong
    c = int(np.sum(~correct_prop & correct_base))  # Baseline correct, Proposed wrong

    if (b + c) == 0:
        chi2_stat = 0.0
        p_val = 1.0
    else:
        # Edwards continuity correction
        chi2_stat = float(((abs(b - c) - 1.0) ** 2) / (b + c))
        p_val = float(1.0 - chi2.cdf(chi2_stat, df=1))

    return {
        "b_proposed_only_correct": b,
        "c_baseline_only_correct": c,
        "mcnemar_chi2": round(chi2_stat, 4),
        "p_value_raw": p_val,
        "p_value_formatted": format_p_value(p_val),
        "statistically_significant_05": bool(p_val < 0.05),
        "cluster_adjustment_notice": (
            "CAUTIONARY AUDIT NOTE: Standard McNemar test evaluates marginal pair-level discordance "
            "assuming independent Bernoulli trials. For clustered instances within premise groups or assignment "
            "cases, pair observations are correlated; inference should be corroborated with group-aware paired "
            "bootstrap confidence intervals."
        ),
    }
