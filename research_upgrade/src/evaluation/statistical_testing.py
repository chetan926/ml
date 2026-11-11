"""
Comprehensive Statistical Evaluation and Hypothesis Testing Module (Phase 3.6+).

Calculates:
- 20 Core Classification Metrics
- Ranking Metrics (ROC-AUC, PR-AUC, Average Precision, Fixed-Recall Precision)
- Paired Bootstrap Confidence Intervals (95% BCa/Percentile CIs, 1,000 resamples)
- McNemar's Test for Paired Classification Significance
"""

import numpy as np
import scipy.stats as stats
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
    average_precision_score,
    cohen_kappa_score,
    matthews_corrcoef,
    brier_score_loss,
)


def compute_comprehensive_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Computes all 20 mandatory classification metrics + ranking metrics.
    """
    y_true = np.asarray(y_true, dtype=np.int32).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int32).ravel()
    N = len(y_true)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)

    # Core rates
    accuracy = (tp + tn) / max(1, N)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    npv = tn / max(1, tn + fn)

    fpr = fp / max(1, fp + tn)
    fnr = fn / max(1, fn + tp)
    fdr = fp / max(1, tp + fp)
    for_rate = fn / max(1, tn + fn)

    # F-scores
    f1 = 2.0 * precision * recall / max(1e-8, precision + recall)
    beta_05_sq = 0.5**2
    f05 = (1.0 + beta_05_sq) * precision * recall / max(1e-8, beta_05_sq * precision + recall)
    beta_2_sq = 2.0**2
    f2 = (1.0 + beta_2_sq) * precision * recall / max(1e-8, beta_2_sq * precision + recall)

    balanced_acc = (recall + specificity) / 2.0
    mcc = float(matthews_corrcoef(y_true, y_pred)) if (len(np.unique(y_pred)) > 1) else 0.0
    kappa = float(cohen_kappa_score(y_true, y_pred))
    youden_j = recall + specificity - 1.0
    g_mean = float(np.sqrt(max(0.0, recall * specificity)))

    # Threat Score / Jaccard
    jaccard = tp / max(1, tp + fp + fn)
    threat_score = jaccard

    # Diagnostic Odds Ratio (clipped to prevent inf)
    dor = (tp * tn) / max(1, fp * fn)

    metrics = {
        "total_samples": N,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "accuracy": round(float(accuracy), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1_score": round(float(f1), 6),
        "f0_5_score": round(float(f05), 6),
        "f2_score": round(float(f2), 6),
        "specificity": round(float(specificity), 6),
        "negative_predictive_value": round(float(npv), 6),
        "false_positive_rate": round(float(fpr), 6),
        "false_negative_rate": round(float(fnr), 6),
        "false_discovery_rate": round(float(fdr), 6),
        "false_omission_rate": round(float(for_rate), 6),
        "balanced_accuracy": round(float(balanced_acc), 6),
        "matthews_corrcoef": round(float(mcc), 6),
        "cohens_kappa": round(float(kappa), 6),
        "youdens_j": round(float(youden_j), 6),
        "geometric_mean": round(float(g_mean), 6),
        "jaccard_score": round(float(jaccard), 6),
        "threat_score": round(float(threat_score), 6),
        "diagnostic_odds_ratio": round(float(dor), 4),
    }

    # Ranking metrics if probabilities provided
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
        if len(np.unique(y_true)) >= 2:
            roc_auc = float(roc_auc_score(y_true, y_prob))
            precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = float(auc(recalls, precisions))
            avg_prec = float(average_precision_score(y_true, y_prob))

            # Precision at fixed recall >= 0.98
            valid_idx = np.where(recalls >= 0.98)[0]
            prec_at_98_recall = float(np.max(precisions[valid_idx])) if len(valid_idx) > 0 else float(precisions[0])

            metrics["roc_auc"] = round(roc_auc, 6)
            metrics["pr_auc"] = round(pr_auc, 6)
            metrics["average_precision"] = round(avg_prec, 6)
            metrics["precision_at_98_recall"] = round(prec_at_98_recall, 6)

    return metrics


def compute_paired_bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Computes percentile bootstrap confidence intervals for primary performance metrics.
    """
    rng = np.random.RandomState(random_seed)
    y_true = np.asarray(y_true, dtype=np.int32).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int32).ravel()
    N = len(y_true)

    alpha_low = (1.0 - confidence_level) / 2.0 * 100.0
    alpha_high = (1.0 + confidence_level) / 2.0 * 100.0

    mcc_dist = []
    f1_dist = []
    rec_dist = []
    prec_dist = []
    acc_dist = []

    for _ in range(n_resamples):
        idx = rng.randint(0, N, size=N)
        b_true = y_true[idx]
        b_pred = y_pred[idx]

        tn, fp, fn, tp = confusion_matrix(b_true, b_pred, labels=[0, 1]).ravel()
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2.0 * prec * rec / max(1e-8, prec + rec)
        acc = (tp + tn) / max(1, N)
        mcc = float(matthews_corrcoef(b_true, b_pred)) if len(np.unique(b_pred)) > 1 else 0.0

        mcc_dist.append(mcc)
        f1_dist.append(f1)
        rec_dist.append(rec)
        prec_dist.append(prec)
        acc_dist.append(acc)

    def get_ci(arr):
        return {
            "mean": round(float(np.mean(arr)), 4),
            "ci_lower": round(float(np.percentile(arr, alpha_low)), 4),
            "ci_upper": round(float(np.percentile(arr, alpha_high)), 4),
            "std_error": round(float(np.std(arr)), 4),
        }

    return {
        "mcc": get_ci(mcc_dist),
        "f1_score": get_ci(f1_dist),
        "recall": get_ci(rec_dist),
        "precision": get_ci(prec_dist),
        "accuracy": get_ci(acc_dist),
    }


def compute_mcnemar_test(
    y_true: np.ndarray,
    y_pred_proposed: np.ndarray,
    y_pred_baseline: np.ndarray,
) -> Dict[str, Any]:
    """
    Computes McNemar's paired chi-squared test for statistical significance.
    Table:
              Baseline Correct | Baseline Incorrect
    Proposed Correct:   a              c
    Proposed Incorrect: b              d
    """
    y_true = np.asarray(y_true, dtype=np.int32).ravel()
    p_prop = (np.asarray(y_pred_proposed, dtype=np.int32).ravel() == y_true)
    p_base = (np.asarray(y_pred_baseline, dtype=np.int32).ravel() == y_true)

    a = int(np.sum(p_prop & p_base))
    b = int(np.sum((~p_prop) & p_base))  # Proposed wrong, baseline right
    c = int(np.sum(p_prop & (~p_base)))  # Proposed right, baseline wrong
    d = int(np.sum((~p_prop) & (~p_base)))

    discordant = b + c
    if discordant == 0:
        return {
            "chi2_statistic": 0.0,
            "p_value": 1.0,
            "significant_at_05": False,
            "contingency_table": {"both_correct": a, "prop_only": c, "base_only": b, "both_wrong": d},
            "note": "Zero discordant pairs; models make identical predictions.",
        }

    # Edwards continuity correction
    chi2 = ((abs(b - c) - 1.0) ** 2) / float(discordant)
    p_val = float(1.0 - stats.chi2.cdf(chi2, df=1))

    return {
        "chi2_statistic": round(float(chi2), 4),
        "p_value": round(p_val, 6),
        "significant_at_05": bool(p_val < 0.05),
        "contingency_table": {"both_correct": a, "prop_only": c, "base_only": b, "both_wrong": d},
        "odds_ratio": round(c / max(1, b), 4),
    }

