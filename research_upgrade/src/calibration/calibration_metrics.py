"""
Calibration Metrics Computation Module (Phase 3.4).

Computes comprehensive calibration quality metrics:
- Brier Score
- Log Loss (Binary Cross-Entropy)
- Expected Calibration Error (ECE - equal-width)
- Adaptive ECE (equal-mass quantile binning)
- Maximum Calibration Error (MCE)
- Calibration Slope and Intercept (via logistic calibration curve fit)
- Reliability Diagram bin statistics
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.linear_model import LogisticRegression


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    adaptive_bins: int = 10,
) -> Dict[str, Any]:
    """
    Computes rigorous probability calibration metrics for binary classification.

    Args:
        y_true: Ground truth binary labels {0, 1}, shape (N,).
        y_prob: Predicted positive-class probabilities in [0, 1], shape (N,).
        n_bins: Number of equal-width bins for standard ECE.
        adaptive_bins: Number of equal-mass quantile bins for adaptive ECE.

    Returns:
        Dict containing all scalar metrics and bin-level reliability diagram data.
    """
    y_true = np.asarray(y_true, dtype=np.int32).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    y_prob = np.clip(y_prob, 1e-7, 1.0 - 1e-7)

    N = len(y_true)
    if N == 0:
        raise ValueError("Cannot compute calibration metrics on empty arrays.")

    # 1. Brier Score
    brier = float(brier_score_loss(y_true, y_prob))

    # 2. Log Loss (Binary Cross-Entropy)
    logloss = float(log_loss(y_true, y_prob))

    # 3. Standard Equal-Width ECE & MCE
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_data = []
    ece = 0.0
    mce = 0.0

    for m in range(n_bins):
        mask = bin_indices == m
        count = int(np.sum(mask))
        if count > 0:
            bin_acc = float(np.mean(y_true[mask]))
            bin_conf = float(np.mean(y_prob[mask]))
            abs_diff = abs(bin_acc - bin_conf)
            weight = count / N
            ece += weight * abs_diff
            if abs_diff > mce:
                mce = abs_diff
            bin_data.append({
                "bin_index": m,
                "bin_lower": float(bin_edges[m]),
                "bin_upper": float(bin_edges[m + 1]),
                "sample_count": count,
                "accuracy": bin_acc,
                "confidence": bin_conf,
                "calibration_gap": abs_diff,
            })
        else:
            bin_data.append({
                "bin_index": m,
                "bin_lower": float(bin_edges[m]),
                "bin_upper": float(bin_edges[m + 1]),
                "sample_count": 0,
                "accuracy": 0.0,
                "confidence": float((bin_edges[m] + bin_edges[m + 1]) / 2.0),
                "calibration_gap": 0.0,
            })

    # 4. Adaptive ECE (Equal-Mass Binning)
    adaptive_ece = 0.0
    try:
        quantiles = np.linspace(0.0, 100.0, adaptive_bins + 1)
        adaptive_edges = np.percentile(y_prob, quantiles)
        # Ensure strictly monotonic edges for digitize
        adaptive_edges = np.unique(adaptive_edges)
        if len(adaptive_edges) > 1:
            adapt_indices = np.digitize(y_prob, adaptive_edges) - 1
            adapt_indices = np.clip(adapt_indices, 0, len(adaptive_edges) - 2)
            n_adapt_bins = len(adaptive_edges) - 1
            for k in range(n_adapt_bins):
                mask = adapt_indices == k
                count = int(np.sum(mask))
                if count > 0:
                    b_acc = float(np.mean(y_true[mask]))
                    b_conf = float(np.mean(y_prob[mask]))
                    adaptive_ece += (count / N) * abs(b_acc - b_conf)
        else:
            adaptive_ece = ece
    except Exception:
        adaptive_ece = ece

    # 5. Calibration Slope and Intercept (via logistic regression of y on logit(p))
    calib_slope = 1.0
    calib_intercept = 0.0
    unique_labels = np.unique(y_true)
    if len(unique_labels) >= 2:
        try:
            logits = np.log(y_prob / (1.0 - y_prob)).reshape(-1, 1)
            calib_lr = LogisticRegression(solver="lbfgs", C=10.0, max_iter=500)
            calib_lr.fit(logits, y_true)
            calib_slope = float(calib_lr.coef_[0][0])
            calib_intercept = float(calib_lr.intercept_[0])
        except Exception:
            calib_slope = 1.0
            calib_intercept = 0.0

    return {
        "brier_score": round(brier, 6),
        "log_loss": round(logloss, 6),
        "expected_calibration_error": round(float(ece), 6),
        "adaptive_ece": round(float(adaptive_ece), 6),
        "maximum_calibration_error": round(float(mce), 6),
        "calibration_slope": round(calib_slope, 4),
        "calibration_intercept": round(calib_intercept, 4),
        "n_bins": n_bins,
        "total_samples": N,
        "reliability_bins": bin_data,
    }
