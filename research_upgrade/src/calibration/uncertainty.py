"""
Uncertainty Estimation and Selective Prediction Module (Phase 3.4).

Computes uncertainty estimators:
- Predictive Entropy (Shannon entropy in bits)
- Maximum Class Probability (Confidence)
- Probability Margin
- Calibration-Aware Selective Classification (Risk-Coverage Curves, AURC, Abstention)
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from scipy.integrate import trapezoid


def compute_predictive_entropy(probabilities: np.ndarray) -> np.ndarray:
    """
    Computes Shannon predictive entropy in bits for binary probabilities:
    H(p) = - [ p * log2(p) + (1-p) * log2(1-p) ]
    Bounded in [0.0, 1.0] where 1.0 indicates maximum uncertainty (p = 0.5).
    """
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    q = 1.0 - p
    entropy = -(p * np.log2(p) + q * np.log2(q))
    return np.clip(entropy, 0.0, 1.0)


def compute_confidence_margin(probabilities: np.ndarray) -> np.ndarray:
    """
    Computes probability margin: |2p - 1| in [0, 1].
    Near 0 indicates borderline/uncertain prediction; near 1 indicates confident prediction.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    return np.abs(2.0 * p - 1.0)


def compute_maximum_class_probability(probabilities: np.ndarray) -> np.ndarray:
    """
    Computes maximum class probability: max(p, 1-p) in [0.5, 1.0].
    """
    p = np.asarray(probabilities, dtype=np.float64)
    return np.maximum(p, 1.0 - p)


def evaluate_selective_prediction(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_thresholds: int = 50,
) -> Dict[str, Any]:
    """
    Evaluates selective classification performance across uncertainty/confidence thresholds.

    Args:
        y_true: Ground truth binary labels {0, 1}.
        y_prob: Predicted positive-class probabilities.
        n_thresholds: Number of evaluation thresholds between 0.50 and 0.99.

    Returns:
        Dict with AURC (Area Under Risk-Coverage Curve), coverage profiles, and abstention stats.
    """
    y_true = np.asarray(y_true, dtype=np.int32).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    conf = compute_maximum_class_probability(y_prob)
    y_pred = (y_prob >= 0.5).astype(np.int32)
    errors = (y_pred != y_true).astype(np.float64)

    N = len(y_true)
    if N == 0:
        raise ValueError("Empty array in selective prediction evaluation.")

    # Threshold grid on confidence (from 0.50 to 0.99)
    thresholds = np.linspace(0.50, 0.99, n_thresholds)
    curve_points = []
    coverages = []
    risks = []

    for theta in thresholds:
        accepted = conf >= theta
        n_acc = int(np.sum(accepted))
        if n_acc > 0:
            cov = n_acc / N
            sel_risk = float(np.mean(errors[accepted]))
            sel_acc = 1.0 - sel_risk
            abstention = 1.0 - cov
        else:
            cov = 0.0
            sel_risk = 0.0
            sel_acc = 1.0
            abstention = 1.0

        coverages.append(cov)
        risks.append(sel_risk)
        curve_points.append({
            "confidence_threshold": round(float(theta), 4),
            "coverage": round(cov, 4),
            "abstention_rate": round(abstention, 4),
            "selective_risk": round(sel_risk, 4),
            "selective_accuracy": round(sel_acc, 4),
            "accepted_samples": n_acc,
        })

    # Sort by coverage ascending for trapezoidal integration of Risk-Coverage curve
    sorted_pairs = sorted(zip(coverages, risks), key=lambda x: x[0])
    cov_sorted = np.array([p[0] for p in sorted_pairs])
    risk_sorted = np.array([p[1] for p in sorted_pairs])

    # AURC (Area Under the Risk-Coverage Curve)
    # Lower is better (lower risk at higher coverage)
    aurc = float(trapezoid(risk_sorted, cov_sorted))

    # Overall base risk (at coverage = 1.0)
    full_risk = float(np.mean(errors))

    return {
        "full_sample_error_rate": round(full_risk, 4),
        "area_under_risk_coverage_curve": round(aurc, 4),
        "mean_predictive_entropy": round(float(np.mean(compute_predictive_entropy(y_prob))), 4),
        "mean_confidence": round(float(np.mean(conf)), 4),
        "risk_coverage_curve": curve_points,
    }
