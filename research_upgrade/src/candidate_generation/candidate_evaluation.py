"""
Candidate Generation Evaluation and Metrics Module (Phase 3.3)
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


@dataclass
class CandidateMetrics:
    threshold: float
    total_pairs: int
    total_positives: int
    total_negatives: int
    escalated_pairs: int
    filtered_pairs: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    candidate_recall: float
    workload_reduction: float
    candidate_precision: float
    retention_rate: float
    f1_score: float
    f2_score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_candidate_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float
) -> CandidateMetrics:
    """
    Evaluates candidate generation performance at a specific threshold tau.
    Pairs with score >= tau are escalated to Stage 2 verification.
    """
    y_true_arr = np.asarray(y_true, dtype=np.int32)
    scores_arr = np.asarray(scores, dtype=np.float64)

    assert len(y_true_arr) == len(scores_arr), "Length mismatch between y_true and scores"
    n_pairs = len(y_true_arr)
    if n_pairs == 0:
        raise ValueError("Cannot evaluate empty predictions.")

    escalated = (scores_arr >= threshold)
    tp = int(np.sum((y_true_arr == 1) & escalated))
    fp = int(np.sum((y_true_arr == 0) & escalated))
    tn = int(np.sum((y_true_arr == 0) & (~escalated)))
    fn = int(np.sum((y_true_arr == 1) & (~escalated)))

    pos_count = int(np.sum(y_true_arr == 1))
    neg_count = int(np.sum(y_true_arr == 0))
    esc_count = int(np.sum(escalated))
    filt_count = n_pairs - esc_count

    recall = float(tp / pos_count) if pos_count > 0 else 1.0
    precision = float(tp / esc_count) if esc_count > 0 else 0.0
    workload_red = float(filt_count / n_pairs) if n_pairs > 0 else 0.0
    retention_rate = float(esc_count / n_pairs) if n_pairs > 0 else 0.0

    if precision + recall > 0:
        f1 = float(2.0 * precision * recall / (precision + recall))
    else:
        f1 = 0.0

    # F2 score weights recall twice as heavily as precision
    beta2 = 4.0
    if (beta2 * precision + recall) > 0:
        f2 = float((1.0 + beta2) * precision * recall / (beta2 * precision + recall))
    else:
        f2 = 0.0

    return CandidateMetrics(
        threshold=round(float(threshold), 6),
        total_pairs=n_pairs,
        total_positives=pos_count,
        total_negatives=neg_count,
        escalated_pairs=esc_count,
        filtered_pairs=filt_count,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        candidate_recall=round(recall, 6),
        workload_reduction=round(workload_red, 6),
        candidate_precision=round(precision, 6),
        retention_rate=round(retention_rate, 6),
        f1_score=round(f1, 6),
        f2_score=round(f2, 6)
    )


def compute_operating_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    num_thresholds: int = 200
) -> List[CandidateMetrics]:
    """
    Computes candidate metrics across a fine grid of candidate score thresholds.
    """
    s_arr = np.asarray(scores, dtype=np.float64)
    min_s = float(np.min(s_arr))
    max_s = float(np.max(s_arr))

    if min_s == max_s:
        thresholds = [min_s]
    else:
        # Extend slightly past bounds to include edge cases (0% and 100% retention)
        eps = 1e-5
        thresholds = np.linspace(min_s - eps, max_s + eps, num_thresholds)

    metrics_list = [
        evaluate_candidate_threshold(y_true, scores, t)
        for t in thresholds
    ]
    return metrics_list

