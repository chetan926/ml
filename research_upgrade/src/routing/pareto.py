"""
Pareto Analysis Module for Budget-Constrained Routing (Phase 3.5).

Generates multi-objective Pareto frontiers mapping:
- Recall vs Computational Cost (Stage-2 calls)
- F1-score vs Computational Cost
- MCC vs Computational Cost
- Latency vs Recall
- Non-dominated Pareto frontier extraction
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.metrics import matthews_corrcoef, f1_score


def is_pareto_efficient(costs: np.ndarray) -> np.ndarray:
    """
    Finds the Pareto-efficient points in a 2D array where col 0 is to be minimized
    and col 1 is to be maximized.

    Args:
        costs: (N, 2) array where costs[:, 0] is minimized, costs[:, 1] is maximized.

    Returns:
        Boolean array of shape (N,) indicating whether each point is Pareto efficient.
    """
    # Transform col 1 to minimization: [-col1]
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # A point j dominates i if c_j0 <= c_i0 and c_j1 >= c_i1, with at least one strict inequality
            dominates = (costs[:, 0] <= c[0]) & (costs[:, 1] >= c[1]) & (
                (costs[:, 0] < c[0]) | (costs[:, 1] > c[1])
            )
            if np.any(dominates):
                is_efficient[i] = False
    return is_efficient


def generate_pareto_frontier_points(
    budget_grid: List[float],
    scores: np.ndarray,
    probs: np.ndarray,
    labels: np.ndarray,
    stage2_scores: np.ndarray,
    base_latency_ms: float = 33.0,
    stage2_latency_ms: float = 45.0,
) -> Dict[str, Any]:
    """
    Evaluates routing trade-offs across varying computational budgets B in (0, 1].
    """
    labels = np.asarray(labels, dtype=np.int32)
    pos_total = int(np.sum(labels == 1))
    N = len(labels)

    points = []
    for b in budget_grid:
        # Fixed quantile cutoff corresponding to budget b
        cutoff = np.percentile(scores, (1.0 - b) * 100.0)
        escalated = scores >= cutoff

        pred = np.where(escalated, (stage2_scores >= 0.5).astype(np.int32), 0)
        tp = int(np.sum((pred == 1) & (labels == 1)))
        fp = int(np.sum((pred == 1) & (labels == 0)))
        tn = int(np.sum((pred == 0) & (labels == 0)))
        fn = int(np.sum((pred == 0) & (labels == 1)))

        recall = tp / max(1, pos_total)
        precision = tp / max(1, tp + fp)
        f1 = float(f1_score(labels, pred, zero_division=0))
        mcc = float(matthews_corrcoef(labels, pred)) if (len(np.unique(pred)) > 1) else 0.0

        stage2_fraction = float(np.mean(escalated))
        effective_latency_ms = base_latency_ms + (stage2_fraction * stage2_latency_ms)

        points.append({
            "budget_fraction": round(b, 2),
            "stage2_invocation_fraction": round(stage2_fraction, 4),
            "effective_latency_ms": round(effective_latency_ms, 2),
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1_score": round(f1, 4),
            "mcc": round(mcc, 4),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        })

    # Pareto efficiency: minimize effective_latency_ms, maximize recall
    cost_matrix = np.array([[p["effective_latency_ms"], p["recall"]] for p in points])
    efficient_mask = is_pareto_efficient(cost_matrix)

    for i, p in enumerate(points):
        p["is_pareto_optimal"] = bool(efficient_mask[i])

    return {
        "budget_evaluation_points": points,
        "pareto_optimal_count": int(np.sum(efficient_mask)),
        "base_stage1_latency_ms": base_latency_ms,
        "stage2_latency_ms": stage2_latency_ms,
    }

