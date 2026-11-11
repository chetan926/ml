"""
Transparent Validation-Based Threshold Selection Module (Phase 3.3)
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from research_upgrade.src.candidate_generation.candidate_evaluation import (
    CandidateMetrics,
    evaluate_candidate_threshold,
    compute_operating_curve
)


class ThresholdSelector:
    """
    Selects candidate scoring thresholds strictly on validation data.
    Enforces the primary research goal: maximizing workload reduction subject
    to a candidate recall constraint (e.g. Target Recall >= 0.98).
    """

    def __init__(self, num_grid_points: int = 500):
        self.num_grid_points = num_grid_points

    def select_recall_constrained_threshold(
        self,
        y_val: np.ndarray,
        val_scores: np.ndarray,
        target_recall: float = 0.98
    ) -> Dict[str, Any]:
        """
        Finds the highest candidate score threshold tau* on validation data such that:
            Validation Recall(tau*) >= target_recall
        Maximizing threshold tau* maximizes Workload Reduction (filters the most pairs safely).
        
        If target_recall cannot be satisfied, selects the threshold achieving maximal recall
        and documents target_achieved = False transparently.
        """
        y_arr = np.asarray(y_val, dtype=np.int32)
        s_arr = np.asarray(val_scores, dtype=np.float64)

        curve = compute_operating_curve(y_arr, s_arr, num_thresholds=self.num_grid_points)

        # Candidates that satisfy recall >= target_recall
        satisfying = [m for m in curve if m.candidate_recall >= target_recall]

        if satisfying:
            # Choose candidate with highest workload reduction (highest threshold)
            best_metric = max(satisfying, key=lambda m: (m.workload_reduction, m.threshold))
            target_achieved = True
            reason = f"Target recall {target_recall:.4f} achieved on validation data."
        else:
            # Fallback: maximal recall achieved
            best_metric = max(curve, key=lambda m: (m.candidate_recall, m.workload_reduction))
            target_achieved = False
            reason = (
                f"Target recall {target_recall:.4f} could not be satisfied on validation data. "
                f"Maximal achievable recall is {best_metric.candidate_recall:.4f}."
            )

        return {
            "selected_threshold": best_metric.threshold,
            "target_recall_constraint": target_recall,
            "target_achieved": target_achieved,
            "validation_metrics": best_metric.to_dict(),
            "explanation": reason
        }

    def select_workload_constrained_threshold(
        self,
        y_val: np.ndarray,
        val_scores: np.ndarray,
        target_workload_reduction: float = 0.50
    ) -> Dict[str, Any]:
        """
        Finds candidate score threshold tau* on validation data such that:
            Validation Workload Reduction(tau*) >= target_workload_reduction
        while maximizing candidate recall.
        """
        y_arr = np.asarray(y_val, dtype=np.int32)
        s_arr = np.asarray(val_scores, dtype=np.float64)

        curve = compute_operating_curve(y_arr, s_arr, num_thresholds=self.num_grid_points)

        satisfying = [m for m in curve if m.workload_reduction >= target_workload_reduction]

        if satisfying:
            best_metric = max(satisfying, key=lambda m: (m.candidate_recall, m.workload_reduction))
            target_achieved = True
            reason = f"Target workload reduction {target_workload_reduction:.4f} achieved."
        else:
            best_metric = max(curve, key=lambda m: (m.workload_reduction, m.candidate_recall))
            target_achieved = False
            reason = (
                f"Target workload reduction {target_workload_reduction:.4f} could not be met. "
                f"Maximal achievable is {best_metric.workload_reduction:.4f}."
            )

        return {
            "selected_threshold": best_metric.threshold,
            "target_workload_constraint": target_workload_reduction,
            "target_achieved": target_achieved,
            "validation_metrics": best_metric.to_dict(),
            "explanation": reason
        }

    def generate_operating_points_profile(
        self,
        y_val: np.ndarray,
        val_scores: np.ndarray,
        recall_targets: Optional[List[float]] = None,
        workload_targets: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Generates a comprehensive multi-point operational profile across standard operating regimes.
        """
        if recall_targets is None:
            recall_targets = [0.99, 0.98, 0.95, 0.90, 0.85]
        if workload_targets is None:
            workload_targets = [0.30, 0.40, 0.50, 0.60, 0.70]

        recall_profiles = [
            self.select_recall_constrained_threshold(y_val, val_scores, target_recall=rt)
            for rt in recall_targets
        ]

        workload_profiles = [
            self.select_workload_constrained_threshold(y_val, val_scores, target_workload_reduction=wt)
            for wt in workload_targets
        ]

        return {
            "recall_constrained_operating_points": recall_profiles,
            "workload_constrained_operating_points": workload_profiles
        }

