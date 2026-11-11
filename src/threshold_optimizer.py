"""
Recall-Constrained Threshold Optimization Module.
Formulation:
    minimize(total_computational_cost)
    subject to: candidate_recall >= target_recall (e.g. 0.95 or 0.98)

Uses validation data strictly.
Evaluates threshold grid, rejects invalid configurations, and freezes optimal thresholds.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class ThresholdOptimizationResult:
    target_recall: float
    best_theta_low: float
    best_theta_high: float
    val_candidate_recall: float
    val_pruning_ratio: float
    val_expensive_call_rate: float
    constraint_satisfied: bool
    evaluated_configurations: int
    search_history: List[Dict[str, Any]]

class RecallConstrainedOptimizer:
    def __init__(
        self,
        cost_cheap_ms: float = 0.05,
        cost_heavy_ms: float = 4.0
    ):
        self.cost_cheap_ms = cost_cheap_ms
        self.cost_heavy_ms = cost_heavy_ms

    def optimize_thresholds(
        self,
        cheap_scores_val: np.ndarray,
        y_val: np.ndarray,
        target_recall: float = 0.95,
        theta_low_grid: np.ndarray = np.linspace(0.01, 0.40, 40),
        theta_high_grid: np.ndarray = np.linspace(0.70, 0.95, 6),
        is_triband: bool = True
    ) -> ThresholdOptimizationResult:
        """
        Searches for the lowest-cost threshold configuration on validation data
        subject to candidate_recall >= target_recall.
        """
        cheap_scores = np.asarray(cheap_scores_val, dtype=float)
        y_true = np.asarray(y_val, dtype=int)
        tot_positives = int((y_true == 1).sum())
        n_pairs = len(y_true)

        if tot_positives == 0 or n_pairs == 0:
            return ThresholdOptimizationResult(
                target_recall=target_recall,
                best_theta_low=0.0,
                best_theta_high=1.0,
                val_candidate_recall=1.0,
                val_pruning_ratio=0.0,
                val_expensive_call_rate=1.0,
                constraint_satisfied=True,
                evaluated_configurations=0,
                search_history=[]
            )

        best_cost = float('inf')
        best_cfg = None
        search_history = []

        if not is_triband:
            # Single threshold binary filter: s < theta -> prune; s >= theta -> heavy call
            for t in theta_low_grid:
                passed_mask = (cheap_scores >= t)
                retained_pos = int((passed_mask & (y_true == 1)).sum())
                cand_rec = float(retained_pos / tot_positives)
                pruning = float((~passed_mask).sum() / n_pairs)
                call_rate = float(passed_mask.sum() / n_pairs)

                # Estimated computational cost per pair in ms
                cost_ms = self.cost_cheap_ms + call_rate * self.cost_heavy_ms
                satisfied = (cand_rec >= target_recall)

                record = {
                    'theta_low': float(t),
                    'theta_high': 1.0,
                    'candidate_recall': cand_rec,
                    'pruning_ratio': pruning,
                    'expensive_call_rate': call_rate,
                    'cost_ms': cost_ms,
                    'satisfied': satisfied
                }
                search_history.append(record)

                if satisfied:
                    if cost_ms < best_cost:
                        best_cost = cost_ms
                        best_cfg = record
        else:
            # Tri-band filter: s < theta_low -> prune; s >= theta_high -> direct flag; else escalate
            for t_low in theta_low_grid:
                for t_high in theta_high_grid:
                    if t_high <= t_low:
                        continue

                    pruned_mask = (cheap_scores < t_low)
                    flagged_mask = (cheap_scores >= t_high)
                    escalate_mask = (~pruned_mask) & (~flagged_mask)

                    retained_pos = int((~pruned_mask & (y_true == 1)).sum())
                    cand_rec = float(retained_pos / tot_positives)
                    pruning = float(pruned_mask.sum() / n_pairs)
                    call_rate = float(escalate_mask.sum() / n_pairs)
                    flag_rate = float(flagged_mask.sum() / n_pairs)

                    cost_ms = self.cost_cheap_ms + call_rate * self.cost_heavy_ms
                    satisfied = (cand_rec >= target_recall)

                    record = {
                        'theta_low': float(t_low),
                        'theta_high': float(t_high),
                        'candidate_recall': cand_rec,
                        'pruning_ratio': pruning,
                        'expensive_call_rate': call_rate,
                        'direct_flag_rate': flag_rate,
                        'cost_ms': cost_ms,
                        'satisfied': satisfied
                    }
                    search_history.append(record)

                    if satisfied:
                        if cost_ms < best_cost:
                            best_cost = cost_ms
                            best_cfg = record

        if best_cfg is not None:
            return ThresholdOptimizationResult(
                target_recall=target_recall,
                best_theta_low=best_cfg['theta_low'],
                best_theta_high=best_cfg['theta_high'],
                val_candidate_recall=best_cfg['candidate_recall'],
                val_pruning_ratio=best_cfg['pruning_ratio'],
                val_expensive_call_rate=best_cfg['expensive_call_rate'],
                constraint_satisfied=True,
                evaluated_configurations=len(search_history),
                search_history=search_history
            )
        else:
            # Fallback: configuration with highest candidate recall
            search_history.sort(key=lambda x: -x['candidate_recall'])
            fallback = search_history[0]
            return ThresholdOptimizationResult(
                target_recall=target_recall,
                best_theta_low=fallback['theta_low'],
                best_theta_high=fallback['theta_high'],
                val_candidate_recall=fallback['candidate_recall'],
                val_pruning_ratio=fallback['pruning_ratio'],
                val_expensive_call_rate=fallback['expensive_call_rate'],
                constraint_satisfied=False,
                evaluated_configurations=len(search_history),
                search_history=search_history
            )

