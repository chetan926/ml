"""
Budget-Constrained Adaptive Routing Engine (Phase 3.5).

Implements multi-objective adaptive routing policies balancing candidate recall,
predictive uncertainty, and computational verification budget:
- DecisionState assignment: LOW_RISK_PRUNED, UNCERTAIN_ESCALATED, HIGH_RISK_FLAGGED
- Threshold optimization on validation data under recall and budget constraints
- Comparative routing baselines (All-Stage-2, Fixed-Threshold, Random-Pruning, Proposed)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from research_upgrade.src.routing.decision_states import DecisionState
from research_upgrade.src.calibration.uncertainty import compute_predictive_entropy


class BudgetAdaptiveRouter:
    """
    Proposed Uncertainty-Aware Budget-Constrained Router.
    Routes candidate pairs into three operational states based on:
    - Continuous candidate score s
    - Calibrated posterior probability p
    - Predictive Shannon entropy H(p)
    - Operational compute budget B in (0, 1]
    """

    def __init__(
        self,
        tau_prune: float = 0.08,
        tau_flag: float = 0.85,
        entropy_threshold: float = 0.65,
        target_recall: float = 0.98,
        max_budget_fraction: float = 0.75,
    ):
        self.tau_prune = tau_prune
        self.tau_flag = tau_flag
        self.entropy_threshold = entropy_threshold
        self.target_recall = target_recall
        self.max_budget_fraction = max_budget_fraction
        self.is_optimized = False

    def route_single_pair(
        self,
        score: float,
        calibrated_prob: float,
        entropy: float,
    ) -> DecisionState:
        """Determines routing state for an individual pair."""
        # Highly uncertain pairs must be escalated to Stage-2 verification
        if entropy >= self.entropy_threshold:
            return DecisionState.UNCERTAIN_ESCALATED

        # Confident low-risk pairs pruned early
        if score < self.tau_prune and calibrated_prob < 0.25:
            return DecisionState.LOW_RISK_PRUNED

        # Confident high-risk pairs flagged directly
        if score >= self.tau_flag and calibrated_prob >= 0.75:
            return DecisionState.HIGH_RISK_FLAGGED

        # All boundary/intermediate cases escalated
        return DecisionState.UNCERTAIN_ESCALATED

    def route_batch(
        self,
        scores: np.ndarray,
        calibrated_probs: np.ndarray,
        entropies: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Routes an array of candidate pairs. Returns array of DecisionState objects.
        """
        scores = np.asarray(scores, dtype=np.float64)
        probs = np.asarray(calibrated_probs, dtype=np.float64)
        if entropies is None:
            entropies = compute_predictive_entropy(probs)
        else:
            entropies = np.asarray(entropies, dtype=np.float64)

        states = []
        for s, p, h in zip(scores, probs, entropies):
            st = self.route_single_pair(s, p, h)
            states.append(st)
        return np.array(states, dtype=object)

    def optimize_thresholds_on_validation(
        self,
        val_scores: np.ndarray,
        val_probs: np.ndarray,
        val_labels: np.ndarray,
        val_entropies: Optional[np.ndarray] = None,
        target_recall: float = 0.98,
        budget_fraction: float = 0.75,
    ) -> Dict[str, Any]:
        """
        Optimizes (tau_prune, tau_flag, entropy_threshold) strictly on validation data
        to maximize workload reduction / minimize Stage-2 calls subject to target recall.
        """
        val_scores = np.asarray(val_scores, dtype=np.float64)
        val_probs = np.asarray(val_probs, dtype=np.float64)
        val_labels = np.asarray(val_labels, dtype=np.int32)
        if val_entropies is None:
            val_entropies = compute_predictive_entropy(val_probs)
        else:
            val_entropies = np.asarray(val_entropies, dtype=np.float64)

        N = len(val_labels)
        pos_total = int(np.sum(val_labels == 1))
        if pos_total == 0:
            self.is_optimized = True
            return {"status": "zero_positives_handled"}

        # Grid search over candidate threshold triplets
        best_obj = -1.0
        best_prune = 0.05
        best_flag = 0.85
        best_ent = 0.65

        prune_candidates = np.linspace(0.01, 0.25, 25)
        flag_candidates = [0.80, 0.85, 0.90, 0.95]
        ent_candidates = [0.55, 0.65, 0.75]

        for p_t in prune_candidates:
            for f_t in flag_candidates:
                for e_t in ent_candidates:
                    # Vectorized routing
                    is_high_ent = val_entropies >= e_t
                    is_prune = (val_scores < p_t) & (val_probs < 0.25) & (~is_high_ent)
                    is_flag = (val_scores >= f_t) & (val_probs >= 0.75) & (~is_high_ent)
                    is_escalate = (~is_prune) & (~is_flag)

                    # Retained pairs for positive class: flag + escalate
                    retained = is_flag | is_escalate
                    tp = np.sum((retained) & (val_labels == 1))
                    recall = tp / pos_total

                    escalated_count = np.sum(is_escalate)
                    escalation_ratio = escalated_count / N

                    # Constraint checks
                    if recall >= (target_recall - 0.005) and escalation_ratio <= budget_fraction:
                        # Objective: maximize pruning of true negatives while minimizing Stage-2 escalation
                        tn_pruned = np.sum((is_prune) & (val_labels == 0))
                        obj = tn_pruned - (escalation_ratio * N * 0.2)
                        if obj > best_obj:
                            best_obj = obj
                            best_prune = float(p_t)
                            best_flag = float(f_t)
                            best_ent = float(e_t)

        self.tau_prune = best_prune
        self.tau_flag = best_flag
        self.entropy_threshold = best_ent
        self.target_recall = target_recall
        self.max_budget_fraction = budget_fraction
        self.is_optimized = True

        return {
            "optimal_tau_prune": self.tau_prune,
            "optimal_tau_flag": self.tau_flag,
            "optimal_entropy_threshold": self.entropy_threshold,
            "validation_target_recall": self.target_recall,
            "validation_budget_fraction": self.max_budget_fraction,
        }


def simulate_routing_baselines(
    scores: np.ndarray,
    probs: np.ndarray,
    labels: np.ndarray,
    stage2_scores: np.ndarray,
    budget_fraction: float = 0.70,
    random_seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluates comparative routing strategies under identical budget constraints:
    1. All-Stage-2 Execution (expensive benchmark)
    2. Fixed-Score Threshold Routing (no uncertainty)
    3. Random Pruning Routing (randomly sub-selects to match budget)
    4. Proposed Uncertainty-Aware Adaptive Routing
    """
    np.random.seed(random_seed)
    N = len(labels)
    pos_total = int(np.sum(labels == 1))
    entropies = compute_predictive_entropy(probs)

    # 1. All-Stage-2
    pred_all = (stage2_scores >= 0.5).astype(np.int32)
    tp_all = int(np.sum((pred_all == 1) & (labels == 1)))
    fp_all = int(np.sum((pred_all == 1) & (labels == 0)))
    rec_all = tp_all / max(1, pos_total)
    prec_all = tp_all / max(1, tp_all + fp_all)

    # 2. Fixed-Score Threshold (select threshold that prunes 1 - budget)
    cutoff = np.percentile(scores, (1.0 - budget_fraction) * 100.0)
    escalated_fixed = scores >= cutoff
    # Stage 2 runs only on escalated; pruned pairs get pred=0
    pred_fixed = np.where(escalated_fixed, (stage2_scores >= 0.5).astype(np.int32), 0)
    tp_fixed = int(np.sum((pred_fixed == 1) & (labels == 1)))
    fp_fixed = int(np.sum((pred_fixed == 1) & (labels == 0)))
    rec_fixed = tp_fixed / max(1, pos_total)
    prec_fixed = tp_fixed / max(1, tp_fixed + fp_fixed)

    # 3. Random Pruning
    escalated_rand = np.random.rand(N) < budget_fraction
    pred_rand = np.where(escalated_rand, (stage2_scores >= 0.5).astype(np.int32), 0)
    tp_rand = int(np.sum((pred_rand == 1) & (labels == 1)))
    fp_rand = int(np.sum((pred_rand == 1) & (labels == 0)))
    rec_rand = tp_rand / max(1, pos_total)
    prec_rand = tp_rand / max(1, tp_rand + fp_rand)

    # 4. Proposed Router
    router = BudgetAdaptiveRouter()
    # Optimize on first half / val split
    half = N // 2
    router.optimize_thresholds_on_validation(
        scores[:half], probs[:half], labels[:half], entropies[:half],
        target_recall=0.98, budget_fraction=budget_fraction
    )
    states = router.route_batch(scores, probs, entropies)
    is_pruned = states == DecisionState.LOW_RISK_PRUNED
    is_flagged = states == DecisionState.HIGH_RISK_FLAGGED
    is_escalated = states == DecisionState.UNCERTAIN_ESCALATED

    # Final prediction: pruned=0, flagged=1, escalated=Stage2>=0.5
    pred_prop = np.zeros(N, dtype=np.int32)
    pred_prop[is_flagged] = 1
    pred_prop[is_escalated] = (stage2_scores[is_escalated] >= 0.5).astype(np.int32)

    tp_prop = int(np.sum((pred_prop == 1) & (labels == 1)))
    fp_prop = int(np.sum((pred_prop == 1) & (labels == 0)))
    rec_prop = tp_prop / max(1, pos_total)
    prec_prop = tp_prop / max(1, tp_prop + fp_prop)
    actual_escalated_ratio = float(np.mean(is_escalated))

    return {
        "all_stage2": {
            "strategy": "all_stage2",
            "escalation_budget": 1.0,
            "actual_stage2_fraction": 1.0,
            "recall": round(rec_all, 4),
            "precision": round(prec_all, 4),
            "f1_score": round(2 * prec_all * rec_all / max(1e-8, prec_all + rec_all), 4),
        },
        "fixed_threshold": {
            "strategy": "fixed_threshold",
            "escalation_budget": budget_fraction,
            "actual_stage2_fraction": round(float(np.mean(escalated_fixed)), 4),
            "recall": round(rec_fixed, 4),
            "precision": round(prec_fixed, 4),
            "f1_score": round(2 * prec_fixed * rec_fixed / max(1e-8, prec_fixed + rec_fixed), 4),
        },
        "random_pruning": {
            "strategy": "random_pruning",
            "escalation_budget": budget_fraction,
            "actual_stage2_fraction": round(float(np.mean(escalated_rand)), 4),
            "recall": round(rec_rand, 4),
            "precision": round(prec_rand, 4),
            "f1_score": round(2 * prec_rand * rec_rand / max(1e-8, prec_rand + rec_rand), 4),
        },
        "proposed_adaptive_routing": {
            "strategy": "proposed_adaptive_routing",
            "escalation_budget": budget_fraction,
            "actual_stage2_fraction": round(actual_escalated_ratio, 4),
            "recall": round(rec_prop, 4),
            "precision": round(prec_prop, 4),
            "f1_score": round(2 * prec_prop * rec_prop / max(1e-8, prec_prop + rec_prop), 4),
            "pruned_count": int(np.sum(is_pruned)),
            "flagged_count": int(np.sum(is_flagged)),
            "escalated_count": int(np.sum(is_escalated)),
        },
    }

