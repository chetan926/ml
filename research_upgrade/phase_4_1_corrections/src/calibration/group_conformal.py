"""
Phase 4.1 Corrected Group Conformal Predictor.
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection.

Methodological Guarantees:
1. Exact Finite-Sample Quantile:
   Calculates rank k = ceil((m + 1) * (1 - alpha)) for m exchangeable calibration groups.
2. Group-Supremum Nonconformity Score:
   Calculates S_g = max_{i in g} (1 - f(x_i, y_i)) to guarantee group-level coverage.
3. Infinite Quantile Fallback:
   If k > m, conservatively assigns q_hat = inf, producing universal prediction sets {0, 1}.
4. Dual Coverage Reporting:
   Reports both empirical group coverage and empirical pair coverage.
5. Calibrator Independence:
   Calibrated on designated validation groups (e.g. 254 premise groups from D_val),
   evaluated on independent held-out test groups (1,015 premise groups from D_test).
"""

from typing import Dict, Any, List, Optional
import math
import numpy as np


class CorrectedGroupConformalPredictor:
    """
    Exact finite-sample Group Conformal Predictor for clustered academic integrity pairs.
    """

    def __init__(self, alpha: float = 0.05):
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"Significance level alpha must be in (0, 1), got {alpha}")
        self.alpha = float(alpha)
        self.m_groups: int = 0
        self.k_rank: int = 0
        self.calib_quantile: float = float("inf")
        self.is_infinite_quantile: bool = True
        self.group_scores: Dict[Any, float] = {}
        self.calibration_metadata: Dict[str, Any] = {}

    def calibrate(
        self,
        group_ids: np.ndarray,
        y_true: np.ndarray,
        y_calibrated_prob: np.ndarray,
    ) -> "CorrectedGroupConformalPredictor":
        """
        Calibrates the nonconformity quantile over disjoint calibration groups.
        """
        group_ids = np.asarray(group_ids)
        y_true = np.asarray(y_true).astype(int)
        y_prob = np.asarray(y_calibrated_prob).astype(float)

        unique_groups = np.unique(group_ids)
        self.m_groups = len(unique_groups)
        if self.m_groups == 0:
            raise ValueError("Cannot calibrate on empty group list.")

        # Compute group-supremum nonconformity score for each calibration group
        scores_list = []
        for g in unique_groups:
            mask = (group_ids == g)
            y_g = y_true[mask]
            p_g = y_prob[mask]
            
            # Nonconformity score for each pair: 1 - prob(true_class)
            prob_true = np.where(y_g == 1, p_g, 1.0 - p_g)
            pair_nonconf = 1.0 - prob_true
            group_supremum = float(np.max(pair_nonconf))
            self.group_scores[g] = group_supremum
            scores_list.append(group_supremum)

        scores_sorted = np.sort(np.asarray(scores_list))

        # Exact finite-sample conformal quantile rank
        # k = ceil((m + 1) * (1 - alpha))
        k = math.ceil((self.m_groups + 1) * (1.0 - self.alpha))
        self.k_rank = k

        if k > self.m_groups:
            self.calib_quantile = float("inf")
            self.is_infinite_quantile = True
        else:
            # 1-indexed rank k corresponds to 0-indexed position k - 1
            self.calib_quantile = float(scores_sorted[k - 1])
            self.is_infinite_quantile = False

        self.calibration_metadata = {
            "alpha": self.alpha,
            "target_coverage": 1.0 - self.alpha,
            "m_groups": self.m_groups,
            "k_rank": self.k_rank,
            "is_infinite_quantile": self.is_infinite_quantile,
            "calib_quantile": self.calib_quantile,
            "min_score": float(scores_sorted[0]),
            "median_score": float(np.median(scores_sorted)),
            "max_score": float(scores_sorted[-1]),
        }
        return self

    def predict_prediction_sets(
        self,
        y_calibrated_prob: np.ndarray,
    ) -> List[List[int]]:
        """
        Constructs conformal prediction sets for evaluation instances:
        C(x_i) = {y in {0, 1} : 1 - prob(y) <= calib_quantile}
        """
        y_prob = np.asarray(y_calibrated_prob).astype(float)
        sets: List[List[int]] = []

        if self.is_infinite_quantile or math.isinf(self.calib_quantile):
            # Universal set when rank exceeds m
            return [[0, 1] for _ in range(len(y_prob))]

        for p in y_prob:
            s: List[int] = []
            nonconf_0 = 1.0 - (1.0 - p) # = p
            nonconf_1 = 1.0 - p

            if nonconf_0 <= self.calib_quantile:
                s.append(0)
            if nonconf_1 <= self.calib_quantile:
                s.append(1)

            if len(s) == 0:
                # Fallback to most likely label if set is empty
                s = [1] if p >= 0.5 else [0]
            sets.append(sorted(s))
        return sets

    def evaluate_test_coverage(
        self,
        group_ids: np.ndarray,
        y_true: np.ndarray,
        y_calibrated_prob: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Evaluates empirical coverage on independent held-out evaluation groups.
        """
        group_ids = np.asarray(group_ids)
        y_true = np.asarray(y_true).astype(int)
        y_prob = np.asarray(y_calibrated_prob).astype(float)

        pred_sets = self.predict_prediction_sets(y_prob)

        # Pair coverage
        pair_covered = [
            (y_true[i] in pred_sets[i]) for i in range(len(y_true))
        ]
        emp_pair_cov = float(np.mean(pair_covered))

        # Group coverage (all pairs in the group must be covered)
        unique_test_groups = np.unique(group_ids)
        group_covered = []
        for g in unique_test_groups:
            mask = (group_ids == g)
            g_pair_cov = [pair_covered[i] for i in np.where(mask)[0]]
            group_covered.append(all(g_pair_cov))
        emp_group_cov = float(np.mean(group_covered))

        set_sizes = [len(s) for s in pred_sets]
        avg_set_size = float(np.mean(set_sizes))
        singleton_pct = float(np.mean([sz == 1 for sz in set_sizes])) * 100.0
        ambiguous_pct = float(np.mean([sz == 2 for sz in set_sizes])) * 100.0
        empty_pct = float(np.mean([sz == 0 for sz in set_sizes])) * 100.0

        return {
            "nominal_target_coverage": 1.0 - self.alpha,
            "empirical_group_coverage": round(emp_group_cov, 4),
            "empirical_pair_coverage": round(emp_pair_cov, 4),
            "calibration_quantile": round(self.calib_quantile, 6),
            "is_infinite_quantile": self.is_infinite_quantile,
            "calibration_k_rank": self.k_rank,
            "calibration_groups_total": self.m_groups,
            "calibration_groups_selected": self.m_groups,
            "test_groups_total": len(unique_test_groups),
            "test_groups_selected": len(unique_test_groups),
            "test_pairs_total": len(y_true),
            "test_pairs_selected": len(y_true),
            "average_prediction_set_size": round(avg_set_size, 4),
            "singleton_set_percentage": round(singleton_pct, 2),
            "ambiguous_pair_percentage": round(ambiguous_pct, 2),
            "empty_set_percentage": round(empty_pct, 2),
            "prediction_sets": pred_sets,
            "exchangeability_unit": "premise_group",
            "declared_limitations": (
                "Finite-sample coverage guarantees apply strictly at the premise-group level. "
                "Unconditional pair-level coverage is not mathematically guaranteed under group clustering."
            ),
        }

