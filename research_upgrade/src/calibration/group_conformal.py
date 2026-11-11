"""
Group Conformal Prediction Module (Phase 3.4).

Implements finite-sample valid group-level conformal prediction for academic integrity detection:
- Text: Premise-group exchangeability units
- Nonconformity scores: Group-maximum nonconformity under Stage-1 frozen selection
- Finite-sample quantile computation with exact ceiling adjustment
- Prediction set generation: {0}, {1}, {0, 1}, or empty set
- Explicit exchangeability assumptions and limitation tracking
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Set, Optional


class TextGroupConformalPredictor:
    """
    Group-Level Conformal Predictor for Text Modality.
    Premise groups act as the primary sampling and exchangeability units.
    """

    def __init__(self, alpha: float = 0.05):
        """
        Args:
            alpha: Significance level (default 0.05 for 95% nominal coverage).
        """
        self.alpha = alpha
        self.calib_quantile: float = 1.0
        self.m_groups: int = 0
        self.is_calibrated: bool = False

    def calibrate(
        self,
        group_ids: np.ndarray,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        selected_mask: Optional[np.ndarray] = None,
    ) -> "TextGroupConformalPredictor":
        """
        Calibrates on validation premise groups.

        Args:
            group_ids: Group identifiers for each pair (e.g. premise_group_id).
            y_true: Ground truth binary labels.
            y_prob: Calibrated positive-class probabilities P(Y=1|x).
            selected_mask: Boolean mask indicating pairs escalated by Stage-1 screening.
        """
        df = pd.DataFrame({
            "group_id": np.asarray(group_ids),
            "y_true": np.asarray(y_true, dtype=np.int32),
            "y_prob": np.asarray(y_prob, dtype=np.float64),
            "selected": (
                np.ones(len(y_true), dtype=bool)
                if selected_mask is None
                else np.asarray(selected_mask, dtype=bool)
            ),
        })

        # Only evaluate on selected pairs (post Stage-1)
        df_sel = df[df["selected"]].copy()
        if len(df_sel) == 0:
            raise ValueError("No pairs survived Stage-1 selection for conformal calibration.")

        # Nonconformity score for true label: 1 - P(Y = y_true | x)
        p_true = np.where(df_sel["y_true"] == 1, df_sel["y_prob"], 1.0 - df_sel["y_prob"])
        df_sel["nonconf"] = 1.0 - p_true

        # Group-maximum nonconformity score: R_g = max_{i in group} (1 - P(Y = y_i | x_i))
        group_scores = df_sel.groupby("group_id")["nonconf"].max().values
        self.m_groups = len(group_scores)

        if self.m_groups == 0:
            raise ValueError("Zero valid nonempty groups found in calibration set.")

        # Finite-sample quantile: ceil((m + 1) * (1 - alpha)) / m
        p_level = np.ceil((self.m_groups + 1) * (1.0 - self.alpha)) / self.m_groups
        p_level = min(1.0, max(0.0, p_level))

        # Quantile via numpy percentile
        self.calib_quantile = float(np.percentile(group_scores, p_level * 100.0, method="higher"))
        self.is_calibrated = True
        return self

    def predict_prediction_sets(
        self,
        y_prob: np.ndarray,
    ) -> List[Set[int]]:
        """
        Generates individual prediction sets C_alpha(x) for each pair:
        C_alpha(x) = { y in {0, 1} : (1 - P(Y=y | x)) <= q_conf }.
        """
        if not self.is_calibrated:
            raise RuntimeError("Conformal predictor must be calibrated before prediction.")

        y_prob = np.asarray(y_prob, dtype=np.float64)
        p0 = 1.0 - y_prob
        p1 = y_prob

        sets = []
        for i in range(len(y_prob)):
            s = set()
            if (1.0 - p0[i]) <= self.calib_quantile:
                s.add(0)
            if (1.0 - p1[i]) <= self.calib_quantile:
                s.add(1)
            sets.append(s)
        return sets

    def evaluate_test_coverage(
        self,
        group_ids: np.ndarray,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        selected_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates empirical coverage at group level and pair level on test partition.
        """
        pred_sets = self.predict_prediction_sets(y_prob)
        y_true = np.asarray(y_true, dtype=np.int32)
        group_ids = np.asarray(group_ids)
        sel = (
            np.ones(len(y_true), dtype=bool)
            if selected_mask is None
            else np.asarray(selected_mask, dtype=bool)
        )

        df_eval = pd.DataFrame({
            "group_id": group_ids,
            "y_true": y_true,
            "selected": sel,
            "pred_set": pred_sets,
        })
        df_eval["covered"] = [y in s for y, s in zip(df_eval["y_true"], df_eval["pred_set"])]
        df_eval["set_size"] = [len(s) for s in df_eval["pred_set"]]

        df_sel = df_eval[df_eval["selected"]].copy()
        if len(df_sel) == 0:
            return {"error": "No test pairs escalated."}

        # Pair-level metrics
        pair_coverage = float(df_sel["covered"].mean())
        avg_set_size = float(df_sel["set_size"].mean())
        singleton_pct = float((df_sel["set_size"] == 1).mean())
        ambiguous_pct = float((df_sel["set_size"] == 2).mean())
        empty_pct = float((df_sel["set_size"] == 0).mean())

        # Group-level coverage: group is covered iff ALL selected pairs in group are covered
        group_covered = df_sel.groupby("group_id")["covered"].all()
        group_coverage = float(group_covered.mean())
        n_test_groups = len(group_covered)

        return {
            "nominal_target_coverage": round(1.0 - self.alpha, 4),
            "empirical_group_coverage": round(group_coverage, 4),
            "empirical_pair_coverage": round(pair_coverage, 4),
            "calibration_quantile": round(self.calib_quantile, 6),
            "calibration_groups_count": self.m_groups,
            "test_groups_count": n_test_groups,
            "test_pairs_evaluated": len(df_sel),
            "average_prediction_set_size": round(avg_set_size, 4),
            "singleton_set_percentage": round(singleton_pct * 100.0, 2),
            "ambiguous_pair_percentage": round(ambiguous_pct * 100.0, 2),
            "empty_set_percentage": round(empty_pct * 100.0, 2),
            "exchangeability_unit": "premise_group",
            "exchangeability_assumptions": (
                "Groups conditionally exchangeable post Stage-1 selection given identical "
                "selection procedure applied to calibration and test partitions."
            ),
            "declared_limitations": (
                "Guarantees valid coverage at premise-group level under post-selection "
                "exchangeability. Does not guarantee unconditional pair-level coverage."
            ),
        }
