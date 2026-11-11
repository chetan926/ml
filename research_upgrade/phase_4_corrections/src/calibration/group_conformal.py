"""
Phase 4 Corrected Group Conformal Prediction Module.
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection.

Methodological Corrections Applied:
1. Exact finite-sample conformal quantile: k = ceil((m + 1) * (1 - alpha)).
2. Exact infinite quantile handling: If k > m, q_hat = inf (sets {0, 1} guaranteed).
3. Independent calibration predictions: Uses independent calibration partitions (not training data).
4. Group supremum nonconformity: Exactly one nonconformity score per valid premise group: R_g = max_{i in g} (1 - P(Y = y_i | x_i)).
5. Symmetric selection filtering: Consistent Stage-1 selection masking applied to both calibration and evaluation cohorts.
6. Separate, honest reporting of group-level and pair-level coverage.
7. Explicit documentation of conditional post-selection exchangeability assumptions and limitations.
"""

from typing import Dict, Any, List, Set, Optional
import numpy as np
import pandas as pd


class CorrectedGroupConformalPredictor:
    """
    Finite-sample valid group-level conformal predictor.
    Guarantees 1 - alpha group-level coverage over exchangeable group sampling units.
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = float(alpha)
        self.calib_quantile: float = float("inf")
        self.k_rank: int = 0
        self.m_groups: int = 0
        self.m_groups_total: int = 0
        self.n_calib_pairs_total: int = 0
        self.n_calib_pairs_selected: int = 0
        self.is_infinite_quantile: bool = False
        self.is_calibrated: bool = False

    def calibrate(
        self,
        group_ids: np.ndarray,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        selected_mask: Optional[np.ndarray] = None,
    ) -> "CorrectedGroupConformalPredictor":
        """
        Calibrates on independent calibration groups using the exact conformal quantile.
        """
        group_ids = np.asarray(group_ids)
        y_true = np.asarray(y_true, dtype=np.int32)
        y_prob = np.asarray(y_prob, dtype=np.float64)

        self.n_calib_pairs_total = len(y_true)
        self.m_groups_total = len(np.unique(group_ids))

        sel = (
            np.ones(len(y_true), dtype=bool)
            if selected_mask is None
            else np.asarray(selected_mask, dtype=bool)
        )
        self.n_calib_pairs_selected = int(np.sum(sel))

        df_cal = pd.DataFrame({
            "group_id": group_ids,
            "y_true": y_true,
            "y_prob": y_prob,
            "selected": sel,
        })

        df_sel = df_cal[df_cal["selected"]].copy()
        if len(df_sel) == 0:
            raise ValueError("No pairs survived selection for conformal calibration.")

        # Nonconformity score for pair: 1 - P(Y = y_true | x)
        p_true = np.where(df_sel["y_true"] == 1, df_sel["y_prob"], 1.0 - df_sel["y_prob"])
        df_sel["nonconf"] = 1.0 - p_true

        # Group-level supremum nonconformity score
        group_scores_series = df_sel.groupby("group_id")["nonconf"].max()
        group_scores = np.sort(group_scores_series.values)
        self.m_groups = len(group_scores)

        if self.m_groups == 0:
            raise ValueError("Zero valid groups present post-selection in calibration cohort.")

        # Exact finite-sample quantile rank: k = ceil((m + 1) * (1 - alpha))
        k = int(np.ceil((self.m_groups + 1) * (1.0 - self.alpha)))
        self.k_rank = k

        if k > self.m_groups:
            # When required rank exceeds sample size, finite-sample coverage requires an infinite quantile
            self.calib_quantile = float("inf")
            self.is_infinite_quantile = True
        else:
            # 1-indexed rank k corresponds to 0-indexed position k - 1
            self.calib_quantile = float(group_scores[k - 1])
            self.is_infinite_quantile = False

        self.is_calibrated = True
        return self

    def predict_prediction_sets(self, y_prob: np.ndarray) -> List[Set[int]]:
        """
        Generates conformal prediction sets C(x) = {y in {0, 1} : (1 - P(Y=y|x)) <= q_hat}.
        """
        if not self.is_calibrated:
            raise RuntimeError("Conformal predictor must be calibrated before generating sets.")

        y_prob = np.asarray(y_prob, dtype=np.float64)
        if self.is_infinite_quantile or not np.isfinite(self.calib_quantile):
            return [{0, 1} for _ in range(len(y_prob))]

        sets: List[Set[int]] = []
        for p in y_prob:
            pset: Set[int] = set()
            # Nonconf(0) = 1 - (1 - p) = p
            if p <= self.calib_quantile:
                pset.add(0)
            # Nonconf(1) = 1 - p
            if (1.0 - p) <= self.calib_quantile:
                pset.add(1)
            sets.append(pset)
        return sets

    def evaluate_test_coverage(
        self,
        group_ids: np.ndarray,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        selected_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates empirical coverage separately at group level and pair level on test partition.
        """
        group_ids = np.asarray(group_ids)
        y_true = np.asarray(y_true, dtype=np.int32)
        y_prob = np.asarray(y_prob, dtype=np.float64)

        n_test_pairs_total = len(y_true)
        n_test_groups_total = len(np.unique(group_ids))

        sel = (
            np.ones(len(y_true), dtype=bool)
            if selected_mask is None
            else np.asarray(selected_mask, dtype=bool)
        )
        n_test_pairs_selected = int(np.sum(sel))

        pred_sets = self.predict_prediction_sets(y_prob)

        df_eval = pd.DataFrame({
            "group_id": group_ids,
            "y_true": y_true,
            "selected": sel,
            "pred_set": pred_sets,
        })
        df_eval["pair_covered"] = [y in s for y, s in zip(df_eval["y_true"], df_eval["pred_set"])]
        df_eval["set_size"] = [len(s) for s in df_eval["pred_set"]]

        df_sel = df_eval[df_eval["selected"]].copy()
        if len(df_sel) == 0:
            return {
                "error": "No test pairs survived selection filter.",
                "test_pairs_total": n_test_pairs_total,
                "test_pairs_selected": 0,
            }

        n_test_groups_selected = len(df_sel["group_id"].unique())

        # Pair-level coverage
        pair_coverage = float(df_sel["pair_covered"].mean())
        avg_set_size = float(df_sel["set_size"].mean())
        singleton_pct = float((df_sel["set_size"] == 1).mean()) * 100.0
        ambiguous_pct = float((df_sel["set_size"] == 2).mean()) * 100.0
        empty_pct = float((df_sel["set_size"] == 0).mean()) * 100.0

        # Group-level coverage: Group g is covered iff ALL pairs in group g are covered
        group_covered = df_sel.groupby("group_id")["pair_covered"].all()
        group_coverage = float(group_covered.mean())

        return {
            "nominal_target_coverage": round(1.0 - self.alpha, 4),
            "empirical_group_coverage": round(group_coverage, 4),
            "empirical_pair_coverage": round(pair_coverage, 4),
            "calibration_quantile": (
                "inf" if self.is_infinite_quantile else round(self.calib_quantile, 6)
            ),
            "is_infinite_quantile": self.is_infinite_quantile,
            "calibration_k_rank": self.k_rank,
            "calibration_groups_total": self.m_groups_total,
            "calibration_groups_selected": self.m_groups,
            "calibration_pairs_total": self.n_calib_pairs_total,
            "calibration_pairs_selected": self.n_calib_pairs_selected,
            "test_groups_total": n_test_groups_total,
            "test_groups_selected": n_test_groups_selected,
            "test_pairs_total": n_test_pairs_total,
            "test_pairs_selected": n_test_pairs_selected,
            "average_prediction_set_size": round(avg_set_size, 4),
            "singleton_set_percentage": round(singleton_pct, 2),
            "ambiguous_pair_percentage": round(ambiguous_pct, 2),
            "empty_set_percentage": round(empty_pct, 2),
            "exchangeability_unit": "premise_group",
            "exchangeability_assumptions": (
                "Premise groups are assumed independently sampled from the target domain. "
                "Post-selection exchangeability holds conditionally if the identical deterministic "
                "selection procedure is applied to both calibration and test cohorts."
            ),
            "declared_limitations": (
                "Finite-sample coverage guarantees apply strictly at the premise-group level. "
                "Unconditional pair-level coverage is not mathematically guaranteed under group clustering "
                "or post-selection truncation."
            ),
        }

