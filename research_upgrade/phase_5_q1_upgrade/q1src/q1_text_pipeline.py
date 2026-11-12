"""
Phase 5 Proposed Text Pipeline (Q1 Upgrade).

Changes vs Phase 4.1 `CorrectedIntegratedTextPipeline`:
    1. Feature space gains the offline LSA dense-semantic view (12 features).
    2. Stage-2 is the heavy `CrossStylePairVerifier` (deep MLP over the
       pair-interaction space) instead of a depth-8 RandomForest.
    3. The tri-band operating point (tau_low, tau_high, entropy threshold,
       Stage-2 decision threshold) is TUNED ON D_val ONLY under an explicit
       Stage-2 call budget cap, instead of being fixed a priori.
    4. Conformal coverage is reported with Wilson score confidence intervals.

Isolation guarantees (unchanged): S1/S2 fitted on D_train; Platt calibrator,
operating point, and group-conformal quantile fitted on D_val; D_test touched
exactly once for final evaluation.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from research_upgrade.src.calibration.calibrators import PlattScaler
from research_upgrade.src.calibration.uncertainty import compute_predictive_entropy
from research_upgrade.src.calibration.calibration_metrics import compute_calibration_metrics
from research_upgrade.phase_4_1_corrections.src.calibration.group_conformal import (
    CorrectedGroupConformalPredictor,
)
from research_upgrade.phase_4_1_corrections.src.evaluation.statistical_testing import (
    compute_comprehensive_classification_metrics,
    compute_group_aware_bootstrap_ci,
)
from research_upgrade.phase_5_q1_upgrade.q1src.heavy_verifiers import CrossStylePairVerifier
from research_upgrade.phase_5_q1_upgrade.q1src.lsa_semantic import LsaSemanticEncoder
from research_upgrade.phase_5_q1_upgrade.q1src.q1_stats_extras import (
    apply_routing,
    tune_routing_operating_point,
    wilson_score_ci,
)

PHASE5_TEXT_FEATURES = [
    "text_word_tfidf_sim", "text_word_jaccard", "text_word_overlap_ratio",
    "text_char_tfidf_sim", "text_char_jaccard_3gram", "text_char_lcs_ratio",
    "text_semantic_minilm_sim", "text_lsa_semantic_sim",
    "text_length_ratio", "text_token_count_delta",
    "text_sentence_delta", "text_ttr_sim",
]


class Phase5TextPipeline:
    """Full proposed framework (C1–C4) with heavy Stage-2 and tuned operating point."""

    def __init__(
        self,
        random_state: int = 42,
        alpha: float = 0.05,
        stage2_budget_cap: float = 0.70,
        n_conformal_groups: int = 254,
    ):
        self.random_state = random_state
        self.alpha = alpha
        self.stage2_budget_cap = stage2_budget_cap
        self.n_conformal_groups = n_conformal_groups

        self.feature_columns = PHASE5_TEXT_FEATURES
        self.lsa_encoder = LsaSemanticEncoder(n_components=128, random_state=random_state)
        self.s1_model = LogisticRegression(C=1.0, max_iter=1000, random_state=random_state)
        self.calibrator = PlattScaler()
        self.conformal_predictor = CorrectedGroupConformalPredictor(alpha=alpha)
        self.s2_model: Optional[CrossStylePairVerifier] = None
        self.operating_point_: Dict[str, Any] = {}
        self.is_fitted = False

    # ------------------------------------------------------------------ fit
    def _build_s2_inputs(
        self,
        df: pd.DataFrame,
        emb_ref: np.ndarray,
        emb_sus: np.ndarray,
    ) -> np.ndarray:
        inter = LsaSemanticEncoder.interaction_features(emb_ref, emb_sus)
        mv = df[self.feature_columns].values.astype(np.float64)
        return np.hstack([inter, mv])

    def fit(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        raw_train_texts: pd.DataFrame,
        raw_val_texts: pd.DataFrame,
    ) -> "Phase5TextPipeline":
        y_tr = df_train["ground_truth_label"].values.astype(int)
        y_val = df_val["ground_truth_label"].values.astype(int)

        # 1. Fit offline dense semantic encoder on TRAIN corpus only (leak-free)
        train_corpus = (
            list(raw_train_texts["reference_text"].values)
            + list(raw_train_texts["suspicious_text"].values)
        )
        self.lsa_encoder.fit(train_corpus)

        emb_tr = (
            self.lsa_encoder.encode(list(raw_train_texts["reference_text"].values)),
            self.lsa_encoder.encode(list(raw_train_texts["suspicious_text"].values)),
        )
        emb_val = (
            self.lsa_encoder.encode(list(raw_val_texts["reference_text"].values)),
            self.lsa_encoder.encode(list(raw_val_texts["suspicious_text"].values)),
        )
        df_train = df_train.copy()
        df_val = df_val.copy()
        df_train["text_lsa_semantic_sim"] = np.round(
            np.sum(emb_tr[0] * emb_tr[1], axis=1), 6
        ).clip(0.0, 1.0)
        df_val["text_lsa_semantic_sim"] = np.round(
            np.sum(emb_val[0] * emb_val[1], axis=1), 6
        ).clip(0.0, 1.0)
        self._train_df_enriched, self._val_df_enriched = df_train, df_val

        # 2. Stage-1 + Platt
        self.s1_model.fit(df_train[self.feature_columns].values, y_tr)
        p_val_s1 = self.s1_model.predict_proba(df_val[self.feature_columns].values)[:, 1]
        self.calibrator.fit(p_val_s1, y_val)

        # 3. Heavy Stage-2 on interaction space (train only)
        X_tr_s2 = self._build_s2_inputs(df_train, emb_tr[0], emb_tr[1])
        self.s2_model = CrossStylePairVerifier(random_state=self.random_state)
        self.s2_model.fit(X_tr_s2, y_tr)

        # 4. Validation operating-point tuning (budget-capped)
        X_val_s2 = self._build_s2_inputs(df_val, emb_val[0], emb_val[1])
        s2_val = self.s2_model.predict_proba(X_val_s2)
        p_val_cal = self.calibrator.predict_positive_proba(p_val_s1)
        ent_val = compute_predictive_entropy(p_val_cal)
        self.operating_point_ = tune_routing_operating_point(
            y_val=y_val,
            p_val_calibrated=p_val_cal,
            entropy_val=ent_val,
            s2_scores_val=s2_val,
            stage2_budget_cap=self.stage2_budget_cap,
        )

        # 5. Group conformal calibration on val groups
        val_prems = raw_val_texts["reference_text"].values
        unique_prems = np.unique(val_prems)
        rng = np.random.RandomState(self.random_state)
        shuffled = rng.permutation(unique_prems)
        cal_groups = shuffled[: self.n_conformal_groups]
        cal_mask = np.isin(val_prems, cal_groups)
        self.conformal_predictor.calibrate(
            val_prems[cal_mask], y_val[cal_mask], p_val_cal[cal_mask]
        )
        self.is_fitted = True
        return self

    # ------------------------------------------------------------ evaluation
    def evaluate_test(
        self,
        df_test: pd.DataFrame,
        raw_test_texts: pd.DataFrame,
        compute_ci: bool = True,
    ) -> Dict[str, Any]:
        if not self.is_fitted:
            raise RuntimeError("Phase5TextPipeline must be fitted first.")
        y_te = df_test["ground_truth_label"].values.astype(int)
        n_test = len(y_te)

        df_test = df_test.copy()
        emb_te = (
            self.lsa_encoder.encode(list(raw_test_texts["reference_text"].values)),
            self.lsa_encoder.encode(list(raw_test_texts["suspicious_text"].values)),
        )
        df_test["text_lsa_semantic_sim"] = np.round(
            np.sum(emb_te[0] * emb_te[1], axis=1), 6
        ).clip(0.0, 1.0)

        s1_scores = self.s1_model.predict_proba(df_test[self.feature_columns].values)[:, 1]
        cal_probs = self.calibrator.predict_positive_proba(s1_scores)
        entropies = compute_predictive_entropy(cal_probs)
        X_te_s2 = self._build_s2_inputs(df_test, emb_te[0], emb_te[1])
        s2_scores_all = self.s2_model.predict_proba(X_te_s2)

        op = self.operating_point_
        final_preds, states, routing_cost = apply_routing(
            cal_probs, entropies, s2_scores_all,
            tau_low=op["tau_low"], tau_high=op["tau_high"],
            entropy_threshold=op["entropy_threshold"],
            stage2_decision_threshold=op["stage2_decision_threshold"],
        )

        test_prems = raw_test_texts["reference_text"].values
        conformal_summary = self.conformal_predictor.evaluate_test_coverage(
            test_prems, y_te, cal_probs
        )
        # Wilson CIs on empirical coverage
        n_groups = int(len(np.unique(test_prems)))
        group_cov_hat = conformal_summary["empirical_group_coverage"]
        k_groups = int(round(group_cov_hat * n_groups))
        _, g_lo, g_hi = wilson_score_ci(k_groups, n_groups)
        pair_cov_hat = conformal_summary["empirical_pair_coverage"]
        _, p_lo, p_hi = wilson_score_ci(int(round(pair_cov_hat * n_test)), n_test)
        conformal_summary["wilson_ci_95"] = {
            "group_coverage": {"n": n_groups, "wilson_lower": round(g_lo, 4),
                                "wilson_upper": round(g_hi, 4),
                                "nominal_target": 1.0 - self.alpha},
            "pair_coverage": {"n": n_test, "wilson_lower": round(p_lo, 4),
                               "wilson_upper": round(p_hi, 4),
                               "nominal_target": 1.0 - self.alpha},
        }

        clf_metrics = compute_comprehensive_classification_metrics(y_te, final_preds, cal_probs)
        calib_metrics = compute_calibration_metrics(y_te, cal_probs)
        boot_ci = (
            compute_group_aware_bootstrap_ci(
                test_prems, y_te, final_preds, n_resamples=1000, random_state=self.random_state
            )
            if compute_ci
            else {}
        )

        # Honest per-pair latencies
        s2_latency = self.s2_model.measure_single_pair_latency_ms(X_te_s2[:1])

        predictions_df = pd.DataFrame({
            "pair_id": df_test.get("pair_id", [f"TXT_{i:06d}" for i in range(n_test)]),
            "premise_group_id": test_prems,
            "ground_truth_label": y_te,
            "stage1_score": np.round(s1_scores, 4),
            "calibrated_prob": np.round(cal_probs, 4),
            "predictive_entropy": np.round(entropies, 4),
            "routing_decision_state": states,
            "stage2_score": np.round(s2_scores_all, 4),
            "final_prediction": final_preds,
        })

        return {
            "classification_metrics": clf_metrics,
            "calibration_metrics": calib_metrics,
            "group_conformal_summary": {k: v for k, v in conformal_summary.items()
                                          if k != "prediction_sets"},
            "group_bootstrap_ci_95": boot_ci,
            "routing_cost": routing_cost,
            "operating_point": op,
            "stage2_heavy_verifier": {
                **self.s2_model.metadata(),
                "measured_single_pair_latency_ms": round(s2_latency, 6),
            },
            "s1_scores": s1_scores,
            "calibrated_probs": cal_probs,
            "entropies": entropies,
            "s2_scores_all": s2_scores_all,
            "final_predictions": final_preds,
            "decision_states": states,
            "predictions_dataframe": predictions_df,
            "_enriched_test": df_test,
        }
