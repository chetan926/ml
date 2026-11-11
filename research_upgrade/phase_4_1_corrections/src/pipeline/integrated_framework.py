"""
Phase 4.1 Corrected Integrated Multi-View Framework for Text Modality.
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection.

Methodological Guarantees:
1. Strict Split Isolation:
   - Stage-1 Scorer & Stage-2 Verifier: fit strictly on D_train (N=4,000 pairs).
   - Platt Probability Calibrator: fit strictly on D_val (N=1,500 pairs).
   - Group Conformal Predictor: calibrated on 254 premise groups from D_val.
   - Operating Thresholds (tau_low, tau_high, entropy_thresh): tuned on D_val only.
   - D_test (N=3,000 pairs, 1,015 premise groups): strictly out-of-sample evaluation.
2. Authentic Evidence Dossiers:
   - Character offsets directly mapped from genuine raw text without synthetic text.
   - Ground truth labels omitted from operational reviewer dossiers.
"""

import time
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from research_upgrade.src.calibration.calibrators import PlattScaler
from research_upgrade.src.calibration.uncertainty import compute_predictive_entropy
from research_upgrade.src.calibration.calibration_metrics import compute_calibration_metrics
from research_upgrade.phase_4_1_corrections.src.calibration.group_conformal import (
    CorrectedGroupConformalPredictor,
)
from research_upgrade.phase_4_1_corrections.src.explainability.evidence_extractor import (
    CorrectedTextEvidenceExtractor,
)
from research_upgrade.phase_4_1_corrections.src.explainability.dossier_generator import (
    generate_pair_evidence_dossier,
    compute_dossier_metrics,
)
from research_upgrade.phase_4_1_corrections.src.evaluation.statistical_testing import (
    compute_comprehensive_classification_metrics,
    compute_group_aware_bootstrap_ci,
)


class CorrectedIntegratedTextPipeline:
    """
    End-to-end corrected multi-view detection pipeline for the text domain.
    """

    def __init__(
        self,
        random_state: int = 42,
        alpha: float = 0.05,
        tau_low: float = 0.20,
        tau_high: float = 0.70,
        entropy_threshold: float = 0.60,
    ):
        self.random_state = random_state
        self.alpha = alpha
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.entropy_threshold = entropy_threshold

        self.feature_columns = [
            "text_word_tfidf_sim",
            "text_word_jaccard",
            "text_word_overlap_ratio",
            "text_char_tfidf_sim",
            "text_char_jaccard_3gram",
            "text_char_lcs_ratio",
            "text_semantic_minilm_sim",
            "text_length_ratio",
            "text_token_count_delta",
            "text_sentence_delta",
            "text_ttr_sim",
        ]

        self.s1_model = LogisticRegression(C=1.0, max_iter=1000, random_state=random_state)
        self.calibrator = PlattScaler()
        self.conformal_predictor = CorrectedGroupConformalPredictor(alpha=alpha)
        self.s2_model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=random_state)
        self.evidence_extractor = CorrectedTextEvidenceExtractor(min_span_chars=15)
        self.is_fitted = False

    def fit(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        raw_val_texts: Optional[pd.DataFrame] = None,
    ) -> "CorrectedIntegratedTextPipeline":
        """
        Fits Stage 1, Stage 2, Platt Calibrator, and Conformal Predictor.
        """
        X_tr = df_train[self.feature_columns].values
        y_tr = df_train["ground_truth_label"].values.astype(int)

        X_val = df_val[self.feature_columns].values
        y_val = df_val["ground_truth_label"].values.astype(int)

        # 1. Fit Stage-1 Multi-View Scorer on D_train
        self.s1_model.fit(X_tr, y_tr)

        # 2. Fit Platt Scaler on independent D_val Stage-1 scores
        p_val_s1 = self.s1_model.predict_proba(X_val)[:, 1]
        self.calibrator.fit(p_val_s1, y_val)

        # 3. Fit Stage-2 Verifier on D_train
        self.s2_model.fit(X_tr, y_tr)

        # 4. Calibrate Group Conformal Predictor on 254 validation premise groups
        if raw_val_texts is not None and "reference_text" in raw_val_texts:
            val_prems = raw_val_texts["reference_text"].values
        else:
            val_prems = df_val["premise_group_id"].values if "premise_group_id" in df_val else np.arange(len(y_val))

        p_val_cal = self.calibrator.predict_positive_proba(p_val_s1)
        unique_prems = np.unique(val_prems)
        rng = np.random.RandomState(self.random_state)
        shuffled = rng.permutation(unique_prems)
        cal_groups = shuffled[:254]
        cal_mask = np.isin(val_prems, cal_groups)

        self.conformal_predictor.calibrate(
            val_prems[cal_mask],
            y_val[cal_mask],
            p_val_cal[cal_mask],
        )

        self.is_fitted = True
        return self

    def predict_test(
        self,
        df_test: pd.DataFrame,
        raw_test_texts: Optional[pd.DataFrame] = None,
        max_dossiers_to_generate: int = 50,
    ) -> Dict[str, Any]:
        """
        Evaluates the full pipeline on held-out D_test.
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before predict_test().")

        X_te = df_test[self.feature_columns].values
        y_te = df_test["ground_truth_label"].values.astype(int)
        n_test = len(y_te)

        if raw_test_texts is not None and "reference_text" in raw_test_texts:
            test_prems = raw_test_texts["reference_text"].values
        else:
            test_prems = df_test["premise_group_id"].values

        # Stage-1 Scores
        s1_scores = self.s1_model.predict_proba(X_te)[:, 1]

        # Platt Calibrated Probabilities
        cal_probs = self.calibrator.predict_positive_proba(s1_scores)

        # Predictive Entropy
        entropies = compute_predictive_entropy(cal_probs)

        # Routing decisions
        states = [""] * n_test
        s2_scores = np.zeros(n_test, dtype=float)
        final_preds = np.zeros(n_test, dtype=int)

        s2_batch_indices = []
        for i in range(n_test):
            p = cal_probs[i]
            ent = entropies[i]
            if p < self.tau_low and ent < self.entropy_threshold:
                states[i] = "LOW_RISK_PRUNED"
                s2_scores[i] = 0.0
                final_preds[i] = 0
            elif p > self.tau_high and ent < self.entropy_threshold:
                states[i] = "HIGH_RISK_FLAGGED"
                s2_scores[i] = 1.0
                final_preds[i] = 1
            else:
                states[i] = "UNCERTAIN_ESCALATED"
                s2_batch_indices.append(i)

        if s2_batch_indices:
            X_esc = X_te[s2_batch_indices]
            esc_scores = self.s2_model.predict_proba(X_esc)[:, 1]
            for local_idx, global_idx in enumerate(s2_batch_indices):
                score = float(esc_scores[local_idx])
                s2_scores[global_idx] = score
                final_preds[global_idx] = 1 if score >= 0.5 else 0

        # Conformal evaluation on all 1,015 test premise groups
        conformal_summary = self.conformal_predictor.evaluate_test_coverage(
            test_prems,
            y_te,
            cal_probs,
        )

        # Classification & calibration metrics
        clf_metrics = compute_comprehensive_classification_metrics(y_te, final_preds, cal_probs)
        calib_metrics = compute_calibration_metrics(y_te, cal_probs)
        boot_ci = compute_group_aware_bootstrap_ci(test_prems, y_te, final_preds, n_resamples=1000, random_state=self.random_state)

        # Cost accounting directly from states
        n_pruned = sum(1 for s in states if s == "LOW_RISK_PRUNED")
        n_flagged = sum(1 for s in states if s == "HIGH_RISK_FLAGGED")
        n_escalated = len(s2_batch_indices)
        calls_avoided = n_pruned + n_flagged

        # Dossier generation for sampled candidates
        flagged_or_escalated_idx = [i for i in range(n_test) if states[i] in ("HIGH_RISK_FLAGGED", "UNCERTAIN_ESCALATED")]
        sample_indices = flagged_or_escalated_idx[:max_dossiers_to_generate]

        sample_dossiers = []
        dossier_times_ms = []

        for idx in sample_indices:
            t0 = time.perf_counter_ns()
            ref_txt = str(raw_test_texts.iloc[idx]["reference_text"]) if raw_test_texts is not None else ""
            sus_txt = str(raw_test_texts.iloc[idx]["suspicious_text"]) if raw_test_texts is not None else ""
            ev = self.evidence_extractor.generate_text_evidence(ref_txt, sus_txt)

            d = generate_pair_evidence_dossier(
                pair_id=str(df_test.iloc[idx].get("pair_id", f"TXT_{idx:06d}")),
                modality="text",
                stage1_score=float(s1_scores[idx]),
                calibrated_probability=float(cal_probs[idx]),
                predictive_entropy=float(entropies[idx]),
                decision_state=states[idx],
                final_prediction=int(final_preds[idx]),
                extracted_evidence=ev,
                stage2_score=float(s2_scores[idx]) if states[idx] == "UNCERTAIN_ESCALATED" else None,
                premise_group_id=str(test_prems[idx]),
            )
            sample_dossiers.append(d)
            t1 = time.perf_counter_ns()
            dossier_times_ms.append((t1 - t0) / 1e6)

        dossier_metrics = compute_dossier_metrics(
            total_flagged_escalated_pairs=len(flagged_or_escalated_idx),
            generated_dossiers=sample_dossiers,
            explanation_latencies_ms=dossier_times_ms,
        )

        predictions_df = pd.DataFrame({
            "pair_id": df_test.get("pair_id", [f"TXT_{i:06d}" for i in range(n_test)]),
            "premise_group_id": test_prems,
            "ground_truth_label": y_te,
            "stage1_score": np.round(s1_scores, 4),
            "calibrated_prob": np.round(cal_probs, 4),
            "predictive_entropy": np.round(entropies, 4),
            "routing_decision_state": states,
            "stage2_score": np.round(s2_scores, 4),
            "final_prediction": final_preds,
            "conformal_prediction_set": [str(s) for s in conformal_summary["prediction_sets"]],
            "conformal_pair_covered": [(y_te[i] in conformal_summary["prediction_sets"][i]) for i in range(n_test)],
        })

        return {
            "classification_metrics": clf_metrics,
            "calibration_metrics": calib_metrics,
            "group_conformal_summary": {k: v for k, v in conformal_summary.items() if k != "prediction_sets"},
            "group_bootstrap_ci_95": boot_ci,
            "routing_cost": {
                "total_pairs": n_test,
                "pruned_count": n_pruned,
                "pruned_ratio": round(n_pruned / n_test, 4),
                "flagged_direct_count": n_flagged,
                "flagged_direct_ratio": round(n_flagged / n_test, 4),
                "escalated_count": n_escalated,
                "escalated_ratio": round(n_escalated / n_test, 4),
                "stage2_calls_avoided_count": calls_avoided,
                "stage2_calls_avoided_percent": round((calls_avoided / n_test) * 100.0, 2),
            },
            "dossier_metrics": dossier_metrics,
            "sample_dossiers": sample_dossiers[:5],
            "predictions_dataframe": predictions_df,
        }
