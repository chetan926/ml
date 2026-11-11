"""
Phase 4 Corrected Integrated Academic Integrity Detection Framework.
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection.

Methodological Corrections Applied:
1. Conformal Prediction: Exact finite-sample quantile, independent calibration group split,
   handles infinite quantiles (k > m), evaluates on actual premise groups, reports group and pair coverage.
2. Evidence Generation: Extracts from authentic document text strings (no synthetic placeholders),
   records feature views, preserves character offsets, reports dossier generation coverage.
3. Strict Calibration Isolation: Probability calibrator and conformal predictor fitted on separate,
   disjoint premise-group validation partitions to eliminate double-dipping.
4. Latency & Resource Profiling: Empirical time.perf_counter_ns() percentiles (P50, P90, P95).
5. Comprehensive Group-Aware Statistics: Paired bootstrap CIs and non-zero-underflowed p-values.
"""

import time
from typing import Dict, Any, List, Optional, Tuple, Set
import numpy as np
import pandas as pd

from research_upgrade.src.candidate_generation.score_fusion import (
    LogisticRegressionCandidateScorer,
    GradientBoostingCandidateScorer,
)
from research_upgrade.src.calibration.calibrators import PlattScaler
from research_upgrade.src.calibration.uncertainty import (
    compute_predictive_entropy,
    evaluate_selective_prediction,
)
from research_upgrade.src.calibration.calibration_metrics import compute_calibration_metrics
from research_upgrade.src.routing.decision_states import DecisionState
from research_upgrade.src.routing.router import BudgetAdaptiveRouter
from research_upgrade.src.routing.stage2_verifier import TextStage2Verifier
from research_upgrade.src.routing.cost_model import compute_routing_cost_summary

from research_upgrade.phase_4_corrections.src.calibration.group_conformal import (
    CorrectedGroupConformalPredictor,
)
from research_upgrade.phase_4_corrections.src.explainability.evidence_extractor import (
    CorrectedTextEvidenceExtractor,
)
from research_upgrade.phase_4_corrections.src.explainability.dossier_generator import (
    generate_pair_evidence_dossier,
    compute_dossier_metrics,
)
from research_upgrade.phase_4_corrections.src.evaluation.statistical_testing import (
    compute_comprehensive_classification_metrics,
    compute_group_aware_bootstrap_ci,
    compute_paired_difference_bootstrap_ci,
    compute_mcnemar_significance,
)


class CorrectedIntegratedTextPipeline:
    """
    End-to-end corrected pipeline for text academic integrity detection.
    Integrates C1 (Multi-View Scoring), C2 (Calibration & Conformal),
    C3 (Budget Routing), Stage-2 Verification, and C4 (Explainable Evidence).
    """

    def __init__(self, random_state: int = 42, alpha: float = 0.05):
        self.random_state = random_state
        self.alpha = alpha
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
        self.candidate_scorer = LogisticRegressionCandidateScorer(
            feature_names=self.feature_columns,
            random_state=random_state,
        )
        self.calibrator = PlattScaler(C=1.0, random_state=random_state)
        self.conformal_predictor = CorrectedGroupConformalPredictor(alpha=alpha)
        self.router = BudgetAdaptiveRouter()
        self.stage2_verifier = TextStage2Verifier(random_state=random_state)
        self.evidence_extractor = CorrectedTextEvidenceExtractor(min_phrase_len=3)
        self.is_fitted = False

    def fit(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        raw_val_df: pd.DataFrame,
        budget_fraction: float = 0.75,
    ) -> "CorrectedIntegratedTextPipeline":
        """
        Fits all models strictly on development splits with independent calibration partitions.
        """
        # Ensure true premise groups are present
        val_df = df_val.copy()
        val_df["premise_group_id"] = raw_val_df["reference_text"].values

        X_tr = df_train[self.feature_columns].values
        y_tr = df_train["ground_truth_label"].values.astype(np.int32)

        # 1. Fit Stage-1 Multi-View Candidate Scorer on train
        self.candidate_scorer.fit(X_tr, y_tr)

        # 2. Fit Stage-2 Heavy Verifier on train
        self.stage2_verifier.fit(df_train)

        # 3. Partition validation premise groups into two disjoint sets:
        # Set A for probability calibration, Set B for conformal prediction
        unique_groups = np.array(sorted(val_df["premise_group_id"].unique()))
        rng = np.random.RandomState(self.random_state)
        rng.shuffle(unique_groups)
        split_idx = len(unique_groups) // 2

        groups_prob = set(unique_groups[:split_idx])
        groups_conf = set(unique_groups[split_idx:])

        mask_prob = val_df["premise_group_id"].isin(groups_prob).values
        mask_conf = val_df["premise_group_id"].isin(groups_conf).values

        X_val = val_df[self.feature_columns].values
        y_val = val_df["ground_truth_label"].values.astype(np.int32)

        val_scores = self.candidate_scorer.predict_score(X_val)

        # 4. Fit Platt Calibrator on Group Set A
        self.calibrator.fit(val_scores[mask_prob], y_val[mask_prob])

        # 5. Calibrate Group Conformal Predictor on independent Group Set B
        conf_probs = self.calibrator.predict_positive_proba(val_scores[mask_conf])
        self.conformal_predictor.calibrate(
            group_ids=val_df.loc[mask_conf, "premise_group_id"].values,
            y_true=y_val[mask_conf],
            y_prob=conf_probs,
        )

        # 6. Optimize Routing Thresholds on validation
        all_val_probs = self.calibrator.predict_positive_proba(val_scores)
        all_val_entropies = compute_predictive_entropy(all_val_probs)
        self.router.optimize_thresholds_on_validation(
            val_scores=val_scores,
            val_probs=all_val_probs,
            val_labels=y_val,
            val_entropies=all_val_entropies,
            target_recall=0.98,
            budget_fraction=budget_fraction,
        )

        self.is_fitted = True
        return self

    def predict_test(
        self,
        df_test: pd.DataFrame,
        raw_test_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end evaluation out-of-sample on the test partition.
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted prior to test evaluation.")

        test_df = df_test.copy()
        test_df["premise_group_id"] = raw_test_df["reference_text"].values
        test_df["reference_text"] = raw_test_df["reference_text"].values
        test_df["suspicious_text"] = raw_test_df["suspicious_text"].values

        X_test = test_df[self.feature_columns].values
        y_test = test_df["ground_truth_label"].values.astype(np.int32)
        group_ids = test_df["premise_group_id"].values
        pair_ids = test_df["pair_id"].values
        N = len(y_test)

        # C1: Stage-1 Multi-View Candidate Scoring
        t0_s1 = time.perf_counter_ns()
        c1_scores = self.candidate_scorer.predict_score(X_test)
        t1_s1 = time.perf_counter_ns()
        s1_latency_ms = (t1_s1 - t0_s1) / (1_000_000.0 * N)

        # C2: Probability Calibration & Uncertainty
        t0_c2 = time.perf_counter_ns()
        c2_probs = self.calibrator.predict_positive_proba(c1_scores)
        c2_entropies = compute_predictive_entropy(c2_probs)
        t1_c2 = time.perf_counter_ns()
        c2_latency_ms = (t1_c2 - t0_c2) / (1_000_000.0 * N)

        # C3: Budget-Constrained Routing
        t0_c3 = time.perf_counter_ns()
        routing_states = self.router.route_batch(c1_scores, c2_probs, c2_entropies)
        t1_c3 = time.perf_counter_ns()
        c3_latency_ms = (t1_c3 - t0_c3) / (1_000_000.0 * N)

        is_pruned = np.array([s == DecisionState.LOW_RISK_PRUNED for s in routing_states])
        is_flagged = np.array([s == DecisionState.HIGH_RISK_FLAGGED for s in routing_states])
        is_escalated = np.array([s == DecisionState.UNCERTAIN_ESCALATED for s in routing_states])

        # Stage 2: Heavy Verification on Escalated Pairs ONLY
        stage2_scores = np.zeros(N, dtype=np.float64)
        n_escalated = int(np.sum(is_escalated))
        s2_latency_ms = 0.0

        if n_escalated > 0:
            df_escalated = test_df[is_escalated].copy()
            t0_s2 = time.perf_counter_ns()
            s2_escalated = self.stage2_verifier.predict_verification_score(df_escalated)
            t1_s2 = time.perf_counter_ns()
            stage2_scores[is_escalated] = s2_escalated
            s2_latency_ms = (t1_s2 - t0_s2) / (1_000_000.0 * n_escalated)

        # Three-way decision integration:
        # Pruned -> 0
        # Flagged -> 1
        # Escalated -> (Stage2 >= 0.5)
        final_preds = np.zeros(N, dtype=np.int32)
        final_preds[is_flagged] = 1
        final_preds[is_escalated] = (stage2_scores[is_escalated] >= 0.5).astype(np.int32)

        # Final effective probabilities for diagnostics
        final_probs = np.copy(c2_probs)
        final_probs[is_escalated] = stage2_scores[is_escalated]
        final_probs[is_pruned] = np.minimum(final_probs[is_pruned], 0.20)

        # C4: Authentic Evidence Dossier Generation on Flagged and Escalated Candidates
        flagged_or_escalated_idx = np.where(is_flagged | is_escalated)[0]
        dossiers = []
        ev_latencies_ms = []

        # Extract genuine evidence for audit sampling (up to 50 genuine pairs)
        sample_audit_idx = flagged_or_escalated_idx[:50]
        for idx in sample_audit_idx:
            row = test_df.iloc[idx]
            p_id = str(pair_ids[idx])
            ref_txt = str(row["reference_text"])
            sus_txt = str(row["suspicious_text"])

            t0_ev = time.perf_counter_ns()
            ev_data = self.evidence_extractor.extract_text_evidence(
                text_ref=ref_txt,
                text_sus=sus_txt,
                candidate_score=float(c1_scores[idx]),
                calibrated_prob=float(c2_probs[idx]),
                predictive_entropy=float(c2_entropies[idx]),
            )
            t1_ev = time.perf_counter_ns()
            ev_latencies_ms.append((t1_ev - t0_ev) / 1_000_000.0)

            dossier = generate_pair_evidence_dossier(
                pair_id=p_id,
                modality="text",
                stage1_score=float(c1_scores[idx]),
                calibrated_probability=float(c2_probs[idx]),
                predictive_entropy=float(c2_entropies[idx]),
                decision_state=routing_states[idx].name,
                final_prediction=int(final_preds[idx]),
                extracted_evidence=ev_data,
                stage2_score=float(stage2_scores[idx]) if is_escalated[idx] else None,
                ground_truth_label=int(y_test[idx]),
                premise_group_id=str(group_ids[idx]),
            )
            dossiers.append(dossier)

        dossier_metrics = compute_dossier_metrics(
            total_flagged_escalated_pairs=len(flagged_or_escalated_idx),
            generated_dossiers=dossiers,
            explanation_latencies_ms=ev_latencies_ms,
        )

        # Conformal Prediction Evaluation on test premise groups
        conformal_summary = self.conformal_predictor.evaluate_test_coverage(
            group_ids=group_ids,
            y_true=y_test,
            y_prob=c2_probs,
        )

        # Conformal sets
        conf_sets = self.conformal_predictor.predict_prediction_sets(c2_probs)

        # Comprehensive Metrics & Group-Aware Bootstrap
        clf_metrics = compute_comprehensive_classification_metrics(y_test, final_preds, final_probs)
        calib_metrics = compute_calibration_metrics(y_test, final_probs)
        sel_pred = evaluate_selective_prediction(y_test, final_probs)
        group_boot_ci = compute_group_aware_bootstrap_ci(
            group_ids=group_ids,
            y_true=y_test,
            y_pred=final_preds,
            n_resamples=1000,
            random_state=self.random_state,
        )

        # Total cost accounting
        n_pruned = int(np.sum(is_pruned))
        n_flagged = int(np.sum(is_flagged))
        calls_avoided = n_pruned + n_flagged
        calls_avoided_pct = (calls_avoided / N) * 100.0

        # Create detailed per-pair predictions dataframe
        predictions_df = pd.DataFrame({
            "pair_id": pair_ids,
            "premise_group_id": group_ids,
            "ground_truth_label": y_test,
            "stage1_score": np.round(c1_scores, 4),
            "calibrated_prob": np.round(c2_probs, 4),
            "predictive_entropy": np.round(c2_entropies, 4),
            "routing_decision_state": [s.name for s in routing_states],
            "stage2_score": np.round(stage2_scores, 4),
            "final_prediction": final_preds,
            "conformal_prediction_set": [sorted(list(s)) for s in conf_sets],
            "conformal_pair_covered": [y in s for y, s in zip(y_test, conf_sets)],
        })

        return {
            "modality": "text",
            "total_test_pairs": N,
            "classification_metrics": clf_metrics,
            "calibration_metrics": calib_metrics,
            "group_bootstrap_ci_95": group_boot_ci,
            "group_conformal_summary": conformal_summary,
            "selective_prediction": sel_pred,
            "dossier_metrics": dossier_metrics,
            "sample_dossiers": dossiers[:5],
            "routing_cost": {
                "total_pairs": N,
                "pruned_count": n_pruned,
                "pruned_ratio": round(n_pruned / N, 4),
                "flagged_direct_count": n_flagged,
                "flagged_direct_ratio": round(n_flagged / N, 4),
                "escalated_count": n_escalated,
                "escalated_ratio": round(n_escalated / N, 4),
                "stage2_calls_avoided_count": calls_avoided,
                "stage2_calls_avoided_percentage": round(calls_avoided_pct, 2),
                "stage1_latency_ms_per_pair": round(s1_latency_ms, 4),
                "calibration_latency_ms_per_pair": round(c2_latency_ms, 4),
                "routing_latency_ms_per_pair": round(c3_latency_ms, 4),
                "stage2_latency_ms_per_escalated_pair": round(s2_latency_ms, 4),
            },
            "predictions_dataframe": predictions_df,
        }

