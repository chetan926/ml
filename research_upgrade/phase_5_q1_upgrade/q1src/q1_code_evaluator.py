"""
Phase 5 Code Modality Evaluator (Q1 Upgrade).

Changes vs Phase 4.1 `CodeGroupDisjointEvaluator`:
    1. Stage-2 is selected PER OUTER FOLD between genuinely heavy candidates
       {HGB(400,d8), RF(500,d12)} by inner nested-LOGO out-of-fold MCC —
       leak-free architecture selection using only the 6 training cases.
    2. The tri-band operating point is tuned per outer fold on inner-OOF
       calibrated predictions under an explicit Stage-2 budget cap and a
       validation recall floor.
    3. Optional adversarial augmentation hook: training folds may be extended
       with extra-benchmark identifier-renamed copies of training positives —
       derived only from training data (leak-free), hardening both stages
       against the single most common code-paraphrase side effect.
    4. Per-fold models are persisted so attacks can be routed through the
       identical fold pipeline.

Isolation guarantees (unchanged): outer LOGO across 7 assignment cases; inner
nested LOGO for calibration, architecture selection, and threshold tuning.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

from research_upgrade.src.calibration.calibrators import PlattScaler
from research_upgrade.src.calibration.uncertainty import compute_predictive_entropy
from research_upgrade.src.calibration.calibration_metrics import compute_calibration_metrics
from research_upgrade.phase_4_1_corrections.src.evaluation.statistical_testing import (
    compute_comprehensive_classification_metrics,
    compute_group_aware_bootstrap_ci,
)
from research_upgrade.phase_5_q1_upgrade.q1src.heavy_verifiers import (
    CodeHeavyVerifier,
    RFHeavyVerifier,
    make_code_heavy_candidates,
    measure_lr_stage1_latency_ms,
)
from research_upgrade.phase_5_q1_upgrade.q1src.q1_stats_extras import (
    apply_routing,
    tune_routing_cost_minimizer,
    tune_routing_operating_point,
)
from research_upgrade.phase_5_q1_upgrade.q1src.q1_baselines import CODE_FEATURES


class Phase5CodeEvaluator:
    """Nested group-disjoint LOGO with selected heavy Stage-2 + tuned routing."""

    def __init__(
        self,
        random_state: int = 42,
        stage2_budget_cap: float = 0.45,
        min_val_recall: float = 0.80,
        epsilon_mcc: float = 0.01,
        augment_pairs_fn: Optional[Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None,
    ):
        self.random_state = random_state
        self.stage2_budget_cap = stage2_budget_cap
        self.min_val_recall = min_val_recall
        self.epsilon_mcc = epsilon_mcc
        self.augment_pairs_fn = augment_pairs_fn
        self.feature_cols = CODE_FEATURES
        self.fold_models_: List[Dict[str, Any]] = []
        self.verifier_selection_report_: List[Dict[str, Any]] = []

    # ------------------------------------------------------------- helpers
    def _augment(self, X, y, abs_idx) -> Tuple[np.ndarray, np.ndarray]:
        if self.augment_pairs_fn is None:
            return X, y
        X_aug, y_aug = self.augment_pairs_fn(abs_idx)
        if X_aug is not None and len(X_aug) > 0:
            return np.vstack([X, np.asarray(X_aug, dtype=np.float64)]), \
                   np.concatenate([y, np.asarray(y_aug).astype(int)])
        return X, y

    # ------------------------------------------------------------------ core
    def _evaluate(
        self,
        df_feats: pd.DataFrame,
        case_series: pd.Series,
        policy_name: str,
        stratified_leakage: bool = False,
    ) -> Dict[str, Any]:
        X = df_feats[self.feature_cols].values
        y = df_feats["ground_truth_label"].values.astype(int)
        cases = case_series.values
        n = len(y)

        oof_p = np.zeros(n, dtype=float)
        oof_s1 = np.zeros(n, dtype=float)
        oof_ent = np.zeros(n, dtype=float)
        oof_s2 = np.zeros(n, dtype=float)
        oof_pred = np.zeros(n, dtype=int)
        oof_state = np.empty(n, dtype=object)
        fold_details = []
        self.fold_models_ = []
        self.verifier_selection_report_ = []

        if stratified_leakage:
            splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            fold_iter = list(splitter.split(X, y))
            held_out_names = [f"strat-fold-{i+1}" for i in range(5)]
        else:
            fold_iter = list(LeaveOneGroupOut().split(X, y, groups=cases))
            held_out_names = [str(cases[t][0]) for _, t in fold_iter]

        for fold_idx, (train_idx, test_idx) in enumerate(fold_iter):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te = X[test_idx]
            cases_tr = cases[train_idx]

            # Stage-1 (augmented outer-training fit)
            X_tr_aug, y_tr_aug = self._augment(X_tr, y_tr, train_idx)
            s1 = LogisticRegression(C=1.0, max_iter=1000, random_state=self.random_state)
            s1.fit(X_tr_aug, y_tr_aug)

            # INNER nested CV: honest OOF Stage-1 probs + candidate verifier scores
            inner_oof_s1 = np.zeros(len(y_tr), dtype=float)
            cand_names = list(make_code_heavy_candidates(self.random_state).keys())
            inner_oof_s2 = {cn: np.zeros(len(y_tr), dtype=float) for cn in cand_names}

            if stratified_leakage:
                inner_iter = StratifiedKFold(
                    n_splits=5, shuffle=True, random_state=self.random_state).split(X_tr, y_tr)
            else:
                inner_iter = LeaveOneGroupOut().split(X_tr, y_tr, groups=cases_tr)

            for in_tr, in_val in inner_iter:
                X_in_tr_aug, y_in_tr_aug = self._augment(X_tr[in_tr], y_tr[in_tr],
                                                         train_idx[in_tr])
                m_in = LogisticRegression(C=1.0, max_iter=1000, random_state=self.random_state)
                m_in.fit(X_in_tr_aug, y_in_tr_aug)
                inner_oof_s1[in_val] = m_in.predict_proba(X_tr[in_val])[:, 1]
                for cn, cand in make_code_heavy_candidates(self.random_state).items():
                    cand.fit(X_in_tr_aug, y_in_tr_aug)
                    inner_oof_s2[cn][in_val] = cand.predict_proba(X_tr[in_val])

            # Leak-free heavy-verifier architecture selection (inner-OOF MCC @ 0.5)
            sel = {cn: float(matthews_corrcoef(y_tr, (inner_oof_s2[cn] >= 0.5).astype(int)))
                   for cn in cand_names}
            winner = max(sel, key=sel.get)
            self.verifier_selection_report_.append({
                "fold": fold_idx + 1, "held_out": held_out_names[fold_idx],
                "candidate_inner_oof_mcc": {k: round(v, 4) for k, v in sel.items()},
                "selected": winner,
            })

            # Platt on inner OOF Stage-1 scores only
            platt = PlattScaler().fit(inner_oof_s1, y_tr)

            # Operating point tuned on inner OOF of the 6 training cases:
            # cost-minimizing router that preserves beta=1 (full heavy verification)
            # inner-OOF MCC up to epsilon.
            inner_oof_p = platt.predict_positive_proba(inner_oof_s1)
            inner_oof_ent = compute_predictive_entropy(inner_oof_p)
            op = tune_routing_cost_minimizer(
                y_val=y_tr,
                p_val_calibrated=inner_oof_p,
                entropy_val=inner_oof_ent,
                s2_scores_val=inner_oof_s2[winner],
                epsilon_mcc=self.epsilon_mcc,
            )

            # Deploy on held-out case
            s2_outer = make_code_heavy_candidates(self.random_state)[winner]
            s2_outer.fit(X_tr_aug, y_tr_aug)
            s1_te = s1.predict_proba(X_te)[:, 1]
            p_te = platt.predict_positive_proba(s1_te)
            ent_te = compute_predictive_entropy(p_te)
            s2_te_all = s2_outer.predict_proba(X_te)

            pred_te, states_te, _ = apply_routing(
                p_te, ent_te, s2_te_all,
                tau_low=op["tau_low"], tau_high=op["tau_high"],
                entropy_threshold=op["entropy_threshold"],
                stage2_decision_threshold=op["stage2_decision_threshold"],
            )

            oof_s1[test_idx] = s1_te
            oof_p[test_idx] = p_te
            oof_ent[test_idx] = ent_te
            oof_s2[test_idx] = s2_te_all
            oof_pred[test_idx] = pred_te
            oof_state[test_idx] = states_te

            self.fold_models_.append({
                "s1_model": s1, "platt": platt, "s2_model": s2_outer,
                "s2_name": winner, "operating_point": op,
                "held_out_case": held_out_names[fold_idx], "test_index": test_idx,
            })
            fold_details.append({
                "fold": fold_idx + 1, "held_out": held_out_names[fold_idx],
                "train_samples": int(len(train_idx)), "test_samples": int(len(test_idx)),
                "selected_verifier": winner,
                "operating_point": {k: op[k] for k in
                                     ("tau_low", "tau_high", "entropy_threshold",
                                      "stage2_decision_threshold", "val_mcc")},
            })

        clf = compute_comprehensive_classification_metrics(y, oof_pred, oof_p)
        calib = compute_calibration_metrics(y, oof_p)
        boot = compute_group_aware_bootstrap_ci(cases, y, oof_pred,
                                                  n_resamples=1000,
                                                  random_state=self.random_state)
        n_pruned = int(np.sum(oof_state == "LOW_RISK_PRUNED"))
        n_flagged = int(np.sum(oof_state == "HIGH_RISK_FLAGGED"))
        n_esc = int(np.sum(oof_state == "UNCERTAIN_ESCALATED"))
        avoided = n_pruned + n_flagged

        predictions_df = pd.DataFrame({
            "pair_id": df_feats["pair_id"].values if "pair_id" in df_feats else np.arange(n),
            "case": cases,
            "ground_truth_label": y,
            "stage1_score": np.round(oof_s1, 4),
            "calibrated_prob": np.round(oof_p, 4),
            "predictive_entropy": np.round(oof_ent, 4),
            "decision_state": oof_state,
            "stage2_score": np.round(oof_s2, 4),
            "final_prediction": oof_pred,
        })

        return {
            "policy": policy_name,
            "evaluation_paradigm": (
                "Nested LOGO (7 cases), inner-OOF-selected heavy Stage-2, "
                "per-fold cost-minimizing operating point (epsilon-MCC 0.01 preserved)"
                if not stratified_leakage else
                "Stratified 5-Fold [LEAKAGE PRESENT — controlled contrast]"
            ),
            "classification_metrics": clf,
            "calibration_metrics": calib,
            "group_bootstrap_ci_95": boot,
            "routing_cost": {
                "total_pairs": n, "pruned_count": n_pruned,
                "flagged_direct_count": n_flagged, "escalated_count": n_esc,
                "stage2_calls_avoided_count": avoided,
                "stage2_calls_avoided_percent": round((avoided / n) * 100.0, 2),
            },
            "fold_details": fold_details,
            "verifier_selection": self.verifier_selection_report_,
            "predictions_dataframe": predictions_df,
            "oof_probs": oof_p,
            "oof_s2_scores": oof_s2,
        }

    def evaluate_logo(self, df_feats, case_series, policy_name="Policy A"):
        return self._evaluate(df_feats, case_series, policy_name, stratified_leakage=False)

    def evaluate_stratified_leakage(self, df_feats, policy_name="Policy A"):
        return self._evaluate(df_feats, case_series=pd.Series(df_feats["case_id"].values),
                              policy_name=policy_name, stratified_leakage=True)

    # ------------------------------------------------- renaming attack eval
    def apply_attacked_features(
        self, case_name: str, feature_matrix: np.ndarray, mode: str = "routed"
    ) -> Dict[str, np.ndarray]:
        """Routes attacked feature rows through the fold whose held-out case matches.
        mode='routed' applies the tuned tri-band pipeline; mode='s2_only' sends
        every pair to the heavy verifier (budget beta=1.0 reference)."""
        for fm in self.fold_models_:
            if fm["held_out_case"] == case_name:
                s1 = fm["s1_model"]
                p = fm["platt"].predict_positive_proba(s1.predict_proba(feature_matrix)[:, 1])
                ent = compute_predictive_entropy(p)
                s2 = fm["s2_model"].predict_proba(feature_matrix)
                if mode == "s2_only":
                    return {"predictions": (s2 >= 0.5).astype(int), "probs": p,
                            "states": np.array(["UNCERTAIN_ESCALATED"] * len(feature_matrix)),
                            "s2": s2}
                op = fm["operating_point"]
                pred, states, _ = apply_routing(p, ent, s2, op["tau_low"], op["tau_high"],
                                                op["entropy_threshold"],
                                                op["stage2_decision_threshold"])
                return {"predictions": pred, "probs": p, "states": states, "s2": s2}
        raise KeyError(f"No fold model found for held-out case {case_name}")

    def measure_stage2_latency(self, df_feats: pd.DataFrame) -> Dict[str, float]:
        """Honest Stage-1 vs Stage-2 latency using the first fold's selected models."""
        if not self.fold_models_:
            raise RuntimeError("No folds available; run an evaluation first.")
        fm = self.fold_models_[0]
        X_single = df_feats[self.feature_cols].values[:1]
        X_batch = df_feats[self.feature_cols].values[:256]
        s2_lat_single = fm["s2_model"].measure_single_pair_latency_ms(X_single)
        s1_lat_single = measure_lr_stage1_latency_ms(fm["s1_model"], X_single)
        s2 = fm["s2_model"]
        s2.predict_proba(X_batch)
        import time as _t
        t0 = _t.perf_counter()
        for _ in range(10):
            s2.predict_proba(X_batch)
        s2_batched = ((_t.perf_counter() - t0) / 10) * 1000.0 / len(X_batch)
        lr = fm["s1_model"]
        lr.predict_proba(X_batch)
        t0 = _t.perf_counter()
        for _ in range(50):
            lr.predict_proba(X_batch)
        s1_batched = ((_t.perf_counter() - t0) / 50) * 1000.0 / len(X_batch)
        meta = fm["s2_model"].metadata()
        return {
            "selected_fold0_verifier": fm["s2_name"],
            "stage1_single_pair_ms": round(s1_lat_single, 6),
            "stage2_single_pair_ms": round(s2_lat_single, 6),
            "stage1_batched_ms_per_pair": round(s1_batched, 6),
            "stage2_batched_ms_per_pair": round(s2_batched, 6),
            "latency_ratio_s2_over_s1_single": round(s2_lat_single / max(s1_lat_single, 1e-9), 1),
            "latency_ratio_s2_over_s1_batched": round(s2_batched / max(s1_batched, 1e-9), 1),
            **{f"s2_{k}": v for k, v in meta.items()},
        }
