"""
Phase 5 Corrected-Protocol Baseline Suite.

Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for
Cost-Efficient Academic Integrity Detection — Q1 Upgrade (Phase 5).

Addresses the Phase 4.1 audit gap: "the code headline table compared against a
degenerate Token-Jaccard baseline only, and adapted SVM/RF peers were absent
from the corrected artifacts."

All baselines are evaluated under EXACTLY the same isolation protocol as the
proposed framework:
    Text — models/scores fitted on D_train, thresholds tuned on D_val only,
           evaluated once on held-out D_test (3,000 pairs / 1,015 premise groups).
    Code — nested Leave-One-Group-Out across the 7 assignment cases; any fitted
           parameters and thresholds come from the 6 training cases of each outer
           fold only (zero program leakage, zero calibration leakage).

Baselines:
    Text:  B-T1 TF-IDF lexical threshold   (Sajid et al., 2025 practice)
           B-T2 MiniLM dense semantic threshold (Setu et al., 2025 practice)
           B-T3 Flat multi-view supervised hybrid (all features, no routing —
               the strongest "existing-style" flat competitor)
    Code:  B-C1 Token Jaccard threshold (footnote baseline, retained for continuity)
           B-C2 RKRGST greedy string tiling threshold
           B-C3 Gandhi-style RBF-SVM on AST/syntactic features (2024)
           B-C4 Mehsen-style RandomForest on token-frequency features (2024)
           B-C5 ES-Plag dynamic pre-filter + RKRGST (Sulistiani & Karnalim, 2019)
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import SVC

from research_upgrade.phase_5_q1_upgrade.q1src.q1_stats_extras import (
    tune_threshold_for_f1,
    tune_threshold_for_iso_recall,
)

TEXT_MULTIVIEW_FEATURES = [
    "text_word_tfidf_sim", "text_word_jaccard", "text_word_overlap_ratio",
    "text_char_tfidf_sim", "text_char_jaccard_3gram", "text_char_lcs_ratio",
    "text_semantic_minilm_sim", "text_length_ratio", "text_token_count_delta",
    "text_sentence_delta", "text_ttr_sim",
]

CODE_FEATURES = [
    "code_token_jaccard", "code_token_frequency_overlap", "code_rkrgst_sim",
    "code_norm_token_jaccard", "code_norm_freq_overlap", "code_norm_rkrgst_sim",
    "code_nesting_depth_sim", "code_control_block_sim", "code_method_structure_sim",
    "code_operator_dist_sim", "code_control_flow_sim", "code_cyclomatic_sim",
    "code_sloc_ratio",
]

# Gandhi-style 7-D syntactic/AST feature subset (adapted to the IR-Plag Java space)
CODE_SVM_FEATURES = [
    "code_norm_token_jaccard", "code_norm_freq_overlap", "code_nesting_depth_sim",
    "code_control_block_sim", "code_method_structure_sim", "code_control_flow_sim",
    "code_cyclomatic_sim",
]

# Mehsen-style token-frequency feature subset
CODE_RF_FEATURES = [
    "code_token_jaccard", "code_token_frequency_overlap", "code_rkrgst_sim",
    "code_norm_token_jaccard", "code_norm_freq_overlap", "code_norm_rkrgst_sim",
    "code_sloc_ratio",
]


def _score_metrics(y_true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "mcc": round(float(matthews_corrcoef(y_true, pred)), 4),
    }


# ---------------------------------------------------------------------------
# TEXT BASELINES
# ---------------------------------------------------------------------------

class TextBaselineSuite:
    """Evaluates the three existing text peers under the corrected protocol."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.thresholds_: Dict[str, float] = {}
        self.flat_model_: Optional[LogisticRegression] = None

    def fit(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
    ) -> "TextBaselineSuite":
        y_tr = df_train["ground_truth_label"].values.astype(int)
        y_val = df_val["ground_truth_label"].values.astype(int)

        # B-T1 / B-T2: threshold-only score baselines (no fitting)
        self.thresholds_["tfidf"], _ = tune_threshold_for_f1(
            y_val, df_val["text_word_tfidf_sim"].values
        )
        self.thresholds_["semantic"], _ = tune_threshold_for_f1(
            y_val, df_val["text_semantic_minilm_sim"].values
        )

        # B-T3: flat multi-view hybrid fitted on train, threshold tuned on val
        self.flat_model_ = LogisticRegression(C=1.0, max_iter=1000, random_state=self.random_state)
        self.flat_model_.fit(df_train[TEXT_MULTIVIEW_FEATURES].values, y_tr)
        p_val = self.flat_model_.predict_proba(df_val[TEXT_MULTIVIEW_FEATURES].values)[:, 1]
        self.thresholds_["flat_hybrid"], _ = tune_threshold_for_f1(y_val, p_val)
        return self

    def evaluate_test(self, df_test: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        if self.flat_model_ is None:
            raise RuntimeError("TextBaselineSuite must be fitted before evaluation.")
        y_te = df_test["ground_truth_label"].values.astype(int)

        pred_t1 = (df_test["text_word_tfidf_sim"].values >= self.thresholds_["tfidf"]).astype(int)
        pred_t2 = (df_test["text_semantic_minilm_sim"].values >= self.thresholds_["semantic"]).astype(int)
        p_t3 = self.flat_model_.predict_proba(df_test[TEXT_MULTIVIEW_FEATURES].values)[:, 1]
        pred_t3 = (p_t3 >= self.thresholds_["flat_hybrid"]).astype(int)

        return {
            "B-T1 TF-IDF Lexical (Sajid 2025)": {
                "scores": df_test["text_word_tfidf_sim"].values,
                "prob_scores": df_test["text_word_tfidf_sim"].values,
                "predictions": pred_t1,
                "tuned_threshold": self.thresholds_["tfidf"],
                **_score_metrics(y_te, pred_t1),
            },
            "B-T2 Dense Semantic MiniLM (Setu 2025)": {
                "scores": df_test["text_semantic_minilm_sim"].values,
                "prob_scores": df_test["text_semantic_minilm_sim"].values,
                "predictions": pred_t2,
                "tuned_threshold": self.thresholds_["semantic"],
                **_score_metrics(y_te, pred_t2),
            },
            "B-T3 Flat Multi-View Hybrid LR": {
                "scores": p_t3,
                "prob_scores": p_t3,
                "predictions": pred_t3,
                "tuned_threshold": self.thresholds_["flat_hybrid"],
                **_score_metrics(y_te, pred_t3),
            },
        }


# ---------------------------------------------------------------------------
# CODE BASELINES (nested group-disjoint LOGO)
# ---------------------------------------------------------------------------

class CodeBaselineSuiteLOGO:
    """Evaluates code peers with all fitting/threshold decisions made per outer fold."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def _fold_threshold(self, y_tr: np.ndarray, scores_tr: np.ndarray, mode: str,
                        iso_target: float = 0.95) -> float:
        if mode == "iso_recall":
            thr, _ = tune_threshold_for_iso_recall(y_tr, scores_tr, recall_target=iso_target)
        else:
            thr, _ = tune_threshold_for_f1(y_tr, scores_tr)
        return thr

    def evaluate(
        self,
        df_feats: pd.DataFrame,
        case_series: pd.Series,
        threshold_mode: str = "f1",
        iso_target: float = 0.95,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Returns per-baseline dicts with pooled OOF predictions/scores and metrics.
        threshold_mode: 'f1' (default operating point) or 'iso_recall'.
        """
        y = df_feats["ground_truth_label"].values.astype(int)
        cases = case_series.values

        oof = {
            "jaccard": np.zeros(len(y)),
            "rkrgst": np.zeros(len(y)),
            "svm": np.zeros(len(y)),
            "rf": np.zeros(len(y)),
        }
        logo = LeaveOneGroupOut()
        for train_idx, test_idx in logo.split(df_feats[CODE_FEATURES].values, y, groups=cases):
            y_tr = y[train_idx]

            # B-C1 / B-C2: score columns; thresholds tuned on the 6 training cases ONLY
            thr_j = self._fold_threshold(y_tr, df_feats["code_token_jaccard"].values[train_idx],
                                         threshold_mode, iso_target)
            thr_r = self._fold_threshold(y_tr, df_feats["code_rkrgst_sim"].values[train_idx],
                                         threshold_mode, iso_target)
            oof["jaccard"][test_idx] = (
                df_feats["code_token_jaccard"].values[test_idx] >= thr_j
            ).astype(int) if threshold_mode == "f1" else (
                df_feats["code_token_jaccard"].values[test_idx] >= thr_j
            ).astype(int)
            # keep raw score separately for stats
            oof["rkrgst"][test_idx] = df_feats["code_rkrgst_sim"].values[test_idx] >= thr_r

            # B-C3 Gandhi-style SVM (fitted on train cases only)
            svm = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=self.random_state)
            svm.fit(df_feats[CODE_SVM_FEATURES].values[train_idx], y_tr)
            oof["svm"][test_idx] = svm.predict(df_feats[CODE_SVM_FEATURES].values[test_idx])

            # B-C4 Mehsen-style RF (fitted on train cases only)
            rf = RandomForestClassifier(
                n_estimators=100, max_depth=None, random_state=self.random_state
            )
            rf.fit(df_feats[CODE_RF_FEATURES].values[train_idx], y_tr)
            oof["rf"][test_idx] = rf.predict(df_feats[CODE_RF_FEATURES].values[test_idx])

        results = {
            "B-C1 Token Jaccard": {"scores": df_feats["code_token_jaccard"].values,
                                     "predictions": oof["jaccard"].astype(int),
                                     **_score_metrics(y, oof["jaccard"].astype(int))},
            "B-C2 RKRGST String Tiling": {"scores": df_feats["code_rkrgst_sim"].values,
                                            "predictions": oof["rkrgst"].astype(int),
                                            **_score_metrics(y, oof["rkrgst"].astype(int))},
            "B-C3 Gandhi RBF-SVM (2024)": {"scores": oof["svm"],
                                             "predictions": oof["svm"].astype(int),
                                             **_score_metrics(y, oof["svm"].astype(int))},
            "B-C4 Mehsen RandomForest (2024)": {"scores": oof["rf"],
                                                  "predictions": oof["rf"].astype(int),
                                                  **_score_metrics(y, oof["rf"].astype(int))},
        }
        return results


def evaluate_esplag_baseline_logo(
    raw_code_df: pd.DataFrame,
    case_series: pd.Series,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    B-C5: faithful ES-Plag (dynamic token-cosine pre-filter + RKRGST with penalty),
    evaluated per outer LOGO fold with the dynamic filter threshold derived from
    the 6 TRAINING cases only.
    """
    from src.code.esplag_detector import ESPlagDetector  # repo-root legacy module

    y = raw_code_df["is_plagiarized"].values.astype(int)
    cases = case_series.values
    preds = np.zeros(len(y), dtype=int)
    scores = np.zeros(len(y), dtype=float)
    retained = np.zeros(len(y), dtype=bool)

    logo = LeaveOneGroupOut()
    for train_idx, test_idx in logo.split(
        raw_code_df[["reference_code", "suspicious_code"]].values, y, groups=cases
    ):
        train_df = raw_code_df.iloc[train_idx].rename(columns={"id": "pair_id_no"})
        detector = ESPlagDetector(mml=2, initial_threshold=0.50, decision_threshold=0.50)

        # Fit-equivalent step: derive the dynamic filter threshold on training pairs
        train_pairs = pd.DataFrame({
            "id": np.arange(len(train_df)),
            "reference_code": train_df["reference_code"].values,
            "suspicious_code": train_df["suspicious_code"].values,
            "is_plagiarized": train_df["is_plagiarized"].values,
            "obfuscation_level": train_df["obfuscation_level"].values
                if "obfuscation_level" in train_df else ["L0"] * len(train_df),
        })
        train_res = detector.evaluate_pairs(train_pairs, apply_filtering=True)
        train_filter_thr = float(train_res.get("filter_threshold", 0.0))

        # Held-out case: filter with the TRAIN-derived threshold, then RKRGST
        test_pairs = pd.DataFrame({
            "id": np.arange(len(test_idx)),
            "reference_code": raw_code_df["reference_code"].values[test_idx],
            "suspicious_code": raw_code_df["suspicious_code"].values[test_idx],
            "is_plagiarized": y[test_idx],
            "obfuscation_level": raw_code_df["obfuscation_level"].values[test_idx]
                if "obfuscation_level" in raw_code_df else ["L0"] * len(test_idx),
        })
        test_res = detector.evaluate_pairs(
            test_pairs, apply_filtering=True, custom_filter_threshold=train_filter_thr
        )
        rec = test_res["pair_records"]
        preds[test_idx] = np.array([r["prediction"] for r in rec], dtype=int)
        scores[test_idx] = np.array([r["final_score"] for r in rec], dtype=float)
        retained[test_idx] = np.array([r["passed_filter"] for r in rec], dtype=bool)

    return {
        "scores": scores,
        "predictions": preds,
        "retained_mask": retained,
        "pairs_passing_dynamic_filter": int(retained.sum()),
        **_score_metrics(y, preds),
    }
