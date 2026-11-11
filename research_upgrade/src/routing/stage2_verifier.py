"""
Stage-2 Heavy Verification Models (Phase 3.5).

Provides high-capacity second-stage verification for pairs escalated from Stage-1 candidate generation:
- TextStage2Verifier: High-precision multi-view non-linear interaction verifier
- CodeStage2Verifier: Deep structural alignment and multi-pass token sequence tiling verifier
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class TextStage2Verifier:
    """
    Stage-2 Heavy Verification Model for Text Modality.
    Combines fine-grained interaction features across lexical, character, dense semantic,
    and structural views with non-linear tree ensemble interactions.
    Simulates high-capacity deep verification (cross-view interaction).
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.85,
            random_state=self.random_state
        )
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
        self.is_fitted = False

    def _engineer_interaction_features(self, df: pd.DataFrame) -> np.ndarray:
        """Constructs higher-order non-linear interaction terms."""
        X_base = df[self.feature_columns].values.astype(np.float64)
        # Interaction terms: semantic * lexical, semantic * char, structural ratio * length
        sem = df["text_semantic_minilm_sim"].values
        lex = df["text_word_jaccard"].values
        char = df["text_char_lcs_ratio"].values
        len_rat = df["text_length_ratio"].values

        interactions = np.column_stack([
            sem * lex,
            sem * char,
            lex * char,
            sem * len_rat,
            np.abs(sem - lex),
        ])
        return np.column_stack([X_base, interactions])

    def fit(self, df_train: pd.DataFrame) -> "TextStage2Verifier":
        X = self._engineer_interaction_features(df_train)
        y = df_train["ground_truth_label"].values.astype(np.int32)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def predict_verification_score(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("TextStage2Verifier must be fitted before scoring.")
        X = self._engineer_interaction_features(df)
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)[:, 1]
        return proba


class CodeStage2Verifier:
    """
    Stage-2 Heavy Verification Model for Source Code Modality.
    Combines normalized token tiling, structural control block matching,
    complexity delta, and non-linear interactions for deep code plagiarism verification.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=4,
            random_state=self.random_state
        )
        self.feature_columns = [
            "code_token_jaccard",
            "code_token_frequency_overlap",
            "code_rkrgst_sim",
            "code_norm_token_jaccard",
            "code_norm_freq_overlap",
            "code_norm_rkrgst_sim",
            "code_nesting_depth_sim",
            "code_control_block_sim",
            "code_method_structure_sim",
            "code_operator_dist_sim",
            "code_control_flow_sim",
            "code_cyclomatic_sim",
            "code_sloc_ratio",
        ]
        self.is_fitted = False

    def _engineer_code_interactions(self, df: pd.DataFrame) -> np.ndarray:
        """Constructs higher-order code interaction features."""
        X_base = df[self.feature_columns].values.astype(np.float64)
        norm_rkr = df["code_norm_rkrgst_sim"].values
        ctrl_block = df["code_control_block_sim"].values
        op_dist = df["code_operator_dist_sim"].values
        sloc_rat = df["code_sloc_ratio"].values

        interactions = np.column_stack([
            norm_rkr * ctrl_block,
            norm_rkr * op_dist,
            ctrl_block * sloc_rat,
            np.abs(norm_rkr - ctrl_block),
        ])
        return np.column_stack([X_base, interactions])

    def fit(self, df_train: pd.DataFrame) -> "CodeStage2Verifier":
        X = self._engineer_code_interactions(df_train)
        y = df_train["ground_truth_label"].values.astype(np.int32)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def predict_verification_score(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("CodeStage2Verifier must be fitted before scoring.")
        X = self._engineer_code_interactions(df)
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)[:, 1]
        return proba

