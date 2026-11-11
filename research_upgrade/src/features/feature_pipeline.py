"""
Unified Multi-View Feature Pipeline Module (Phase 3.2, Module E)
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection
"""

import time
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from research_upgrade.src.features.text_lexical import TextLexicalExtractor
from research_upgrade.src.features.text_character import TextCharacterExtractor
from research_upgrade.src.features.text_semantic import TextSemanticExtractor
from research_upgrade.src.features.text_structural import TextStructuralExtractor

from research_upgrade.src.features.code_token import CodeTokenExtractor
from research_upgrade.src.features.code_identifier import CodeIdentifierNormalizedExtractor
from research_upgrade.src.features.code_ast import CodeStructuralExtractor
from research_upgrade.src.features.code_complexity import CodeComplexityExtractor


TEXT_FEATURE_COLUMNS = [
    # View 1: Lexical
    "text_word_tfidf_sim",
    "text_word_jaccard",
    "text_word_overlap_ratio",
    # View 2: Character
    "text_char_tfidf_sim",
    "text_char_jaccard_3gram",
    "text_char_lcs_ratio",
    # View 3: Semantic
    "text_semantic_minilm_sim",
    # View 4: Structural
    "text_length_ratio",
    "text_token_count_delta",
    "text_sentence_delta",
    "text_ttr_sim"
]

CODE_FEATURE_COLUMNS = [
    # View 1: Raw Token
    "code_token_jaccard",
    "code_token_frequency_overlap",
    "code_rkrgst_sim",
    # View 2: Identifier-Normalized
    "code_norm_token_jaccard",
    "code_norm_freq_overlap",
    "code_norm_rkrgst_sim",
    # View 3: Structural Block
    "code_nesting_depth_sim",
    "code_control_block_sim",
    "code_method_structure_sim",
    # View 4: Complexity
    "code_operator_dist_sim",
    "code_control_flow_sim",
    "code_cyclomatic_sim",
    "code_sloc_ratio"
]


class TextMultiViewPipeline:
    """
    Unified multi-view feature extraction pipeline for academic text pairs.
    Integrates Word Lexical, Character n-gram, Dense Semantic, and Structural views.
    Enforces strict train-only fitting for learned transformations.
    """

    def __init__(self, semantic_model_name: str = "all-MiniLM-L6-v2"):
        self.lexical_extractor = TextLexicalExtractor()
        self.character_extractor = TextCharacterExtractor()
        self.semantic_extractor = TextSemanticExtractor(model_name=semantic_model_name)
        self.structural_extractor = TextStructuralExtractor()
        self._is_fitted = False

    def fit(self, df_train: pd.DataFrame) -> "TextMultiViewPipeline":
        """
        Fits lexical and character TF-IDF vectorizers strictly on training texts.
        Uses reference and suspicious texts from D_train only.
        """
        assert "reference_text" in df_train.columns and "suspicious_text" in df_train.columns, (
            "Training DataFrame must contain 'reference_text' and 'suspicious_text'"
        )
        training_corpus = list(df_train["reference_text"].dropna()) + list(df_train["suspicious_text"].dropna())
        self.lexical_extractor.fit(training_corpus)
        self.character_extractor.fit(training_corpus)
        self._is_fitted = True
        return self

    def extract_pair(self, ref_text: str, sus_text: str) -> Dict[str, float]:
        """Extracts all 11 text features for a single pair."""
        if not self._is_fitted:
            raise RuntimeError("TextMultiViewPipeline must be fitted on training data before extraction.")

        f_lex = self.lexical_extractor.extract(ref_text, sus_text)
        f_char = self.character_extractor.extract(ref_text, sus_text)
        f_sem = self.semantic_extractor.extract(ref_text, sus_text)
        f_struct = self.structural_extractor.extract(ref_text, sus_text)

        combined = {}
        combined.update(f_lex)
        combined.update(f_char)
        combined.update(f_sem)
        combined.update(f_struct)
        return combined

    def transform_dataframe(
        self,
        df: pd.DataFrame,
        partition_name: str,
        dataset_id: str = "snli_text_benchmark"
    ) -> pd.DataFrame:
        """
        Extracts multi-view features for a full DataFrame partition.
        Preserves original pair ID, label, partition metadata, and records extraction time.
        Labels are NEVER used as input features.
        """
        if not self._is_fitted:
            raise RuntimeError("TextMultiViewPipeline must be fitted on training data before transform.")

        records = []
        # Precompute batch semantic embeddings for high throughput
        ref_list = list(df["reference_text"].astype(str))
        sus_list = list(df["suspicious_text"].astype(str))
        sem_sims = self.semantic_extractor.extract_batch(ref_list, sus_list)

        for idx, (_, row) in enumerate(df.iterrows()):
            t0 = time.perf_counter()

            ref = row["reference_text"]
            sus = row["suspicious_text"]

            f_lex = self.lexical_extractor.extract(ref, sus)
            f_char = self.character_extractor.extract(ref, sus)
            f_struct = self.structural_extractor.extract(ref, sus)

            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            rec = {
                "pair_id": row["id"],
                "dataset_id": dataset_id,
                "modality": "text",
                "ground_truth_label": int(row["is_plagiarized"]),
                "partition": partition_name,
                "premise_group_id": str(row["premise_group_id"]) if "premise_group_id" in row else "",
                "extraction_time_ms": round(elapsed_ms, 4),
                "extraction_status": "SUCCESS"
            }
            rec.update(f_lex)
            rec.update(f_char)
            rec["text_semantic_minilm_sim"] = sem_sims[idx]
            rec.update(f_struct)
            records.append(rec)

        return pd.DataFrame(records)

    @property
    def feature_names(self) -> List[str]:
        return list(TEXT_FEATURE_COLUMNS)

    def get_feature_schema(self) -> Dict[str, Any]:
        return {
            "modality": "text",
            "total_features": len(TEXT_FEATURE_COLUMNS),
            "views": {
                "view_1_lexical": ["text_word_tfidf_sim", "text_word_jaccard", "text_word_overlap_ratio"],
                "view_2_character": ["text_char_tfidf_sim", "text_char_jaccard_3gram", "text_char_lcs_ratio"],
                "view_3_semantic": ["text_semantic_minilm_sim"],
                "view_4_structural": ["text_length_ratio", "text_token_count_delta", "text_sentence_delta", "text_ttr_sim"]
            },
            "numerical_range": "[0.0, 1.0]",
            "train_only_fitted_views": ["view_1_lexical", "view_2_character"]
        }


class CodeMultiViewPipeline:
    """
    Unified multi-view feature extraction pipeline for Java source code pairs.
    Integrates Raw Token, Identifier-Normalized, Structural Block, and Complexity views.
    """

    def __init__(self):
        self.token_extractor = CodeTokenExtractor()
        self.identifier_extractor = CodeIdentifierNormalizedExtractor()
        self.structural_extractor = CodeStructuralExtractor()
        self.complexity_extractor = CodeComplexityExtractor()

    def extract_pair(self, ref_code: str, sus_code: str) -> Dict[str, float]:
        """Extracts all 13 code features for a single code pair."""
        f_tok = self.token_extractor.extract(ref_code, sus_code)
        f_norm = self.identifier_extractor.extract(ref_code, sus_code)
        f_struct = self.structural_extractor.extract(ref_code, sus_code)
        f_comp = self.complexity_extractor.extract(ref_code, sus_code)

        combined = {}
        combined.update(f_tok)
        combined.update(f_norm)
        combined.update(f_struct)
        combined.update(f_comp)
        return combined

    def transform_dataframe(
        self,
        df: pd.DataFrame,
        policy: str = "policy_a",
        dataset_id: str = "ir_plag_code_benchmark"
    ) -> pd.DataFrame:
        """
        Extracts multi-view features for a code benchmark DataFrame.
        Preserves original pair ID, case, label, policy, and records extraction time.
        Labels are NEVER used as input features.
        """
        records = []
        for _, row in df.iterrows():
            t0 = time.perf_counter()

            ref = row["reference_code"]
            sus = row["suspicious_code"]

            feats = self.extract_pair(ref, sus)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            rec = {
                "pair_id": row["id"],
                "dataset_id": dataset_id,
                "modality": "code",
                "case_id": row["case"],
                "ref_type": row.get("ref_type", "unknown"),
                "sus_type": row.get("sus_type", "unknown"),
                "obfuscation_level": str(row.get("obfuscation_level", "NONE")),
                "ground_truth_label": int(row["is_plagiarized"]),
                "policy": policy,
                "extraction_time_ms": round(elapsed_ms, 4),
                "extraction_status": "SUCCESS"
            }
            rec.update(feats)
            records.append(rec)

        return pd.DataFrame(records)

    @property
    def feature_names(self) -> List[str]:
        return list(CODE_FEATURE_COLUMNS)

    def get_feature_schema(self) -> Dict[str, Any]:
        return {
            "modality": "code",
            "total_features": len(CODE_FEATURE_COLUMNS),
            "views": {
                "view_1_token": ["code_token_jaccard", "code_token_frequency_overlap", "code_rkrgst_sim"],
                "view_2_identifier_normalized": ["code_norm_token_jaccard", "code_norm_freq_overlap", "code_norm_rkrgst_sim"],
                "view_3_structural": ["code_nesting_depth_sim", "code_control_block_sim", "code_method_structure_sim"],
                "view_4_complexity": ["code_operator_dist_sim", "code_control_flow_sim", "code_cyclomatic_sim", "code_sloc_ratio"]
            },
            "numerical_range": "[0.0, 1.0]",
            "ast_parser_fallback_disclosed": not self.structural_extractor.ast_parser_available
        }
