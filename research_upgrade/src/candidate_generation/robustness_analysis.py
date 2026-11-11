"""
Robustness and Stress Testing Analysis Module (Phase 3.3, Workstream D)
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from research_upgrade.src.candidate_generation.candidate_evaluation import evaluate_candidate_threshold


def evaluate_text_robustness_subsets(
    df_val_feats: pd.DataFrame,
    val_scores: np.ndarray,
    applied_threshold: float
) -> Dict[str, Any]:
    """
    Evaluates candidate generation robustness across difficult linguistic subsets
    using strictly validation partition pairs.
    """
    df = df_val_feats.copy()
    df["candidate_score"] = val_scores
    y_val = df["ground_truth_label"].to_numpy()

    # 1. Zero-Word-Overlap Positives in Validation Set
    zero_overlap_mask = (df["text_word_jaccard"] == 0.0) & (df["ground_truth_label"] == 1)
    df_zero_overlap = df[zero_overlap_mask]
    n_zero_overlap = len(df_zero_overlap)
    if n_zero_overlap > 0:
        esc_zero = int(np.sum(df_zero_overlap["candidate_score"] >= applied_threshold))
        recall_zero = float(esc_zero / n_zero_overlap)
    else:
        esc_zero = 0
        recall_zero = 1.0

    # 2. Low Lexical Overlap Positives (TF-IDF < 0.2)
    low_lex_mask = (df["text_word_tfidf_sim"] < 0.20) & (df["ground_truth_label"] == 1)
    df_low_lex = df[low_lex_mask]
    n_low_lex = len(df_low_lex)
    if n_low_lex > 0:
        esc_low = int(np.sum(df_low_lex["candidate_score"] >= applied_threshold))
        recall_low = float(esc_low / n_low_lex)
    else:
        esc_low = 0
        recall_low = 1.0

    # 3. High Semantic Similarity Positives (MiniLM > 0.70)
    high_sem_mask = (df["text_semantic_minilm_sim"] >= 0.70) & (df["ground_truth_label"] == 1)
    df_high_sem = df[high_sem_mask]
    n_high_sem = len(df_high_sem)
    if n_high_sem > 0:
        esc_high_sem = int(np.sum(df_high_sem["candidate_score"] >= applied_threshold))
        recall_high_sem = float(esc_high_sem / n_high_sem)
    else:
        esc_high_sem = 0
        recall_high_sem = 1.0

    # 4. Length-Mismatched Pairs (Length ratio < 0.50)
    len_mismatch_mask = (df["text_length_ratio"] < 0.50)
    df_len_mismatch = df[len_mismatch_mask]
    if len(df_len_mismatch) > 0:
        m_len = evaluate_candidate_threshold(
            df_len_mismatch["ground_truth_label"].to_numpy(),
            df_len_mismatch["candidate_score"].to_numpy(),
            applied_threshold
        )
        len_mismatch_dict = m_len.to_dict()
    else:
        len_mismatch_dict = {}

    return {
        "applied_threshold": applied_threshold,
        "zero_word_overlap_positives": {
            "total_pairs_in_val": n_zero_overlap,
            "escalated_pairs": esc_zero,
            "missed_pairs": n_zero_overlap - esc_zero,
            "candidate_recall": round(recall_zero, 4),
            "historical_distinction_note": (
                "The historical 119 zero-overlap positive cases belong to the historical stress test distribution, "
                f"while the active validation set contains {n_zero_overlap} zero-word-overlap positive pairs."
            )
        },
        "low_lexical_overlap_positives": {
            "total_pairs_in_val": n_low_lex,
            "escalated_pairs": esc_low,
            "candidate_recall": round(recall_low, 4)
        },
        "high_semantic_similarity_positives": {
            "total_pairs_in_val": n_high_sem,
            "escalated_pairs": esc_high_sem,
            "candidate_recall": round(recall_high_sem, 4)
        },
        "length_mismatched_pairs": len_mismatch_dict
    }


def evaluate_code_robustness_by_obfuscation(
    df_code_feats: pd.DataFrame,
    code_scores: np.ndarray,
    applied_threshold: float
) -> Dict[str, Any]:
    """
    Evaluates candidate generation performance across IR-Plag obfuscation levels.
    """
    df = df_code_feats.copy()
    df["candidate_score"] = code_scores

    obf_levels = sorted(df["obfuscation_level"].unique())
    obf_results = {}

    for lvl in obf_levels:
        sub = df[df["obfuscation_level"] == lvl]
        if len(sub) == 0:
            continue
        m = evaluate_candidate_threshold(
            sub["ground_truth_label"].to_numpy(),
            sub["candidate_score"].to_numpy(),
            applied_threshold
        )
        obf_results[str(lvl)] = {
            "obfuscation_level": str(lvl),
            "total_pairs": len(sub),
            "positive_pairs": m.total_positives,
            "negative_pairs": m.total_negatives,
            "candidate_recall": m.candidate_recall,
            "workload_reduction": m.workload_reduction,
            "escalated_pairs": m.escalated_pairs
        }

    # Renaming impact analysis: Normalized vs Raw token similarity on obfuscated code
    norm_adv = float(np.mean(df["code_norm_token_jaccard"] - df["code_token_jaccard"]))

    return {
        "applied_threshold": applied_threshold,
        "obfuscation_level_breakdown": obf_results,
        "renaming_invariance_analysis": {
            "mean_token_jaccard_raw": round(float(np.mean(df["code_token_jaccard"])), 4),
            "mean_token_jaccard_normalized": round(float(np.mean(df["code_norm_token_jaccard"])), 4),
            "mean_identifier_normalization_gain": round(norm_adv, 4)
        }
    }

