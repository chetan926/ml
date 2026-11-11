"""
Structural Feature Extractor (Phase 3.2, Text View 4)
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection
"""

import re
from typing import Dict, Any, List


def split_sentences(text: str) -> List[str]:
    """Splits text into sentences using standard punctuation heuristics."""
    if not isinstance(text, str) or not text.strip():
        return []
    s = re.split(r"[.!?]+(?:\s+|$)", text.strip())
    return [sent.strip() for sent in s if sent.strip()]


def compute_ttr(tokens: List[str]) -> float:
    """Computes Type-Token Ratio (lexical diversity) in [0, 1]."""
    if not tokens:
        return 0.0
    return float(len(set(tokens)) / len(tokens))


class TextStructuralExtractor:
    """
    Extracts structural and length-based features:
    1. Text character length ratio: min(L1, L2) / max(L1, L2).
    2. Word token count delta: |T1 - T2| / max(T1, T2).
    3. Sentence count delta: |S1 - S2| / max(S1, S2).
    4. Type-Token Ratio (TTR) lexical diversity similarity: 1 - |TTR1 - TTR2|.
    """

    def __init__(self):
        pass

    def extract(self, ref_text: str, sus_text: str) -> Dict[str, float]:
        """Extracts structural features for a single text pair."""
        ref_str = str(ref_text).strip() if ref_text is not None else ""
        sus_str = str(sus_text).strip() if sus_text is not None else ""

        len_ref = len(ref_str)
        len_sus = len(sus_str)

        # 1. Length Ratio
        max_len = max(len_ref, len_sus)
        if max_len == 0:
            length_ratio = 1.0  # Both empty -> identical length
        else:
            length_ratio = float(min(len_ref, len_sus) / max_len)

        # 2. Token Count Delta
        tokens_ref = re.findall(r"\b\w+\b", ref_str.lower())
        tokens_sus = re.findall(r"\b\w+\b", sus_str.lower())

        max_tokens = max(len(tokens_ref), len(tokens_sus))
        if max_tokens == 0:
            token_delta = 0.0
        else:
            token_delta = float(abs(len(tokens_ref) - len(tokens_sus)) / max_tokens)

        # 3. Sentence Count Delta
        sents_ref = split_sentences(ref_str)
        sents_sus = split_sentences(sus_str)

        max_sents = max(len(sents_ref), len(sents_sus))
        if max_sents == 0:
            sent_delta = 0.0
        else:
            sent_delta = float(abs(len(sents_ref) - len(sents_sus)) / max_sents)

        # 4. TTR Lexical Diversity Similarity
        ttr_ref = compute_ttr(tokens_ref)
        ttr_sus = compute_ttr(tokens_sus)
        ttr_sim = float(max(0.0, 1.0 - abs(ttr_ref - ttr_sus)))

        return {
            "text_length_ratio": round(length_ratio, 6),
            "text_token_count_delta": round(token_delta, 6),
            "text_sentence_delta": round(sent_delta, 6),
            "text_ttr_sim": round(ttr_sim, 6)
        }

    @property
    def feature_names(self) -> List[str]:
        return [
            "text_length_ratio",
            "text_token_count_delta",
            "text_sentence_delta",
            "text_ttr_sim"
        ]

