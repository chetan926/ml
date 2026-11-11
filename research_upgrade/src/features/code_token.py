"""
Raw Token Feature Extractor (Phase 3.2, Code View 1)
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection
"""

from collections import Counter
from typing import Dict, Any, List
import numpy as np

# Reuse verified baseline components from src/ without modifying them
from src.code.esplag_tokenizer import remove_comments_safe, tokenize_code
from src.code.rkr_gst import rkr_gst


class CodeTokenExtractor:
    """
    Extracts raw Java token similarity features:
    1. Token Jaccard similarity: |T1 cap T2| / |T1 cup T2|.
    2. Token frequency overlap (multiset Counter intersection).
    3. Running Karp-Rabin Greedy String Tiling (RKR-GST) sequence coverage.
    """

    def __init__(self, mml: int = 3):
        self.mml = mml

    def extract(self, ref_code: str, sus_code: str) -> Dict[str, float]:
        """Extracts token similarity features for a Java code pair."""
        ref_src = str(ref_code) if ref_code is not None else ""
        sus_src = str(sus_code) if sus_code is not None else ""

        # Safe comment stripping and tokenization
        tokens_ref = tokenize_code(remove_comments_safe(ref_src))
        tokens_sus = tokenize_code(remove_comments_safe(sus_src))

        # 1. Token Jaccard Similarity
        set_ref = set(tokens_ref)
        set_sus = set(tokens_sus)

        if not set_ref or not set_sus:
            jaccard = 0.0
        else:
            inter = len(set_ref.intersection(set_sus))
            union = len(set_ref.union(set_sus))
            jaccard = float(inter / union) if union > 0 else 0.0

        # 2. Token Frequency Overlap (Multiset Intersection)
        if not tokens_ref or not tokens_sus:
            freq_overlap = 0.0
        else:
            c_ref = Counter(tokens_ref)
            c_sus = Counter(tokens_sus)
            common_tokens = set(c_ref.keys()).intersection(set(c_sus.keys()))
            min_overlap = sum(min(c_ref[t], c_sus[t]) for t in common_tokens)
            max_total = max(len(tokens_ref), len(tokens_sus))
            freq_overlap = float(min_overlap / max_total) if max_total > 0 else 0.0

        # 3. RKR-GST Sequence Similarity
        if not tokens_ref or not tokens_sus:
            rkrgst_sim = 0.0
        else:
            try:
                res = rkr_gst(tokens_ref, tokens_sus, mml=self.mml)
                rkrgst_sim = float(res.standard_similarity)
                rkrgst_sim = max(0.0, min(1.0, rkrgst_sim))
            except Exception:
                rkrgst_sim = 0.0

        return {
            "code_token_jaccard": round(jaccard, 6),
            "code_token_frequency_overlap": round(freq_overlap, 6),
            "code_rkrgst_sim": round(rkrgst_sim, 6)
        }

    @property
    def feature_names(self) -> List[str]:
        return [
            "code_token_jaccard",
            "code_token_frequency_overlap",
            "code_rkrgst_sim"
        ]
