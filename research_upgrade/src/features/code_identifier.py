"""
Identifier-Normalized Feature Extractor (Phase 3.2, Code View 2)
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection
"""

from collections import Counter
from typing import Dict, Any, List
import re

from src.code.esplag_tokenizer import remove_comments_safe, tokenize_code, JAVA_KEYWORDS
from src.code.rkr_gst import rkr_gst


def normalize_identifiers(tokens: List[str]) -> List[str]:
    """
    Replaces user-defined identifiers with canonical placeholder tokens ('IDENT')
    while strictly preserving Java keywords, operators, and literal delimiters.
    """
    normalized = []
    for tok in tokens:
        # If token is a keyword or operator/delimiter, preserve it
        if tok in JAVA_KEYWORDS or re.match(r"^[^a-zA-Z0-9_$]+$", tok) or re.match(r"^\d+$", tok) or tok.startswith('"') or tok.startswith("'"):
            normalized.append(tok)
        else:
            # User-defined identifier
            normalized.append("IDENT")
    return normalized


class CodeIdentifierNormalizedExtractor:
    """
    Extracts identifier-normalized similarity features:
    1. Normalized token Jaccard similarity.
    2. Normalized token frequency overlap.
    3. Normalized RKR-GST sequence similarity (renaming-invariant structural matching).
    """

    def __init__(self, mml: int = 3):
        self.mml = mml

    def extract(self, ref_code: str, sus_code: str) -> Dict[str, float]:
        """Extracts normalized token features for a Java code pair."""
        ref_src = str(ref_code) if ref_code is not None else ""
        sus_src = str(sus_code) if sus_code is not None else ""

        tokens_ref = tokenize_code(remove_comments_safe(ref_src))
        tokens_sus = tokenize_code(remove_comments_safe(sus_src))

        norm_ref = normalize_identifiers(tokens_ref)
        norm_sus = normalize_identifiers(tokens_sus)

        # 1. Normalized Token Jaccard
        set_ref = set(norm_ref)
        set_sus = set(norm_sus)
        if not set_ref or not set_sus:
            jaccard = 0.0
        else:
            inter = len(set_ref.intersection(set_sus))
            union = len(set_ref.union(set_sus))
            jaccard = float(inter / union) if union > 0 else 0.0

        # 2. Normalized Frequency Overlap
        if not norm_ref or not norm_sus:
            freq_overlap = 0.0
        else:
            c_ref = Counter(norm_ref)
            c_sus = Counter(norm_sus)
            common = set(c_ref.keys()).intersection(set(c_sus.keys()))
            min_overlap = sum(min(c_ref[t], c_sus[t]) for t in common)
            max_total = max(len(norm_ref), len(norm_sus))
            freq_overlap = float(min_overlap / max_total) if max_total > 0 else 0.0

        # 3. Normalized RKR-GST Similarity
        if not norm_ref or not norm_sus:
            rkrgst_sim = 0.0
        else:
            try:
                res = rkr_gst(norm_ref, norm_sus, mml=self.mml)
                rkrgst_sim = float(res.standard_similarity)
                rkrgst_sim = max(0.0, min(1.0, rkrgst_sim))
            except Exception:
                rkrgst_sim = 0.0

        return {
            "code_norm_token_jaccard": round(jaccard, 6),
            "code_norm_freq_overlap": round(freq_overlap, 6),
            "code_norm_rkrgst_sim": round(rkrgst_sim, 6)
        }

    @property
    def feature_names(self) -> List[str]:
        return [
            "code_norm_token_jaccard",
            "code_norm_freq_overlap",
            "code_norm_rkrgst_sim"
        ]
