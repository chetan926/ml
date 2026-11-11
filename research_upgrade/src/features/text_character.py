"""
Character-Level Feature Extractor (Phase 3.2, Text View 2)
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection
"""

from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extract_char_ngrams(text: str, n: int = 3) -> set:
    """Extracts sliding character n-grams from string."""
    clean = text.lower().strip()
    if len(clean) < n:
        return {clean} if clean else set()
    return {clean[i:i + n] for i in range(len(clean) - n + 1)}


def longest_common_substring(s1: str, s2: str) -> int:
    """Computes length of longest common contiguous substring using DP."""
    if not s1 or not s2:
        return 0
    # Length optimization: keep only previous and current row
    m, n = len(s1), len(s2)
    # Ensure s1 is shorter to optimize memory
    if m > n:
        s1, s2 = s2, s1
        m, n = n, m

    prev = [0] * (n + 1)
    max_len = 0
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1] + 1
                if curr[j] > max_len:
                    max_len = curr[j]
        prev = curr
    return max_len


class TextCharacterExtractor:
    """
    Extracts character-level n-gram features:
    1. Character n-gram TF-IDF similarity (3-5 grams, fitted strictly on training data).
    2. Character 3-gram Jaccard similarity (spelling/punctuation variation resilience).
    3. Longest Common Substring (LCS) ratio.
    """

    def __init__(self, ngram_range=(3, 5), max_features: int = 15000):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.vectorizer: Optional[TfidfVectorizer] = None
        self._is_fitted = False

    def fit(self, training_texts: List[str]) -> "TextCharacterExtractor":
        """Fits character TF-IDF vectorizer strictly on training texts."""
        if not training_texts:
            raise ValueError("Cannot fit character TF-IDF on empty training corpus.")
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=True
        )
        self.vectorizer.fit(training_texts)
        self._is_fitted = True
        return self

    def extract(self, ref_text: str, sus_text: str) -> Dict[str, float]:
        """Extracts character-level features for a single text pair."""
        if not self._is_fitted:
            raise RuntimeError(
                "TextCharacterExtractor must be fitted on training data before feature extraction."
            )

        ref_str = str(ref_text) if ref_text is not None else ""
        sus_str = str(sus_text) if sus_text is not None else ""

        # 1. Character TF-IDF Cosine Similarity
        if not ref_str.strip() or not sus_str.strip():
            char_tfidf_sim = 0.0
        else:
            try:
                vecs = self.vectorizer.transform([ref_str, sus_str])
                sim = float(cosine_similarity(vecs[0], vecs[1])[0, 0])
                char_tfidf_sim = max(0.0, min(1.0, sim))
            except Exception:
                char_tfidf_sim = 0.0

        # 2. Character 3-Gram Jaccard
        grams_ref = extract_char_ngrams(ref_str, n=3)
        grams_sus = extract_char_ngrams(sus_str, n=3)

        if not grams_ref or not grams_sus:
            char_jaccard = 0.0
        else:
            inter = len(grams_ref.intersection(grams_sus))
            union = len(grams_ref.union(grams_sus))
            char_jaccard = float(inter / union) if union > 0 else 0.0

        # 3. Longest Common Substring Ratio
        max_len = max(len(ref_str), len(sus_str))
        if max_len == 0:
            lcs_ratio = 0.0
        else:
            lcs_len = longest_common_substring(ref_str.lower(), sus_str.lower())
            lcs_ratio = float(lcs_len / max_len)

        return {
            "text_char_tfidf_sim": round(char_tfidf_sim, 6),
            "text_char_jaccard_3gram": round(char_jaccard, 6),
            "text_char_lcs_ratio": round(lcs_ratio, 6)
        }

    @property
    def feature_names(self) -> List[str]:
        return [
            "text_char_tfidf_sim",
            "text_char_jaccard_3gram",
            "text_char_lcs_ratio"
        ]

