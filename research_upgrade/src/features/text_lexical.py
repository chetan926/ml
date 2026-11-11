"""
Word-Level Lexical Feature Extractor (Phase 3.2, Text View 1)
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection
"""

import re
from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def tokenize_words(text: str) -> List[str]:
    """Tokenizes text into lowercased alphanumeric words."""
    if not isinstance(text, str):
        return []
    return re.findall(r"\b\w+\b", text.lower())


class TextLexicalExtractor:
    """
    Extracts word-level lexical features:
    1. Word-level TF-IDF cosine similarity (fitted strictly on training corpus).
    2. Word set Jaccard similarity.
    3. Word overlap ratio (containment).
    """

    def __init__(self, ngram_range=(1, 2), max_features: int = 10000):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.vectorizer: Optional[TfidfVectorizer] = None
        self._is_fitted = False

    def fit(self, training_texts: List[str]) -> "TextLexicalExtractor":
        """
        Fits TF-IDF vectorizer strictly on training texts.
        Rejects fitting on test or validation splits.
        """
        if not training_texts:
            raise ValueError("Cannot fit TF-IDF vectorizer on empty training corpus.")
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=True,
            strip_accents="unicode"
        )
        self.vectorizer.fit(training_texts)
        self._is_fitted = True
        return self

    def extract(self, ref_text: str, sus_text: str) -> Dict[str, float]:
        """Extracts lexical feature dictionary for a single text pair."""
        if not self._is_fitted:
            raise RuntimeError(
                "TextLexicalExtractor must be fitted on training data before feature extraction."
            )

        ref_str = str(ref_text) if ref_text is not None else ""
        sus_str = str(sus_text) if sus_text is not None else ""

        # 1. TF-IDF Cosine Similarity
        if not ref_str.strip() or not sus_str.strip():
            tfidf_sim = 0.0
        else:
            try:
                vecs = self.vectorizer.transform([ref_str, sus_str])
                sim_matrix = cosine_similarity(vecs[0], vecs[1])
                tfidf_sim = float(sim_matrix[0, 0])
                tfidf_sim = max(0.0, min(1.0, tfidf_sim))
            except Exception:
                tfidf_sim = 0.0

        # 2. Word Token Jaccard and Overlap Ratio
        tokens_ref = tokenize_words(ref_str)
        tokens_sus = tokenize_words(sus_str)

        set_ref = set(tokens_ref)
        set_sus = set(tokens_sus)

        if not set_ref or not set_sus:
            jaccard = 0.0
            overlap_ratio = 0.0
        else:
            intersection = len(set_ref.intersection(set_sus))
            union = len(set_ref.union(set_sus))
            jaccard = float(intersection / union) if union > 0 else 0.0

            min_len = min(len(set_ref), len(set_sus))
            overlap_ratio = float(intersection / min_len) if min_len > 0 else 0.0

        return {
            "text_word_tfidf_sim": round(tfidf_sim, 6),
            "text_word_jaccard": round(jaccard, 6),
            "text_word_overlap_ratio": round(overlap_ratio, 6)
        }

    @property
    def feature_names(self) -> List[str]:
        return [
            "text_word_tfidf_sim",
            "text_word_jaccard",
            "text_word_overlap_ratio"
        ]

