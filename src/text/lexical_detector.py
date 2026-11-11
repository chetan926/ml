"""
Sublinear TF-IDF Lexical Similarity Detector.
Based on Salton & Buckley (1988) and benchmarked in Sajid et al. (2025) / Setu et al. (2025).
"""

import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import paired_cosine_distances

class LexicalTFIDFDetector:
    def __init__(self, max_features: int = 5000, sublinear_tf: bool = True):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            sublinear_tf=sublinear_tf,
            stop_words='english',
            token_pattern=r'(?u)\b\w+\b'
        )
        self.is_fitted = False

    def fit(self, corpus: list[str]):
        """Fits the TF-IDF vocabulary on reference text corpus."""
        self.vectorizer.fit(corpus)
        self.is_fitted = True
        return self

    def compute_similarity(self, ref_texts: list[str], sus_texts: list[str]) -> np.ndarray:
        """Computes pairwise cosine similarities for paired text arrays."""
        if not self.is_fitted:
            # Fallback fit on union of texts
            self.fit(ref_texts + sus_texts)

        X_ref = self.vectorizer.transform(ref_texts)
        X_sus = self.vectorizer.transform(sus_texts)

        # paired_cosine_distances computes 1 - cosine_similarity
        dists = paired_cosine_distances(X_ref, X_sus)
        sims = np.clip(1.0 - dists, 0.0, 1.0)
        return sims

