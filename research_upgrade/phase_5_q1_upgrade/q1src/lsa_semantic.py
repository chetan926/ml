"""
Phase 5 Offline Dense Semantic Encoder (LSA).

Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for
Cost-Efficient Academic Integrity Detection — Q1 Upgrade (Phase 5).

Methodological Role:
    The Phase 3/4 text feature store contains a `text_semantic_minilm_sim` column
    computed with `all-MiniLM-L6-v2` SentenceTransformer weights. Those weights
    cannot be re-downloaded inside the network-isolated audit sandbox (no outbound
    access to model registries), which makes attack-time feature recomputation
    impossible for the MiniLM view.

    This module provides `LsaSemanticEncoder`, a fully offline dense semantic
    representation: word-level TF-IDF (sublinear) projected by Truncated SVD onto
    a k-dimensional latent semantic space with L2 normalization. The encoder is
    fitted STRICTLY on the training partition (leak-free), then used for
    (a) interaction-feature construction for the heavy Stage-2 cross-style
        pair-interaction verifier, and
    (b) the adversarial paraphrase robustness study where all multi-view features
        must be recomputed on transformed texts.
"""

from typing import List, Optional, Tuple

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class LsaSemanticEncoder:
    """Leak-free latent-semantic (TF-IDF + SVD) dense text encoder."""

    def __init__(
        self,
        n_components: int = 128,
        max_features: int = 20000,
        ngram_range: Tuple[int, int] = (1, 2),
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.random_state = random_state

        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self._is_fitted = False

    def fit(self, training_texts: List[str]) -> "LsaSemanticEncoder":
        """Fits TF-IDF and SVD strictly on the training corpus."""
        texts = [str(t) if t is not None else "" for t in training_texts]
        if not texts:
            raise ValueError("Cannot fit LSA encoder on an empty corpus.")
        mat = self.vectorizer.fit_transform(texts)
        self.svd.fit(mat)
        self._is_fitted = True
        return self

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encodes texts into L2-normalized dense latent vectors."""
        if not self._is_fitted:
            raise RuntimeError("LsaSemanticEncoder must be fitted before encode().")
        texts = [str(t) if t is not None else "" for t in texts]
        mat = self.vectorizer.transform(texts)
        dense = self.svd.transform(mat)
        return normalize(dense, norm="l2")

    def pair_cosine(self, ref_text: str, sus_text: str) -> float:
        """Bounded [0,1] cosine similarity for a single pair."""
        emb = self.encode([ref_text, sus_text])
        sim = float(np.dot(emb[0], emb[1]))
        return max(0.0, min(1.0, sim))

    @staticmethod
    def interaction_features(emb_ref: np.ndarray, emb_sus: np.ndarray) -> np.ndarray:
        """
        Builds the pair-interaction representation used by the heavy
        cross-style Stage-2 verifier:
            [ e_ref ; e_sus ; |e_ref - e_sus| ; e_ref * e_sus ; cos(e_ref, e_sus) ]
        """
        emb_ref = np.asarray(emb_ref, dtype=np.float64)
        emb_sus = np.asarray(emb_sus, dtype=np.float64)
        if emb_ref.ndim == 1:
            emb_ref = emb_ref.reshape(1, -1)
        if emb_sus.ndim == 1:
            emb_sus = emb_sus.reshape(1, -1)
        diff = np.abs(emb_ref - emb_sus)
        prod = emb_ref * emb_sus
        cos = np.sum(emb_ref * emb_sus, axis=1, keepdims=True)
        return np.hstack([emb_ref, emb_sus, diff, prod, cos])

    @property
    def interaction_dim(self) -> int:
        return int(self.n_components) * 4 + 1
