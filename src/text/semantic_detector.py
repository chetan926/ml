"""
Contextual Semantic Similarity Detector using Sentence-BERT.
Based on Setu et al. (2025), "A comprehensive strategy for identifying plagiarism in academic submissions",
Journal of Umm Al-Qura University for Engineering and Architecture, Springer Nature.
DOI: 10.1007/s43995-025-00108-1.
"""

import os
import numpy as np

# Enforce offline mode for Hugging Face
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from sentence_transformers import SentenceTransformer

class SemanticSBERTDetector:
    _cached_model = None

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = 'cpu'):
        self.model_name = model_name
        self.device = device
        if SemanticSBERTDetector._cached_model is None:
            SemanticSBERTDetector._cached_model = SentenceTransformer(model_name, device=device)
        self.model = SemanticSBERTDetector._cached_model

    def compute_embeddings(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Computes dense normalized contextual sequence embeddings."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True
        )

    def compute_similarity(self, ref_texts: list[str], sus_texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Computes pairwise cosine similarities using row-wise dot products of normalized vectors."""
        ref_vecs = self.compute_embeddings(ref_texts, batch_size=batch_size)
        sus_vecs = self.compute_embeddings(sus_texts, batch_size=batch_size)

        sims = (ref_vecs * sus_vecs).sum(axis=1)
        return np.clip(sims, -1.0, 1.0)

