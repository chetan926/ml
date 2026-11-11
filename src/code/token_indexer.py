"""
ES-Plag Token Frequency Indexer and Dynamic Cosine Filter.
Based on Sulistiani & Karnalim (2019), "ES-Plag", Comput. Appl. Eng. Educ.
DOI: 10.1002/cae.22066.

Features:
- Inverted index for fast token multiset representation.
- Pairwise cosine similarity calculation over token frequency vectors.
- Dynamic filtering threshold calculation:
    threshold = sim_min + initial_threshold * (sim_max - sim_min)
"""

import math
from collections import Counter
import numpy as np

def compute_cosine_similarity(tokens_a: list[str], tokens_b: list[str]) -> float:
    """
    Computes set-theoretic cosine similarity between token frequency distributions of two code files.
    """
    if not tokens_a or not tokens_b:
        return 0.0

    counts_a = Counter(tokens_a)
    counts_b = Counter(tokens_b)

    common_tokens = set(counts_a.keys()).intersection(counts_b.keys())
    if not common_tokens:
        return 0.0

    dot_product = sum(counts_a[t] * counts_b[t] for t in common_tokens)
    norm_a = math.sqrt(sum(v * v for v in counts_a.values()))
    norm_b = math.sqrt(sum(v * v for v in counts_b.values()))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))

def compute_dynamic_esplag_threshold(
    similarities: list[float] | np.ndarray,
    initial_threshold: float = 0.50
) -> float:
    """
    Computes the dynamic filtering threshold proposed in ES-Plag:
        threshold = sim_min + initial_threshold * (sim_max - sim_min)
    
    If all similarities are equal, returns sim_min.
    """
    if len(similarities) == 0:
        return 0.0

    sim_min = float(np.min(similarities))
    sim_max = float(np.max(similarities))

    if sim_max <= sim_min:
        return sim_min

    threshold = sim_min + initial_threshold * (sim_max - sim_min)
    return float(np.clip(threshold, 0.0, 1.0))

class TokenFrequencyIndex:
    """
    Maintains an in-memory index of tokenized submissions for fast pairwise scoring.
    """
    def __init__(self):
        self.doc_tokens: dict[str, list[str]] = {}
        self.doc_counts: dict[str, Counter] = {}
        self.doc_norms: dict[str, float] = {}

    def add_document(self, doc_id: str, tokens: list[str]):
        """Indexes a document by its token sequence."""
        self.doc_tokens[doc_id] = tokens
        counts = Counter(tokens)
        self.doc_counts[doc_id] = counts
        norm = math.sqrt(sum(v * v for v in counts.values())) if counts else 0.0
        self.doc_norms[doc_id] = norm

    def get_similarity(self, doc_id_a: str, doc_id_b: str) -> float:
        """Computes cosine similarity between two indexed documents."""
        if doc_id_a not in self.doc_counts or doc_id_b not in self.doc_counts:
            return 0.0

        counts_a = self.doc_counts[doc_id_a]
        counts_b = self.doc_counts[doc_id_b]
        norm_a = self.doc_norms[doc_id_a]
        norm_b = self.doc_norms[doc_id_b]

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        common = set(counts_a.keys()).intersection(counts_b.keys())
        if not common:
            return 0.0

        dot = sum(counts_a[t] * counts_b[t] for t in common)
        return float(dot / (norm_a * norm_b))

