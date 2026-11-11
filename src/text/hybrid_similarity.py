"""
Hybrid Lexical-Semantic Similarity Fusion.
Based on Setu et al. (2025), "A comprehensive strategy for identifying plagiarism in academic submissions",
Algorithm 1 & Section 3.2. DOI: 10.1007/s43995-025-00108-1.

Formulas:
- Hybrid Similarity:
    hybrid_score = max(tfidf_score, sbert_score)
- Similarity Categorization:
    category = 'direct_copy_like' if tfidf_score >= sbert_score else 'paraphrase_like'
"""

import numpy as np

def compute_hybrid_similarity(
    tfidf_scores: np.ndarray,
    sbert_scores: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    """
    Computes element-wise maximum hybrid similarity and categorizes similarity type.
    
    Returns:
        hybrid_scores: np.ndarray of max(tfidf, sbert)
        categories: list of 'direct_copy_like' or 'paraphrase_like'
    """
    tfidf_arr = np.asarray(tfidf_scores, dtype=float)
    sbert_arr = np.asarray(sbert_scores, dtype=float)

    hybrid_scores = np.maximum(tfidf_arr, sbert_arr)
    categories = [
        'direct_copy_like' if t >= s else 'paraphrase_like'
        for t, s in zip(tfidf_arr, sbert_arr)
    ]
    return hybrid_scores, categories

