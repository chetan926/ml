"""
ES-Plag Penalty-Based Similarity Normalization.
Based on Sulistiani & Karnalim (2019), "ES-Plag", Comput. Appl. Eng. Educ.
DOI: 10.1002/cae.22066.

Formulas:
- Penalized Max:
    penalized_max = (matched_tokens - (num_subsequences - 1)) / min(len(a), len(b))
- Penalized Average:
    penalized_average = 2 * (matched_tokens - (num_subsequences - 1)) / (len(a) + len(b))

The penalty mechanism penalizes fragmented matches (rearranged statements / scrambled code)
while rewarding contiguous verbatim subsequence blocks.
"""

import numpy as np

def compute_penalized_similarities(
    matched_tokens: int,
    num_subsequences: int,
    len_a: int,
    len_b: int
) -> dict[str, float]:
    """
    Computes both Penalized Max and Penalized Average similarities from RKRGST tiling output.
    
    Args:
        matched_tokens: Total number of matched tokens in tiles.
        num_subsequences: Number of distinct tiles (subsequences).
        len_a: Token count of submission A.
        len_b: Token count of submission B.
        
    Returns:
        dict with keys:
            'penalized_max': float in [0.0, 1.0]
            'penalized_average': float in [0.0, 1.0]
            'raw_gst_average': float in [0.0, 1.0]
            'penalty_deducted': int
    """
    if matched_tokens <= 0 or num_subsequences <= 0 or len_a <= 0 or len_b <= 0:
        return {
            'penalized_max': 0.0,
            'penalized_average': 0.0,
            'raw_gst_average': 0.0,
            'penalty_deducted': 0
        }

    penalty = num_subsequences - 1
    penalized_numerator = max(0.0, float(matched_tokens - penalty))

    min_len = min(len_a, len_b)
    sum_len = len_a + len_b

    penalized_max = float(penalized_numerator / min_len) if min_len > 0 else 0.0
    penalized_avg = float((2.0 * penalized_numerator) / sum_len) if sum_len > 0 else 0.0
    raw_gst_avg = float((2.0 * matched_tokens) / sum_len) if sum_len > 0 else 0.0

    return {
        'penalized_max': float(np.clip(penalized_max, 0.0, 1.0)),
        'penalized_average': float(np.clip(penalized_avg, 0.0, 1.0)),
        'raw_gst_average': float(np.clip(raw_gst_avg, 0.0, 1.0)),
        'penalty_deducted': penalty
    }

