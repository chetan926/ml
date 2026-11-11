"""
Running-Karp-Rabin Greedy-String-Tiling (RKR-GST) Algorithm.
Based on:
- Wise, M. J. (1996). "YAP3: Improved detection of similarities in computer program and other texts".
- Sulistiani, L., & Karnalim, O. (2019). "ES-Plag: Efficient and sensitive source code plagiarism detection tool".
  Comput. Appl. Eng. Educ., 27(1), 166-178. DOI: 10.1002/cae.22066.

Features:
- Configurable minimum matching length m (default m = 2).
- Greedy tiling: iteratively extracts non-overlapping maximal matching token substrings.
- Returns matched token count, number of tiles (subsequences), and tile coordinates.
"""

from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class MatchTile:
    pos_a: int
    pos_b: int
    length: int

@dataclass
class RKRGSTResult:
    matched_tokens: int
    num_subsequences: int
    tiles: List[MatchTile]
    standard_similarity: float
    len_a: int
    len_b: int

def rkr_gst(tokens_a: list[str], tokens_b: list[str], mml: int = 2) -> RKRGSTResult:
    """
    Executes Greedy String Tiling over two token sequences with minimum matching length (mml).
    
    Args:
        tokens_a: First token sequence.
        tokens_b: Second token sequence.
        mml: Minimum matching length (reference setting mml = 2).
        
    Returns:
        RKRGSTResult containing matched_tokens, num_subsequences, tiles, and standard_similarity.
    """
    n_a = len(tokens_a)
    n_b = len(tokens_b)

    if n_a == 0 or n_b == 0 or n_a < mml or n_b < mml:
        return RKRGSTResult(
            matched_tokens=0,
            num_subsequences=0,
            tiles=[],
            standard_similarity=0.0,
            len_a=n_a,
            len_b=n_b
        )

    marked_a = [False] * n_a
    marked_b = [False] * n_b
    tiles: List[MatchTile] = []

    # Rolling search for maximal matches
    while True:
        max_match = mml - 1
        matches_at_max: List[Tuple[int, int, int]] = []

        # Find the length of maximal unmarked matches
        for i in range(n_a):
            if marked_a[i]:
                continue
            for j in range(n_b):
                if marked_b[j]:
                    continue

                # Measure contiguous unmarked match starting at (i, j)
                k = 0
                while (
                    i + k < n_a and 
                    j + k < n_b and 
                    not marked_a[i + k] and 
                    not marked_b[j + k] and 
                    tokens_a[i + k] == tokens_b[j + k]
                ):
                    k += 1

                if k > max_match:
                    max_match = k
                    matches_at_max = [(i, j, k)]
                elif k == max_match and k >= mml:
                    matches_at_max.append((i, j, k))

        # Termination condition: no unmarked match >= mml found
        if max_match < mml or not matches_at_max:
            break

        # Greedy tiling: sort matches to prioritize longest and earliest, then mark
        # Sort by length descending, then earliest indices
        matches_at_max.sort(key=lambda x: (-x[2], x[0], x[1]))

        for i, j, k in matches_at_max:
            # Check if any token in this candidate tile has already been marked in this round
            is_valid = True
            for offset in range(k):
                if marked_a[i + offset] or marked_b[j + offset]:
                    is_valid = False
                    break

            if is_valid:
                for offset in range(k):
                    marked_a[i + offset] = True
                    marked_b[j + offset] = True
                tiles.append(MatchTile(pos_a=i, pos_b=j, length=k))

    total_matched_tokens = sum(tile.length for tile in tiles)
    num_subsequences = len(tiles)
    standard_similarity = (2.0 * total_matched_tokens) / (n_a + n_b) if (n_a + n_b) > 0 else 0.0

    return RKRGSTResult(
        matched_tokens=total_matched_tokens,
        num_subsequences=num_subsequences,
        tiles=tiles,
        standard_similarity=float(standard_similarity),
        len_a=n_a,
        len_b=n_b
    )

