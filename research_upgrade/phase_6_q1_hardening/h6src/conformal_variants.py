"""Phase 6 hardening — group-conformal coverage-repair variants.

The phase-5 audit found group coverage 0.930 vs the nominal 0.95 target
(Wilson 95% CI excludes 0.95). This module implements three reviewer-grade
repairs and evaluates each on held-out test groups:

  A. STANDARD          — baseline group-supremum conformal (reproduction ref).
  B. MONDRIAN-SIZE     — Mondrian conformal conditioned on calibration-group
                         size bin (targets premise-group heteroscedasticity).
  C. MONDRIAN-PARITY   — conditioned on majority class of the group.
  D. ALPHA-TUNED       — split-sample significance tuning: grid search alpha* on
                         a held-out half of calibration groups, then recalibrate
                         the full calibration pool at alpha* (conservative max).

Group-exchangeability unit = premise group. Nonconformity = group supremum
S_g = max_{i in g} (1 - p̂(y_i)). Coverage verdicts carry Wilson 95% CIs.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - r) / d, (c + r) / d


def group_supremum_scores(
    group_ids: np.ndarray, y_true: np.ndarray, p: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (unique_groups, S_g per group, group_size per group)."""
    group_ids = np.asarray(group_ids)
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p, dtype=float)
    uniq, sizes = np.unique(group_ids, return_counts=True)
    scores = np.empty(len(uniq), dtype=float)
    p_true = np.where(y_true == 1, p, 1.0 - p)  # p̂(y_true)
    for gi, g in enumerate(uniq):
        scores[gi] = float(np.max(1.0 - p_true[group_ids == g]))
    return uniq, scores, sizes


def exact_quantile(scores: np.ndarray, alpha: float) -> float:
    """k = ceil((m+1)(1-alpha)); return k-th smallest score; +inf fallback."""
    m = len(scores)
    if m == 0:
        return float("inf")
    k = int(math.ceil((m + 1) * (1.0 - alpha)))
    if k > m:
        return float("inf")
    return float(np.sort(scores)[k - 1])


def group_covered_mask(
    unique_g: np.ndarray, S_test: np.ndarray, qhat: float
) -> np.ndarray:
    return S_test <= qhat


def evaluate_group_coverage(
    test_group_ids: np.ndarray,
    y_test: np.ndarray,
    p_test: np.ndarray,
    qhat: float,
) -> Dict[str, Any]:
    ug, S_test, sizes = group_supremum_scores(test_group_ids, y_test, p_test)
    cov = group_covered_mask(ug, S_test, qhat)
    k, n = int(cov.sum()), len(cov)
    lo, hi = wilson_ci(k, n)
    return {"n_groups": n, "coverage": round(float(k / n), 6),
            "wilson95": [round(lo, 6), round(hi, 6)], "qhat": qhat,
            "group_sizes_median": float(np.median(sizes))}


def mondrian_size_qhats(
    cal_group_ids: np.ndarray, y_cal: np.ndarray, p_cal: np.ndarray,
    alpha: float,
) -> Dict[str, Any]:
    """Mondrian over calibration-group size (<= median vs > median)."""
    ug, S_cal, sizes = group_supremum_scores(cal_group_ids, y_cal, p_cal)
    med = float(np.median(sizes))
    small = S_cal[sizes <= med]
    large = S_cal[sizes > med]
    return {"median_size": med, "qhat_small": exact_quantile(small, alpha),
            "qhat_large": exact_quantile(large, alpha),
            "m_small": int(len(small)), "m_large": int(len(large))}


def evaluate_mondrian_size(
    cal_group_ids: np.ndarray, y_cal: np.ndarray, p_cal: np.ndarray,
    test_group_ids: np.ndarray, y_test: np.ndarray, p_test: np.ndarray,
    alpha: float,
) -> Dict[str, Any]:
    qb = mondrian_size_qhats(cal_group_ids, y_cal, p_cal, alpha)
    ug, S_test, sizes = group_supremum_scores(test_group_ids, y_test, p_test)
    qh = np.where(sizes <= qb["median_size"], qb["qhat_small"], qb["qhat_large"])
    cov = S_test <= qh
    k, n = int(cov.sum()), len(cov)
    lo, hi = wilson_ci(k, n)
    return {"n_groups": n, "coverage": round(float(k / n), 6),
            "wilson95": [round(lo, 6), round(hi, 6)], **qb}


def mondrian_parity_qhats(
    cal_group_ids: np.ndarray, y_cal: np.ndarray, p_cal: np.ndarray,
    alpha: float,
) -> Dict[str, Any]:
    """Mondrian over group majority class (positive-dominant vs not)."""
    ug, S_cal, _ = group_supremum_scores(cal_group_ids, y_cal, p_cal)
    y_cal = np.asarray(y_cal).astype(int)
    dom = np.zeros(len(ug), dtype=bool)
    for gi, g in enumerate(ug):
        dom[gi] = y_cal[cal_group_ids == g].mean() > 0.5
    return {"qhat_posdom": exact_quantile(S_cal[dom], alpha),
            "qhat_negdom": exact_quantile(S_cal[~dom], alpha),
            "m_posdom": int(dom.sum()), "m_negdom": int((~dom).sum())}


def evaluate_mondrian_parity(
    cal_group_ids: np.ndarray, y_cal: np.ndarray, p_cal: np.ndarray,
    test_group_ids: np.ndarray, y_test: np.ndarray, p_test: np.ndarray,
    alpha: float,
) -> Dict[str, Any]:
    qb = mondrian_parity_qhats(cal_group_ids, y_cal, p_cal, alpha)
    ug, S_test, _ = group_supremum_scores(test_group_ids, y_test, p_test)
    y_test = np.asarray(y_test).astype(int)
    dom_te = np.array([y_test[test_group_ids == g].mean() > 0.5 for g in ug])
    qh = np.where(dom_te, qb["qhat_posdom"], qb["qhat_negdom"])
    cov = S_test <= qh
    k, n = int(cov.sum()), len(cov)
    lo, hi = wilson_ci(k, n)
    return {"n_groups": n, "coverage": round(float(k / n), 6),
            "wilson95": [round(lo, 6), round(hi, 6)], **qb}


def alpha_split_sample_tuning(
    cal_group_ids: np.ndarray, y_cal: np.ndarray, p_cal: np.ndarray,
    target_coverage: float, random_state: int = 42,
    alpha_grid: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Split calibration groups 50/50; grid-search alpha* so that coverage on the
    check half >= target using quantiles computed on the tune half; swap halves and
    repeat; final alpha* = max (conservative). Returns alpha* and both estimates."""
    if alpha_grid is None:
        alpha_grid = np.round(np.arange(0.01, 0.201, 0.01), 4)
    ug, S_cal, _ = group_supremum_scores(cal_group_ids, y_cal, p_cal)
    rng = np.random.RandomState(random_state)
    perm = rng.permutation(len(ug))
    half = len(ug) // 2
    A, B = perm[:half], perm[half:]
    stars = []

    def tune(tune_idx, check_idx):
        S_tune, S_check = S_cal[tune_idx], S_cal[check_idx]
        for a in alpha_grid:
            q = exact_quantile(S_tune, a)
            if np.mean(S_check <= q) >= target_coverage:
                return float(a)
        return float(alpha_grid[-1])

    stars.append(tune(A, B))
    stars.append(tune(B, A))
    return {"alpha_star_conservative": max(stars), "alpha_star_first": stars[0],
            "alpha_star_swap": stars[1], "target": target_coverage,
            "m_groups_total": int(len(ug)), "half": int(half)}
