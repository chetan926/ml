"""Tests for phase-6 group-conformal coverage-repair variants."""
import numpy as np
import pytest

from research_upgrade.phase_6_q1_hardening.h6src.conformal_variants import (
    alpha_split_sample_tuning,
    evaluate_group_coverage,
    evaluate_mondrian_parity,
    evaluate_mondrian_size,
    exact_quantile,
    group_supremum_scores,
    wilson_ci,
)


def _make_groups(rng, n_groups=120, sizes=(4, 14)):
    gid, y, p = [], [], []
    for gi in range(n_groups):
        k = rng.randint(sizes[0], sizes[1] + 1)
        yy = (rng.rand(k) < 0.4).astype(int)
        pp = np.clip(0.2 + 0.55 * yy + rng.randn(k) * 0.15, 0.01, 0.99)
        gid.extend([f"g{gi}"] * k); y.extend(yy); p.extend(pp)
    return np.asarray(gid), np.asarray(y), np.asarray(p)


def test_exact_quantile_finite_sample_and_fallback():
    s = np.array([0.1, 0.4, 0.7])
    assert exact_quantile(s, 0.5) == pytest.approx(0.4)   # k=ceil(4*0.5)=2
    assert exact_quantile(s, 0.01) == float("inf")        # k=4 > m
    assert exact_quantile(np.array([]), 0.5) == float("inf")


def test_wilson_bounds_contain_point_estimate():
    lo, hi = wilson_ci(93, 100)
    assert lo <= 0.93 <= hi and 0.80 < lo


def test_standard_coverage_reasonable():
    rng = np.random.RandomState(7)
    gid, y, p = _make_groups(rng)
    gid_t, y_t, p_t = _make_groups(rng)
    _, S_cal, _ = group_supremum_scores(gid, y, p)
    res = evaluate_group_coverage(gid_t, y_t, p_t, exact_quantile(S_cal, 0.05))
    assert 0.85 <= res["coverage"] <= 1.0
    assert res["n_groups"] == 120


def test_mondrian_size_runs_and_covers():
    rng = np.random.RandomState(11)
    gid, y, p = _make_groups(rng)
    gid_t, y_t, p_t = _make_groups(rng)
    res = evaluate_mondrian_size(gid, y, p, gid_t, y_t, p_t, 0.05)
    assert 0.85 <= res["coverage"] <= 1.0
    assert res["m_small"] + res["m_large"] == 120


def test_mondrian_parity_runs():
    rng = np.random.RandomState(13)
    gid, y, p = _make_groups(rng)
    gid_t, y_t, p_t = _make_groups(rng)
    res = evaluate_mondrian_parity(gid, y, p, gid_t, y_t, p_t, 0.05)
    assert 0.8 <= res["coverage"] <= 1.0


def test_alpha_tuning_is_conservative_max():
    rng = np.random.RandomState(17)
    gid, y, p = _make_groups(rng, n_groups=160)
    out = alpha_split_sample_tuning(gid, y, p, 0.95, random_state=5)
    assert out["alpha_star_conservative"] == max(
        out["alpha_star_first"], out["alpha_star_swap"])
    assert 0.0 < out["alpha_star_conservative"] <= 0.2
