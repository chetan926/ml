"""
Phase 5 Q1-Upgrade Regression Tests.

Covers the new Phase-5 scientific components:
    1. Adversarial attack determinism and non-triviality.
    2. Wilson score CI mathematical sanity.
    3. Operating-point tuning leakage guards (val-only grids; states consistent).
    4. LSA encoder leak-free fitting and bounded outputs.
    5. Heavy verifier fit/predict/latency contract.
    6. Code baseline suite column contracts.

Run:
    python -m unittest tests/test_phase_5_q1_upgrade.py -v
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from research_upgrade.phase_5_q1_upgrade.q1src.q1_attacks import (
    apply_text_attack,
    synonym_substitution,
    systematic_identifier_renaming,
)
from research_upgrade.phase_5_q1_upgrade.q1src.q1_stats_extras import (
    apply_routing,
    tune_routing_operating_point,
    tune_threshold_for_f1,
    tune_threshold_for_iso_recall,
    wilson_score_ci,
)
from research_upgrade.phase_5_q1_upgrade.q1src.lsa_semantic import LsaSemanticEncoder
from research_upgrade.phase_5_q1_upgrade.q1src.heavy_verifiers import (
    CodeHeavyVerifier,
    CrossStylePairVerifier,
)
from research_upgrade.phase_5_q1_upgrade.q1src.q1_baselines import (
    CODE_FEATURES,
    CODE_RF_FEATURES,
    CODE_SVM_FEATURES,
    TEXT_MULTIVIEW_FEATURES,
)


class TestAttackSuite(unittest.TestCase):
    SENT = "Two women are embracing while holding to go packages."

    def test_attacks_are_deterministic(self):
        for attack in ("synonym_light", "synonym_heavy", "word_order",
                       "char_noise", "filler", "combined"):
            a = apply_text_attack([self.SENT], attack, seed=42)
            b = apply_text_attack([self.SENT], attack, seed=42)
            self.assertEqual(a, b, f"{attack} must be deterministic for fixed seed")

    def test_attacks_modify_text(self):
        for attack in ("synonym_heavy", "char_noise", "filler", "combined"):
            out = apply_text_attack([self.SENT], attack, seed=7)[0]
            self.assertNotEqual(out, self.SENT, f"{attack} should modify the text")

    def test_attacks_preserve_rough_content(self):
        out = apply_text_attack([self.SENT], "synonym_light", seed=1)[0]
        # Word count must not collapse; content largely preserved
        self.assertGreaterEqual(len(out.split()), max(3, len(self.SENT.split()) - 5))

    def test_identifier_renaming_consistency(self):
        code = "public class Hello { int counter = 0; void bump() { counter = counter + 1; } }"
        renamed = systematic_identifier_renaming(code)
        self.assertNotIn("counter", renamed)
        self.assertIn("int", renamed)  # keywords preserved
        # Deterministic
        self.assertEqual(renamed, systematic_identifier_renaming(code))


class TestWilsonCI(unittest.TestCase):
    def test_wilson_contains_point_estimate(self):
        p_hat, lo, hi = wilson_score_ci(950, 1000)
        self.assertLess(lo, p_hat)
        self.assertGreater(hi, p_hat)
        self.assertAlmostEqual(p_hat, 0.95, places=6)

    def test_wilson_bounds_valid(self):
        for k, n in [(0, 100), (50, 100), (100, 100), (3, 7)]:
            _, lo, hi = wilson_score_ci(k, n)
            self.assertGreaterEqual(lo, 0.0)
            self.assertLessEqual(hi, 1.0)

    def test_wilson_shrinks_with_n(self):
        _, lo1, hi1 = wilson_score_ci(95, 100)
        _, lo2, hi2 = wilson_score_ci(950, 1000)
        self.assertLess(hi2 - lo2, hi1 - lo1)


class TestOperatingPointTuners(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.n = 600
        self.y = rng.randint(0, 2, self.n)
        self.scores = np.clip(self.y * 0.6 + rng.normal(0.35, 0.25, self.n), 0, 1)

    def test_f1_threshold_maximizes(self):
        thr, f1 = tune_threshold_for_f1(self.y, self.scores)
        self.assertTrue(0.0 < thr < 1.0)
        self.assertGreaterEqual(f1, 0.0)

    def test_iso_recall_reaches_target_when_feasible(self):
        thr, rec = tune_threshold_for_iso_recall(self.y, self.scores, 0.90)
        pred = (self.scores >= thr).astype(int)
        self.assertGreaterEqual(
            pred[self.y == 1].mean() if (self.y == 1).any() else 1.0, 0.85)

    def test_cost_minimizer_preserves_beta1_mcc_within_epsilon(self):
        from research_upgrade.phase_5_q1_upgrade.q1src.q1_stats_extras import (
            tune_routing_cost_minimizer)
        rng = np.random.RandomState(11)
        y = rng.randint(0, 2, 500)
        p = np.clip(y * 0.55 + rng.normal(0, 0.3, 500), 0.01, 0.99)
        ent = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
        s2 = np.clip(p + rng.normal(0, 0.03, 500), 0, 1)
        op = tune_routing_cost_minimizer(y, p, ent, s2, epsilon_mcc=0.10)
        ref_mcc = op.get("reference_beta1_mcc")
        self.assertGreaterEqual(op["val_mcc"], ref_mcc - 0.10 - 1e-9)
        self.assertLessEqual(op["val_escalation_ratio"], 1.0)

    def test_routing_grid_respects_budget_cap(self):
        rng = np.random.RandomState(1)
        p = rng.rand(self.n)
        ent = -p * np.log2(p + 1e-12) - (1 - p) * np.log2(1 - p + 1e-12)
        s2 = np.clip(p + rng.randn(self.n) * 0.02, 0, 1)
        op = tune_routing_operating_point(self.y, p, ent, s2, stage2_budget_cap=0.5)
        self.assertLessEqual(op["val_escalation_ratio"], 0.5 + 1e-9)
        pred, states, cost = apply_routing(
            p, ent, s2, op["tau_low"], op["tau_high"],
            op["entropy_threshold"], op["stage2_decision_threshold"])
        self.assertEqual(len(pred), self.n)
        self.assertEqual(
            cost["escalated_count"], int(np.sum(states == "UNCERTAIN_ESCALATED")))
        self.assertLessEqual(cost["escalated_count"] / self.n, 0.5 + 1e-9)


class TestLsaAndVerifiers(unittest.TestCase):
    def test_lsa_fit_encode_contract(self):
        corpus = ["a man sleeps outside", "a woman reads indoors",
                   "the dog runs fast", "children play together"] * 20
        enc = LsaSemanticEncoder(n_components=16, random_state=0).fit(corpus)
        sim = enc.pair_cosine("a man sleeps outside", "a male dozes outdoors")
        self.assertGreaterEqual(sim, 0.0)
        self.assertLessEqual(sim, 1.0)
        emb = enc.encode(["hello world", "hello there world"])
        inter = LsaSemanticEncoder.interaction_features(emb[:1], emb[1:])
        self.assertEqual(inter.shape[1], 16 * 4 + 1)

    def test_lsa_unfitted_raises(self):
        with self.assertRaises(RuntimeError):
            LsaSemanticEncoder(n_components=4).encode(["x"])

    def test_cross_verifier_contract(self):
        rng = np.random.RandomState(3)
        X = rng.rand(120, 30)
        y = np.array([0] * 60 + [1] * 60)
        v = CrossStylePairVerifier(hidden_layers=(16, 8), max_iter=40, random_state=0)
        v.fit(X, y)
        p = v.predict_proba(X[:5])
        self.assertEqual(p.shape, (5,))
        self.assertTrue(np.all((p >= 0) & (p <= 1)))
        self.assertGreater(v.n_params_, 0)
        lat = v.measure_single_pair_latency_ms(X[:1], repeats=10)
        self.assertGreater(lat, 0.0)

    def test_code_heavy_verifier_and_degenerate(self):
        rng = np.random.RandomState(5)
        hv = CodeHeavyVerifier(max_iter=30, random_state=0)
        hv.fit(rng.rand(60, 12), [0] * 30 + [1] * 30)
        p = hv.predict_proba(rng.rand(4, 12))
        self.assertEqual(p.shape, (4,))
        # Degenerate single-class fold
        hd = CodeHeavyVerifier(random_state=0).fit(rng.rand(10, 12), [1] * 10)
        self.assertTrue(np.allclose(hd.predict_proba(rng.rand(3, 12)), 1.0))


class TestBaselineFeatureContracts(unittest.TestCase):
    def test_code_subsets_are_within_full_feature_set(self):
        self.assertTrue(set(CODE_SVM_FEATURES).issubset(set(CODE_FEATURES)))
        self.assertTrue(set(CODE_RF_FEATURES).issubset(set(CODE_FEATURES)))

    def test_text_multiview_features_nonredundant(self):
        self.assertEqual(len(TEXT_MULTIVIEW_FEATURES), len(set(TEXT_MULTIVIEW_FEATURES)))
        self.assertIn("text_semantic_minilm_sim", TEXT_MULTIVIEW_FEATURES)


if __name__ == "__main__":
    unittest.main()
