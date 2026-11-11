"""
Unit tests for Routing and Cost Optimization Module (Phase 3.5).
"""

import unittest
import numpy as np
import pandas as pd
from research_upgrade.src.routing import (
    DecisionState,
    TextStage2Verifier,
    CodeStage2Verifier,
    BudgetAdaptiveRouter,
    simulate_routing_baselines,
    generate_pareto_frontier_points,
    is_pareto_efficient,
    ComputationalCostProfiler,
    compute_routing_cost_summary,
)


class TestRouting(unittest.TestCase):
    """Test suite for Phase 3.5 routing engine, cost profiler, and Pareto frontier."""

    def setUp(self):
        np.random.seed(42)
        self.N = 100
        self.scores = np.random.uniform(0.0, 1.0, size=self.N)
        self.probs = np.random.uniform(0.0, 1.0, size=self.N)
        self.labels = np.random.binomial(1, 0.4, size=self.N)
        self.stage2_scores = np.random.uniform(0.0, 1.0, size=self.N)

    def test_decision_states(self):
        self.assertEqual(DecisionState.LOW_RISK_PRUNED.value, "LOW_RISK_PRUNED")
        self.assertEqual(DecisionState.UNCERTAIN_ESCALATED.value, "UNCERTAIN_ESCALATED")
        self.assertEqual(DecisionState.HIGH_RISK_FLAGGED.value, "HIGH_RISK_FLAGGED")
        desc = DecisionState.get_description(DecisionState.LOW_RISK_PRUNED)
        self.assertIn("Excluded from expensive", desc)

    def test_budget_adaptive_router_single(self):
        router = BudgetAdaptiveRouter(tau_prune=0.1, tau_flag=0.8, entropy_threshold=0.7)
        # Low risk, low entropy
        st = router.route_single_pair(score=0.05, calibrated_prob=0.1, entropy=0.2)
        self.assertEqual(st, DecisionState.LOW_RISK_PRUNED)

        # High entropy -> escalate
        st = router.route_single_pair(score=0.05, calibrated_prob=0.1, entropy=0.9)
        self.assertEqual(st, DecisionState.UNCERTAIN_ESCALATED)

        # High risk, low entropy
        st = router.route_single_pair(score=0.9, calibrated_prob=0.85, entropy=0.3)
        self.assertEqual(st, DecisionState.HIGH_RISK_FLAGGED)

    def test_budget_adaptive_router_batch_and_optimize(self):
        router = BudgetAdaptiveRouter()
        res = router.optimize_thresholds_on_validation(
            self.scores, self.probs, self.labels, budget_fraction=0.8
        )
        self.assertTrue(router.is_optimized)
        batch_states = router.route_batch(self.scores, self.probs)
        self.assertEqual(len(batch_states), self.N)

    def test_simulate_routing_baselines(self):
        results = simulate_routing_baselines(
            self.scores, self.probs, self.labels, self.stage2_scores, budget_fraction=0.7
        )
        self.assertIn("all_stage2", results)
        self.assertIn("fixed_threshold", results)
        self.assertIn("random_pruning", results)
        self.assertIn("proposed_adaptive_routing", results)
        self.assertGreaterEqual(results["proposed_adaptive_routing"]["recall"], 0.0)

    def test_pareto_analysis(self):
        pareto_res = generate_pareto_frontier_points(
            [0.2, 0.4, 0.6, 0.8, 1.0],
            self.scores,
            self.probs,
            self.labels,
            self.stage2_scores,
        )
        self.assertIn("budget_evaluation_points", pareto_res)
        self.assertGreaterEqual(pareto_res["pareto_optimal_count"], 1)

    def test_cost_profiler_and_summary(self):
        profiler = ComputationalCostProfiler()
        timings = [10.0, 12.0, 11.5, 13.0, 9.5]
        dist = profiler.compute_latency_distribution(timings)
        self.assertAlmostEqual(dist["mean_ms"], 11.2, places=1)
        self.assertGreater(dist["p90_ms"], 10.0)

        cost_sum = compute_routing_cost_summary(
            total_pairs=1000,
            pruned_count=300,
            escalated_count=500,
            flagged_direct_count=200,
            feature_extraction_ms_per_pair=30.0,
            candidate_scoring_ms_per_pair=0.3,
            routing_ms_per_pair=0.1,
            stage2_ms_per_pair=40.0,
            evidence_ms_per_pair=5.0,
        )
        self.assertIn("heavy_model_reduction_percentage", cost_sum)
        self.assertEqual(cost_sum["heavy_model_reduction_percentage"], 50.0)
        self.assertGreater(cost_sum["latency_reduction_percentage"], 20.0)


if __name__ == "__main__":
    unittest.main()

