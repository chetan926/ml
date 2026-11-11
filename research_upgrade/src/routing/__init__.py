"""
Routing Package Exports (Phase 3.5).
"""

from research_upgrade.src.routing.decision_states import DecisionState
from research_upgrade.src.routing.stage2_verifier import TextStage2Verifier, CodeStage2Verifier
from research_upgrade.src.routing.cost_model import (
    ComputationalCostProfiler,
    compute_routing_cost_summary,
)
from research_upgrade.src.routing.router import (
    BudgetAdaptiveRouter,
    simulate_routing_baselines,
)
from research_upgrade.src.routing.pareto import (
    generate_pareto_frontier_points,
    is_pareto_efficient,
)

__all__ = [
    "DecisionState",
    "TextStage2Verifier",
    "CodeStage2Verifier",
    "ComputationalCostProfiler",
    "compute_routing_cost_summary",
    "BudgetAdaptiveRouter",
    "simulate_routing_baselines",
    "generate_pareto_frontier_points",
    "is_pareto_efficient",
]

