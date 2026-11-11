"""
Computational Cost Accounting and Latency Profiling Module (Phase 3.5).

Measures and profiles fine-grained latency and computational resource usage across:
- Offline vs Online pipeline stages
- Stage-1 feature extraction and candidate scoring
- Budget-constrained routing decisions
- Stage-2 heavy verification model invocations
- Explainable evidence generation
- Statistical distributions: Mean, Median (P50), P90, P95, P99, Std Dev, Throughput
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional, Callable


class ComputationalCostProfiler:
    """
    Profiles component and end-to-end execution latencies.
    """

    def __init__(self):
        self.stage_timings: Dict[str, List[float]] = {}

    def record_timing(self, stage_name: str, duration_ms: float):
        if stage_name not in self.stage_timings:
            self.stage_timings[stage_name] = []
        self.stage_timings[stage_name].append(duration_ms)

    def benchmark_callable(
        self,
        func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        warmup_runs: int = 5,
        benchmark_runs: int = 20,
    ) -> List[float]:
        """Runs warmup and timed benchmark iterations, returning latency per run in ms."""
        kwargs = kwargs or {}
        for _ in range(warmup_runs):
            func(*args, **kwargs)

        timings = []
        for _ in range(benchmark_runs):
            t0 = time.perf_counter()
            func(*args, **kwargs)
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)
        return timings

    @staticmethod
    def compute_latency_distribution(timings_ms: List[float]) -> Dict[str, float]:
        """Calculates standard percentile latency metrics in milliseconds."""
        arr = np.asarray(timings_ms, dtype=np.float64)
        if len(arr) == 0:
            return {
                "mean_ms": 0.0,
                "median_ms": 0.0,
                "std_ms": 0.0,
                "p90_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
            }
        return {
            "mean_ms": round(float(np.mean(arr)), 4),
            "median_ms": round(float(np.median(arr)), 4),
            "std_ms": round(float(np.std(arr)), 4),
            "p90_ms": round(float(np.percentile(arr, 90)), 4),
            "p95_ms": round(float(np.percentile(arr, 95)), 4),
            "p99_ms": round(float(np.percentile(arr, 99)), 4),
            "min_ms": round(float(np.min(arr)), 4),
            "max_ms": round(float(np.max(arr)), 4),
        }

    @staticmethod
    def calculate_system_throughput(mean_latency_ms_per_pair: float) -> float:
        """Returns pairs processed per second."""
        if mean_latency_ms_per_pair <= 0:
            return 0.0
        return round(1000.0 / mean_latency_ms_per_pair, 2)


def compute_routing_cost_summary(
    total_pairs: int,
    pruned_count: int,
    escalated_count: int,
    flagged_direct_count: int,
    feature_extraction_ms_per_pair: float,
    candidate_scoring_ms_per_pair: float,
    routing_ms_per_pair: float,
    stage2_ms_per_pair: float,
    evidence_ms_per_pair: float,
) -> Dict[str, Any]:
    """
    Computes rigorous end-to-end computational cost accounting comparing:
    1. Unrouted All-Expensive Baseline (all pairs invoke Stage-2)
    2. Proposed Budget-Constrained Uncertainty-Aware Routed System
    """
    # Baseline: Every pair executes Stage-1 features, Stage-2 heavy model, and evidence for flagged
    baseline_stage2_calls = total_pairs
    baseline_total_ms = (
        total_pairs * feature_extraction_ms_per_pair
        + total_pairs * candidate_scoring_ms_per_pair
        + baseline_stage2_calls * stage2_ms_per_pair
        + (total_pairs * 0.35) * evidence_ms_per_pair  # approximate positive rate evidence
    )
    baseline_ms_per_pair = baseline_total_ms / max(1, total_pairs)

    # Proposed Routed System
    actual_stage2_calls = escalated_count
    calls_avoided = baseline_stage2_calls - actual_stage2_calls
    calls_avoided_pct = (calls_avoided / baseline_stage2_calls) * 100.0 if baseline_stage2_calls > 0 else 0.0

    flagged_total = flagged_direct_count + int(actual_stage2_calls * 0.5)  # estimated flagged from escalated
    proposed_total_ms = (
        total_pairs * feature_extraction_ms_per_pair
        + total_pairs * candidate_scoring_ms_per_pair
        + total_pairs * routing_ms_per_pair
        + actual_stage2_calls * stage2_ms_per_pair
        + flagged_total * evidence_ms_per_pair
    )
    proposed_ms_per_pair = proposed_total_ms / max(1, total_pairs)

    latency_reduction_pct = (
        ((baseline_ms_per_pair - proposed_ms_per_pair) / baseline_ms_per_pair) * 100.0
        if baseline_ms_per_pair > 0 else 0.0
    )

    return {
        "total_pairs_evaluated": total_pairs,
        "pruned_pairs": pruned_count,
        "pruning_ratio": round(pruned_count / max(1, total_pairs), 4),
        "escalated_pairs_to_stage2": escalated_count,
        "escalation_ratio": round(escalated_count / max(1, total_pairs), 4),
        "flagged_direct_pairs": flagged_direct_count,
        "heavy_model_calls_baseline": baseline_stage2_calls,
        "heavy_model_calls_proposed": actual_stage2_calls,
        "heavy_model_calls_avoided": calls_avoided,
        "heavy_model_reduction_percentage": round(calls_avoided_pct, 2),
        "baseline_mean_latency_ms_per_pair": round(baseline_ms_per_pair, 4),
        "proposed_mean_latency_ms_per_pair": round(proposed_ms_per_pair, 4),
        "latency_reduction_percentage": round(latency_reduction_pct, 2),
        "baseline_throughput_pairs_per_sec": ComputationalCostProfiler.calculate_system_throughput(baseline_ms_per_pair),
        "proposed_throughput_pairs_per_sec": ComputationalCostProfiler.calculate_system_throughput(proposed_ms_per_pair),
        "computational_cost_per_1000_pairs_sec": round((proposed_total_ms / total_pairs), 3),
    }

