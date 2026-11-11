"""
Unified Computational Cost Profiler & Accounting Module.
Measures and calculates:
- Candidate Recall: retained_positives / total_actual_positives
- Pruning Ratio: pruned_candidates / total_candidates
- Expensive Detector Call Rate: expensive_calls / total_candidates
- Runtime Reduction: (baseline_runtime - pipeline_runtime) / baseline_runtime * 100
- Latency per candidate and peak RAM tracking.
"""

import time
import os
import tracemalloc
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class CostProfile:
    total_candidates: int
    pruned_candidates: int
    expensive_calls: int
    direct_flagged: int
    pruning_ratio: float
    expensive_call_rate: float
    direct_flag_rate: float
    candidate_recall: float
    baseline_runtime_sec: float
    pipeline_runtime_sec: float
    runtime_reduction_pct: float
    avg_latency_ms: float
    peak_memory_mb: float

class CostProfiler:
    def __init__(self):
        self.start_time = 0.0
        self.is_tracking_memory = False

    def start_profiling(self):
        """Starts timing and memory tracking."""
        tracemalloc.start()
        self.is_tracking_memory = True
        self.start_time = time.perf_counter()

    def stop_profiling(
        self,
        total_candidates: int,
        pruned_candidates: int,
        expensive_calls: int,
        direct_flagged: int,
        candidate_recall: float,
        baseline_runtime_sec: float = 0.0
    ) -> CostProfile:
        """Stops timing and memory tracking, computing standardized cost metrics."""
        elapsed_sec = time.perf_counter() - self.start_time
        peak_bytes = 0
        if self.is_tracking_memory:
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            self.is_tracking_memory = False

        peak_mb = float(peak_bytes / (1024 * 1024))
        pruning_ratio = float(pruned_candidates / total_candidates) if total_candidates > 0 else 0.0
        call_rate = float(expensive_calls / total_candidates) if total_candidates > 0 else 0.0
        flag_rate = float(direct_flagged / total_candidates) if total_candidates > 0 else 0.0

        if baseline_runtime_sec > 0:
            runtime_reduction_pct = float((baseline_runtime_sec - elapsed_sec) / baseline_runtime_sec * 100.0)
        else:
            runtime_reduction_pct = 0.0

        avg_latency_ms = float((elapsed_sec / total_candidates) * 1000.0) if total_candidates > 0 else 0.0

        return CostProfile(
            total_candidates=total_candidates,
            pruned_candidates=pruned_candidates,
            expensive_calls=expensive_calls,
            direct_flagged=direct_flagged,
            pruning_ratio=pruning_ratio,
            expensive_call_rate=call_rate,
            direct_flag_rate=flag_rate,
            candidate_recall=candidate_recall,
            baseline_runtime_sec=float(baseline_runtime_sec),
            pipeline_runtime_sec=float(elapsed_sec),
            runtime_reduction_pct=runtime_reduction_pct,
            avg_latency_ms=avg_latency_ms,
            peak_memory_mb=peak_mb
        )

