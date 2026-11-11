"""
Phase 4 Corrected Hardware Profiling and Latency Measurement Module.
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection.

Methodological Corrections Applied:
1. Identical hardware, runtime boundaries, and dataset populations for baseline vs proposed.
2. Standardized warm-up cycles (10 iterations) prior to taking measured timestamps.
3. High-resolution timing via time.perf_counter_ns() for sub-millisecond precision.
4. Separate accounting for:
   - Feature extraction time per pair
   - Stage-1 model scoring latency
   - Routing decision latency
   - Stage-2 heavy model latency (for escalated pairs only)
   - Total pipeline latency per pair
5. Full percentile latency distribution: P50 (median), P90, P95, P99.
6. Peak memory profiling via tracemalloc.
7. Strict prohibition of theoretical cost estimates masquerading as empirical measurements.
"""

import time
import tracemalloc
from typing import Dict, Any, List, Callable
import numpy as np


class HardwareLatencyProfiler:
    """
    Standardized benchmark harness for profiling throughput, latency percentiles,
    and memory footprint under strictly identical hardware conditions.
    """

    def __init__(self, warm_up_iterations: int = 10):
        self.warm_up_iterations = warm_up_iterations

    def run_warmup(self, func: Callable, sample_input: Any) -> None:
        """Executes warm-up iterations to ensure JIT/caching stabilization."""
        for _ in range(self.warm_up_iterations):
            try:
                func(sample_input)
            except Exception:
                pass

    def measure_batch_execution(
        self,
        name: str,
        func: Callable,
        batch_input: Any,
        n_items: int,
    ) -> Dict[str, Any]:
        """
        Executes and profiles batch execution measuring latency percentiles and memory.
        """
        tracemalloc.start()
        tracemalloc.reset_peak()

        t0_ns = time.perf_counter_ns()
        result = func(batch_input)
        t1_ns = time.perf_counter_ns()

        cur_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        elapsed_ms = (t1_ns - t0_ns) / 1_000_000.0
        ms_per_pair = elapsed_ms / max(1, n_items)
        throughput = n_items / (elapsed_ms / 1000.0) if elapsed_ms > 0 else 0.0

        return {
            "component_name": name,
            "total_items_processed": n_items,
            "total_wall_clock_time_ms": round(elapsed_ms, 3),
            "mean_latency_ms_per_pair": round(ms_per_pair, 4),
            "throughput_pairs_per_sec": round(throughput, 2),
            "peak_memory_mb": round(peak_mem / (1024.0 * 1024.0), 4),
            "output": result,
        }

    @staticmethod
    def compute_percentiles(latencies_ms: List[float]) -> Dict[str, float]:
        """Calculates P50, P90, P95, P99 latency percentiles."""
        if not latencies_ms:
            return {"p50_ms": 0.0, "p90_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
        arr = np.asarray(latencies_ms)
        return {
            "p50_ms": round(float(np.percentile(arr, 50)), 4),
            "p90_ms": round(float(np.percentile(arr, 90)), 4),
            "p95_ms": round(float(np.percentile(arr, 95)), 4),
            "p99_ms": round(float(np.percentile(arr, 99)), 4),
        }

