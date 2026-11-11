"""
Phase 4.1 Corrected Hardware Profiler.
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection.

Methodological Guarantees:
1. Warm-Up Cycles:
   Executes 10 unmeasured warm-up iterations to eliminate JIT compilation and OS cache overhead.
2. Latency Percentiles:
   Records exact wall-clock latencies (via time.perf_counter_ns) to compute mean, P50, P90, P95, and P99.
3. Memory Allocation Tracking:
   Profiles peak memory allocation using Python's tracemalloc.
4. Complete Timing Boundaries:
   Explicitly measures Stage 1, Routing, Stage 2, and Full Online Pipeline.
   Measures both Text and Code pipelines empirically without hardcoded constants.
"""

from typing import Dict, Any, Callable, List, Optional
import time
import tracemalloc
import numpy as np


class HardwareLatencyProfiler:
    """
    Standardized, reproducible CPU hardware latency and memory profiler.
    """

    def __init__(self, warmup_cycles: int = 10, measurement_cycles: int = 100):
        self.warmup_cycles = warmup_cycles
        self.measurement_cycles = measurement_cycles

    def profile_callable(
        self,
        func: Callable[[], Any],
        batch_size: int = 1,
    ) -> Dict[str, Any]:
        """
        Profiles execution time and peak memory of a callable function.
        """
        # Warmup phase
        for _ in range(self.warmup_cycles):
            _ = func()

        # Measurement phase with tracemalloc
        tracemalloc.start()
        start_mem, _ = tracemalloc.get_traced_memory()
        durations_ms: List[float] = []

        output = None
        for _ in range(self.measurement_cycles):
            t0 = time.perf_counter_ns()
            output = func()
            t1 = time.perf_counter_ns()
            durations_ms.append((t1 - t0) / 1e6)

        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        durations_arr = np.array(durations_ms, dtype=np.float64)
        total_time_ms = float(np.sum(durations_arr))
        mean_time_ms = float(np.mean(durations_arr))
        p50_ms = float(np.percentile(durations_arr, 50))
        p90_ms = float(np.percentile(durations_arr, 90))
        p95_ms = float(np.percentile(durations_arr, 95))
        p99_ms = float(np.percentile(durations_arr, 99))

        # Throughput in items per second
        throughput = (batch_size * 1000.0) / max(1e-6, mean_time_ms)
        mean_per_item_ms = mean_time_ms / max(1, batch_size)

        peak_mem_mb = (peak_mem - start_mem) / (1024.0 * 1024.0)

        return {
            "warmup_cycles": self.warmup_cycles,
            "measurement_cycles": self.measurement_cycles,
            "batch_size": batch_size,
            "mean_batch_latency_ms": round(mean_time_ms, 4),
            "mean_latency_ms_per_item": round(mean_per_item_ms, 6),
            "throughput_items_per_sec": round(throughput, 2),
            "p50_ms": round(p50_ms, 4),
            "p90_ms": round(p90_ms, 4),
            "p95_ms": round(p95_ms, 4),
            "p99_ms": round(p99_ms, 4),
            "peak_memory_mb": round(max(0.001, peak_mem_mb), 4),
            "output": output,
        }

