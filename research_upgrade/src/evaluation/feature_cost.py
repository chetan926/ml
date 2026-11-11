"""
Feature Extraction Cost and Latency Benchmarking Module (Phase 3.2, Module G)
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection
"""

import time
import tracemalloc
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


def benchmark_feature_extraction_cost(
    extractor_func,
    sample_pairs: List[tuple],
    modality: str,
    extractor_name: str,
    warmup_runs: int = 5,
    model_load_time_ms: float = 0.0
) -> Dict[str, Any]:
    """
    Measures empirical runtime latency and memory footprint for feature extraction.
    
    Standards:
    - Separates model loading time from per-pair forward inference.
    - Records mean, median, 95th percentile, min, and max latency across evaluated pairs.
    - Measures peak memory allocation using tracemalloc.
    - Explicitly labels all values as empirical physical measurements (not simulated).
    """
    if not sample_pairs:
        raise ValueError("Cannot benchmark cost on empty sample pairs.")

    # Warmup runs to stabilize JIT/cache
    for i in range(min(warmup_runs, len(sample_pairs))):
        p1, p2 = sample_pairs[i]
        _ = extractor_func(p1, p2)

    latencies_ms: List[float] = []

    tracemalloc.start()
    t_start = time.perf_counter()

    for p1, p2 in sample_pairs:
        t0 = time.perf_counter()
        _ = extractor_func(p1, p2)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)

    total_time_s = time.perf_counter() - t_start
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    latencies_arr = np.array(latencies_ms)

    return {
        "modality": modality,
        "extractor_name": extractor_name,
        "sample_size_evaluated": len(sample_pairs),
        "total_extraction_time_seconds": round(total_time_s, 4),
        "model_loading_time_ms": round(model_load_time_ms, 2),
        "latency_per_pair_ms": {
            "mean": round(float(np.mean(latencies_arr)), 4),
            "median": round(float(np.median(latencies_arr)), 4),
            "p95": round(float(np.percentile(latencies_arr, 95)), 4),
            "min": round(float(np.min(latencies_arr)), 4),
            "max": round(float(np.max(latencies_arr)), 4),
            "std": round(float(np.std(latencies_arr)), 4)
        },
        "memory_usage_mb": {
            "peak_traced_mb": round(peak_mem / (1024 * 1024), 4),
            "current_traced_mb": round(current_mem / (1024 * 1024), 4)
        },
        "measurement_type": "EMPIRICAL_PHYSICAL_MEASUREMENT",
        "notes": "Model loading time is isolated and recorded separately from per-pair feature extraction."
    }

