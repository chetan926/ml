"""
Evaluation and benchmarking package (Phase 3.2).
"""

from research_upgrade.src.evaluation.feature_quality import validate_feature_matrix_quality
from research_upgrade.src.evaluation.feature_cost import benchmark_feature_extraction_cost

__all__ = [
    "validate_feature_matrix_quality",
    "benchmark_feature_extraction_cost"
]

