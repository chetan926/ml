"""
Candidate Generation Package (Phase 3.3)
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection
"""

from research_upgrade.src.candidate_generation.candidate_evaluation import (
    CandidateMetrics,
    evaluate_candidate_threshold,
    compute_operating_curve
)
from research_upgrade.src.candidate_generation.threshold_selection import (
    ThresholdSelector
)
from research_upgrade.src.candidate_generation.score_fusion import (
    BaseCandidateScorer,
    SingleViewThresholdScorer,
    WeightedMultiViewScorer,
    LogisticRegressionCandidateScorer,
    RandomForestCandidateScorer,
    GradientBoostingCandidateScorer
)
from research_upgrade.src.candidate_generation.text_candidate_scorer import (
    TextCandidateScorerPipeline
)
from research_upgrade.src.candidate_generation.code_candidate_scorer import (
    CodeCandidateScorerPipeline
)
from research_upgrade.src.candidate_generation.ablation_experiments import (
    run_text_ablation_experiments,
    run_code_ablation_experiments
)
from research_upgrade.src.candidate_generation.robustness_analysis import (
    evaluate_text_robustness_subsets,
    evaluate_code_robustness_by_obfuscation
)

__all__ = [
    "CandidateMetrics",
    "evaluate_candidate_threshold",
    "compute_operating_curve",
    "ThresholdSelector",
    "BaseCandidateScorer",
    "SingleViewThresholdScorer",
    "WeightedMultiViewScorer",
    "LogisticRegressionCandidateScorer",
    "RandomForestCandidateScorer",
    "GradientBoostingCandidateScorer",
    "TextCandidateScorerPipeline",
    "CodeCandidateScorerPipeline",
    "run_text_ablation_experiments",
    "run_code_ablation_experiments",
    "evaluate_text_robustness_subsets",
    "evaluate_code_robustness_by_obfuscation"
]

