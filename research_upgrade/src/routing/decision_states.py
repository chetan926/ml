"""
Three-Way Decision States for Adaptive Routing (Phase 3.5).

Defines the frozen operational state taxonomy:
- LOW_RISK_PRUNED: Excluded from expensive Stage-2 verification.
- UNCERTAIN_ESCALATED: Forwarded to high-capacity Stage-2 verification.
- HIGH_RISK_FLAGGED: Identified directly for human academic review.
"""

from enum import Enum
from typing import Dict, Any


class DecisionState(str, Enum):
    """
    Standard three-way routing decision states.
    """
    LOW_RISK_PRUNED = "LOW_RISK_PRUNED"
    UNCERTAIN_ESCALATED = "UNCERTAIN_ESCALATED"
    HIGH_RISK_FLAGGED = "HIGH_RISK_FLAGGED"

    @classmethod
    def get_description(cls, state: "DecisionState") -> str:
        descriptions = {
            cls.LOW_RISK_PRUNED: (
                "Excluded from expensive downstream Stage-2 verification based on validated "
                "low-risk score and low prediction uncertainty. (Pruned pairs are not guaranteed "
                "to be non-plagiarized; reflects high-throughput cost efficiency)."
            ),
            cls.UNCERTAIN_ESCALATED: (
                "Forwarded to computationally intensive Stage-2 deep verification model due to "
                "intermediate risk, boundary proximity, or elevated predictive entropy."
            ),
            cls.HIGH_RISK_FLAGGED: (
                "High-risk plagiarism suspect directly flagged for human academic review. "
                "(High-risk flags represent statistical alerts for review, not definitive proof "
                "of academic misconduct)."
            ),
        }
        return descriptions.get(state, "Unknown routing state.")

