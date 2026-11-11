"""
Evidence Dossier Generation Module (Phase 3.6).

Assembles comprehensive, structured, audit-ready evidence dossiers for candidate pairs
to facilitate human review by academic integrity panels.
"""

import datetime
from typing import Dict, Any, List, Optional
from research_upgrade.src.routing.decision_states import DecisionState


def generate_pair_evidence_dossier(
    pair_id: str,
    modality: str,
    stage1_score: float,
    calibrated_probability: float,
    predictive_entropy: float,
    decision_state: DecisionState,
    final_prediction: int,
    stage2_score: Optional[float] = None,
    ground_truth_label: Optional[int] = None,
    feature_contributions: Optional[Dict[str, float]] = None,
    extracted_evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Constructs an evidence dossier for an individual pair.
    """
    dossier = {
        "dossier_metadata": {
            "pair_id": pair_id,
            "modality": modality,
            "generation_timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "framework_version": "Phase 3.6 Integrated Proposed Framework",
            "audit_compliance": "GOVERNANCE_STRICT",
        },
        "classification_and_routing": {
            "stage1_candidate_score": round(float(stage1_score), 4),
            "calibrated_positive_probability": round(float(calibrated_probability), 4),
            "predictive_entropy_bits": round(float(predictive_entropy), 4),
            "stage2_verification_score": (
                round(float(stage2_score), 4) if stage2_score is not None else None
            ),
            "routing_decision_state": str(decision_state),
            "decision_state_description": DecisionState.get_description(decision_state),
            "final_predicted_label": int(final_prediction),
            "ground_truth_label": int(ground_truth_label) if ground_truth_label is not None else None,
            "human_review_required": bool(final_prediction == 1 or decision_state == DecisionState.HIGH_RISK_FLAGGED),
        },
        "feature_view_attributions": feature_contributions or {},
        "traceable_evidence": extracted_evidence or {},
        "scientific_disclaimer": (
            "NOTICE: Algorithmic similarity measurements, calibrated probabilities, and "
            "automated evidence spans represent statistical decision-support indicators. "
            "They do not constitute legal proof or definitive adjudication of academic misconduct. "
            "Final determinations must be rendered by human academic integrity committees."
        ),
    }
    return dossier

