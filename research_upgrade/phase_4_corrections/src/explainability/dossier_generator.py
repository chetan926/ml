"""
Phase 4 Corrected Evidence Dossier Generator.
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection.
"""

from typing import Dict, Any, List, Optional
import time


def generate_pair_evidence_dossier(
    pair_id: str,
    modality: str,
    stage1_score: float,
    calibrated_probability: float,
    predictive_entropy: float,
    decision_state: str,
    final_prediction: int,
    extracted_evidence: Dict[str, Any],
    stage2_score: Optional[float] = None,
    ground_truth_label: Optional[int] = None,
    premise_group_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Generates an audit-ready, immutable JSON Evidence Dossier for an individual pair."""
    dossier = {
        "dossier_schema_version": "4.0.0_PHASE4_CORRECTED",
        "pair_id": str(pair_id),
        "premise_group_id": str(premise_group_id) if premise_group_id is not None else None,
        "modality": modality,
        "decision_metadata": {
            "stage1_screening_score": round(float(stage1_score), 4),
            "calibrated_positive_probability": round(float(calibrated_probability), 4),
            "predictive_entropy_bits": round(float(predictive_entropy), 4),
            "routing_decision_state": str(decision_state),
            "stage2_verification_score": (
                round(float(stage2_score), 4) if stage2_score is not None else None
            ),
            "final_predicted_label": int(final_prediction),
            "ground_truth_label": int(ground_truth_label) if ground_truth_label is not None else None,
        },
        "extracted_evidence": extracted_evidence,
        "legal_and_adjudicatory_notice": (
            "NOTICE TO ACADEMIC INTEGRITY REVIEW BOARD: Algorithmic similarity metrics and "
            "automated span alignments provided in this dossier constitute statistical indicators. "
            "They do not establish intentional academic misconduct, plagiarism, or contract cheating. "
            "Institutional review policies require independent qualitative assessment by qualified academic evaluators."
        ),
    }
    return dossier


def compute_dossier_metrics(
    total_flagged_escalated_pairs: int,
    generated_dossiers: List[Dict[str, Any]],
    explanation_latencies_ms: List[float],
) -> Dict[str, Any]:
    """
    Computes rigorous metrics for the evidence generation module:
    - Dossier generation coverage (% of flagged/escalated pairs successfully profiled)
    - Latency distribution
    - Declares explicit omission of heuristic precision/recall/F1.
    """
    n_generated = len(generated_dossiers)
    coverage = (n_generated / max(1, total_flagged_escalated_pairs)) * 100.0

    mean_lat = float(sum(explanation_latencies_ms) / max(1, len(explanation_latencies_ms)))
    p95_lat = (
        float(sorted(explanation_latencies_ms)[int(len(explanation_latencies_ms) * 0.95)])
        if explanation_latencies_ms
        else 0.0
    )

    return {
        "total_flagged_or_escalated_pairs": total_flagged_escalated_pairs,
        "dossiers_generated_count": n_generated,
        "dossier_generation_coverage_percent": round(coverage, 2),
        "mean_explanation_latency_ms": round(mean_lat, 3),
        "p95_explanation_latency_ms": round(p95_lat, 3),
        "ground_truth_annotations_status": "UNAVAILABLE",
        "evidence_precision_recall_f1_status": (
            "OMITTED: In accordance with scientific integrity guidelines, span-level precision, "
            "recall, and F1 are not reported because no independently annotated ground-truth "
            "character-level span dataset exists for this corpus."
        ),
    }

