"""
Evidence Quality and Explainability Evaluation Metrics (Phase 3.6).

Computes operational metrics for generated explainability artifacts:
- Evidence Precision
- Evidence Recall
- Evidence F1-score
- Evidence Coverage Ratio
- Explanation Generation Latency Overhead
"""

import numpy as np
from typing import Dict, Any, List


def compute_evidence_metrics(
    dossiers: List[Dict[str, Any]],
    generation_latencies_ms: List[float],
) -> Dict[str, Any]:
    """
    Computes summary explainability performance across a collection of pair dossiers.
    """
    total_dossiers = len(dossiers)
    if total_dossiers == 0:
        return {"total_dossiers": 0}

    non_empty_evidence_count = 0
    valid_overlap_count = 0
    total_flagged_positives = 0
    captured_evidence_for_positives = 0

    for d in dossiers:
        clf = d.get("classification_and_routing", {})
        ev = d.get("traceable_evidence", {})
        y_true = clf.get("ground_truth_label")
        is_pos = (y_true == 1) if y_true is not None else False
        final_pred = clf.get("final_predicted_label", 0)

        # Check evidence presence
        has_phrases = len(ev.get("matching_phrases", [])) > 0
        has_lines = len(ev.get("matched_line_ranges", [])) > 0
        has_tokens = len(ev.get("salient_shared_tokens", [])) > 0
        has_evidence = has_phrases or has_lines or has_tokens

        if has_evidence:
            non_empty_evidence_count += 1
            # Check overlap precision
            cov = ev.get("token_evidence_coverage_ratio", ev.get("line_coverage_ratio", 0.0))
            if cov > 0.0:
                valid_overlap_count += 1

        if is_pos and final_pred == 1:
            total_flagged_positives += 1
            if has_evidence:
                captured_evidence_for_positives += 1

    # Evidence coverage: fraction of generated dossiers with evidence
    evidence_coverage = non_empty_evidence_count / total_dossiers

    # Evidence precision: fraction of evidence packages with verified non-zero overlap
    evidence_precision = valid_overlap_count / max(1, non_empty_evidence_count)

    # Evidence recall: fraction of true positives that yield traceable evidence
    evidence_recall = (
        captured_evidence_for_positives / max(1, total_flagged_positives)
        if total_flagged_positives > 0 else 1.0
    )

    evidence_f1 = (
        2.0 * evidence_precision * evidence_recall / max(1e-8, evidence_precision + evidence_recall)
    )

    mean_latency = float(np.mean(generation_latencies_ms)) if generation_latencies_ms else 0.0

    return {
        "total_evaluated_dossiers": total_dossiers,
        "evidence_coverage_ratio": round(evidence_coverage, 4),
        "evidence_precision": round(evidence_precision, 4),
        "evidence_recall": round(evidence_recall, 4),
        "evidence_f1_score": round(evidence_f1, 4),
        "mean_explanation_latency_ms": round(mean_latency, 4),
        "overhead_acceptable": bool(mean_latency < 25.0),  # requirement: negligible overhead
    }

