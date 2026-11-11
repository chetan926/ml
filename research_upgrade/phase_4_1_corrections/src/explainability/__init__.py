"""Explainability package for Phase 4.1."""
from .evidence_extractor import CorrectedTextEvidenceExtractor, CorrectedCodeEvidenceExtractor
from .dossier_generator import generate_pair_evidence_dossier, compute_dossier_metrics

__all__ = [
    "CorrectedTextEvidenceExtractor",
    "CorrectedCodeEvidenceExtractor",
    "generate_pair_evidence_dossier",
    "compute_dossier_metrics",
]

