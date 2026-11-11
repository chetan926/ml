"""
Explainability Package Exports (Phase 3.6).
"""

from research_upgrade.src.explainability.evidence_extractor import (
    TextEvidenceExtractor,
    CodeEvidenceExtractor,
)
from research_upgrade.src.explainability.dossier_generator import (
    generate_pair_evidence_dossier,
)
from research_upgrade.src.explainability.evidence_metrics import (
    compute_evidence_metrics,
)

__all__ = [
    "TextEvidenceExtractor",
    "CodeEvidenceExtractor",
    "generate_pair_evidence_dossier",
    "compute_evidence_metrics",
]

