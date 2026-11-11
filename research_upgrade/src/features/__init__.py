"""
Multi-view feature extraction package (Phase 3.2).
"""

from research_upgrade.src.features.text_lexical import TextLexicalExtractor
from research_upgrade.src.features.text_character import TextCharacterExtractor
from research_upgrade.src.features.text_semantic import TextSemanticExtractor
from research_upgrade.src.features.text_structural import TextStructuralExtractor
from research_upgrade.src.features.code_token import CodeTokenExtractor
from research_upgrade.src.features.code_identifier import CodeIdentifierNormalizedExtractor
from research_upgrade.src.features.code_ast import CodeStructuralExtractor
from research_upgrade.src.features.code_complexity import CodeComplexityExtractor
from research_upgrade.src.features.feature_pipeline import (
    TextMultiViewPipeline,
    CodeMultiViewPipeline,
    TEXT_FEATURE_COLUMNS,
    CODE_FEATURE_COLUMNS
)

__all__ = [
    "TextLexicalExtractor",
    "TextCharacterExtractor",
    "TextSemanticExtractor",
    "TextStructuralExtractor",
    "CodeTokenExtractor",
    "CodeIdentifierNormalizedExtractor",
    "CodeStructuralExtractor",
    "CodeComplexityExtractor",
    "TextMultiViewPipeline",
    "CodeMultiViewPipeline",
    "TEXT_FEATURE_COLUMNS",
    "CODE_FEATURE_COLUMNS"
]

