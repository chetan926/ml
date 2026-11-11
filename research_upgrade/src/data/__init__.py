"""
Data loading and partition validation package (Phase 3.2).
"""

from research_upgrade.src.data.schema_validation import validate_text_schema, validate_code_schema
from research_upgrade.src.data.partition_validation import validate_text_partitions, validate_code_partitions
from research_upgrade.src.data.text_loader import TextBenchmarkDataLoader
from research_upgrade.src.data.code_loader import CodeBenchmarkDataLoader

__all__ = [
    "validate_text_schema",
    "validate_code_schema",
    "validate_text_partitions",
    "validate_code_partitions",
    "TextBenchmarkDataLoader",
    "CodeBenchmarkDataLoader",
]

