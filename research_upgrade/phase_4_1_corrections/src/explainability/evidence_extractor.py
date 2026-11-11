"""
Phase 4.1 Corrected Authentic Evidence Extractor for Text and Code Modalities.
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection.

Defect Fixes Implemented:
1. True Physical Line Numbers for Code:
   Preserves raw blank lines in original Java source files so physical line numbers
   match exactly without position shifts.
2. Authentic Raw Span Offsets for Text:
   Extracts genuine character offsets [start_char, end_char] directly from original documents.
3. No Synthetic Evidence:
   All extracted snippets and line blocks are drawn strictly from actual input pairs.
"""

from typing import Dict, Any, List, Tuple
import re
import difflib


class CorrectedTextEvidenceExtractor:
    """
    Extracts authentic text evidence spans directly from genuine suspicious/reference documents.
    """

    def __init__(self, min_span_chars: int = 15):
        self.min_span_chars = min_span_chars

    def extract_matching_spans(
        self, text_ref: str, text_sus: str
    ) -> List[Dict[str, Any]]:
        """
        Extracts verbatim or near-verbatim matching character spans with exact offsets.
        """
        if not text_ref or not text_sus:
            return []

        matcher = difflib.SequenceMatcher(None, text_ref, text_sus)
        matching_spans: List[Dict[str, Any]] = []

        for block in matcher.get_matching_blocks():
            if block.size >= self.min_span_chars:
                ref_sub = text_ref[block.a : block.a + block.size]
                sus_sub = text_sus[block.b : block.b + block.size]
                
                # Ignore whitespace-only or punctuation-only blocks
                if not re.search(r"\w{3,}", ref_sub):
                    continue

                matching_spans.append({
                    "feature_view": "lexical_surface_span_view",
                    "reference_char_start": block.a,
                    "reference_char_end": block.a + block.size,
                    "suspicious_char_start": block.b,
                    "suspicious_char_end": block.b + block.size,
                    "span_length_chars": block.size,
                    "reference_text_snippet": ref_sub[:120],
                    "suspicious_text_snippet": sus_sub[:120],
                })
        return matching_spans

    def extract_lexical_overlap_tokens(
        self, text_ref: str, text_sus: str
    ) -> Dict[str, Any]:
        """Extracts shared word stems and tokens."""
        tokens_ref = set(re.findall(r"\b[A-Za-z]{3,}\b", text_ref.lower()))
        tokens_sus = set(re.findall(r"\b[A-Za-z]{3,}\b", text_sus.lower()))
        shared = sorted(list(tokens_ref.intersection(tokens_sus)))
        jaccard = len(shared) / max(1, len(tokens_ref.union(tokens_sus)))

        return {
            "feature_view": "token_jaccard_view",
            "shared_tokens_count": len(shared),
            "shared_tokens_sample": shared[:15],
            "token_jaccard_similarity": round(jaccard, 4),
        }

    def generate_text_evidence(
        self, text_ref: str, text_sus: str
    ) -> Dict[str, Any]:
        """Compiles complete multi-view evidence for a text pair."""
        return {
            "surface_spans": self.extract_matching_spans(text_ref, text_sus),
            "token_overlap": self.extract_lexical_overlap_tokens(text_ref, text_sus),
            "evidence_extraction_notice": (
                "Offsets are 0-indexed character slices into original raw document texts. "
                "Documented overlaps do not establish intentional misconduct and require human review."
            ),
        }


class CorrectedCodeEvidenceExtractor:
    """
    Extracts line numbers, token tiles, and identifier mappings directly from genuine Java code pairs.
    Preserves exact physical line numbers by accounting for blank lines without position shifting.
    """

    def __init__(self, min_line_block: int = 2):
        self.min_line_block = min_line_block

    def extract_line_range_matches(
        self, code_ref: str, code_sus: str
    ) -> List[Dict[str, Any]]:
        """
        Extracts matching line ranges with true 1-indexed Java physical line numbers.
        Preserves blank lines in original source code to avoid line number drift.
        """
        if not code_ref or not code_sus:
            return []

        raw_lines_ref = code_ref.splitlines()
        raw_lines_sus = code_sus.splitlines()

        # Retain non-empty stripped lines alongside their true 0-indexed physical line numbers
        ref_indexed = [(idx, line.strip()) for idx, line in enumerate(raw_lines_ref) if line.strip()]
        sus_indexed = [(idx, line.strip()) for idx, line in enumerate(raw_lines_sus) if line.strip()]

        if not ref_indexed or not sus_indexed:
            return []

        ref_tokens = [item[1] for item in ref_indexed]
        sus_tokens = [item[1] for item in sus_indexed]

        matcher = difflib.SequenceMatcher(None, ref_tokens, sus_tokens)
        line_matches: List[Dict[str, Any]] = []

        for block in matcher.get_matching_blocks():
            if block.size >= self.min_line_block:
                ref_phys_start = ref_indexed[block.a][0] + 1
                ref_phys_end = ref_indexed[block.a + block.size - 1][0] + 1
                sus_phys_start = sus_indexed[block.b][0] + 1
                sus_phys_end = sus_indexed[block.b + block.size - 1][0] + 1

                line_matches.append({
                    "feature_view": "code_structural_line_view",
                    "reference_line_start": ref_phys_start,
                    "reference_line_end": ref_phys_end,
                    "suspicious_line_start": sus_phys_start,
                    "suspicious_line_end": sus_phys_end,
                    "matched_line_count": block.size,
                    "sample_matched_content": raw_lines_ref[ref_phys_start - 1][:100],
                })
        return line_matches

    def extract_identifier_mappings(
        self, code_ref: str, code_sus: str
    ) -> Dict[str, Any]:
        """Identifies identifier alignments and potential renaming substitutions."""
        keywords = {
            "public", "private", "protected", "class", "void", "static", "int", "double",
            "float", "boolean", "if", "else", "for", "while", "return", "new", "import",
            "package", "true", "false", "null", "this", "super", "try", "catch", "finally"
        }
        id_ref = set(re.findall(r"\b[A-Za-z_]\w*\b", code_ref)) - keywords
        id_sus = set(re.findall(r"\b[A-Za-z_]\w*\b", code_sus)) - keywords

        exact_matches = sorted(list(id_ref.intersection(id_sus)))
        potential_renames: Dict[str, str] = {}
        unmatched_ref = id_ref - id_sus
        for r_name in unmatched_ref:
            close = difflib.get_close_matches(r_name, list(id_sus), n=1, cutoff=0.7)
            if close:
                potential_renames[r_name] = close[0]

        return {
            "feature_view": "identifier_mapping_view",
            "exact_matching_identifiers": exact_matches[:10],
            "exact_matching_count": len(exact_matches),
            "potential_renamed_identifiers": potential_renames,
        }

    def generate_code_evidence(
        self, code_ref: str, code_sus: str
    ) -> Dict[str, Any]:
        """Compiles complete multi-view evidence for a code pair."""
        return {
            "line_matches": self.extract_line_range_matches(code_ref, code_sus),
            "identifier_mappings": self.extract_identifier_mappings(code_ref, code_sus),
            "evidence_extraction_notice": (
                "Line numbers are 1-based physical lines in the original source files. "
                "Structural alignments provide suspicious indicators and require human review."
            ),
        }

