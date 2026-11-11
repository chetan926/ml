"""
Unit tests for Explainable Evidence Aggregation (Phase 3.6).
"""

import unittest
from research_upgrade.src.routing.decision_states import DecisionState
from research_upgrade.src.explainability import (
    TextEvidenceExtractor,
    CodeEvidenceExtractor,
    generate_pair_evidence_dossier,
    compute_evidence_metrics,
)


class TestExplainability(unittest.TestCase):
    """Test suite for Phase 3.6 evidence extractors, dossiers, and metrics."""

    def test_text_evidence_extractor(self):
        extractor = TextEvidenceExtractor(min_phrase_len=3)
        t_a = "The quick brown fox jumps over the lazy dog in the sunny park."
        t_b = "A quick brown fox jumps over the sleeping cat in the sunny park."
        ev = extractor.extract_text_evidence(t_a, t_b, score=0.85, calibrated_prob=0.88, entropy=0.3)

        self.assertEqual(ev["modality"], "text")
        self.assertGreater(ev["matching_phrase_count"], 0)
        self.assertGreater(len(ev["matching_phrases"]), 0)
        self.assertIn("quick brown fox jumps over the", ev["matching_phrases"][0]["matched_phrase"])
        self.assertGreater(ev["token_evidence_coverage_ratio"], 0.2)

    def test_code_evidence_extractor(self):
        extractor = CodeEvidenceExtractor(min_token_tile_len=3)
        c_a = """
        public class MathUtil {
            public int add(int a, int b) {
                int sum = a + b;
                return sum;
            }
        }
        """
        c_b = """
        public class MathUtilRenamed {
            public int addNumbers(int x, int y) {
                int sum = x + y;
                return sum;
            }
        }
        """
        ev = extractor.extract_code_evidence(c_a, c_b, score=0.82, calibrated_prob=0.80, entropy=0.4)
        self.assertEqual(ev["modality"], "code")
        self.assertGreater(ev["matched_line_blocks_count"], 0)
        self.assertIn("identifier_mappings", ev)

    def test_generate_dossier(self):
        dossier = generate_pair_evidence_dossier(
            pair_id="pair_1234",
            modality="text",
            stage1_score=0.92,
            calibrated_probability=0.91,
            predictive_entropy=0.25,
            decision_state=DecisionState.HIGH_RISK_FLAGGED,
            final_prediction=1,
            ground_truth_label=1,
        )
        self.assertIn("dossier_metadata", dossier)
        self.assertIn("classification_and_routing", dossier)
        self.assertEqual(dossier["classification_and_routing"]["final_predicted_label"], 1)
        self.assertIn("scientific_disclaimer", dossier)

    def test_evidence_metrics(self):
        dossiers = [
            {
                "classification_and_routing": {"final_predicted_label": 1, "ground_truth_label": 1},
                "traceable_evidence": {"matching_phrases": [{"matched_phrase": "test"}], "token_evidence_coverage_ratio": 0.5},
            },
            {
                "classification_and_routing": {"final_predicted_label": 0, "ground_truth_label": 0},
                "traceable_evidence": {},
            },
        ]
        metrics = compute_evidence_metrics(dossiers, [5.0, 4.5])
        self.assertEqual(metrics["total_evaluated_dossiers"], 2)
        self.assertGreaterEqual(metrics["evidence_coverage_ratio"], 0.5)
        self.assertTrue(metrics["overhead_acceptable"])


if __name__ == "__main__":
    unittest.main()

