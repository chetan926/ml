"""
Automated Test Suite for Phase 4 Corrected Components.
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection.
"""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure project root in sys.path
import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from research_upgrade.phase_4_corrections.src.calibration.group_conformal import (
    CorrectedGroupConformalPredictor,
)
from research_upgrade.phase_4_corrections.src.explainability.evidence_extractor import (
    CorrectedTextEvidenceExtractor,
    CorrectedCodeEvidenceExtractor,
)
from research_upgrade.phase_4_corrections.src.explainability.dossier_generator import (
    generate_pair_evidence_dossier,
    compute_dossier_metrics,
)
from research_upgrade.phase_4_corrections.src.evaluation.statistical_testing import (
    compute_comprehensive_classification_metrics,
    compute_group_aware_bootstrap_ci,
    compute_paired_difference_bootstrap_ci,
    compute_mcnemar_significance,
    format_p_value,
)
from research_upgrade.phase_4_corrections.src.evaluation.profiling import (
    HardwareLatencyProfiler,
)


class TestPhase4ConformalPrediction(unittest.TestCase):
    """Tests exact finite-sample conformal prediction and boundary handling."""

    def test_finite_sample_quantile_exact(self):
        predictor = CorrectedGroupConformalPredictor(alpha=0.05)
        # m = 39 groups: k = ceil(40 * 0.95) = 38
        groups = np.repeat(np.arange(39), 2)
        y_true = np.ones(78, dtype=int)
        # Probabilities decreasing from 0.95 to 0.50
        y_prob = np.repeat(np.linspace(0.95, 0.50, 39), 2)

        predictor.calibrate(group_ids=groups, y_true=y_true, y_prob=y_prob)
        self.assertEqual(predictor.m_groups, 39)
        self.assertEqual(predictor.k_rank, 38)
        self.assertFalse(predictor.is_infinite_quantile)
        self.assertLess(predictor.calib_quantile, 1.0)

    def test_infinite_quantile_boundary(self):
        predictor = CorrectedGroupConformalPredictor(alpha=0.05)
        # m = 10 groups: k = ceil(11 * 0.95) = ceil(10.45) = 11 > 10
        groups = np.arange(10)
        y_true = np.ones(10, dtype=int)
        y_prob = np.full(10, 0.8)

        predictor.calibrate(group_ids=groups, y_true=y_true, y_prob=y_prob)
        self.assertEqual(predictor.k_rank, 11)
        self.assertTrue(predictor.is_infinite_quantile)
        self.assertEqual(predictor.calib_quantile, float("inf"))

        # When quantile is infinite, all prediction sets must be {0, 1}
        sets = predictor.predict_prediction_sets(np.array([0.1, 0.5, 0.9]))
        for s in sets:
            self.assertEqual(s, {0, 1})


class TestPhase4Explainability(unittest.TestCase):
    """Tests authentic evidence extraction without synthetic strings."""

    def test_text_char_spans(self):
        extractor = CorrectedTextEvidenceExtractor(min_phrase_len=3)
        ref = "The quick brown fox jumps over the lazy dog."
        sus = "A quick brown fox leaps over a lazy dog."
        ev = extractor.extract_text_evidence(
            text_ref=ref,
            text_sus=sus,
            candidate_score=0.85,
            calibrated_prob=0.82,
            predictive_entropy=0.67,
        )
        self.assertGreater(ev["matched_phrases_count"], 0)
        phrase = ev["matching_phrases"][0]
        self.assertEqual(phrase["matched_phrase"], "quick brown fox")
        self.assertIn("reference_char_span", phrase)
        self.assertIn("suspicious_char_span", phrase)
        self.assertEqual(phrase["feature_view"], "lexical_ngram_view")

    def test_dossier_generation_coverage(self):
        dossier = generate_pair_evidence_dossier(
            pair_id="PAIR_001",
            modality="text",
            stage1_score=0.75,
            calibrated_probability=0.80,
            predictive_entropy=0.72,
            decision_state="HIGH_RISK_FLAGGED",
            final_prediction=1,
            extracted_evidence={"matched_phrases_count": 2},
        )
        self.assertEqual(dossier["dossier_schema_version"], "4.0.0_PHASE4_CORRECTED")
        self.assertIn("NOTICE TO ACADEMIC INTEGRITY REVIEW BOARD", dossier["legal_and_adjudicatory_notice"])

        metrics = compute_dossier_metrics(10, [dossier], [1.5])
        self.assertEqual(metrics["dossier_generation_coverage_percent"], 10.0)
        self.assertEqual(metrics["ground_truth_annotations_status"], "UNAVAILABLE")


class TestPhase4StatisticalTesting(unittest.TestCase):
    """Tests group-aware bootstrap, paired difference CIs, and p-value formatting."""

    def test_format_p_value(self):
        self.assertEqual(format_p_value(0.0), "< 1e-16")
        self.assertEqual(format_p_value(1e-25), "< 1e-16")
        self.assertEqual(format_p_value(0.0001234), "1.2340e-04")
        self.assertEqual(format_p_value(0.045), "0.045000")

    def test_group_aware_bootstrap(self):
        groups = np.array(["g1", "g1", "g2", "g2", "g3", "g3", "g4", "g4"])
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 1, 0, 0, 1])

        ci = compute_group_aware_bootstrap_ci(groups, y_true, y_pred, n_resamples=50)
        self.assertIn("mcc", ci)
        self.assertIn("ci_lower", ci["mcc"])
        self.assertIn("ci_upper", ci["mcc"])

    def test_paired_difference_ci(self):
        groups = np.array(["g1", "g1", "g2", "g2", "g3", "g3", "g4", "g4"])
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_prop = np.array([1, 1, 0, 0, 1, 1, 0, 0])  # perfect
        y_base = np.array([0, 0, 1, 1, 0, 0, 1, 1])  # inverted

        paired = compute_paired_difference_bootstrap_ci(groups, y_true, y_prop, y_base, n_resamples=50)
        self.assertIn("delta_mcc", paired)
        self.assertGreater(paired["delta_mcc"]["mean_difference"], 0.0)


class TestPhase4Profiling(unittest.TestCase):
    """Tests hardware profiling and percentile latency distribution."""

    def test_percentile_computation(self):
        latencies = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        perc = HardwareLatencyProfiler.compute_percentiles(latencies)
        self.assertAlmostEqual(perc["p50_ms"], 0.55, places=2)
        self.assertGreater(perc["p95_ms"], perc["p50_ms"])


if __name__ == "__main__":
    unittest.main()

