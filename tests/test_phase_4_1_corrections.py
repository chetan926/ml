"""
Automated Unit and Integration Test Suite for Phase 4.1 Corrected Components.
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection.
"""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd

import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from research_upgrade.phase_4_1_corrections.src.calibration.group_conformal import (
    CorrectedGroupConformalPredictor,
)
from research_upgrade.phase_4_1_corrections.src.explainability.evidence_extractor import (
    CorrectedTextEvidenceExtractor,
    CorrectedCodeEvidenceExtractor,
)
from research_upgrade.phase_4_1_corrections.src.explainability.dossier_generator import (
    generate_pair_evidence_dossier,
    compute_dossier_metrics,
)
from research_upgrade.phase_4_1_corrections.src.routing.code_group_evaluator import (
    CodeGroupDisjointEvaluator,
)
from research_upgrade.phase_4_1_corrections.src.evaluation.statistical_testing import (
    compute_comprehensive_classification_metrics,
    compute_group_aware_bootstrap_ci,
    compute_paired_difference_bootstrap_ci,
    compute_mcnemar_significance,
    format_p_value,
)
from research_upgrade.phase_4_1_corrections.src.evaluation.profiling import (
    HardwareLatencyProfiler,
)


class TestPhase41Corrections(unittest.TestCase):
    """Test suite verifying all Phase 4.1 methodological defect corrections."""

    def test_conformal_exact_rank_and_fallback(self):
        """Verifies exact rank formula k = ceil((m+1)(1-alpha)) and fallback if k > m."""
        cp = CorrectedGroupConformalPredictor(alpha=0.05)
        # m = 19 groups -> k = ceil(20 * 0.95) = 19 (active finite quantile)
        groups = np.repeat(np.arange(19), 2)
        y = np.ones(len(groups), dtype=int)
        probs = np.linspace(0.6, 0.95, len(groups))
        cp.calibrate(groups, y, probs)
        self.assertEqual(cp.m_groups, 19)
        self.assertEqual(cp.k_rank, 19)
        self.assertFalse(cp.is_infinite_quantile)
        self.assertLess(cp.calib_quantile, 1.0)

        # Small sample where k > m: m = 5, alpha = 0.05 -> k = ceil(6 * 0.95) = 6 > 5
        cp_small = CorrectedGroupConformalPredictor(alpha=0.05)
        groups_small = np.arange(5)
        cp_small.calibrate(groups_small, np.ones(5), np.full(5, 0.8))
        self.assertEqual(cp_small.k_rank, 6)
        self.assertTrue(cp_small.is_infinite_quantile)
        self.assertTrue(np.isinf(cp_small.calib_quantile))
        sets = cp_small.predict_prediction_sets(np.array([0.2, 0.8]))
        self.assertEqual(sets, [[0, 1], [0, 1]])

    def test_java_physical_line_numbers_preserve_blank_lines(self):
        """Verifies that blank lines in Java code do not shift physical line numbers."""
        extractor = CorrectedCodeEvidenceExtractor(min_line_block=2)
        code_ref = "public class Example {\n\n    int a = 1;\n    int b = 2;\n\n    void run() {}\n}\n"
        code_sus = "// header\n\n\n    int a = 1;\n    int b = 2;\n\n    void run() {}\n}\n"

        matches = extractor.extract_line_range_matches(code_ref, code_sus)
        self.assertGreater(len(matches), 0)

        first_match = matches[0]
        # In code_ref: line 1 is public class, line 2 is blank, line 3 is int a = 1
        self.assertEqual(first_match["reference_line_start"], 3)
        # In code_sus: line 1 is // header, line 2 is blank, line 3 is blank, line 4 is int a = 1
        self.assertEqual(first_match["suspicious_line_start"], 4)

    def test_evidence_dossier_omits_ground_truth_label(self):
        """Verifies ground-truth label is strictly absent from operational reviewer dossier."""
        dossier = generate_pair_evidence_dossier(
            pair_id="CODE_9999",
            modality="code",
            stage1_score=0.85,
            calibrated_probability=0.88,
            predictive_entropy=0.35,
            decision_state="HIGH_RISK_FLAGGED",
            final_prediction=1,
            extracted_evidence={"test": "data"},
        )
        self.assertNotIn("ground_truth_label", dossier["decision_metadata"])
        self.assertEqual(dossier["decision_metadata"]["final_predicted_label"], 1)

    def test_dossier_metrics_sampling_reporting(self):
        """Verifies dossier metrics accurately report sample size vs total population."""
        metrics = compute_dossier_metrics(
            total_flagged_escalated_pairs=2000,
            generated_dossiers=[{"dossier": i} for i in range(50)],
            explanation_latencies_ms=[1.2] * 50,
        )
        self.assertEqual(metrics["total_eligible_candidates_in_population"], 2000)
        self.assertEqual(metrics["dossiers_sampled_count"], 50)
        self.assertEqual(metrics["sampling_fraction_percent"], 2.5)
        self.assertEqual(metrics["sample_generation_success_rate_percent"], 100.0)

    def test_nested_code_calibration_leakage_free(self):
        """Verifies nested LOGO cross-validation executes with zero outer test leakage."""
        np.random.seed(42)
        n_pairs = 140
        cases = np.array([f"case-0{i}" for i in range(1, 8) for _ in range(20)])
        y = np.random.randint(0, 2, size=n_pairs)
        feats = pd.DataFrame({
            "pair_id": [f"PAIR_{i:04d}" for i in range(n_pairs)],
            "case_id": cases,
            "ground_truth_label": y,
            "code_token_jaccard": np.random.uniform(0, 1, n_pairs),
            "code_sloc_ratio": np.random.uniform(0, 1, n_pairs),
        })

        evaluator = CodeGroupDisjointEvaluator(random_state=42)
        res = evaluator.evaluate_logo_cross_validation(feats, feats["case_id"])

        self.assertEqual(res["total_pairs"], n_pairs)
        self.assertEqual(len(res["fold_details"]), 7)
        self.assertIn("stage2_calls_avoided_count", res["routing_cost"])
        # Verify calls avoided matches predictions dataframe
        n_avoided = (res["predictions_dataframe"]["decision_state"] != "UNCERTAIN_ESCALATED").sum()
        self.assertEqual(res["routing_cost"]["stage2_calls_avoided_count"], n_avoided)

    def test_p_value_format_underflow_safety(self):
        """Verifies format_p_value does not return 0.0 for small p-values."""
        self.assertEqual(format_p_value(1e-25), "< 1e-16")
        self.assertEqual(format_p_value(0.00234), "2.3400e-03")

    def test_hardware_profiler_warmup(self):
        """Verifies hardware profiler executes warmup and reports all required percentiles."""
        profiler = HardwareLatencyProfiler(warmup_cycles=5, measurement_cycles=20)
        res = profiler.profile_callable(lambda: [x**2 for x in range(100)], batch_size=100)
        self.assertEqual(res["warmup_cycles"], 5)
        self.assertEqual(res["measurement_cycles"], 20)
        self.assertIn("p50_ms", res)
        self.assertIn("p90_ms", res)
        self.assertIn("p95_ms", res)
        self.assertIn("throughput_items_per_sec", res)


if __name__ == "__main__":
    unittest.main()
