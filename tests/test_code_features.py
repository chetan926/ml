"""
Unit Tests for Code Multi-View Feature Extractors (Phase 3.2)
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection
"""

import unittest
from research_upgrade.src.features.code_token import CodeTokenExtractor
from research_upgrade.src.features.code_identifier import CodeIdentifierNormalizedExtractor, normalize_identifiers
from research_upgrade.src.features.code_ast import CodeStructuralExtractor
from research_upgrade.src.features.code_complexity import (
    CodeComplexityExtractor,
    count_sloc,
    estimate_cyclomatic_complexity
)


JAVA_SNIPPET_A = """
public class Calculator {
    public static int add(int a, int b) {
        int sum = a + b;
        return sum;
    }
    public static void main(String[] args) {
        int result = add(5, 10);
        if (result > 10) {
            System.out.println("Greater");
        }
    }
}
"""

JAVA_SNIPPET_B_RENAMED = """
public class CalculatorRenamed {
    public static int add(int x, int y) {
        int total = x + y;
        return total;
    }
    public static void main(String[] arguments) {
        int output = add(5, 10);
        if (output > 10) {
            System.out.println("Greater");
        }
    }
}
"""

JAVA_SNIPPET_C_DIFFERENT = """
public class Matrix {
    public double[][] multiply(double[][] m1, double[][] m2) {
        int n = m1.length;
        double[][] res = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    res[i][j] += m1[i][k] * m2[k][j];
                }
            }
        }
        return res;
    }
}
"""


class TestCodeTokenExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = CodeTokenExtractor()

    def test_feature_names(self):
        names = self.extractor.feature_names
        self.assertEqual(len(names), 3)
        self.assertIn("code_token_jaccard", names)
        self.assertIn("code_token_frequency_overlap", names)
        self.assertIn("code_rkrgst_sim", names)

    def test_identical_code(self):
        res = self.extractor.extract(JAVA_SNIPPET_A, JAVA_SNIPPET_A)
        self.assertEqual(res["code_token_jaccard"], 1.0)
        self.assertEqual(res["code_token_frequency_overlap"], 1.0)
        self.assertAlmostEqual(res["code_rkrgst_sim"], 1.0, places=4)

    def test_empty_and_none_code(self):
        res = self.extractor.extract("", "")
        self.assertEqual(res["code_token_jaccard"], 0.0)
        self.assertEqual(res["code_token_frequency_overlap"], 0.0)
        self.assertEqual(res["code_rkrgst_sim"], 0.0)

        res2 = self.extractor.extract(None, JAVA_SNIPPET_A)
        self.assertEqual(res2["code_token_jaccard"], 0.0)

    def test_value_ranges(self):
        res = self.extractor.extract(JAVA_SNIPPET_A, JAVA_SNIPPET_C_DIFFERENT)
        for k, v in res.items():
            self.assertGreaterEqual(v, 0.0, f"{k} < 0.0")
            self.assertLessEqual(v, 1.0, f"{k} > 1.0")


class TestCodeIdentifierNormalizedExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = CodeIdentifierNormalizedExtractor()

    def test_feature_names(self):
        names = self.extractor.feature_names
        self.assertEqual(len(names), 3)
        self.assertIn("code_norm_token_jaccard", names)
        self.assertIn("code_norm_freq_overlap", names)
        self.assertIn("code_norm_rkrgst_sim", names)

    def test_renaming_invariance_benefit(self):
        raw_extractor = CodeTokenExtractor()
        raw_res = raw_extractor.extract(JAVA_SNIPPET_A, JAVA_SNIPPET_B_RENAMED)
        norm_res = self.extractor.extract(JAVA_SNIPPET_A, JAVA_SNIPPET_B_RENAMED)

        # Identifier normalization should yield higher or equal similarity when identifiers are renamed
        self.assertGreater(norm_res["code_norm_token_jaccard"], raw_res["code_token_jaccard"])
        self.assertGreaterEqual(norm_res["code_norm_freq_overlap"], raw_res["code_token_frequency_overlap"])
        self.assertGreaterEqual(norm_res["code_norm_rkrgst_sim"], raw_res["code_rkrgst_sim"])

    def test_identifier_normalizer_function(self):
        tokens = ["public", "static", "int", "myVariable", "=", "42", ";"]
        norm = normalize_identifiers(tokens)
        self.assertEqual(norm[0], "public")
        self.assertEqual(norm[1], "static")
        self.assertEqual(norm[2], "int")
        self.assertEqual(norm[3], "IDENT")
        self.assertEqual(norm[4], "=")
        self.assertEqual(norm[5], "42")
        self.assertEqual(norm[6], ";")


class TestCodeStructuralExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = CodeStructuralExtractor()

    def test_feature_names(self):
        names = self.extractor.feature_names
        self.assertEqual(len(names), 3)
        self.assertIn("code_nesting_depth_sim", names)
        self.assertIn("code_control_block_sim", names)
        self.assertIn("code_method_structure_sim", names)

    def test_fallback_disclosure(self):
        metadata = self.extractor.get_metadata()
        self.assertIn("ast_parser_available", metadata)
        self.assertIn("methodology", metadata)
        # In current environment, javalang is not installed
        if not metadata["ast_parser_available"]:
            self.assertEqual(metadata["methodology"], "BLOCK_STRUCTURE_HEURISTIC_FALLBACK")
            self.assertIn("not installed", metadata["limitation"])

    def test_identical_and_differing_code(self):
        res_ident = self.extractor.extract(JAVA_SNIPPET_A, JAVA_SNIPPET_A)
        self.assertEqual(res_ident["code_nesting_depth_sim"], 1.0)
        self.assertEqual(res_ident["code_control_block_sim"], 1.0)
        self.assertEqual(res_ident["code_method_structure_sim"], 1.0)

        res_diff = self.extractor.extract(JAVA_SNIPPET_A, JAVA_SNIPPET_C_DIFFERENT)
        for k, v in res_diff.items():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)


class TestCodeComplexityExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = CodeComplexityExtractor()

    def test_feature_names(self):
        names = self.extractor.feature_names
        self.assertEqual(len(names), 4)
        self.assertIn("code_operator_dist_sim", names)
        self.assertIn("code_control_flow_sim", names)
        self.assertIn("code_cyclomatic_sim", names)
        self.assertIn("code_sloc_ratio", names)

    def test_helpers(self):
        sloc = count_sloc(JAVA_SNIPPET_A)
        self.assertGreater(sloc, 0)
        cc = estimate_cyclomatic_complexity(JAVA_SNIPPET_A)
        self.assertGreaterEqual(cc, 1)

    def test_identical_and_empty(self):
        res = self.extractor.extract(JAVA_SNIPPET_A, JAVA_SNIPPET_A)
        self.assertEqual(res["code_operator_dist_sim"], 1.0)
        self.assertEqual(res["code_control_flow_sim"], 1.0)
        self.assertEqual(res["code_cyclomatic_sim"], 1.0)
        self.assertEqual(res["code_sloc_ratio"], 1.0)

        res_empty = self.extractor.extract("", "")
        self.assertEqual(res_empty["code_operator_dist_sim"], 0.0)
        self.assertEqual(res_empty["code_sloc_ratio"], 0.0)


if __name__ == "__main__":
    unittest.main()

