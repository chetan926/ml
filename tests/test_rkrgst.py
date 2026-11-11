"""
Unit tests for RKRGST algorithm, ES-Plag comment stripping, and penalty-based similarity normalization.
"""

import unittest
from src.code.esplag_tokenizer import remove_comments_safe, tokenize_code
from src.code.rkr_gst import rkr_gst, RKRGSTResult
from src.code.penalty_similarity import compute_penalized_similarities


class TestESPlagTokenizer(unittest.TestCase):
    def test_comment_removal_safe(self):
        # Case 1: Line comment removed
        code1 = "int x = 10; // this is a line comment\nint y = 20;"
        clean1 = remove_comments_safe(code1)
        self.assertNotIn("// this is a line comment", clean1)
        self.assertIn("int x = 10;", clean1)
        self.assertIn("int y = 20;", clean1)

        # Case 2: Block comment removed
        code2 = "/* Block comment line 1\n line 2 */\nint z = 30;"
        clean2 = remove_comments_safe(code2)
        self.assertNotIn("Block comment", clean2)
        self.assertIn("int z = 30;", clean2)

        # Case 3: String literal preserved even if containing comment markers
        code3 = 'String s = "/* not a comment */ // also literal";'
        clean3 = remove_comments_safe(code3)
        self.assertIn('"/* not a comment */ // also literal"', clean3)

        # Case 4: Character literal preserved
        code4 = "char slash = '/'; // trailing comment"
        clean4 = remove_comments_safe(code4)
        self.assertIn("char slash = '/'", clean4)
        self.assertNotIn("trailing comment", clean4)

    def test_tokenize_code(self):
        code = 'public static void main(String[] args) {\n    System.out.println("Hello");\n}'
        tokens = tokenize_code(code)
        self.assertIn("public", tokens)
        self.assertIn("static", tokens)
        self.assertIn("void", tokens)
        self.assertIn("main", tokens)
        self.assertIn('"Hello"', tokens)


class TestRKRGST(unittest.TestCase):
    def test_identical_sequences(self):
        tokens = ["public", "static", "void", "main", "(", ")"]
        res = rkr_gst(tokens, tokens, mml=2)
        self.assertEqual(res.matched_tokens, len(tokens))
        self.assertEqual(res.num_subsequences, 1)
        self.assertAlmostEqual(res.standard_similarity, 1.0, places=5)
        self.assertEqual(len(res.tiles), 1)
        self.assertEqual(res.tiles[0].pos_a, 0)
        self.assertEqual(res.tiles[0].pos_b, 0)
        self.assertEqual(res.tiles[0].length, len(tokens))

    def test_scrambled_substrings(self):
        # Tile 1: "int", "a", "=", "1", ";" (len 5)
        # Tile 2: "int", "b", "=", "2", ";" (len 5)
        tokens_a = ["int", "a", "=", "1", ";", "int", "b", "=", "2", ";"]
        tokens_b = ["int", "b", "=", "2", ";", "int", "a", "=", "1", ";"]
        res = rkr_gst(tokens_a, tokens_b, mml=2)
        self.assertEqual(res.matched_tokens, 10)
        self.assertEqual(res.num_subsequences, 2)
        self.assertAlmostEqual(res.standard_similarity, 1.0, places=5)

    def test_disjoint_sequences(self):
        tokens_a = ["int", "x", "=", "10", ";"]
        tokens_b = ["String", "name", "=", '"Alice"', ";;"]
        res = rkr_gst(tokens_a, tokens_b, mml=2)
        self.assertEqual(res.matched_tokens, 0)
        self.assertEqual(res.num_subsequences, 0)
        self.assertEqual(res.standard_similarity, 0.0)

    def test_mml_threshold(self):
        # Single isolated token match is ignored when mml=2
        tokens_a = ["a", "x", "b"]
        tokens_b = ["c", "x", "d"]
        res = rkr_gst(tokens_a, tokens_b, mml=2)
        self.assertEqual(res.matched_tokens, 0)
        self.assertEqual(res.num_subsequences, 0)

        # But matched when mml=1
        res1 = rkr_gst(tokens_a, tokens_b, mml=1)
        self.assertEqual(res1.matched_tokens, 1)
        self.assertEqual(res1.num_subsequences, 1)


class TestPenaltySimilarity(unittest.TestCase):
    def test_boundary_zero_and_empty(self):
        res_zero = compute_penalized_similarities(matched_tokens=0, num_subsequences=0, len_a=10, len_b=10)
        self.assertEqual(res_zero['penalized_max'], 0.0)
        self.assertEqual(res_zero['penalized_average'], 0.0)
        self.assertEqual(res_zero['raw_gst_average'], 0.0)
        self.assertEqual(res_zero['penalty_deducted'], 0)

        res_empty = compute_penalized_similarities(matched_tokens=5, num_subsequences=1, len_a=0, len_b=10)
        self.assertEqual(res_empty['penalized_max'], 0.0)
        self.assertEqual(res_empty['penalized_average'], 0.0)

    def test_single_subsequence_no_penalty(self):
        # 10 tokens matched in 1 continuous tile out of 10 tokens each
        res = compute_penalized_similarities(matched_tokens=10, num_subsequences=1, len_a=10, len_b=10)
        self.assertEqual(res['penalty_deducted'], 0)
        self.assertAlmostEqual(res['penalized_max'], 1.0, places=5)
        self.assertAlmostEqual(res['penalized_average'], 1.0, places=5)
        self.assertAlmostEqual(res['raw_gst_average'], 1.0, places=5)

    def test_multiple_subsequences_penalty_applied(self):
        # 10 tokens matched across 3 tiles (penalty = 3 - 1 = 2)
        # penalized numerator = 10 - 2 = 8
        # len_a = 10, len_b = 10
        # penalized_max = 8 / 10 = 0.8
        # penalized_avg = (2 * 8) / 20 = 0.8
        # raw_gst_avg = (2 * 10) / 20 = 1.0
        res = compute_penalized_similarities(matched_tokens=10, num_subsequences=3, len_a=10, len_b=10)
        self.assertEqual(res['penalty_deducted'], 2)
        self.assertAlmostEqual(res['penalized_max'], 0.8, places=5)
        self.assertAlmostEqual(res['penalized_average'], 0.8, places=5)
        self.assertAlmostEqual(res['raw_gst_average'], 1.0, places=5)
        self.assertLess(res['penalized_average'], res['raw_gst_average'])


if __name__ == '__main__':
    unittest.main()

