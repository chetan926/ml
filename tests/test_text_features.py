"""
Text Multi-View Feature Extractors Automated Test Suite (Phase 3.2)
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection
"""

import unittest
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from research_upgrade.src.features.text_lexical import TextLexicalExtractor
from research_upgrade.src.features.text_character import TextCharacterExtractor
from research_upgrade.src.features.text_semantic import TextSemanticExtractor
from research_upgrade.src.features.text_structural import TextStructuralExtractor


class TestTextFeatures(unittest.TestCase):
    """Unit tests for all 4 text feature extractors."""

    @classmethod
    def setUpClass(cls):
        cls.train_corpus = [
            "A small dog plays in the green grass outside.",
            "Students write academic research papers in universities.",
            "Algorithms detect plagiarism in source code and written essays."
        ]
        cls.lexical = TextLexicalExtractor().fit(cls.train_corpus)
        cls.character = TextCharacterExtractor().fit(cls.train_corpus)
        cls.semantic = TextSemanticExtractor()
        cls.structural = TextStructuralExtractor()

    def test_lexical_unfitted_raises_runtime_error(self):
        """Asserts that calling extract() on an unfitted LexicalExtractor raises RuntimeError."""
        unfitted = TextLexicalExtractor()
        with self.assertRaises(RuntimeError):
            unfitted.extract("text a", "text b")

    def test_lexical_identical_and_disjoint_strings(self):
        """Validates lexical similarity boundaries."""
        # Identical strings
        s = "Students write academic research papers in universities."
        res = self.lexical.extract(s, s)
        self.assertAlmostEqual(res["text_word_jaccard"], 1.0, places=4)
        self.assertAlmostEqual(res["text_word_overlap_ratio"], 1.0, places=4)
        self.assertAlmostEqual(res["text_word_tfidf_sim"], 1.0, places=4)

        # Disjoint strings
        s1 = "Quantum computing physics."
        s2 = "Culinary baking pastry."
        res_dis = self.lexical.extract(s1, s2)
        self.assertEqual(res_dis["text_word_jaccard"], 0.0)
        self.assertEqual(res_dis["text_word_overlap_ratio"], 0.0)

        # Empty strings
        res_empty = self.lexical.extract("", "")
        self.assertEqual(res_empty["text_word_jaccard"], 0.0)
        self.assertEqual(res_empty["text_word_tfidf_sim"], 0.0)

    def test_character_extractor_boundaries(self):
        """Validates character n-gram similarity boundaries."""
        s = "academic research"
        res = self.character.extract(s, s)
        self.assertAlmostEqual(res["text_char_jaccard_3gram"], 1.0, places=4)
        self.assertAlmostEqual(res["text_char_lcs_ratio"], 1.0, places=4)
        self.assertAlmostEqual(res["text_char_tfidf_sim"], 1.0, places=4)

        # Minor typo resilience: 'academic' vs 'academik'
        res_typo = self.character.extract("academic research", "academik research")
        self.assertGreater(res_typo["text_char_jaccard_3gram"], 0.5)
        self.assertGreater(res_typo["text_char_lcs_ratio"], 0.4)

        # Empty inputs
        res_empty = self.character.extract("", "")
        self.assertEqual(res_empty["text_char_jaccard_3gram"], 0.0)
        self.assertEqual(res_empty["text_char_lcs_ratio"], 0.0)

    def test_semantic_extractor_boundaries(self):
        """Validates dense embedding cosine similarity."""
        s = "A small dog plays in the green grass outside."
        res = self.semantic.extract(s, s)
        self.assertAlmostEqual(res["text_semantic_minilm_sim"], 1.0, places=2)

        # Paraphrase alignment
        s1 = "A small puppy is running in the yard."
        res_para = self.semantic.extract(s, s1)
        self.assertGreater(res_para["text_semantic_minilm_sim"], 0.6)

        # Empty string handling
        res_empty = self.semantic.extract("", "test")
        self.assertEqual(res_empty["text_semantic_minilm_sim"], 0.0)

    def test_structural_extractor_boundaries(self):
        """Validates structural and length delta features."""
        s = "Hello world. This is sentence two."
        res = self.structural.extract(s, s)
        self.assertAlmostEqual(res["text_length_ratio"], 1.0, places=4)
        self.assertAlmostEqual(res["text_token_count_delta"], 0.0, places=4)
        self.assertAlmostEqual(res["text_sentence_delta"], 0.0, places=4)
        self.assertAlmostEqual(res["text_ttr_sim"], 1.0, places=4)

        # Different lengths
        s1 = "Hello."
        s2 = "Hello world this is a much longer paragraph with many words."
        res_diff = self.structural.extract(s1, s2)
        self.assertLess(res_diff["text_length_ratio"], 0.5)
        self.assertGreater(res_diff["text_token_count_delta"], 0.5)


if __name__ == "__main__":
    unittest.main()

