from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mcse.asr_scoring import corpus_score, normalize_english, score_words


class ASRScoringTests(unittest.TestCase):
    def test_punctuation_apostrophes_and_numbers_have_fixed_rules(self):
        self.assertEqual(normalize_english("‘Don't’ re-enter café, 12!"),
                         ["DON'T", "RE", "ENTER", "CAFE", "12"])
        self.assertEqual(score_words("12", "twelve")["S"], 1)

    def test_deletion_and_insertion_have_correct_reference_positions(self):
        deleted = score_words("ONE TWO THREE", "ONE THREE")
        self.assertEqual((deleted["S"], deleted["D"], deleted["I"]), (0, 1, 0))
        self.assertEqual([a["reference_index"] for a in deleted["alignment"] if a["op"] == "D"], [1])
        inserted = score_words("ONE THREE", "ONE TWO THREE")
        self.assertEqual((inserted["S"], inserted["D"], inserted["I"]), (0, 0, 1))

    def test_ambiguous_alignment_uses_substitution_before_delete_insert(self):
        score = score_words("A B", "B A")
        self.assertEqual((score["S"], score["D"], score["I"]), (2, 0, 0))

    def test_empty_hypothesis_is_all_deletions(self):
        score = score_words("KEEP THE END", "")
        self.assertEqual(score["D"], 3)
        self.assertEqual(score["wer"], 1)

    def test_wer_can_exceed_one(self):
        self.assertEqual(score_words("A", "A B C D")["wer"], 3)

    def test_corpus_uses_reference_word_denominator(self):
        small = score_words("A", "")
        large = score_words("B C D", "B C D")
        self.assertEqual(corpus_score([small, large])["wer"], 0.25)

    def test_empty_reference_is_not_silently_scored(self):
        with self.assertRaises(ValueError):
            score_words("...", "NOISE")
        self.assertIsNone(corpus_score([])["wer"])


if __name__ == "__main__":
    unittest.main()

