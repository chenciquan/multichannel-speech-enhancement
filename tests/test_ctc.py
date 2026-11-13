import unittest
from mcse.ctc import greedy_ctc_tokens


class GreedyCTCTest(unittest.TestCase):
    labels = ('-', '|', 'A', 'B', "'")

    def test_collapse_before_removing_blank(self):
        self.assertEqual(greedy_ctc_tokens([2, 2, 0, 2, 2], self.labels), 'AA')

    def test_boundary_apostrophe_and_repeats(self):
        self.assertEqual(greedy_ctc_tokens([0, 2, 4, 4, 3, 1, 1, 2, 0], self.labels), "A'B A")

    def test_empty_and_blank_only(self):
        self.assertEqual(greedy_ctc_tokens([], self.labels), '')
        self.assertEqual(greedy_ctc_tokens([0, 0], self.labels), '')

    def test_invalid_indices_rejected(self):
        for ids in ([5], [-1], [1.5]):
            with self.assertRaises(ValueError):
                greedy_ctc_tokens(ids, self.labels)


if __name__ == '__main__':
    unittest.main()
