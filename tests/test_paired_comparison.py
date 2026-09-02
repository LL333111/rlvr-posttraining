import unittest

from src.analysis.paired_comparison import exact_mcnemar_p, paired_statistics


class PairedComparisonTests(unittest.TestCase):
    def test_counts_and_delta(self):
        primary = [
            {"id": "a", "correct": True},
            {"id": "b", "correct": True},
            {"id": "c", "correct": False},
            {"id": "d", "correct": False},
        ]
        comparator = [
            {"id": "a", "correct": True},
            {"id": "b", "correct": False},
            {"id": "c", "correct": True},
            {"id": "d", "correct": False},
        ]
        result = paired_statistics(
            primary, comparator, bootstrap_samples=100, seed=42
        )
        self.assertEqual(result["accuracy_delta"], 0.0)
        self.assertEqual(
            result["outcomes"],
            {"both_correct": 1, "primary_only": 1, "comparator_only": 1, "both_wrong": 1},
        )

    def test_exact_mcnemar_is_symmetric(self):
        self.assertEqual(exact_mcnemar_p(3, 3), 1.0)
        self.assertEqual(exact_mcnemar_p(0, 0), 1.0)


if __name__ == "__main__":
    unittest.main()
