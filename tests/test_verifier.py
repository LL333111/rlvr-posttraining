import unittest

from src.rewards.metrics import exact_match_accuracy, reward_group_counts
from src.rewards.verifier import verify_answer


class VerifierTests(unittest.TestCase):
    def test_binary_reward(self):
        correct = verify_answer("Reasoning\n#### 6.0", "6")
        wrong = verify_answer("#### 7", "6")
        self.assertTrue(correct.correct)
        self.assertEqual(correct.reward, 1.0)
        self.assertFalse(wrong.correct)
        self.assertEqual(wrong.reward, 0.0)

    def test_reward_groups(self):
        summary = reward_group_counts([0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1], 4)
        self.assertEqual(summary["all_wrong"], 1)
        self.assertEqual(summary["mixed"], 1)
        self.assertEqual(summary["all_correct"], 1)
        self.assertAlmostEqual(summary["mixed_fraction"], 1 / 3)

    def test_metrics_reject_partial_group(self):
        with self.assertRaises(ValueError):
            reward_group_counts([0, 1, 0], 4)
        self.assertEqual(exact_match_accuracy([True, False, True]), 2 / 3)


if __name__ == "__main__":
    unittest.main()
