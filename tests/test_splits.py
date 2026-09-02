import unittest

from src.data.splits import make_splits


class SplitTests(unittest.TestCase):
    def test_deterministic_and_disjoint(self):
        first = make_splits(100, {"sft": 50, "rl": 20, "validation": 10}, 42)
        second = make_splits(100, {"sft": 50, "rl": 20, "validation": 10}, 42)
        self.assertEqual(first, second)
        sets = [set(values) for values in first.values()]
        self.assertFalse(sets[0] & sets[1])
        self.assertFalse(sets[0] & sets[2])
        self.assertFalse(sets[1] & sets[2])

    def test_oversubscribed(self):
        with self.assertRaises(ValueError):
            make_splits(5, {"a": 4, "b": 2}, 1)


if __name__ == "__main__":
    unittest.main()
