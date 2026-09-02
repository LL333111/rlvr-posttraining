import unittest

from src.rewards.answer_parser import extract_final_answer, normalize_answer


class AnswerParserTests(unittest.TestCase):
    def test_explicit_markers(self):
        cases = {
            "work\n#### 42": "42",
            r"Therefore, \boxed{-3}": "-3",
            "The final answer is 1,250.": "1250",
            "Answer: 0.50": "1/2",
            "The answer is 3/4": "3/4",
            r"Therefore, \boxed{\frac{3}{4}}": "3/4",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(extract_final_answer(text), expected)

    def test_last_number_fallback(self):
        self.assertEqual(extract_final_answer("First 2, then 7"), "7")

    def test_equivalent_numeric_formats(self):
        for value in ("5", "5.0", "10/2", " 5.000 "):
            self.assertEqual(normalize_answer(value), "5")

    def test_invalid_values(self):
        self.assertIsNone(extract_final_answer("no numeric answer"))
        self.assertIsNone(normalize_answer("1/0"))


if __name__ == "__main__":
    unittest.main()
