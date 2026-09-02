import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.evaluation.evaluator import evaluate


class EvaluatorTests(unittest.TestCase):
    def test_evaluator_writes_traceable_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "test.jsonl"
            rows = [
                {"id": "a", "question": "q1", "ground_truth": "2"},
                {"id": "b", "question": "q2", "ground_truth": "3"},
            ]
            dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            config = {
                "model": {"name": "unused", "dtype": "float32"},
                "evaluation": {
                    "per_device_batch_size": 2,
                    "max_new_tokens": 5,
                    "seed": 42,
                    "bootstrap_samples": 20,
                    "limit": None,
                },
                "benchmarks": {"gsm8k": str(dataset)},
                "checkpoints": {"base": None},
                "output_dir": str(root / "results"),
                "generations_dir": str(root / "generations"),
            }
            fake = [
                {**rows[0], "completion": "#### 2", "completion_tokens": 2},
                {**rows[1], "completion": "#### 4", "completion_tokens": 2},
            ]
            with patch("src.evaluation.evaluator.generate_completions", return_value=fake):
                summary = evaluate(config=config, checkpoint_key="base", benchmark="gsm8k")
            self.assertEqual(summary["accuracy"], 0.5)
            self.assertTrue((root / "results" / "base_gsm8k.json").is_file())
            generated = (root / "generations" / "base_gsm8k.jsonl").read_text()
            self.assertIn('"correct": true', generated)


if __name__ == "__main__":
    unittest.main()
