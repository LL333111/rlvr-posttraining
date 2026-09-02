import tempfile
import unittest
from pathlib import Path

from src.utils.config import load_config, require_keys


class ConfigTests(unittest.TestCase):
    def test_load_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("training:\n  seed: 42\n", encoding="utf-8")
            self.assertEqual(load_config(path)["training"]["seed"], 42)

    def test_reject_non_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("- one\n- two\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)

    def test_required_keys(self):
        with self.assertRaises(ValueError):
            require_keys({"a": 1}, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
