from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation.evaluator import evaluate
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/eval.yaml"))
    parser.add_argument("--checkpoint-key", required=True)
    args = parser.parse_args()
    evaluate(config=load_config(args.config), checkpoint_key=args.checkpoint_key, benchmark="gsm8k")


if __name__ == "__main__":
    main()
