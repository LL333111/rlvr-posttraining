"""Evaluate saved branch checkpoints on a frozen GSM8K subset."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from src.evaluation.evaluator import evaluate
from src.utils.config import load_config
from src.utils.logging import write_json


def _step(path: Path, final_step: int) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else final_step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/eval.yaml"))
    parser.add_argument("--branch", choices=["continued_sft", "grpo"], required=True)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config["checkpoints"][args.branch])
    metadata_path = Path("results") / args.branch / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    final_step = int(metadata["optimizer_steps"])
    candidates_by_step = {
        _step(path, final_step): path for path in root.glob("checkpoint-*")
    }
    candidates_by_step[final_step] = root
    candidates = [candidates_by_step[step] for step in sorted(candidates_by_step)]
    log_rows = []
    log_path = Path("results") / args.branch / "train_log.jsonl"
    with log_path.open(encoding="utf-8") as handle:
        log_rows = [json.loads(line) for line in handle if line.strip()]

    def gpu_hours_at_step(step: int) -> float:
        exact_or_prior = [row for row in log_rows if int(row.get("step", -1)) <= step]
        if not exact_or_prior:
            raise RuntimeError(f"No elapsed-time log at or before step {step}")
        closest = max(exact_or_prior, key=lambda row: int(row["step"]))
        return (
            float(closest["elapsed_seconds"])
            * int(metadata["training_gpu_count"])
            / 3600.0
        )
    records = []
    for checkpoint in candidates:
        step = _step(checkpoint, final_step)
        summary = evaluate(
            config=config,
            checkpoint_key=args.branch,
            benchmark="gsm8k",
            adapter_path=checkpoint,
            output_suffix=f"sweep_step_{step}",
            limit=args.limit,
        )
        records.append(
            {
                "step": step,
                "accuracy": summary["accuracy"],
                "examples": summary["examples"],
                "gpu_hours": gpu_hours_at_step(step),
                "checkpoint": str(checkpoint),
            }
        )
    write_json(Path(config["output_dir"]) / f"{args.branch}_gsm8k_sweep.json", records)


if __name__ == "__main__":
    main()
