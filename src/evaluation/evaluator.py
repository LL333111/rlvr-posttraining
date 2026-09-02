from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from src.evaluation.generate import generate_completions
from src.rewards.metrics import bootstrap_accuracy_interval, exact_match_accuracy
from src.rewards.verifier import verify_answer
from src.training.common import load_jsonl
from src.utils.config import load_config
from src.utils.hardware import hardware_snapshot
from src.utils.logging import write_json
from src.utils.reproducibility import set_seed


def evaluate(
    *,
    config: dict[str, Any],
    checkpoint_key: str,
    benchmark: str,
    adapter_path: str | Path | None = None,
    output_suffix: str = "",
    limit: int | None = None,
) -> dict[str, Any]:
    if benchmark not in config["benchmarks"]:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    if checkpoint_key not in config["checkpoints"] and adapter_path is None:
        raise ValueError(f"Unknown checkpoint: {checkpoint_key}")
    evaluation = config["evaluation"]
    set_seed(int(evaluation["seed"]))
    rows = load_jsonl(config["benchmarks"][benchmark])
    effective_limit = limit if limit is not None else evaluation.get("limit")
    if effective_limit is not None:
        rows = rows[: int(effective_limit)]
    selected_adapter = adapter_path
    if selected_adapter is None:
        selected_adapter = config["checkpoints"][checkpoint_key]

    started = time.monotonic()
    generations = generate_completions(
        rows,
        model_config=config["model"],
        adapter_path=selected_adapter,
        batch_size=int(evaluation["per_device_batch_size"]),
        max_new_tokens=int(evaluation["max_new_tokens"]),
    )
    scored = []
    for item in generations:
        result = verify_answer(item["completion"], item["ground_truth"])
        scored.append({**item, **result.to_dict()})
    correct = [bool(item["correct"]) for item in scored]
    low, high = bootstrap_accuracy_interval(
        correct,
        samples=int(evaluation["bootstrap_samples"]),
        seed=int(evaluation["seed"]),
    )
    suffix = f"_{output_suffix}" if output_suffix else ""
    stem = f"{checkpoint_key}_{benchmark}{suffix}"
    generation_path = Path(config["generations_dir"]) / f"{stem}.jsonl"
    generation_path.parent.mkdir(parents=True, exist_ok=True)
    with generation_path.open("w", encoding="utf-8") as handle:
        for row in scored:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "checkpoint": checkpoint_key,
        "adapter_path": str(selected_adapter) if selected_adapter else None,
        "benchmark": benchmark,
        "examples": len(scored),
        "correct": sum(correct),
        "accuracy": exact_match_accuracy(correct),
        "bootstrap_95_ci": [low, high],
        "mean_completion_tokens": (
            sum(item["completion_tokens"] for item in scored) / len(scored) if scored else 0.0
        ),
        "wall_clock_seconds": time.monotonic() - started,
        "decoding": {"do_sample": False, "num_beams": 1},
        "hardware": hardware_snapshot(),
        "raw_generations": str(generation_path),
    }
    write_json(Path(config["output_dir"]) / f"{stem}.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/eval.yaml"))
    parser.add_argument(
        "--checkpoint-key", choices=["base", "sft", "continued_sft", "grpo"], required=True
    )
    parser.add_argument("--benchmark", choices=["gsm8k", "svamp"], required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    summary = evaluate(
        config=config,
        checkpoint_key=args.checkpoint_key,
        benchmark=args.benchmark,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
