"""Create a reproducible, deliberately unfilled manual review sheet."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from src.training.common import load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", type=Path, default=Path("generations/evaluation/sft_gsm8k.jsonl"))
    parser.add_argument(
        "--grpo", type=Path, default=Path("generations/evaluation/grpo_gsm8k.jsonl")
    )
    parser.add_argument("--output", type=Path, default=Path("results/analysis/failure_review.csv"))
    parser.add_argument("--examples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    sft = {row["id"]: row for row in load_jsonl(args.sft)}
    grpo = {row["id"]: row for row in load_jsonl(args.grpo)}
    shared = sorted(set(sft) & set(grpo))
    changed = [key for key in shared if bool(sft[key]["correct"]) != bool(grpo[key]["correct"])]
    unchanged_failures = [
        key for key in shared if not sft[key]["correct"] and not grpo[key]["correct"]
    ]
    rng = random.Random(args.seed)
    rng.shuffle(changed)
    rng.shuffle(unchanged_failures)
    selected = (changed + unchanged_failures)[: args.examples]
    if len(selected) < args.examples:
        remaining = [key for key in shared if key not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: args.examples - len(selected)])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id", "question", "ground_truth", "sft_correct", "grpo_correct",
        "sft_completion", "grpo_completion", "manual_category", "reviewer_notes",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in selected:
            writer.writerow(
                {
                    "id": key,
                    "question": sft[key]["question"],
                    "ground_truth": sft[key]["ground_truth"],
                    "sft_correct": sft[key]["correct"],
                    "grpo_correct": grpo[key]["correct"],
                    "sft_completion": sft[key]["completion"],
                    "grpo_completion": grpo[key]["completion"],
                    "manual_category": "",
                    "reviewer_notes": "",
                }
            )
    manifest = {
        "seed": args.seed,
        "requested_examples": args.examples,
        "selected_ids": selected,
        "category_options": [
            "arithmetic error", "wrong strategy", "correct reasoning / wrong extraction",
            "incomplete reasoning", "overlong reasoning", "answer-format error",
            "RL-specific degeneration", "successful corrected reasoning",
        ],
        "status": "manual review required; no categories were auto-claimed",
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
