from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.analysis.plots import grouped_bar, line_plot
from src.training.common import load_jsonl
from src.utils.logging import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward-groups", type=Path, default=Path("results/grpo/reward_groups.jsonl"))
    parser.add_argument("--rollouts", type=Path, default=Path("generations/grpo/rollouts.jsonl"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    groups = load_jsonl(args.reward_groups)
    if not groups:
        raise ValueError("No GRPO reward-group records found")
    line_plot(
        groups,
        x="step",
        y="mean_reward",
        group=None,
        xlabel="Optimizer step",
        ylabel="Mean verifier reward",
        title="GRPO reward during training",
        output=args.figures_dir / "grpo_reward_curve.png",
    )
    totals = {key: sum(int(row[key]) for row in groups) for key in ("all_wrong", "mixed", "all_correct")}
    denominator = sum(totals.values())
    fractions = {key: value / denominator if denominator else 0.0 for key, value in totals.items()}
    grouped_bar(
        ["All wrong", "Mixed", "All correct"],
        [fractions["all_wrong"], fractions["mixed"], fractions["all_correct"]],
        ylabel="Fraction of rollout groups",
        title="Quality of the relative reward signal",
        output=args.figures_dir / "reward_group_composition.png",
    )

    rollouts = load_jsonl(args.rollouts)
    correct_lengths = [row["completion_tokens"] for row in rollouts if row["correct"]]
    incorrect_lengths = [row["completion_tokens"] for row in rollouts if not row["correct"]]
    summary = {
        "reward_group_counts": totals,
        "reward_group_fractions": fractions,
        "rollouts": len(rollouts),
        "mean_completion_tokens": sum(row["completion_tokens"] for row in rollouts) / len(rollouts) if rollouts else 0.0,
        "mean_correct_completion_tokens": sum(correct_lengths) / len(correct_lengths) if correct_lengths else None,
        "mean_incorrect_completion_tokens": sum(incorrect_lengths) / len(incorrect_lengths) if incorrect_lengths else None,
    }
    write_json("results/analysis/training_dynamics.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
