"""Paired held-out comparisons for checkpoints evaluated on identical examples."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def exact_mcnemar_p(primary_only: int, comparator_only: int) -> float:
    discordant = primary_only + comparator_only
    if discordant == 0:
        return 1.0
    tail = min(primary_only, comparator_only)
    probability = sum(math.comb(discordant, value) for value in range(tail + 1))
    return min(1.0, 2.0 * probability / (2**discordant))


def paired_statistics(
    primary_rows: list[dict[str, Any]],
    comparator_rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    primary = {str(row["id"]): row for row in primary_rows}
    comparator = {str(row["id"]): row for row in comparator_rows}
    if set(primary) != set(comparator):
        raise ValueError("Paired evaluations must contain identical example IDs")

    differences: list[int] = []
    counts = {"both_correct": 0, "primary_only": 0, "comparator_only": 0, "both_wrong": 0}
    for identifier in sorted(primary):
        primary_correct = bool(primary[identifier]["correct"])
        comparator_correct = bool(comparator[identifier]["correct"])
        differences.append(int(primary_correct) - int(comparator_correct))
        if primary_correct and comparator_correct:
            counts["both_correct"] += 1
        elif primary_correct:
            counts["primary_only"] += 1
        elif comparator_correct:
            counts["comparator_only"] += 1
        else:
            counts["both_wrong"] += 1

    examples = len(differences)
    if examples == 0:
        raise ValueError("Paired evaluation is empty")
    delta = sum(differences) / examples
    generator = random.Random(seed)
    bootstrapped = []
    for _ in range(bootstrap_samples):
        total = sum(differences[generator.randrange(examples)] for _ in range(examples))
        bootstrapped.append(total / examples)
    bootstrapped.sort()
    lower = bootstrapped[int(0.025 * (bootstrap_samples - 1))]
    upper = bootstrapped[int(0.975 * (bootstrap_samples - 1))]

    return {
        "examples": examples,
        "accuracy_delta": delta,
        "accuracy_delta_percentage_points": 100.0 * delta,
        "paired_bootstrap_95_ci": [lower, upper],
        "paired_bootstrap_95_ci_percentage_points": [100.0 * lower, 100.0 * upper],
        "mcnemar_exact_p": exact_mcnemar_p(
            counts["primary_only"], counts["comparator_only"]
        ),
        "outcomes": counts,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
    }


def run(generations_dir: Path, output_dir: Path, bootstrap_samples: int, seed: int) -> None:
    comparisons = [
        ("sft", "base"),
        ("grpo", "base"),
        ("grpo", "sft"),
        ("grpo", "continued_sft"),
    ]
    benchmarks = ["gsm8k", "svamp"]
    results: dict[str, Any] = {}
    for primary, comparator in comparisons:
        key = f"{primary}_vs_{comparator}"
        results[key] = {}
        for benchmark in benchmarks:
            primary_rows = load_jsonl(generations_dir / f"{primary}_{benchmark}.jsonl")
            comparator_rows = load_jsonl(
                generations_dir / f"{comparator}_{benchmark}.jsonl"
            )
            results[key][benchmark] = paired_statistics(
                primary_rows,
                comparator_rows,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison_statistics.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "| Comparison | Benchmark | Accuracy delta | Paired 95% CI | McNemar exact p |",
        "|---|---|---:|---:|---:|",
    ]
    for comparison, by_benchmark in results.items():
        for benchmark, stats in by_benchmark.items():
            low, high = stats["paired_bootstrap_95_ci_percentage_points"]
            lines.append(
                f"| {comparison.replace('_', ' ')} | {benchmark.upper()} | "
                f"{stats['accuracy_delta_percentage_points']:+.3f} pp | "
                f"[{low:+.3f}, {high:+.3f}] pp | {stats['mcnemar_exact_p']:.4g} |"
            )
    (output_dir / "comparison_statistics.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generations-dir", type=Path, default=Path("generations/evaluation")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis"))
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
