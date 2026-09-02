"""Metrics shared by training and held-out evaluation."""

from __future__ import annotations

import random
from collections.abc import Sequence


def exact_match_accuracy(correct: Sequence[bool | int | float]) -> float:
    return sum(bool(item) for item in correct) / len(correct) if correct else 0.0


def reward_group_counts(
    rewards: Sequence[float], generations_per_prompt: int
) -> dict[str, int | float]:
    if generations_per_prompt <= 0:
        raise ValueError("generations_per_prompt must be positive")
    if len(rewards) % generations_per_prompt:
        raise ValueError("reward count must be divisible by generations_per_prompt")

    counts = {"all_wrong": 0, "mixed": 0, "all_correct": 0}
    for start in range(0, len(rewards), generations_per_prompt):
        group = rewards[start : start + generations_per_prompt]
        positives = sum(float(value) > 0.0 for value in group)
        if positives == 0:
            counts["all_wrong"] += 1
        elif positives == generations_per_prompt:
            counts["all_correct"] += 1
        else:
            counts["mixed"] += 1
    total = sum(counts.values())
    return {
        **counts,
        "total": total,
        "all_wrong_fraction": counts["all_wrong"] / total if total else 0.0,
        "mixed_fraction": counts["mixed"] / total if total else 0.0,
        "all_correct_fraction": counts["all_correct"] / total if total else 0.0,
    }


def bootstrap_accuracy_interval(
    correct: Sequence[bool | int],
    *,
    samples: int = 2_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    if not correct:
        return 0.0, 0.0
    rng = random.Random(seed)
    values = [float(bool(item)) for item in correct]
    estimates = []
    for _ in range(samples):
        estimates.append(sum(rng.choice(values) for _ in values) / len(values))
    estimates.sort()
    tail = (1.0 - confidence) / 2.0
    low = estimates[int(tail * (samples - 1))]
    high = estimates[int((1.0 - tail) * (samples - 1))]
    return low, high
