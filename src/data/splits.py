"""Deterministic, disjoint dataset splits."""

from __future__ import annotations

import json
import random
from pathlib import Path


def make_splits(
    population_size: int,
    sizes: dict[str, int],
    seed: int,
) -> dict[str, list[int]]:
    if population_size < 0 or any(size < 0 for size in sizes.values()):
        raise ValueError("population and split sizes must be non-negative")
    if sum(sizes.values()) > population_size:
        raise ValueError("requested splits exceed the dataset size")
    indices = list(range(population_size))
    random.Random(seed).shuffle(indices)
    output: dict[str, list[int]] = {}
    cursor = 0
    for name, size in sizes.items():
        output[name] = indices[cursor : cursor + size]
        cursor += size
    return output


def save_split_manifest(
    path: str | Path,
    *,
    dataset_id: str,
    dataset_revision: str | None,
    seed: int,
    splits: dict[str, list[int]],
    metadata: dict[str, object] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "seed": seed,
        "splits": splits,
        "metadata": metadata or {},
    }
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
