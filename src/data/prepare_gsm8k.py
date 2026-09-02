"""Download GSM8K and create frozen train/validation/RL splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .splits import make_splits, save_split_manifest


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _target(answer: str) -> str:
    return answer.rsplit("####", maxsplit=1)[-1].strip()


def prepare(
    output_dir: Path,
    *,
    sft_size: int,
    rl_size: int,
    validation_size: int,
    seed: int,
    dataset_revision: str | None = None,
) -> None:
    from datasets import load_dataset

    dataset_id = "openai/gsm8k"
    kwargs = {"revision": dataset_revision} if dataset_revision else {}
    dataset = load_dataset(dataset_id, "main", **kwargs)
    train = dataset["train"]
    splits = make_splits(
        len(train),
        {"sft": sft_size, "rl": rl_size, "validation": validation_size},
        seed,
    )
    save_split_manifest(
        output_dir / "splits.json",
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        seed=seed,
        splits=splits,
        metadata={
            "config_name": "main",
            "train_fingerprint": train._fingerprint,
            "test_fingerprint": dataset["test"]._fingerprint,
            "train_rows": len(train),
            "test_rows": len(dataset["test"]),
        },
    )

    for split_name, indices in splits.items():
        rows = []
        for index in indices:
            item = train[index]
            rows.append(
                {
                    "id": f"gsm8k-train-{index}",
                    "source_index": index,
                    "question": item["question"],
                    "reasoning_solution": item["answer"],
                    "ground_truth": _target(item["answer"]),
                }
            )
        _write_jsonl(output_dir / f"gsm8k_{split_name}.jsonl", rows)

    test_rows = [
        {
            "id": f"gsm8k-test-{index}",
            "source_index": index,
            "question": item["question"],
            "reasoning_solution": item["answer"],
            "ground_truth": _target(item["answer"]),
        }
        for index, item in enumerate(dataset["test"])
    ]
    _write_jsonl(output_dir / "gsm8k_test.jsonl", test_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--sft-size", type=int, default=3000)
    parser.add_argument("--rl-size", type=int, default=1000)
    parser.add_argument("--validation-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-revision")
    args = parser.parse_args()
    prepare(**vars(args))


if __name__ == "__main__":
    main()
