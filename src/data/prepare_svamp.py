"""Download the fixed SVAMP transfer benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def prepare(output_file: Path, dataset_revision: str | None = None) -> None:
    from datasets import load_dataset

    kwargs = {"revision": dataset_revision} if dataset_revision else {}
    dataset = load_dataset("ChilleD/SVAMP", **kwargs)
    split = dataset["test"] if "test" in dataset else next(iter(dataset.values()))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        for index, item in enumerate(split):
            body = str(item.get("Body", item.get("body", ""))).strip()
            question = str(item.get("Question", item.get("question", ""))).strip()
            target = item.get("Answer", item.get("answer"))
            row = {
                "id": f"svamp-{index}",
                "source_index": index,
                "question": f"{body} {question}".strip(),
                "ground_truth": str(target),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "dataset_id": "ChilleD/SVAMP",
        "dataset_revision": dataset_revision,
        "split": "test" if "test" in dataset else "first_available",
        "fingerprint": split._fingerprint,
        "rows": len(split),
    }
    output_file.with_name("svamp_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-file", type=Path, default=Path("data/processed/svamp_test.jsonl")
    )
    parser.add_argument("--dataset-revision")
    args = parser.parse_args()
    prepare(args.output_file, args.dataset_revision)


if __name__ == "__main__":
    main()
