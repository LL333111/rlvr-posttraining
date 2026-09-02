from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.analysis.plots import line_plot


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    metadata = {
        branch: _load(args.results_dir / branch / "run_metadata.json")
        for branch in ("continued_sft", "grpo")
    }
    source_hashes = {item["source_checkpoint_sha256"] for item in metadata.values()}
    if len(source_hashes) != 1:
        raise RuntimeError("Branch provenance failure: SFT source checkpoint hashes do not match")
    rows = []
    for branch, item in metadata.items():
        rows.append(
            {
                "method": "Continued SFT" if branch == "continued_sft" else "GRPO / RLVR",
                "extra_gpu_hours": item["training_gpu_hours"],
                "optimizer_steps": item["optimizer_steps"],
                "peak_vram_gb": item["peak_vram_bytes"] / 1e9 if item["peak_vram_bytes"] else None,
                "training_tokens": item.get("training_tokens_seen"),
            }
        )
    output = args.results_dir / "analysis" / "compute_table.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| Method | Extra GPU hours | Optimizer steps | Training tokens | Peak VRAM (GB) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        vram = f"{row['peak_vram_gb']:.2f}" if row["peak_vram_gb"] is not None else "n/a"
        training_tokens = row["training_tokens"] or "n/a"
        lines.append(
            f"| {row['method']} | {row['extra_gpu_hours']:.3f} | "
            f"{row['optimizer_steps']} | {training_tokens} | {vram} |"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    eval_dir = args.results_dir / "evaluation"
    eval_payloads = {}
    for checkpoint in ("base", "sft", "continued_sft", "grpo"):
        for benchmark in ("gsm8k", "svamp"):
            path = eval_dir / f"{checkpoint}_{benchmark}.json"
            if path.is_file():
                eval_payloads[(checkpoint, benchmark)] = _load(path)
    if len(eval_payloads) == 8:
        labels = {
            "base": "Base",
            "sft": "Reasoning SFT",
            "continued_sft": "Continued SFT",
            "grpo": "GRPO / RLVR",
        }
        result_lines = [
            "| Model | GSM8K exact match | SVAMP exact match | Extra GPU hours |",
            "|---|---:|---:|---:|",
        ]
        for checkpoint in ("base", "sft", "continued_sft", "grpo"):
            extra = "—"
            if checkpoint in metadata:
                extra = f"{metadata[checkpoint]['training_gpu_hours']:.3f}"
            gsm8k = eval_payloads[(checkpoint, "gsm8k")]["accuracy"]
            svamp = eval_payloads[(checkpoint, "svamp")]["accuracy"]
            result_lines.append(
                f"| {labels[checkpoint]} | {gsm8k:.3%} | {svamp:.3%} | {extra} |"
            )
        (args.results_dir / "analysis" / "main_results.md").write_text(
            "\n".join(result_lines) + "\n", encoding="utf-8"
        )

    sweep_rows = []
    for branch in ("continued_sft", "grpo"):
        path = args.results_dir / "evaluation" / f"{branch}_gsm8k_sweep.json"
        if path.is_file():
            for row in _load(path):
                method = "Continued SFT" if branch == "continued_sft" else "GRPO / RLVR"
                sweep_rows.append({**row, "method": method})
    if sweep_rows:
        line_plot(
            sweep_rows,
            x="gpu_hours",
            y="accuracy",
            group="method",
            xlabel="Additional GPU hours",
            ylabel="GSM8K subset exact match",
            title="Performance versus post-training budget",
            output=args.figures_dir / "performance_vs_gpu_time.png",
        )
    print(output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
