from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from src.utils.hardware import hardware_snapshot
from src.utils.logging import sha256_tree, write_json


def prepare_run(config_path: str | Path, output_dir: str | Path, results_dir: str | Path) -> None:
    output = Path(output_dir)
    results = Path(results_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Refusing to mix a new run with existing checkpoint artifacts in {output}"
        )
    if results.exists() and any(results.iterdir()):
        raise FileExistsError(
            f"Refusing to mix a new run with existing result artifacts in {results}"
        )
    output.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, results / "config.yaml")


def reset_peak_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def peak_memory_bytes() -> int | None:
    try:
        import torch

        return torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    except ImportError:
        return None


def train_and_record(
    trainer: Any,
    *,
    output_dir: str | Path,
    results_dir: str | Path,
    config_path: str | Path,
    source_checkpoint: str | Path | None = None,
    budget_callback: Any = None,
) -> dict[str, Any]:
    reset_peak_memory()
    run_started = time.monotonic()
    train_started = time.monotonic()
    train_output = trainer.train()
    training_seconds = time.monotonic() - train_started
    trainer.save_model(str(output_dir))
    if getattr(trainer, "processing_class", None) is not None:
        trainer.processing_class.save_pretrained(str(output_dir))

    hardware = hardware_snapshot()
    gpu_count = int(os.environ.get("WORLD_SIZE", "1")) if hardware.get("cuda_available") else 0
    tokens_seen = getattr(trainer.state, "num_input_tokens_seen", None)
    if tokens_seen is None:
        for row in reversed(trainer.state.log_history):
            if "num_input_tokens_seen" in row or "num_tokens" in row:
                tokens_seen = row.get("num_input_tokens_seen", row.get("num_tokens"))
                break
    if isinstance(tokens_seen, (int, float)):
        tokens_seen = int(tokens_seen)
    source_hash = sha256_tree(source_checkpoint) if source_checkpoint else None
    model_config = getattr(trainer.model, "config", None)
    metadata = {
        "config_path": str(config_path),
        "source_checkpoint": str(source_checkpoint) if source_checkpoint else None,
        "source_checkpoint_sha256": source_hash,
        "output_checkpoint": str(output_dir),
        "output_checkpoint_sha256": sha256_tree(output_dir),
        "upstream_model": getattr(model_config, "_name_or_path", None),
        "upstream_model_commit": getattr(model_config, "_commit_hash", None),
        "optimizer_steps": int(trainer.state.global_step),
        "training_tokens_seen": tokens_seen,
        "training_wall_clock_seconds": training_seconds,
        "training_gpu_hours": training_seconds * gpu_count / 3600.0,
        "training_gpu_count": gpu_count,
        "total_run_wall_clock_seconds": time.monotonic() - run_started,
        "peak_vram_bytes": peak_memory_bytes(),
        "stopped_for_budget": bool(
            budget_callback and getattr(budget_callback, "stopped_for_budget", False)
        ),
        "train_metrics": dict(getattr(train_output, "metrics", {})),
        "hardware": hardware,
    }
    write_json(Path(results_dir) / "run_metadata.json", metadata)
    write_json(Path(results_dir) / "trainer_state.json", trainer.state.log_history)
    return metadata


def matched_wall_seconds(metadata_file: str | Path, fallback_gpu_hours: float) -> float:
    path = Path(metadata_file)
    if not path.is_file():
        return fallback_gpu_hours * 3600.0
    payload = json.loads(path.read_text(encoding="utf-8"))
    target_gpu_hours = float(payload["training_gpu_hours"])
    snapshot = hardware_snapshot()
    current_gpu_count = int(os.environ.get("WORLD_SIZE", "1")) if snapshot.get("cuda_available") else 0
    if current_gpu_count <= 0:
        raise RuntimeError("Compute-matched training requires at least one GPU")
    return target_gpu_hours * 3600.0 / current_gpu_count


def wall_seconds_for_gpu_hours(gpu_hours: float) -> float:
    snapshot = hardware_snapshot()
    gpu_count = int(os.environ.get("WORLD_SIZE", "1")) if snapshot.get("cuda_available") else 0
    if gpu_count <= 0:
        raise RuntimeError("Training requires at least one CUDA GPU")
    return gpu_hours * 3600.0 / gpu_count
