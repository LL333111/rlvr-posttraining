from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

from src.utils.logging import append_jsonl


class JsonlLogCallback(TrainerCallback):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.started = time.monotonic()
        self.last_elapsed = 0.0
        self.last_tokens = 0.0

    def on_train_begin(self, args: Any, state: Any, control: Any, **_: Any):
        self.started = time.monotonic()
        self.last_elapsed = 0.0
        self.last_tokens = 0.0
        return control

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: dict[str, Any] | None = None,
        **_: Any,
    ):
        if state.is_world_process_zero and logs:
            elapsed = time.monotonic() - self.started
            telemetry: dict[str, Any] = {"elapsed_seconds": elapsed}
            token_value = logs.get("num_input_tokens_seen", logs.get("num_tokens"))
            if isinstance(token_value, (int, float)) and elapsed > self.last_elapsed:
                telemetry["tokens_per_second_since_last_log"] = (
                    float(token_value) - self.last_tokens
                ) / (elapsed - self.last_elapsed)
                self.last_tokens = float(token_value)
                self.last_elapsed = elapsed
            try:
                import torch

                if torch.cuda.is_available():
                    telemetry.update(
                        {
                            "gpu_memory_allocated_bytes": torch.cuda.memory_allocated(),
                            "gpu_memory_reserved_bytes": torch.cuda.memory_reserved(),
                        }
                    )
            except ImportError:
                pass
            append_jsonl(self.path, {"step": state.global_step, **telemetry, **logs})
        return control


class WallClockBudgetCallback(TrainerCallback):
    """Stop after the first completed optimizer step past a wall-clock budget."""

    def __init__(self, budget_seconds: float | None) -> None:
        self.budget_seconds = budget_seconds
        self.started: float | None = None
        self.stopped_for_budget = False

    def on_train_begin(self, args: Any, state: Any, control: Any, **_: Any):
        self.started = time.monotonic()
        return control

    def on_step_end(self, args: Any, state: Any, control: Any, **_: Any):
        if (
            self.budget_seconds is not None
            and self.started is not None
            and time.monotonic() - self.started >= self.budget_seconds
        ):
            control.should_training_stop = True
            control.should_save = True
            self.stopped_for_budget = True
        return control
