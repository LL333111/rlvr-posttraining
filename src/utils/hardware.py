from __future__ import annotations

import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from typing import Any


def _git(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *command], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_state() -> dict[str, Any]:
    commit = _git(["rev-parse", "HEAD"])
    dirty = _git(["status", "--porcelain"])
    return {"commit": commit, "dirty": bool(dirty) if dirty is not None else None}


def hardware_snapshot() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "git": git_state(),
        "cuda_available": False,
        "gpu_count": 0,
        "gpus": [],
        "packages": {},
    }
    for package in ("accelerate", "datasets", "peft", "safetensors", "torch", "transformers", "trl"):
        try:
            payload["packages"][package] = version(package)
        except PackageNotFoundError:
            payload["packages"][package] = None
    try:
        import torch

        payload["torch"] = torch.__version__
        payload["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            payload["cuda_version"] = torch.version.cuda
            payload["gpu_count"] = torch.cuda.device_count()
            payload["gpus"] = [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ]
    except ImportError:
        payload["torch"] = None
    return payload
