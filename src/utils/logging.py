from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .hardware import hardware_snapshot


def write_json(path: str | Path, payload: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: str | Path, payload: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def sha256_tree(path: str | Path) -> str:
    root = Path(path)
    digest = hashlib.sha256()
    for file_path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(file_path.relative_to(root)).encode())
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


class RunTimer:
    def __init__(self) -> None:
        self.started = time.monotonic()

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started

    def metadata(self, **extra: Any) -> dict[str, Any]:
        hardware = hardware_snapshot()
        elapsed = self.elapsed_seconds
        gpu_count = int(hardware.get("gpu_count", 0))
        return {
            "wall_clock_seconds": elapsed,
            "gpu_hours": elapsed * gpu_count / 3600.0,
            "hardware": hardware,
            **extra,
        }
