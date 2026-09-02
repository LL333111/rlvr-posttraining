from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be a mapping")
    return payload


def require_keys(mapping: dict[str, Any], keys: list[str], section: str = "config") -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"missing keys in {section}: {', '.join(missing)}")
