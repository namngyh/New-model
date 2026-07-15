"""Configuration loading and validation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration and resolve only project-relative paths."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    required = {"project", "data", "hmm", "random_forest", "simulation"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Thiếu nhóm cấu hình: {sorted(missing)}")
    out = deepcopy(config)
    out["_config_path"] = str(path)
    return out
