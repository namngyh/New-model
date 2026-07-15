"""Artifact persistence and reproducibility helpers."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
import numpy as np


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def save_model(path: str | Path, model: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)


def library_versions() -> dict[str, str]:
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "arch", "hmmlearn", "matplotlib"]
    result: dict[str, str] = {"python": platform.python_version()}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


def run_metadata(data_path: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path),
        "data_hash": sha256_file(data_path),
        "config_path": config.get("_config_path"),
        "seed": config["project"]["seed"],
        "library_versions": library_versions(),
        "model_version": "1.0.0",
    }
