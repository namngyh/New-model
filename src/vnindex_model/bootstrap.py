"""IID, moving, stationary, and regime-conditioned residual bootstrap."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def iid_bootstrap(residuals: np.ndarray, size: int | tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    pool = np.asarray(residuals)[np.isfinite(residuals)]
    if len(pool) == 0:
        raise ValueError("Residual pool rỗng")
    return rng.choice(pool, size=size, replace=True)


def moving_block_bootstrap(
    residuals: np.ndarray, sample_length: int, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    pool = np.asarray(residuals)[np.isfinite(residuals)]
    if block_length < 1 or block_length > len(pool):
        raise ValueError("block_length phải nằm trong [1, len(residuals)]")
    output: list[np.ndarray] = []
    while sum(len(block) for block in output) < sample_length:
        start = int(rng.integers(0, len(pool) - block_length + 1))
        output.append(pool[start : start + block_length])
    return np.concatenate(output)[:sample_length]


def stationary_bootstrap(
    residuals: np.ndarray, sample_length: int, mean_block_length: int, rng: np.random.Generator
) -> np.ndarray:
    pool = np.asarray(residuals)[np.isfinite(residuals)]
    if mean_block_length < 1:
        raise ValueError("mean_block_length phải dương")
    output = np.empty(sample_length)
    index = int(rng.integers(0, len(pool)))
    restart_probability = 1 / mean_block_length
    for position in range(sample_length):
        if position == 0 or rng.random() < restart_probability:
            index = int(rng.integers(0, len(pool)))
        else:
            index = (index + 1) % len(pool)
        output[position] = pool[index]
    return output


def stationary_bootstrap_indices(
    sample_length: int,
    mean_block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Stationary-bootstrap indices; consecutive observations retain original adjacency."""
    if sample_length < 1 or mean_block_length < 1:
        raise ValueError("sample_length và mean_block_length phải dương")
    output = np.empty(int(sample_length), dtype=int)
    index = int(rng.integers(0, sample_length))
    for position in range(int(sample_length)):
        if position == 0 or rng.random() < 1.0 / int(mean_block_length):
            index = int(rng.integers(0, sample_length))
        else:
            index = (index + 1) % int(sample_length)
        output[position] = index
    return output


def regime_conditioned_sample(
    residuals: np.ndarray,
    probabilities: np.ndarray,
    state: int,
    size: int,
    rng: np.random.Generator,
    minimum_effective_size: float = 30,
) -> tuple[np.ndarray, bool]:
    residuals = np.asarray(residuals, dtype=float)
    weights = np.asarray(probabilities, dtype=float)[:, state]
    mask = np.isfinite(residuals) & np.isfinite(weights)
    residuals, weights = residuals[mask], weights[mask]
    effective = weights.sum() ** 2 / max(np.square(weights).sum(), 1e-12)
    fallback = effective < minimum_effective_size or weights.sum() <= 0
    if fallback:
        return rng.choice(residuals, size=size, replace=True), True
    weights = weights / weights.sum()
    return rng.choice(residuals, size=size, replace=True, p=weights), False


def choose_block_length(residuals: np.ndarray, candidates: list[int] | None = None) -> int:
    """Data-driven rule chosen without test labels: minimize lag-ACF reconstruction error."""
    pool = np.asarray(residuals, dtype=float)
    pool = pool[np.isfinite(pool)]
    candidates = candidates or [5, 10, 15, 20]
    lag_one = abs(np.corrcoef(pool[:-1], pool[1:])[0, 1]) if len(pool) > 2 else 0.0
    target = max(2, int(round(1 / max(1 - lag_one, 0.05))))
    return min(candidates, key=lambda value: abs(value - target))


def _outer_record_metrics(records: pd.DataFrame) -> dict[str, float]:
    actual = records["actual_return"].to_numpy(dtype=float)
    old = records["old_center"].to_numpy(dtype=float)
    new = records["improved_center"].to_numpy(dtype=float)
    lower, upper = records["lower_95"].to_numpy(dtype=float), records["upper_95"].to_numpy(dtype=float)
    old_lower = records["old_lower_95"].to_numpy(dtype=float)
    old_upper = records["old_upper_95"].to_numpy(dtype=float)
    covered = (actual >= lower) & (actual <= upper)
    old_covered = (actual >= old_lower) & (actual <= old_upper)
    alpha = 0.05
    interval_score = upper - lower + 2 / alpha * (lower - actual) * (actual < lower) + 2 / alpha * (actual - upper) * (actual > upper)
    old_interval_score = old_upper - old_lower + 2 / alpha * (old_lower - actual) * (actual < old_lower) + 2 / alpha * (actual - old_upper) * (actual > old_upper)
    violations = actual < records["conformal_var_95"].to_numpy(dtype=float)
    actual_tail = actual <= np.quantile(actual, 0.05)
    realized_es = float(actual[actual_tail].mean()) if actual_tail.any() else np.nan
    predicted_es = float(records["conformal_es_95"].mean())
    metrics = {
        "old_rmse": float(np.sqrt(np.mean(np.square(actual - old)))),
        "new_rmse": float(np.sqrt(np.mean(np.square(actual - new)))),
        "old_mae": float(np.mean(np.abs(actual - old))),
        "new_mae": float(np.mean(np.abs(actual - new))),
        "brier_score": float(records["brier_loss"].mean()),
        "coverage_95": float(covered.mean()),
        "old_coverage_95": float(old_covered.mean()),
        "coverage_improvement": float(covered.mean() - old_covered.mean()),
        "interval_score_95": float(interval_score.mean()),
        "old_interval_score_95": float(old_interval_score.mean()),
        "interval_score_change": float(interval_score.mean() - old_interval_score.mean()),
        "var_exceedance_95": float(violations.mean()),
        "expected_shortfall_calibration_error": predicted_es - realized_es,
        "drawdown_probability_5": float((records["actual_drawdown"] <= -0.05).mean()),
    }
    metrics["rmse_difference_new_minus_old"] = metrics["new_rmse"] - metrics["old_rmse"]
    metrics["mae_difference_new_minus_old"] = metrics["new_mae"] - metrics["old_mae"]
    return metrics


def outer_stationary_bootstrap_quick(
    records: pd.DataFrame,
    replications: int = 2000,
    mean_block_length: int = 20,
    seed: int = 55,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap complete OOS forecast records to preserve cross-metric dependence."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    for replication in range(int(replications)):
        indices = stationary_bootstrap_indices(len(records), int(mean_block_length), rng)
        rows.append({"replication": replication, **_outer_record_metrics(records.iloc[indices])})
    replicates = pd.DataFrame(rows)
    summary_rows = []
    for metric in replicates.columns.drop("replication"):
        values = replicates[metric].to_numpy(dtype=float)
        summary_rows.append(
            {
                "metric": metric,
                "mean": float(np.mean(values)),
                "standard_error": float(np.std(values, ddof=1)),
                "ci_lower": float(np.quantile(values, 0.025)),
                "ci_upper": float(np.quantile(values, 0.975)),
                "replications": int(replications),
                "mean_block_length": int(mean_block_length),
            }
        )
    return pd.DataFrame(summary_rows), replicates


def outer_bootstrap_full(
    replications: int,
    refit_callback,
    checkpoint_directory: str | Path,
) -> pd.DataFrame:
    """Run model-refit outer bootstrap with an atomic checkpoint per replication."""
    root = Path(checkpoint_directory)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for replication in range(int(replications)):
        checkpoint = root / f"replication_{replication:04d}.json"
        if checkpoint.exists():
            rows.append(json.loads(checkpoint.read_text(encoding="utf-8")))
            continue
        result = {"replication": replication, **dict(refit_callback(replication))}
        temporary = checkpoint.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(checkpoint)
        rows.append(result)
    return pd.DataFrame(rows)
