"""Validation-only protection layer for the predictive distribution center."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CenterSelection:
    alpha: float
    selected_center: str
    reason: str
    validation_table: pd.DataFrame


def apply_center_blend(ml_prediction: np.ndarray, drift_prediction: np.ndarray, alpha: float) -> np.ndarray:
    """Blend an ML prediction with the locked random-walk-drift protection center."""
    ml = np.asarray(ml_prediction, dtype=float)
    drift = np.asarray(drift_prediction, dtype=float)
    if ml.shape != drift.shape:
        raise ValueError("ML và drift prediction phải cùng shape")
    return float(alpha) * ml + (1.0 - float(alpha)) * drift


def select_validation_gated_center(
    actual: np.ndarray,
    ml_prediction: np.ndarray,
    drift_prediction: np.ndarray,
    alpha_grid: list[float],
    minimum_relative_improvement: float = 0.01,
    use_one_standard_error_rule: bool = True,
    effective_block_length: int = 1,
) -> CenterSelection:
    """Select alpha using validation records only, with shrinkage toward alpha=0."""
    actual = np.asarray(actual, dtype=float)
    ml_prediction = np.asarray(ml_prediction, dtype=float)
    drift_prediction = np.asarray(drift_prediction, dtype=float)
    if not (actual.shape == ml_prediction.shape == drift_prediction.shape):
        raise ValueError("Actual, ML và drift prediction phải cùng shape")
    grid = sorted({float(value) for value in alpha_grid})
    if not grid or grid[0] != 0.0 or grid[-1] > 1.0 or grid[0] < 0.0:
        raise ValueError("alpha_grid phải chứa 0 và nằm trong [0, 1]")
    rows: list[dict[str, float]] = []
    effective_n = max(len(actual) // max(int(effective_block_length), 1), 2)
    for alpha in grid:
        prediction = apply_center_blend(ml_prediction, drift_prediction, alpha)
        error = actual - prediction
        squared = np.square(error)
        rmse = float(np.sqrt(np.mean(squared)))
        mse_se = float(np.std(squared, ddof=1) / np.sqrt(effective_n)) if len(squared) > 1 else 0.0
        rmse_se = mse_se / max(2.0 * rmse, 1e-12)
        rows.append(
            {
                "alpha": alpha,
                "rmse": rmse,
                "rmse_standard_error": rmse_se,
                "mae": float(np.mean(np.abs(error))),
                "directional_accuracy": float(np.mean(np.sign(actual) == np.sign(prediction))),
            }
        )
    table = pd.DataFrame(rows)
    baseline_rmse = float(table.loc[table["alpha"] == 0.0, "rmse"].iloc[0])
    best_row = table.loc[table["rmse"].idxmin()]
    relative_improvement = (baseline_rmse - float(best_row["rmse"])) / max(baseline_rmse, 1e-12)
    if relative_improvement < float(minimum_relative_improvement):
        selected_alpha = 0.0
        reason = (
            f"fallback: validation improvement {relative_improvement:.4%} < "
            f"minimum {minimum_relative_improvement:.4%}"
        )
    elif use_one_standard_error_rule:
        threshold = float(best_row["rmse"] + best_row["rmse_standard_error"])
        selected_alpha = float(table.loc[table["rmse"] <= threshold, "alpha"].min())
        reason = f"one-standard-error shrinkage; best alpha={float(best_row['alpha']):.2f}"
    else:
        selected_alpha = float(best_row["alpha"])
        reason = "minimum validation RMSE"
    table["relative_improvement_vs_drift"] = (baseline_rmse - table["rmse"]) / max(baseline_rmse, 1e-12)
    table["selected"] = np.isclose(table["alpha"], selected_alpha)
    table["selection_reason"] = reason
    selected_center = "random_walk_drift_fallback" if selected_alpha == 0.0 else "validation_gated_blend"
    return CenterSelection(selected_alpha, selected_center, reason, table)
