"""Block jackknife on out-of-sample forecast records."""

from __future__ import annotations

import numpy as np
import pandas as pd


def jackknife_summary(values: np.ndarray, groups: pd.Series, metric_name: str) -> dict[str, object]:
    values = np.asarray(values, dtype=float)
    estimates: list[float] = []
    labels: list[str] = []
    for group in pd.unique(groups):
        mask = np.asarray(groups != group)
        if mask.sum() < 2:
            continue
        estimates.append(float(np.mean(values[mask])))
        labels.append(str(group))
    estimates_array = np.asarray(estimates)
    full = float(np.mean(values))
    count = len(estimates_array)
    bias = float((count - 1) * (estimates_array.mean() - full)) if count else np.nan
    variance = (
        float((count - 1) / max(count, 1) * np.sum((estimates_array - estimates_array.mean()) ** 2))
        if count
        else np.nan
    )
    standard_error = np.sqrt(variance) if np.isfinite(variance) else np.nan
    return {
        "metric": metric_name,
        "full_estimate": full,
        "blocks": count,
        "jackknife_bias": bias,
        "jackknife_variance": variance,
        "ci_lower": float(full - 1.96 * standard_error),
        "ci_upper": float(full + 1.96 * standard_error),
        "block_labels": labels,
        "leave_one_out_estimates": estimates,
    }


def block_jackknife_table(records: pd.DataFrame) -> pd.DataFrame:
    """Evaluate error stability by month, quarter, regime episode, and crisis-like block."""
    date = pd.to_datetime(records["date"])
    specifications = {
        "leave_one_month_out_mae": date.dt.to_period("M").astype(str),
        "leave_one_quarter_out_mae": date.dt.to_period("Q").astype(str),
        "leave_one_regime_episode_out_mae": records["predicted_regime"]
        .ne(records["predicted_regime"].shift())
        .cumsum()
        .astype(str),
        "leave_one_crisis_like_block_out_mae": (
            (records["actual_drawdown"] < -0.05).ne((records["actual_drawdown"] < -0.05).shift()).cumsum()
        ).astype(str),
    }
    metric_values = {
        "mae": np.abs(records["actual_return"] - records["predicted_return"]).to_numpy(),
        "brier_score": records["brier_loss"].to_numpy(),
        "interval_95_coverage": records["interval_95_covered"].to_numpy(),
        "var_95": records["var_95"].to_numpy(),
        "expected_shortfall_95": records["expected_shortfall_95"].to_numpy(),
        "predicted_max_drawdown": records["predicted_drawdown"].to_numpy(),
    }
    rows = []
    for block_name, groups in specifications.items():
        for metric_name, values in metric_values.items():
            result = jackknife_summary(values, groups, f"{block_name}:{metric_name}")
            rows.append(
                {key: value for key, value in result.items() if key not in {"block_labels", "leave_one_out_estimates"}}
            )
    return pd.DataFrame(rows)
