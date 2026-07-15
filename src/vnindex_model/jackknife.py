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


def _classification_recall(records: pd.DataFrame, label: str) -> float:
    mask = records["actual_regime"] == label
    return float((records.loc[mask, "predicted_regime"] == label).mean()) if mask.any() else np.nan


def _sensitivity_metrics(records: pd.DataFrame) -> dict[str, float]:
    actual = records["actual_return"].to_numpy(dtype=float)
    prediction = records["improved_center"].to_numpy(dtype=float)
    lower = records["lower_95"].to_numpy(dtype=float)
    upper = records["upper_95"].to_numpy(dtype=float)
    var = records["conformal_var_95"].to_numpy(dtype=float)
    realized_tail = actual <= np.quantile(actual, 0.05)
    realized_es = actual[realized_tail].mean() if realized_tail.any() else np.nan
    return {
        "h20_rmse": float(np.sqrt(np.mean(np.square(actual - prediction)))),
        "coverage_95": float(np.mean((actual >= lower) & (actual <= upper))),
        "var_95_mean": float(np.mean(var)),
        "var_95_exceedance": float(np.mean(actual < var)),
        "expected_shortfall_calibration_error": float(records["conformal_es_95"].mean() - realized_es),
        "probability_maximum_drawdown_5": float(np.mean(records["actual_drawdown"] <= -0.05)),
        "bear_recall": _classification_recall(records, "Bear"),
        "stress_recall": _classification_recall(records, "Stress"),
    }


def delete_block_jackknife(records: pd.DataFrame) -> pd.DataFrame:
    """Rank month, quarter, regime episode, and large-drawdown block influence."""
    records = records.reset_index(drop=True).copy()
    date = pd.to_datetime(records["date"])
    regime_episode = records["actual_regime"].ne(records["actual_regime"].shift()).cumsum()
    drawdown_flag = records["actual_drawdown"] <= -0.05
    drawdown_episode = drawdown_flag.ne(drawdown_flag.shift()).cumsum()
    specifications: list[tuple[str, pd.Series, np.ndarray]] = [
        ("month", date.dt.to_period("M").astype(str), np.ones(len(records), dtype=bool)),
        ("quarter", date.dt.to_period("Q").astype(str), np.ones(len(records), dtype=bool)),
        ("regime_episode", regime_episode.astype(str), np.ones(len(records), dtype=bool)),
        ("large_drawdown_block", drawdown_episode.astype(str), drawdown_flag.to_numpy()),
    ]
    full = _sensitivity_metrics(records)
    rows: list[dict[str, float | int | str]] = []
    for block_type, groups, eligible in specifications:
        for group in pd.unique(groups[eligible]):
            omitted = (groups == group).to_numpy() & eligible
            retained = records.loc[~omitted]
            if len(retained) < 30:
                continue
            estimate = _sensitivity_metrics(retained)
            for metric, full_value in full.items():
                value = estimate[metric]
                delta = value - full_value
                rows.append(
                    {
                        "block_type": block_type,
                        "block_label": str(group),
                        "metric": metric,
                        "observations_removed": int(omitted.sum()),
                        "full_estimate": full_value,
                        "leave_block_out_estimate": value,
                        "influence": delta,
                        "absolute_influence": abs(delta),
                    }
                )
    table = pd.DataFrame(rows)
    if len(table):
        table["influence_rank"] = table.groupby("metric")["absolute_influence"].rank(
            method="dense", ascending=False
        )
    return table.sort_values(["metric", "absolute_influence"], ascending=[True, False]).reset_index(drop=True)


def feature_importance_delete_block_jackknife(
    records: pd.DataFrame,
    transformed_features: np.ndarray,
    actual_return: np.ndarray,
    predictor,
    feature_names: list[str],
    selected_features: list[str],
    seed: int = 55,
) -> pd.DataFrame:
    """Measure fixed-model permutation-importance sensitivity to deleting temporal blocks."""
    records = records.reset_index(drop=True)
    features = np.asarray(transformed_features, dtype=float)
    actual = np.asarray(actual_return, dtype=float)
    selected_positions = [feature_names.index(name) for name in selected_features]
    date = pd.to_datetime(records["date"])
    regime_episode = records["actual_regime"].ne(records["actual_regime"].shift()).cumsum().astype(str)
    drawdown_flag = records["actual_drawdown"] <= -0.05
    drawdown_episode = drawdown_flag.ne(drawdown_flag.shift()).cumsum().astype(str)
    specifications = [
        ("month", date.dt.to_period("M").astype(str), np.ones(len(records), dtype=bool)),
        ("quarter", date.dt.to_period("Q").astype(str), np.ones(len(records), dtype=bool)),
        ("regime_episode", regime_episode, np.ones(len(records), dtype=bool)),
        ("large_drawdown_block", drawdown_episode, drawdown_flag.to_numpy()),
    ]
    rng = np.random.default_rng(seed)

    def importance(mask: np.ndarray, position: int) -> float:
        baseline = np.mean(np.abs(actual[mask] - predictor(features[mask])))
        permuted = features[mask].copy()
        permuted[:, position] = rng.permutation(permuted[:, position])
        return float(np.mean(np.abs(actual[mask] - predictor(permuted))) - baseline)

    full_mask = np.ones(len(records), dtype=bool)
    full_importance = {
        name: importance(full_mask, position) for name, position in zip(selected_features, selected_positions, strict=True)
    }
    rows: list[dict[str, float | int | str]] = []
    for block_type, groups, eligible in specifications:
        for group in pd.unique(groups[eligible]):
            omitted = (groups == group).to_numpy() & eligible
            retained = ~omitted
            if retained.sum() < 30:
                continue
            for name, position in zip(selected_features, selected_positions, strict=True):
                estimate = importance(retained, position)
                delta = estimate - full_importance[name]
                rows.append(
                    {
                        "block_type": block_type,
                        "block_label": str(group),
                        "feature": name,
                        "observations_removed": int(omitted.sum()),
                        "full_permutation_importance": full_importance[name],
                        "leave_block_out_permutation_importance": estimate,
                        "influence": delta,
                        "absolute_influence": abs(delta),
                    }
                )
    return pd.DataFrame(rows).sort_values("absolute_influence", ascending=False).reset_index(drop=True)
