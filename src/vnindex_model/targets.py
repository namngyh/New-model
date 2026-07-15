"""Forward return, price, regime, and drawdown targets."""

from __future__ import annotations

import numpy as np
import pandas as pd

CLASS_NAMES = ["Bull", "Sideway", "Bear", "Stress"]


def assign_regime_labels(
    returns: pd.Series,
    drawdown: pd.Series,
    scale: pd.Series,
    lambda_bull: float,
    lambda_bear: float,
    lambda_stress: float,
) -> pd.Series:
    """Assign four economic classes using fixed, volatility-adjusted thresholds."""
    labels = pd.Series("Sideway", index=returns.index, dtype=object)
    labels[returns > lambda_bull * scale] = "Bull"
    labels[returns < -lambda_bear * scale] = "Bear"
    labels[(drawdown < -lambda_stress * scale) | (returns < -lambda_stress * scale)] = "Stress"
    labels[returns.isna() | scale.isna()] = np.nan
    return labels


def select_regime_thresholds(
    returns: pd.Series,
    drawdown: pd.Series,
    scale: pd.Series,
    train_index: np.ndarray,
    validation_index: np.ndarray,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Select lambdas on validation without inspecting test class proportions."""
    desired = pd.Series({"Bull": 0.30, "Sideway": 0.40, "Bear": 0.15, "Stress": 0.15})
    rows = []
    for direction_lambda in [0.35, 0.55, 0.75]:
        for stress_lambda in [1.10, 1.35, 1.60, 1.85]:
            labels = assign_regime_labels(
                returns,
                drawdown,
                scale,
                direction_lambda,
                direction_lambda,
                stress_lambda,
            )
            train_counts = labels.iloc[train_index].value_counts().reindex(CLASS_NAMES, fill_value=0)
            validation_counts = labels.iloc[validation_index].value_counts().reindex(CLASS_NAMES, fill_value=0)
            train_share = train_counts / max(train_counts.sum(), 1)
            validation_share = validation_counts / max(validation_counts.sum(), 1)
            distribution_penalty = float(np.square(validation_share - desired).sum())
            drift_penalty = float(np.abs(validation_share - train_share).sum())
            sufficiency_penalty = 10.0 * int(train_counts.min() < 80 or validation_counts.min() < 25)
            objective = distribution_penalty + 0.5 * drift_penalty + sufficiency_penalty
            rows.append(
                {
                    "lambda_bull": direction_lambda,
                    "lambda_bear": direction_lambda,
                    "lambda_stress": stress_lambda,
                    "objective": objective,
                    **{f"train_{name.lower()}": int(train_counts[name]) for name in CLASS_NAMES},
                    **{f"validation_{name.lower()}": int(validation_counts[name]) for name in CLASS_NAMES},
                }
            )
    table = pd.DataFrame(rows)
    selected_index = table["objective"].idxmin()
    table["selected"] = table.index == selected_index
    selected = table.loc[selected_index, ["lambda_bull", "lambda_bear", "lambda_stress"]].to_dict()
    return {key: float(value) for key, value in selected.items()}, table


def forward_max_drawdown(close: pd.Series, horizon: int) -> pd.Series:
    values = close.to_numpy(dtype=float)
    result = np.full(len(values), np.nan)
    for origin in range(len(values) - horizon):
        path = np.concatenate(([values[origin]], values[origin + 1 : origin + horizon + 1]))
        drawdown = path / np.maximum.accumulate(path) - 1
        result[origin] = drawdown.min()
    return pd.Series(result, index=close.index)


def build_targets(frame: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    close = frame["close"].astype(float)
    log_close = np.log(close)
    daily_vol = log_close.diff().ewm(span=40, adjust=False, min_periods=20).std().shift(1)
    out = pd.DataFrame(index=frame.index)
    for horizon in horizons:
        returns = log_close.shift(-horizon) - log_close
        scale = daily_vol * np.sqrt(horizon)
        drawdown = forward_max_drawdown(close, horizon)
        labels = assign_regime_labels(returns, drawdown, scale, 0.55, 0.55, 1.35)
        out[f"forward_return_{horizon}"] = returns
        out[f"normalized_return_{horizon}"] = returns / scale.clip(lower=1e-6)
        out[f"future_price_{horizon}"] = close.shift(-horizon)
        out[f"forward_max_drawdown_{horizon}"] = drawdown
        out[f"forecast_scale_{horizon}"] = scale
        out[f"regime_{horizon}"] = pd.Categorical(labels, categories=CLASS_NAMES)
        out[f"target_end_date_{horizon}"] = frame["date"].shift(-horizon)
    return out
