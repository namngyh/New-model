"""Causal statistical and machine-learning point baselines."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import Holt


def baseline_predictions(
    close, returns, forward_target, features, train_index, test_index, horizon: int
) -> dict[str, np.ndarray]:
    close = np.asarray(close, dtype=float)
    returns = np.asarray(returns, dtype=float)
    train_daily = returns[train_index]
    historical_mean = np.nanmean(train_daily) * horizon
    drift = np.log(close[train_index[-1]] / close[train_index[0]]) / max(len(train_index), 1) * horizon
    rolling = np.array([np.nanmean(returns[max(0, idx - 60) : idx]) * horizon for idx in test_index])
    ar_x = returns[train_index[:-1]]
    ar_y = returns[train_index[1:]]
    valid = np.isfinite(ar_x) & np.isfinite(ar_y)
    coefficient = np.polyfit(ar_x[valid], ar_y[valid], 1) if valid.sum() > 10 else np.array([0.0, 0.0])
    ar = np.polyval(coefficient, np.nan_to_num(returns[test_index], nan=0.0)) * horizon
    train_series = train_daily[np.isfinite(train_daily)]
    arima_candidates = []
    for order in [(0, 0, 0), (1, 0, 0), (2, 0, 0)]:
        try:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                fitted = ARIMA(train_series, order=order, trend="c").fit()
            if bool(getattr(fitted, "mle_retvals", {}).get("converged", True)):
                arima_candidates.append((float(fitted.aic), fitted))
        except (ValueError, np.linalg.LinAlgError):
            continue
    if arima_candidates:
        arima_result = min(arima_candidates, key=lambda item: item[0])[1]
        parameters = np.asarray(arima_result.params, dtype=float)
        constant = float(parameters[0])
        coefficients = parameters[1:-1]
        arima_prediction = np.full(len(test_index), constant * horizon)
        for position, index in enumerate(test_index):
            lags = returns[max(0, index - len(coefficients) + 1) : index + 1][::-1]
            if len(coefficients):
                arima_prediction[position] = (
                    constant + np.dot(coefficients[: len(lags)], np.nan_to_num(lags))
                ) * horizon
    else:
        arima_prediction = ar
    ridge = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))
    ridge.fit(features[train_index], np.asarray(forward_target)[train_index])
    ridge_prediction = ridge.predict(features[test_index])
    try:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            holt_fit = Holt(close[train_index], damped_trend=True, initialization_method="estimated").fit(
                optimized=True
            )
        alpha = float(holt_fit.params["smoothing_level"])
        beta = float(holt_fit.params["smoothing_trend"])
        phi = float(holt_fit.params["damping_trend"])
        level = float(holt_fit.level[0])
        trend = float(holt_fit.trend[0])
        filtered_level = np.empty(len(close))
        filtered_trend = np.empty(len(close))
        start_index = int(train_index[0])
        filtered_level[:start_index] = close[:start_index]
        filtered_trend[:start_index] = 0.0
        for index in range(start_index, len(close)):
            value = close[index]
            if index == start_index:
                filtered_level[index] = level
                filtered_trend[index] = trend
            else:
                previous_level = filtered_level[index - 1]
                previous_trend = filtered_trend[index - 1]
                filtered_level[index] = alpha * value + (1 - alpha) * (previous_level + phi * previous_trend)
                filtered_trend[index] = (
                    beta * (filtered_level[index] - previous_level) + (1 - beta) * phi * previous_trend
                )
        damped_multiplier = sum(phi**step for step in range(1, horizon + 1))
        holt_price = filtered_level[test_index] + damped_multiplier * filtered_trend[test_index]
        holt_return = np.log(np.maximum(holt_price, 1e-8) / close[test_index])
    except (ValueError, RuntimeError, np.linalg.LinAlgError, KeyError):
        holt_return = np.full(len(test_index), drift * 0.8)
    return {
        "random_walk": np.zeros(len(test_index)),
        "random_walk_drift": np.full(len(test_index), drift),
        "historical_mean_return": np.full(len(test_index), historical_mean),
        "rolling_mean_return": rolling,
        "ar_1": ar,
        "arima_small_grid": arima_prediction,
        "holt_damped_trend": holt_return,
        "ridge": ridge_prediction,
    }
