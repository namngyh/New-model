"""Causal technical, risk, drawdown, volume, and calendar features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import linregress


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    return series.rolling(window).apply(lambda values: linregress(x, values).slope, raw=True)


def _streak(signs: pd.Series) -> pd.Series:
    groups = signs.ne(signs.shift()).cumsum()
    return signs.groupby(groups).cumcount().add(1).mul(signs)


def _historical_es(series: pd.Series, alpha: float = 0.05) -> float:
    values = np.asarray(series)
    threshold = np.quantile(values, alpha)
    tail = values[values <= threshold]
    return float(tail.mean()) if len(tail) else float(threshold)


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build features using current and past rows only; no backfill is used."""
    close = frame["close"].astype(float)
    log_price = np.log(close)
    log_return = log_price.diff()
    simple_return = close.pct_change()
    out = pd.DataFrame(index=frame.index)
    out["simple_return"] = simple_return
    out["log_return"] = log_return
    for lag in range(1, 21):
        out[f"return_lag_{lag}"] = log_return.shift(lag)
    for window in [2, 3, 5, 10, 20, 40, 60]:
        out[f"cumulative_return_{window}"] = log_price.diff(window)
    for window in [5, 10, 20, 50, 100, 200]:
        sma = close.rolling(window).mean()
        out[f"distance_sma_{window}"] = close / sma - 1
    for window in [5, 10, 12, 20, 26, 50]:
        ema = close.ewm(span=window, adjust=False, min_periods=window).mean()
        out[f"distance_ema_{window}"] = close / ema - 1
    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    out["macd"] = (ema12 - ema26) / close
    signal = (ema12 - ema26).ewm(span=9, adjust=False, min_periods=9).mean()
    out["macd_signal"] = signal / close
    out["macd_histogram"] = ((ema12 - ema26) - signal) / close
    for window in [5, 10, 20, 60]:
        out[f"log_price_slope_{window}"] = _rolling_slope(log_price, window)
        out[f"up_ratio_{window}"] = (log_return > 0).rolling(window).mean()
    out["return_streak"] = _streak(np.sign(log_return).fillna(0))
    for window in [5, 10, 20, 40, 60]:
        out[f"rolling_volatility_{window}"] = log_return.rolling(window).std(ddof=1)
    out["ewma_volatility"] = log_return.ewm(span=40, adjust=False, min_periods=10).std()
    for name, mask in {"downside": log_return.clip(upper=0), "upside": log_return.clip(lower=0)}.items():
        out[f"{name}_volatility"] = mask.rolling(20).std(ddof=1)
    out["semivariance"] = log_return.clip(upper=0).pow(2).rolling(20).mean()
    out["rolling_skewness"] = log_return.rolling(60).skew()
    out["rolling_excess_kurtosis"] = log_return.rolling(60).kurt()
    out["historical_var_95"] = log_return.rolling(252, min_periods=60).quantile(0.05)
    out["historical_es_95"] = log_return.rolling(252, min_periods=60).apply(_historical_es, raw=True)
    out["jump_proxy"] = log_return.abs() / out["rolling_volatility_20"].clip(lower=1e-8)
    out["volatility_of_volatility"] = out["rolling_volatility_20"].rolling(20).std()
    rolling_max = close.cummax()
    out["current_drawdown"] = close / rolling_max - 1
    for window in [20, 60, 120, 252]:
        peak = close.rolling(window).max()
        out[f"distance_peak_{window}"] = close / peak - 1
    out["rolling_max_drawdown"] = out["current_drawdown"].rolling(252, min_periods=20).min()
    is_peak = close.eq(rolling_max)
    peak_group = is_peak.cumsum()
    out["drawdown_duration"] = (~is_peak).groupby(peak_group).cumsum()
    out["days_since_peak"] = out["drawdown_duration"]
    out["recovery_speed"] = out["current_drawdown"].diff(5) / 5
    if {"high", "low"}.issubset(frame.columns):
        high, low = frame["high"].astype(float), frame["low"].astype(float)
        out["realized_range"] = np.log(high / low.clip(lower=1e-8))
        out["parkinson_volatility"] = np.sqrt(out["realized_range"].pow(2).rolling(20).mean() / (4 * np.log(2)))
        previous = close.shift(1)
        true_range = pd.concat([high - low, (high - previous).abs(), (low - previous).abs()], axis=1).max(axis=1)
        out["atr_normalized"] = true_range.rolling(14).mean() / close
        if "open" in frame:
            open_ = frame["open"].astype(float)
            gk = 0.5 * np.log(high / low).pow(2) - (2 * np.log(2) - 1) * np.log(close / open_).pow(2)
            out["garman_klass_volatility"] = np.sqrt(gk.clip(lower=0).rolling(20).mean())
    middle = close.rolling(20).mean()
    out["bollinger_width"] = 4 * close.rolling(20).std() / middle
    if "volume" in frame and frame["volume"].notna().any():
        volume = frame["volume"].astype(float).replace(0, np.nan)
        log_volume = np.log1p(volume)
        out["log_volume"] = log_volume
        out["volume_change"] = log_volume.diff()
        mean20, std20 = log_volume.rolling(20).mean(), log_volume.rolling(20).std()
        out["volume_zscore"] = (log_volume - mean20) / std20
        out["volume_average_20"] = volume / volume.rolling(20).mean() - 1
        out["volume_volatility"] = log_volume.diff().rolling(20).std()
        out["price_volume_interaction"] = log_return * out["volume_zscore"]
        out["obv"] = (np.sign(simple_return).fillna(0) * volume.fillna(0)).cumsum()
        out["obv_trend"] = out["obv"].diff(20) / volume.rolling(20).mean()
    date = pd.to_datetime(frame["date"])
    out["day_of_week"] = date.dt.dayofweek
    out["month"] = date.dt.month
    out["quarter"] = date.dt.quarter
    out["is_month_end"] = date.dt.is_month_end.astype(int)
    out["is_month_start"] = date.dt.is_month_start.astype(int)
    return out.replace([np.inf, -np.inf], np.nan)


def select_train_features(
    features: pd.DataFrame, train_index: np.ndarray, correlation_threshold: float = 0.995
) -> list[str]:
    """Select nonconstant, nonduplicate features using train rows only."""
    train = features.iloc[train_index]
    nonconstant = train.columns[train.nunique(dropna=True) > 1].tolist()
    corr = train[nonconstant].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = {column for column in upper.columns if (upper[column] > correlation_threshold).any()}
    return [column for column in nonconstant if column not in drop]
