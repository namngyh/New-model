"""Path-dependent drawdown forecasts, calibration, and Monte Carlo diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

DEFAULT_THRESHOLDS = (0.03, 0.05, 0.07, 0.10, 0.15)
DEFAULT_LEVELS = (0.80, 0.90, 0.95, 0.975, 0.99)


@dataclass
class DrawdownForecastResult:
    """Drawdown paths and summaries for one peak-anchor definition."""

    drawdown_paths: np.ndarray
    severity_paths: np.ndarray
    running_maximum_drawdown_paths: np.ndarray
    term_structure: pd.DataFrame
    summary: dict[str, object]


def _as_path_matrix(price_paths: np.ndarray) -> np.ndarray:
    values = np.asarray(price_paths, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("price_paths phải là ma trận paths × horizon không rỗng")
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("price_paths phải hữu hạn và dương")
    return values


def compute_drawdown_paths(
    price_paths: np.ndarray,
    initial_price: float,
    anchor_mode: str = "origin_peak",
    historical_peak: float | None = None,
    rolling_peak: float | None = None,
) -> np.ndarray:
    """Compute negative drawdown returns under an explicit starting peak."""
    prices = _as_path_matrix(price_paths)
    initial = float(initial_price)
    if not np.isfinite(initial) or initial <= 0:
        raise ValueError("initial_price phải hữu hạn và dương")
    if anchor_mode == "origin_peak":
        starting_peak = initial
    elif anchor_mode == "historical_peak":
        if historical_peak is None:
            raise ValueError("historical_peak bắt buộc cho historical_peak anchor")
        starting_peak = max(float(historical_peak), initial)
    elif anchor_mode == "rolling_peak":
        if rolling_peak is None:
            raise ValueError("rolling_peak bắt buộc cho rolling_peak anchor")
        starting_peak = max(float(rolling_peak), initial)
    else:
        raise ValueError(f"Drawdown anchor không hỗ trợ: {anchor_mode}")
    full = np.column_stack([np.full(len(prices), starting_peak), prices])
    running_peak = np.maximum.accumulate(full, axis=1)[:, 1:]
    return np.minimum(prices / running_peak - 1.0, 0.0)


def compute_running_maximum_drawdown(drawdown_paths: np.ndarray) -> np.ndarray:
    """Convert negative drawdowns to non-decreasing maximum severity paths."""
    values = np.asarray(drawdown_paths, dtype=float)
    if values.ndim != 2:
        raise ValueError("drawdown_paths phải là ma trận")
    severity = np.maximum(-values, 0.0)
    return np.maximum.accumulate(severity, axis=1)


def first_passage_times(
    severity_paths: np.ndarray,
    thresholds: tuple[float, ...] | list[float] = DEFAULT_THRESHOLDS,
) -> dict[float, np.ndarray]:
    """Return one-based first breach step; unbreached paths remain NaN/right-censored."""
    severity = np.asarray(severity_paths, dtype=float)
    if severity.ndim != 2:
        raise ValueError("severity_paths phải là ma trận")
    output: dict[float, np.ndarray] = {}
    for threshold in thresholds:
        reached = severity >= float(threshold)
        times = np.full(len(severity), np.nan)
        breached = reached.any(axis=1)
        times[breached] = np.argmax(reached[breached], axis=1) + 1
        output[float(threshold)] = times
    return output


def _threshold_label(threshold: float) -> str:
    return str(int(round(100 * float(threshold))))


def _level_label(level: float) -> str:
    return "975" if np.isclose(level, 0.975) else str(int(round(100 * float(level))))


def drawdown_term_structure(
    severity_paths: np.ndarray,
    thresholds: tuple[float, ...] | list[float] = DEFAULT_THRESHOLDS,
) -> pd.DataFrame:
    """Pointwise severity fan plus cumulative first-passage probabilities."""
    severity = np.asarray(severity_paths, dtype=float)
    if severity.ndim != 2 or not np.isfinite(severity).all() or np.any(severity < -1e-12):
        raise ValueError("drawdown severity phải là ma trận hữu hạn không âm")
    running = np.maximum.accumulate(severity, axis=1)
    quantile_levels = (0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975)
    quantiles = {level: np.quantile(severity, level, axis=0) for level in quantile_levels}
    running_quantiles = {level: np.quantile(running, level, axis=0) for level in quantile_levels}
    table = pd.DataFrame(
        {
            "step": np.arange(1, severity.shape[1] + 1),
            "drawdown_mean": severity.mean(axis=0),
            "drawdown_median": quantiles[0.50],
            "drawdown_q025": quantiles[0.025],
            "drawdown_q05": quantiles[0.05],
            "drawdown_q10": quantiles[0.10],
            "drawdown_q25": quantiles[0.25],
            "drawdown_q75": quantiles[0.75],
            "drawdown_q90": quantiles[0.90],
            "drawdown_q95": quantiles[0.95],
            "drawdown_q975": quantiles[0.975],
            "drawdown_lower_50": quantiles[0.25],
            "drawdown_upper_50": quantiles[0.75],
            "drawdown_lower_80": quantiles[0.10],
            "drawdown_upper_80": quantiles[0.90],
            "drawdown_lower_90": quantiles[0.05],
            "drawdown_upper_90": quantiles[0.95],
            "drawdown_lower_95": quantiles[0.025],
            "drawdown_upper_95": quantiles[0.975],
            "running_mdd_mean": running.mean(axis=0),
            "running_mdd_median": running_quantiles[0.50],
            "running_mdd_q90": running_quantiles[0.90],
            "running_mdd_q95": running_quantiles[0.95],
            "running_mdd_q99": np.quantile(running, 0.99, axis=0),
        }
    )
    for threshold in thresholds:
        table[f"probability_breach_{_threshold_label(threshold)}"] = np.mean(running >= float(threshold), axis=0)
    return table


def maximum_drawdown_at_risk(
    maximum_severity: np.ndarray,
    levels: tuple[float, ...] | list[float] = DEFAULT_LEVELS,
) -> dict[str, float]:
    values = np.asarray(maximum_severity, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0 or np.any(values < 0):
        raise ValueError("maximum severity phải có giá trị hữu hạn không âm")
    return {f"mdar_{_level_label(level)}": float(np.quantile(values, level)) for level in levels}


def conditional_expected_drawdown(
    maximum_severity: np.ndarray,
    levels: tuple[float, ...] | list[float] = (0.90, 0.95, 0.99),
) -> dict[str, float]:
    values = np.asarray(maximum_severity, dtype=float)
    values = values[np.isfinite(values)]
    output: dict[str, float] = {}
    for level in levels:
        cutoff = float(np.quantile(values, level))
        tail = values[values >= cutoff]
        output[f"ced_{_level_label(level)}"] = float(tail.mean())
    return output


def recovery_statistics(price_paths: np.ndarray, historical_peak: float) -> dict[str, object]:
    """Recovery times preserve NaN for paths censored at the forecast horizon."""
    prices = _as_path_matrix(price_paths)
    peak = float(historical_peak)
    recovered = prices >= peak
    times = np.full(len(prices), np.nan)
    observed = recovered.any(axis=1)
    times[observed] = np.argmax(recovered[observed], axis=1) + 1
    probability_by_step = np.mean(np.maximum.accumulate(recovered, axis=1), axis=0)
    conditional_mean = float(np.nanmean(times)) if observed.any() else np.nan
    median = float(np.nanmedian(times)) if observed.any() else np.nan
    return {
        "probability_recovery_by_step": probability_by_step,
        "probability_recovery_by_horizon": float(observed.mean()),
        "median_recovery_time": median,
        "conditional_expected_recovery_time": conditional_mean,
        "probability_new_high": float(np.mean(np.max(prices, axis=1) > peak)),
        "recovery_times": times,
        "right_censored": ~observed,
        "unrecovered_survival_by_step": 1.0 - probability_by_step,
    }


def drawdown_duration_statistics(drawdown_paths: np.ndarray) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compute trough timing and underwater duration without inventing recovery times."""
    drawdown = np.asarray(drawdown_paths, dtype=float)
    if drawdown.ndim != 2:
        raise ValueError("drawdown_paths phải là ma trận")
    rows: list[dict[str, float]] = []
    for path in drawdown:
        underwater = path < -1e-12
        durations: list[int] = []
        current = 0
        for flag in underwater:
            current = current + 1 if flag else 0
            durations.append(current)
        trough = int(np.argmin(path)) + 1
        recovered_after = np.flatnonzero(~underwater[trough:])
        recovery = float(trough + recovered_after[0] + 1) if len(recovered_after) else np.nan
        rows.append(
            {
                "time_to_maximum_drawdown": float(trough),
                "maximum_underwater_duration": float(max(durations, default=0)),
                "ending_underwater_duration": float(durations[-1]),
                "time_of_trough": float(trough),
                "time_of_recovery": recovery,
            }
        )
    frame = pd.DataFrame(rows)
    summary: dict[str, float] = {}
    for column in frame:
        values = frame[column].dropna()
        for label, function in {
            "mean": np.mean,
            "median": np.median,
            "q80": lambda x: np.quantile(x, 0.80),
            "q90": lambda x: np.quantile(x, 0.90),
            "q95": lambda x: np.quantile(x, 0.95),
        }.items():
            summary[f"{column}_{label}"] = float(function(values)) if len(values) else np.nan
    return frame, summary


def _wilson_interval(estimate: float, n: int, confidence: float) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    denominator = 1 + z * z / n
    center = (estimate + z * z / (2 * n)) / denominator
    radius = z * np.sqrt(estimate * (1 - estimate) / n + z * z / (4 * n * n)) / denominator
    return float(max(0.0, center - radius)), float(min(1.0, center + radius))


def drawdown_probability_intervals(
    events: dict[str, np.ndarray],
    weights: np.ndarray | None = None,
    seed: int = 55,
    bootstrap_replications: int = 500,
    batches: int = 20,
) -> pd.DataFrame:
    """Wilson intervals for ordinary MC; weighted batch bootstrap for IS paths."""
    rows: list[dict[str, float | int | str]] = []
    rng = np.random.default_rng(seed)
    for name, indicator_values in events.items():
        indicator = np.asarray(indicator_values, dtype=float)
        if indicator.ndim != 1:
            raise ValueError("Mỗi event indicator phải là vector")
        n = len(indicator)
        if weights is None:
            estimate = float(indicator.mean())
            mcse = float(np.sqrt(estimate * (1 - estimate) / max(n, 1)))
            ci90 = _wilson_interval(estimate, n, 0.90)
            ci95 = _wilson_interval(estimate, n, 0.95)
            ess = float(n)
            maximum_weight = 1.0 / max(n, 1)
        else:
            normalized = np.asarray(weights, dtype=float)
            if normalized.shape != indicator.shape or np.any(normalized < 0):
                raise ValueError("weights phải không âm và cùng shape với event")
            normalized = normalized / normalized.sum()
            estimate = float(np.sum(normalized * indicator))
            mcse = float(np.sqrt(np.sum(np.square(normalized) * np.square(indicator - estimate))))
            ess = float(1.0 / np.sum(np.square(normalized)))
            maximum_weight = float(normalized.max())
            batch_ids = np.array_split(np.arange(n), min(int(batches), n))
            batch_estimates = np.array(
                [
                    np.sum(normalized[index] * indicator[index]) / max(np.sum(normalized[index]), 1e-15)
                    for index in batch_ids
                ]
            )
            replicate = np.array(
                [
                    np.mean(rng.choice(batch_estimates, len(batch_estimates), replace=True))
                    for _ in range(bootstrap_replications)
                ]
            )
            ci90 = tuple(np.quantile(replicate, [0.05, 0.95]))
            ci95 = tuple(np.quantile(replicate, [0.025, 0.975]))
        rows.append(
            {
                "statistic": name,
                "estimate": estimate,
                "mcse": mcse,
                "relative_mcse": mcse / max(estimate, 1e-15),
                "ci_90_lower": float(ci90[0]),
                "ci_90_upper": float(ci90[1]),
                "ci_95_lower": float(ci95[0]),
                "ci_95_upper": float(ci95[1]),
                "effective_sample_size": ess,
                "maximum_normalized_weight": maximum_weight,
                "number_of_tail_events": int(indicator.sum()),
                "paths": n,
                "weighted": weights is not None,
            }
        )
    return pd.DataFrame(rows)


def eligible_drawdown_scores(target_end: np.ndarray, forecast_origin: int | float) -> np.ndarray:
    """Indices whose drawdown target has matured by the forecast origin."""
    values = np.asarray(target_end)
    return np.flatnonzero(values <= forecast_origin)


def finite_sample_upper_correction(scores: np.ndarray, alpha: float) -> float:
    values = np.sort(np.asarray(scores, dtype=float)[np.isfinite(scores)])
    if len(values) == 0:
        return 0.0
    k = min(int(np.ceil((len(values) + 1) * (1 - float(alpha)))), len(values))
    return float(values[max(k - 1, 0)])


def direct_drawdown_conformal_upper(
    calibration_actual_severity: np.ndarray,
    calibration_mc_upper: np.ndarray,
    evaluation_mc_upper: np.ndarray,
    alpha: float,
    calibration_target_end: np.ndarray | None = None,
    evaluation_origins: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calibrate a one-sided MC drawdown upper bound using only matured scores."""
    actual = np.asarray(calibration_actual_severity, dtype=float)
    predicted = np.asarray(calibration_mc_upper, dtype=float)
    evaluation = np.asarray(evaluation_mc_upper, dtype=float)
    scores = actual - predicted
    corrections = np.empty(len(evaluation), dtype=float)
    if calibration_target_end is None or evaluation_origins is None:
        correction = finite_sample_upper_correction(scores, alpha)
        corrections.fill(correction)
    else:
        target_end = np.asarray(calibration_target_end)
        origins = np.asarray(evaluation_origins)
        for position, origin in enumerate(origins):
            eligible = eligible_drawdown_scores(target_end, origin)
            corrections[position] = finite_sample_upper_correction(scores[eligible], alpha)
    return np.maximum(evaluation + corrections, 0.0), corrections


def pointwise_drawdown_band(severity_paths: np.ndarray, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    severity = np.asarray(severity_paths, dtype=float)
    alpha = 1 - float(level)
    return np.quantile(severity, alpha / 2, axis=0), np.quantile(severity, 1 - alpha / 2, axis=0)


def simultaneous_drawdown_band(
    calibration_actual_paths: np.ndarray,
    calibration_center_paths: np.ndarray,
    calibration_scale_paths: np.ndarray,
    forecast_center_path: np.ndarray,
    forecast_scale_path: np.ndarray,
    level: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Conformal band targeting coverage of the complete drawdown trajectory."""
    actual = np.asarray(calibration_actual_paths, dtype=float)
    center = np.asarray(calibration_center_paths, dtype=float)
    scale = np.maximum(np.asarray(calibration_scale_paths, dtype=float), 1e-8)
    scores = np.max(np.abs(actual - center) / scale, axis=1)
    multiplier = finite_sample_upper_correction(scores, 1 - float(level))
    forecast_center = np.asarray(forecast_center_path, dtype=float)
    forecast_scale = np.asarray(forecast_scale_path, dtype=float)
    lower = np.maximum(forecast_center - multiplier * forecast_scale, 0.0)
    upper = forecast_center + multiplier * forecast_scale
    return lower, upper, multiplier


def build_drawdown_forecast(
    price_paths: np.ndarray,
    initial_price: float,
    anchor_mode: str,
    historical_peak: float | None = None,
    rolling_peak: float | None = None,
    thresholds: tuple[float, ...] | list[float] = DEFAULT_THRESHOLDS,
) -> DrawdownForecastResult:
    drawdown = compute_drawdown_paths(
        price_paths,
        initial_price,
        anchor_mode,
        historical_peak=historical_peak,
        rolling_peak=rolling_peak,
    )
    severity = np.maximum(-drawdown, 0.0)
    running = compute_running_maximum_drawdown(drawdown)
    term = drawdown_term_structure(severity, thresholds)
    maximum = running[:, -1]
    passage = first_passage_times(running, thresholds)
    summary: dict[str, object] = {
        "anchor_mode": anchor_mode,
        "expected_ending_drawdown_severity": float(severity[:, -1].mean()),
        "median_ending_drawdown_severity": float(np.median(severity[:, -1])),
        "expected_maximum_drawdown_severity": float(maximum.mean()),
        "median_maximum_drawdown_severity": float(np.median(maximum)),
        **maximum_drawdown_at_risk(maximum),
        **conditional_expected_drawdown(maximum),
        "first_passage": {},
    }
    for threshold, times in passage.items():
        breached = np.isfinite(times)
        summary["first_passage"][str(threshold)] = {
            "probability_breach_by_horizon": float(breached.mean()),
            "probability_no_breach_by_horizon": float(1 - breached.mean()),
            "median_time_to_breach": float(np.nanmedian(times)) if breached.any() else np.nan,
            "conditional_expected_time_to_breach": float(np.nanmean(times)) if breached.any() else np.nan,
            **{f"probability_breach_within_{step}": float(np.mean(times <= step)) for step in (5, 10, 20)},
        }
    return DrawdownForecastResult(drawdown, severity, running, term, summary)


def realized_drawdown_severity(
    close: np.ndarray,
    horizon: int,
    anchor_mode: str = "origin_peak",
    rolling_peak_window: int = 252,
) -> np.ndarray:
    """Realized forward maximum severity using the same anchor as the forecast."""
    prices = np.asarray(close, dtype=float)
    result = np.full(len(prices), np.nan)
    for origin in range(len(prices) - int(horizon)):
        future = prices[origin + 1 : origin + int(horizon) + 1][None, :]
        if anchor_mode == "historical_peak":
            peak = float(np.max(prices[: origin + 1]))
        elif anchor_mode == "rolling_peak":
            peak = float(np.max(prices[max(0, origin - rolling_peak_window + 1) : origin + 1]))
        else:
            peak = float(prices[origin])
        drawdown = compute_drawdown_paths(
            future,
            float(prices[origin]),
            anchor_mode,
            historical_peak=peak,
            rolling_peak=peak,
        )
        result[origin] = float(np.max(-drawdown))
    return result


def _standardized_shocks(
    method: str,
    paths: int,
    horizon: int,
    residuals: np.ndarray,
    degrees_of_freedom: float,
    seed: int,
    block_length: int = 10,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    residual_pool = np.asarray(residuals, dtype=float)
    residual_pool = residual_pool[np.isfinite(residual_pool)]
    lower, upper = np.quantile(residual_pool, [0.001, 0.999])
    residual_pool = np.clip(residual_pool, max(lower, -12.0), min(upper, 12.0))
    student = rng.standard_t(float(degrees_of_freedom), size=(int(paths), int(horizon)))
    student *= np.sqrt((float(degrees_of_freedom) - 2) / float(degrees_of_freedom))
    restart = rng.random((int(paths), int(horizon))) < 1 / max(int(block_length), 1)
    indices = np.empty((int(paths), int(horizon)), dtype=int)
    indices[:, 0] = rng.integers(0, len(residual_pool), int(paths))
    for step in range(1, int(horizon)):
        continued = (indices[:, step - 1] + 1) % len(residual_pool)
        indices[:, step] = np.where(restart[:, step], rng.integers(0, len(residual_pool), int(paths)), continued)
    empirical = residual_pool[indices]
    if method == "egarch_student_t":
        return student
    if method == "residual_block_bootstrap":
        return empirical
    if method in {"hybrid_monte_carlo", "hybrid_importance_sampling", "hybrid_direct_conformal"}:
        mixture = rng.random((int(paths), int(horizon))) < 0.35
        return np.where(mixture, student, empirical)
    raise ValueError(f"Drawdown backtest method không hỗ trợ: {method}")


def _origin_distributions(
    centers: np.ndarray,
    daily_volatility: np.ndarray,
    shocks: np.ndarray,
    current_drawdown: np.ndarray | None = None,
    anchor_mode: str = "origin_peak",
) -> np.ndarray:
    """Simulate maximum severity at each origin with common random numbers."""
    centers = np.asarray(centers, dtype=float)
    volatility = np.asarray(daily_volatility, dtype=float)
    paths, horizon = shocks.shape
    output = np.empty((len(centers), paths), dtype=np.float32)
    for position, (center, sigma) in enumerate(zip(centers, volatility, strict=True)):
        returns = float(center) / horizon + max(float(sigma), 1e-6) * shocks
        relative_price = np.exp(np.clip(np.cumsum(returns, axis=1), -5, 5))
        if anchor_mode == "historical_peak":
            severity = max(-float(np.asarray(current_drawdown)[position]), 0.0)
            starting_peak = 1.0 / max(1.0 - severity, 1e-8)
            drawdown = compute_drawdown_paths(relative_price, 1.0, "historical_peak", historical_peak=starting_peak)
        else:
            drawdown = compute_drawdown_paths(relative_price, 1.0, "origin_peak")
        output[position] = np.max(-drawdown, axis=1)
    return output


def _pinball(actual: np.ndarray, predicted: np.ndarray, level: float) -> float:
    error = np.asarray(actual) - np.asarray(predicted)
    return float(np.mean(np.maximum(float(level) * error, (float(level) - 1) * error)))


def _crps_from_samples(actual: np.ndarray, samples: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed)
    count = min(samples.shape[1], 160)
    subset = samples[:, rng.choice(samples.shape[1], count, replace=False)].astype(float)
    first = np.mean(np.abs(subset - np.asarray(actual)[:, None]), axis=1)
    order = np.sort(subset, axis=1)
    coefficient = 2 * np.arange(1, count + 1) - count - 1
    second = np.sum(order * coefficient[None, :], axis=1) / (count * count)
    return float(np.mean(first - second))


def _stratum_keys(method: str, drawdown_bin: np.ndarray, volatility_bin: np.ndarray, regime: np.ndarray) -> np.ndarray:
    if method == "current_drawdown_stratified":
        return np.asarray([f"d{value}" for value in drawdown_bin])
    if method == "volatility_stratified":
        return np.asarray([f"v{value}" for value in volatility_bin])
    if method == "regime_stratified":
        return np.asarray([f"r{value}" for value in regime])
    if method == "current_drawdown_regime":
        return np.asarray([f"d{d}_r{r}" for d, r in zip(drawdown_bin, regime, strict=True)])
    if method == "volatility_regime":
        return np.asarray([f"v{v}_r{r}" for v, r in zip(volatility_bin, regime, strict=True)])
    return np.full(len(regime), "global", dtype=object)


def _group_corrections(
    scores: np.ndarray,
    calibration_keys: np.ndarray,
    evaluation_keys: np.ndarray,
    alpha: float,
    minimum_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    global_correction = finite_sample_upper_correction(scores, alpha)
    corrections = np.full(len(evaluation_keys), global_correction)
    used = np.full(len(evaluation_keys), "global", dtype=object)
    for key in np.unique(evaluation_keys):
        calibration = np.flatnonzero(calibration_keys == key)
        evaluation = np.flatnonzero(evaluation_keys == key)
        if len(calibration) >= int(minimum_size):
            corrections[evaluation] = finite_sample_upper_correction(scores[calibration], alpha)
            used[evaluation] = key
    return corrections, used


def drawdown_backtest(
    actual_validation: np.ndarray,
    actual_test: np.ndarray,
    center_validation: np.ndarray,
    center_test: np.ndarray,
    daily_volatility_validation: np.ndarray,
    daily_volatility_test: np.ndarray,
    current_drawdown_validation: np.ndarray,
    current_drawdown_test: np.ndarray,
    regime_validation: np.ndarray,
    regime_test: np.ndarray,
    training_severity: np.ndarray,
    residuals: np.ndarray,
    degrees_of_freedom: float,
    horizon: int,
    paths: int,
    minimum_stratum_size: int,
    anchor_mode: str = "origin_peak",
    thresholds: tuple[float, ...] | list[float] = (0.03, 0.05, 0.07, 0.10),
    seed: int = 55,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Path-based OOS drawdown backtest with validation-only conformal selection."""
    actual_validation = np.asarray(actual_validation, dtype=float)
    actual_test = np.asarray(actual_test, dtype=float)
    calibration_stop = max(int(len(actual_validation) * 0.60), 1)
    drawdown_edges = np.quantile(
        np.maximum(-np.asarray(current_drawdown_validation[:calibration_stop]), 0), [1 / 3, 2 / 3]
    )
    volatility_edges = np.quantile(np.asarray(daily_volatility_validation[:calibration_stop]), [1 / 3, 2 / 3])
    drawdown_bins_validation = np.digitize(np.maximum(-np.asarray(current_drawdown_validation), 0), drawdown_edges)
    drawdown_bins_test = np.digitize(np.maximum(-np.asarray(current_drawdown_test), 0), drawdown_edges)
    volatility_bins_validation = np.digitize(daily_volatility_validation, volatility_edges)
    volatility_bins_test = np.digitize(daily_volatility_test, volatility_edges)
    methods = ["egarch_student_t", "residual_block_bootstrap", "hybrid_monte_carlo"]
    validation_distributions: dict[str, np.ndarray] = {}
    test_distributions: dict[str, np.ndarray] = {}
    for offset, method in enumerate(methods):
        shocks = _standardized_shocks(
            method, paths, horizon, residuals, degrees_of_freedom, seed + 100 * horizon + offset
        )
        validation_distributions[method] = _origin_distributions(
            center_validation, daily_volatility_validation, shocks, current_drawdown_validation, anchor_mode
        )
        test_distributions[method] = _origin_distributions(
            center_test, daily_volatility_test, shocks, current_drawdown_test, anchor_mode
        )
    training = np.asarray(training_severity, dtype=float)
    training = training[np.isfinite(training)]
    historical_validation = np.tile(training, (len(actual_validation), 1)).astype(np.float32)
    historical_test = np.tile(training, (len(actual_test), 1)).astype(np.float32)
    validation_distributions["historical_empirical"] = historical_validation
    test_distributions["historical_empirical"] = historical_test
    validation_distributions["hybrid_importance_sampling"] = validation_distributions["hybrid_monte_carlo"]
    test_distributions["hybrid_importance_sampling"] = test_distributions["hybrid_monte_carlo"]
    candidate_methods = [
        "global",
        "current_drawdown_stratified",
        "volatility_stratified",
        "regime_stratified",
        "current_drawdown_regime",
        "volatility_regime",
    ]
    selection_rows: list[dict[str, float | int | str | bool]] = []
    selected_by_level: dict[float, str] = {}
    hybrid_validation = validation_distributions["hybrid_monte_carlo"]
    hybrid_test = test_distributions["hybrid_monte_carlo"]
    calibrated_test_quantiles: dict[float, np.ndarray] = {}
    for level in (0.80, 0.90, 0.95, 0.99):
        alpha = 1 - level
        validation_upper = np.quantile(hybrid_validation, level, axis=1)
        calibration_scores = actual_validation[:calibration_stop] - validation_upper[:calibration_stop]
        candidate_objectives: list[float] = []
        for candidate in candidate_methods:
            calibration_keys = _stratum_keys(
                candidate,
                drawdown_bins_validation[:calibration_stop],
                volatility_bins_validation[:calibration_stop],
                regime_validation[:calibration_stop],
            )
            selection_keys = _stratum_keys(
                candidate,
                drawdown_bins_validation[calibration_stop:],
                volatility_bins_validation[calibration_stop:],
                regime_validation[calibration_stop:],
            )
            corrections, used = _group_corrections(
                calibration_scores,
                calibration_keys,
                selection_keys,
                alpha,
                minimum_stratum_size,
            )
            bound = np.maximum(validation_upper[calibration_stop:] + corrections, 0)
            selection_actual = actual_validation[calibration_stop:]
            coverage = float(np.mean(selection_actual <= bound))
            pinball = _pinball(selection_actual, bound, level)
            objective = abs(coverage - level) + pinball + 0.01 * float(bound.mean())
            candidate_objectives.append(objective)
            selection_rows.append(
                {
                    "horizon": horizon,
                    "anchor_mode": anchor_mode,
                    "level": level,
                    "method": candidate,
                    "coverage": coverage,
                    "coverage_error": coverage - level,
                    "pinball_loss": pinball,
                    "average_upper_bound": float(bound.mean()),
                    "objective": objective,
                    "fallback_rate": float(np.mean(used == "global")) if candidate != "global" else 0.0,
                    "selected": False,
                }
            )
        selected = candidate_methods[int(np.argmin(candidate_objectives))]
        selected_by_level[level] = selected
        selection_rows[-len(candidate_methods) + int(np.argmin(candidate_objectives))]["selected"] = True
        full_scores = actual_validation - validation_upper
        calibration_keys = _stratum_keys(
            selected, drawdown_bins_validation, volatility_bins_validation, regime_validation
        )
        test_keys = _stratum_keys(selected, drawdown_bins_test, volatility_bins_test, regime_test)
        correction, _ = _group_corrections(full_scores, calibration_keys, test_keys, alpha, minimum_stratum_size)
        calibrated_test_quantiles[level] = np.maximum(np.quantile(hybrid_test, level, axis=1) + correction, 0)
    metrics_rows: list[dict[str, float | int | str]] = []
    probability_rows: list[dict[str, float | int | str]] = []
    interval_rows: list[dict[str, float | int | str | bool]] = []
    detail = pd.DataFrame(
        {
            "actual_severity": actual_test,
            "horizon": horizon,
            "anchor_mode": anchor_mode,
            "regime": regime_test,
            "volatility_bin": volatility_bins_test,
            "current_drawdown_bin": drawdown_bins_test,
            "current_drawdown_severity": np.maximum(-np.asarray(current_drawdown_test), 0),
        }
    )
    for method, distribution in test_distributions.items():
        median = np.median(distribution, axis=1)
        detail[f"median_{method}"] = median
        row: dict[str, float | int | str] = {
            "horizon": horizon,
            "anchor_mode": anchor_mode,
            "method": method,
            "mae_median_drawdown": float(np.mean(np.abs(actual_test - median))),
            "rmse_median_drawdown": float(np.sqrt(np.mean(np.square(actual_test - median)))),
            "crps": _crps_from_samples(actual_test, distribution, seed + horizon),
        }
        for level in (0.80, 0.90, 0.95, 0.99):
            upper = np.quantile(distribution, level, axis=1)
            row[f"pinball_q{int(level * 100)}"] = _pinball(actual_test, upper, level)
            row[f"mdar_coverage_{int(level * 100)}"] = float(np.mean(actual_test <= upper))
            tail = np.where(distribution >= upper[:, None], distribution, np.nan)
            predicted_ced = np.nanmean(tail, axis=1)
            realized_cutoff = np.quantile(actual_test, level)
            realized_ced = float(actual_test[actual_test >= realized_cutoff].mean())
            row[f"ced_error_{int(level * 100)}"] = float(np.mean(predicted_ced) - realized_ced)
        metrics_rows.append(row)
        for threshold in thresholds:
            probability = np.mean(distribution >= float(threshold), axis=1)
            event = (actual_test >= float(threshold)).astype(float)
            clipped = np.clip(probability, 1e-6, 1 - 1e-6)
            probability_rows.append(
                {
                    "horizon": horizon,
                    "anchor_mode": anchor_mode,
                    "method": method,
                    "threshold": float(threshold),
                    "brier_score": float(np.mean(np.square(probability - event))),
                    "log_loss": float(-np.mean(event * np.log(clipped) + (1 - event) * np.log(1 - clipped))),
                    "mean_probability": float(probability.mean()),
                    "event_rate": float(event.mean()),
                    "reliability_gap": float(probability.mean() - event.mean()),
                }
            )
    conformal_row: dict[str, float | int | str] = {
        "horizon": horizon,
        "anchor_mode": anchor_mode,
        "method": "hybrid_direct_drawdown_conformal",
        "mae_median_drawdown": metrics_rows[-1]["mae_median_drawdown"],
        "rmse_median_drawdown": metrics_rows[-1]["rmse_median_drawdown"],
        "crps": metrics_rows[-1]["crps"],
    }
    for level, upper in calibrated_test_quantiles.items():
        exceedance = (actual_test > upper).astype(float)
        lag_correlation = (
            float(np.corrcoef(exceedance[:-1], exceedance[1:])[0, 1])
            if len(exceedance) > 2 and np.std(exceedance[:-1]) > 0 and np.std(exceedance[1:]) > 0
            else np.nan
        )
        conformal_row[f"pinball_q{int(level * 100)}"] = _pinball(actual_test, upper, level)
        conformal_row[f"mdar_coverage_{int(level * 100)}"] = float(np.mean(actual_test <= upper))
        conformal_row[f"ced_error_{int(level * 100)}"] = np.nan
        interval_rows.append(
            {
                "horizon": horizon,
                "anchor_mode": anchor_mode,
                "level": level,
                "method": "hybrid_direct_drawdown_conformal",
                "selected_conformal_method": selected_by_level[level],
                "upper_bound_coverage": float(np.mean(actual_test <= upper)),
                "coverage_error": float(np.mean(actual_test <= upper) - level),
                "average_upper_bound": float(upper.mean()),
                "pinball_loss": _pinball(actual_test, upper, level),
                "exceedance_lag1_correlation": lag_correlation,
            }
        )
        detail[f"conformal_upper_{int(level * 100)}"] = upper
        detail[f"conformal_covered_{int(level * 100)}"] = 1.0 - exceedance
    metrics_rows.append(conformal_row)
    return (
        pd.DataFrame(metrics_rows),
        pd.DataFrame(interval_rows),
        pd.DataFrame(probability_rows),
        pd.DataFrame(selection_rows),
        detail,
    )
