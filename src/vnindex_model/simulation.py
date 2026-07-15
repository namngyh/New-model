"""Student-t, bootstrap, and hybrid regime-aware scenario generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .targets import CLASS_NAMES
from .validation import assert_quantile_monotonicity
from .volatility import standardized_student_t


@dataclass
class SimulationResult:
    price_paths: np.ndarray
    return_paths: np.ndarray
    regime_paths: np.ndarray
    forecast: pd.DataFrame
    summary: dict[str, object]


@dataclass
class StratifiedSimulationResult:
    estimates: dict[str, float]
    allocation: pd.DataFrame
    terminal_returns: np.ndarray
    maximum_drawdowns: np.ndarray
    path_weights: np.ndarray


def maximum_drawdown(paths: np.ndarray) -> np.ndarray:
    running = np.maximum.accumulate(paths, axis=1)
    return np.min(paths / running - 1, axis=1)


def _class_to_state_probabilities(class_probability: np.ndarray, economic_labels: list[str]) -> np.ndarray:
    state = np.array(
        [class_probability[CLASS_NAMES.index(label)] if label in CLASS_NAMES else 0.0 for label in economic_labels]
    )
    return state / max(state.sum(), 1e-12)


def simulate_paths(
    last_close: float,
    horizon: int,
    paths: int,
    daily_drift: float,
    daily_volatility: float,
    degrees_of_freedom: float,
    residuals: np.ndarray,
    historical_regime_probabilities: np.ndarray,
    transition_matrix: np.ndarray,
    current_regime_probability: np.ndarray,
    rf_class_probability: np.ndarray,
    economic_labels: list[str],
    method: str = "hybrid",
    student_weight: float = 0.35,
    block_length: int = 10,
    seed: int = 55,
) -> SimulationResult:
    rng = np.random.default_rng(seed)
    residuals = np.asarray(residuals, dtype=float)
    finite_residuals = residuals[np.isfinite(residuals)]
    lower_residual, upper_residual = np.quantile(finite_residuals, [0.001, 0.999])
    residuals = np.clip(residuals, max(lower_residual, -12.0), min(upper_residual, 12.0))
    daily_volatility = float(np.clip(daily_volatility, 1e-5, 0.15))
    state_count = transition_matrix.shape[0]
    rf_state_probability = _class_to_state_probabilities(rf_class_probability, economic_labels)
    regime_paths = np.empty((paths, horizon), dtype=np.int16)
    returns = np.empty((paths, horizon), dtype=float)
    current = rng.choice(state_count, size=paths, p=current_regime_probability / current_regime_probability.sum())
    fallback_count = 0
    block_positions = np.full(paths, -1, dtype=int)
    previous_state = np.full(paths, -1, dtype=int)
    state_scale = np.ones(state_count)
    for state in range(state_count):
        weights = historical_regime_probabilities[:, state]
        denominator = max(weights.sum(), 1e-12)
        state_scale[state] = np.sqrt(np.sum(weights * np.square(residuals)) / denominator)
    state_scale = np.clip(
        state_scale / max(np.average(state_scale, weights=current_regime_probability), 1e-8), 0.65, 1.8
    )
    for step in range(horizon):
        eta = 0.15 + 0.55 * (step + 1) / horizon
        next_state = np.empty(paths, dtype=int)
        for origin_state in range(state_count):
            mask = current == origin_state
            if not mask.any():
                continue
            adjusted = np.power(np.maximum(transition_matrix[origin_state], 1e-12), 1 - eta) * np.power(
                np.maximum(rf_state_probability, 1e-12), eta
            )
            adjusted /= adjusted.sum()
            next_state[mask] = rng.choice(state_count, size=int(mask.sum()), p=adjusted)
        regime_paths[:, step] = next_state
        shocks = np.empty(paths)
        for state in range(state_count):
            mask = next_state == state
            count = int(mask.sum())
            if not count:
                continue
            weights = historical_regime_probabilities[:, state].astype(float)
            valid_pool = np.isfinite(residuals) & np.isfinite(weights)
            effective = weights[valid_pool].sum() ** 2 / max(np.square(weights[valid_pool]).sum(), 1e-12)
            fallback = effective < 30 or weights[valid_pool].sum() <= 0
            fallback_count += int(fallback)
            member_indices = np.flatnonzero(mask)
            restart = (
                (previous_state[member_indices] != state)
                | (block_positions[member_indices] < 0)
                | (rng.random(count) < 1 / max(block_length, 1))
            )
            available = np.flatnonzero(valid_pool)
            if fallback:
                starts = rng.choice(available, size=int(restart.sum()), replace=True)
            else:
                state_weights = weights[available] / weights[available].sum()
                starts = rng.choice(available, size=int(restart.sum()), replace=True, p=state_weights)
            block_positions[member_indices[restart]] = starts
            continuing = member_indices[~restart]
            block_positions[continuing] = (block_positions[continuing] + 1) % len(residuals)
            empirical = residuals[block_positions[member_indices]]
            student = standardized_student_t(rng, degrees_of_freedom, count)
            if method == "student_t":
                selected = student
            elif method in {"bootstrap", "regime_block_bootstrap"}:
                selected = empirical
            elif method == "hybrid":
                mixture = rng.random(count) < student_weight
                selected = np.where(mixture, student, empirical)
            else:
                raise ValueError(f"Simulation method không hỗ trợ: {method}")
            shocks[mask] = selected * state_scale[state]
        returns[:, step] = daily_drift + daily_volatility * shocks
        previous_state = current
        current = next_state
    cumulative = np.clip(np.cumsum(returns, axis=1), -5.0, 5.0)
    price_paths = last_close * np.exp(cumulative)
    full_prices = np.column_stack([np.full(paths, last_close), price_paths])
    max_drawdowns = maximum_drawdown(full_prices)
    terminal_returns = cumulative[:, -1]
    quantiles = {
        level: np.quantile(price_paths, level, axis=0)
        for level in [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975]
    }
    forecast = pd.DataFrame(
        {
            "step": np.arange(1, horizon + 1),
            "mean": price_paths.mean(axis=0),
            "median": quantiles[0.50],
            "lower_50": quantiles[0.25],
            "upper_50": quantiles[0.75],
            "lower_80": quantiles[0.10],
            "upper_80": quantiles[0.90],
            "lower_90": quantiles[0.05],
            "upper_90": quantiles[0.95],
            "lower_95": quantiles[0.025],
            "upper_95": quantiles[0.975],
            "expected_volatility": daily_volatility * np.sqrt(np.arange(1, horizon + 1)),
        }
    )
    daily_regime = np.stack([(regime_paths == state).mean(axis=0) for state in range(state_count)], axis=1)
    for class_name in CLASS_NAMES:
        matching = [state for state, label in enumerate(economic_labels) if label == class_name]
        forecast[f"probability_{class_name.lower()}"] = daily_regime[:, matching].sum(axis=1) if matching else 0.0
    assert_quantile_monotonicity(forecast)
    var95 = float(np.quantile(terminal_returns, 0.05))
    var99 = float(np.quantile(terminal_returns, 0.01))
    summary = {
        "expected_terminal_close": float(price_paths[:, -1].mean()),
        "median_terminal_close": float(np.median(price_paths[:, -1])),
        "expected_return": float(np.mean(np.exp(terminal_returns) - 1)),
        "median_return": float(np.median(np.exp(terminal_returns) - 1)),
        "probability_positive_return": float(np.mean(terminal_returns > 0)),
        "probability_negative_return": float(np.mean(terminal_returns < 0)),
        "var_95": var95,
        "var_99": var99,
        "expected_shortfall_95": float(terminal_returns[terminal_returns <= var95].mean()),
        "expected_shortfall_99": float(terminal_returns[terminal_returns <= var99].mean()),
        "expected_maximum_drawdown": float(max_drawdowns.mean()),
        "median_maximum_drawdown": float(np.median(max_drawdowns)),
        "maximum_drawdown_quantiles": {
            str(level): float(np.quantile(max_drawdowns, level)) for level in [0.05, 0.5, 0.95]
        },
        "drawdown_probabilities": {
            str(threshold): float(np.mean(max_drawdowns <= -threshold)) for threshold in [0.03, 0.05, 0.07, 0.10]
        },
        "simulation_method": method,
        "number_of_paths": paths,
        "random_seed": seed,
        "student_weight": student_weight,
        "regime_pool_fallback_count": fallback_count,
    }
    return SimulationResult(price_paths, returns, regime_paths, forecast, summary)


def allocate_stratified_paths(
    probabilities: np.ndarray,
    total_paths: int,
    allocation: str = "proportional",
    pilot_standard_deviation: np.ndarray | None = None,
    minimum_paths_per_regime: int = 1,
) -> np.ndarray:
    """Allocate paths while retaining true stratum probabilities for estimation."""
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities / probabilities.sum()
    minimum = int(minimum_paths_per_regime)
    if int(total_paths) < len(probabilities) * minimum:
        raise ValueError("total_paths nhỏ hơn tổng minimum paths của các regime")
    if allocation == "equal":
        score = np.ones_like(probabilities)
    elif allocation == "proportional":
        score = probabilities
    elif allocation == "neyman":
        if pilot_standard_deviation is None:
            raise ValueError("Neyman allocation cần pilot_standard_deviation")
        score = probabilities * np.maximum(np.asarray(pilot_standard_deviation, dtype=float), 1e-8)
    else:
        raise ValueError(f"Allocation không hỗ trợ: {allocation}")
    remaining = int(total_paths) - len(probabilities) * minimum
    raw = remaining * score / score.sum()
    counts = np.floor(raw).astype(int) + minimum
    remainder = int(total_paths) - int(counts.sum())
    if remainder:
        order = np.argsort(-(raw - np.floor(raw)))
        counts[order[:remainder]] += 1
    return counts


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    values = np.asarray(values, dtype=float)[order]
    weights = np.asarray(weights, dtype=float)[order]
    cumulative = np.cumsum(weights) / weights.sum()
    return float(values[min(np.searchsorted(cumulative, quantile, side="left"), len(values) - 1)])


def stratified_estimator(stratum_estimates: np.ndarray, true_probabilities: np.ndarray) -> float:
    """Combine stratum means with true probabilities, not raw path proportions."""
    probability = np.asarray(true_probabilities, dtype=float)
    probability = probability / probability.sum()
    return float(np.sum(probability * np.asarray(stratum_estimates, dtype=float)))


def run_stratified_simulation(
    total_paths: int,
    allocation: str,
    pilot_paths: int,
    minimum_paths_per_regime: int,
    simulation_kwargs: dict,
) -> StratifiedSimulationResult:
    """Stratify by initial Filtered-HMM state and combine with true state weights."""
    probabilities = np.asarray(simulation_kwargs["current_regime_probability"], dtype=float)
    probabilities = probabilities / probabilities.sum()
    state_count = len(probabilities)
    pilot_std = np.ones(state_count)
    if allocation == "neyman":
        pilot_each = max(int(pilot_paths) // state_count, 50)
        for state in range(state_count):
            kwargs = dict(simulation_kwargs)
            initial = np.zeros(state_count)
            initial[state] = 1.0
            kwargs["current_regime_probability"] = initial
            pilot_seed = int(kwargs.pop("seed", 55)) + 100 + state
            pilot = simulate_paths(paths=pilot_each, seed=pilot_seed, **kwargs)
            pilot_std[state] = max(float(np.std(pilot.return_paths.sum(axis=1))), 1e-8)
    counts = allocate_stratified_paths(
        probabilities,
        int(total_paths),
        allocation,
        pilot_std,
        int(minimum_paths_per_regime),
    )
    terminals: list[np.ndarray] = []
    drawdowns: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    rows: list[dict[str, float | int | str]] = []
    for state, count in enumerate(counts):
        kwargs = dict(simulation_kwargs)
        initial = np.zeros(state_count)
        initial[state] = 1.0
        kwargs["current_regime_probability"] = initial
        stratum_seed = int(kwargs.pop("seed", 55)) + 1000 + state
        result = simulate_paths(paths=int(count), seed=stratum_seed, **kwargs)
        terminal = result.return_paths.sum(axis=1)
        mdd = maximum_drawdown(np.column_stack([np.ones(int(count)), np.exp(np.cumsum(result.return_paths, axis=1))]))
        terminals.append(terminal)
        drawdowns.append(mdd)
        weights.append(np.full(int(count), probabilities[state] / int(count)))
        rows.append(
            {
                "state": state,
                "economic_label": simulation_kwargs["economic_labels"][state],
                "true_probability": probabilities[state],
                "allocated_paths": int(count),
                "pilot_standard_deviation": pilot_std[state],
                "path_weight": probabilities[state] / int(count),
                "allocation": allocation,
            }
        )
    terminal = np.concatenate(terminals)
    mdd = np.concatenate(drawdowns)
    path_weights = np.concatenate(weights)
    path_weights /= path_weights.sum()
    var95 = weighted_quantile(terminal, path_weights, 0.05)
    estimates = {
        "probability_negative_return": float(np.sum(path_weights * (terminal < 0))),
        "probability_mdd_3": float(np.sum(path_weights * (mdd <= -0.03))),
        "probability_mdd_5": float(np.sum(path_weights * (mdd <= -0.05))),
        "probability_mdd_7": float(np.sum(path_weights * (mdd <= -0.07))),
        "probability_mdd_10": float(np.sum(path_weights * (mdd <= -0.10))),
        "var_95": var95,
        "var_99": weighted_quantile(terminal, path_weights, 0.01),
        "expected_shortfall_95": float(np.sum(path_weights[terminal <= var95] * terminal[terminal <= var95]) / path_weights[terminal <= var95].sum()),
    }
    return StratifiedSimulationResult(estimates, pd.DataFrame(rows), terminal, mdd, path_weights)


def adaptive_simulate_paths(
    last_close: float,
    economic_labels: list[str],
    simulation_kwargs: dict,
    stopping_config: dict,
) -> tuple[SimulationResult, pd.DataFrame]:
    """Generate independent batches until probability MCSE and quantiles stabilize."""
    batch_size = int(stopping_config["batch_size"])
    minimum_paths = int(stopping_config["minimum_paths"])
    maximum_paths = int(stopping_config["maximum_paths"])
    tolerance = float(stopping_config["probability_mcse_tolerance"])
    rare_relative = float(stopping_config["rare_probability_relative_mcse"])
    quantile_tolerance = float(stopping_config["quantile_stability_tolerance"])
    required_stable = int(stopping_config["consecutive_stable_batches"])
    price_batches: list[np.ndarray] = []
    return_batches: list[np.ndarray] = []
    regime_batches: list[np.ndarray] = []
    history: list[dict[str, float | int | bool]] = []
    previous_quantiles: np.ndarray | None = None
    stable_batches = 0
    base_seed = int(simulation_kwargs.get("seed", 55))
    for batch in range(int(np.ceil(maximum_paths / batch_size))):
        count = min(batch_size, maximum_paths - sum(len(values) for values in return_batches))
        kwargs = dict(simulation_kwargs)
        kwargs["seed"] = base_seed + batch
        result = simulate_paths(paths=count, **kwargs)
        price_batches.append(result.price_paths)
        return_batches.append(result.return_paths)
        regime_batches.append(result.regime_paths)
        returns = np.concatenate(return_batches)
        terminal = returns.sum(axis=1)
        mdd = maximum_drawdown(np.column_stack([np.ones(len(returns)), np.exp(np.cumsum(returns, axis=1))]))
        probabilities = np.array(
            [np.mean(terminal < 0), *(np.mean(mdd <= -threshold) for threshold in [0.03, 0.05, 0.07, 0.10])]
        )
        mcse = np.sqrt(probabilities * (1 - probabilities) / len(terminal))
        quantiles = np.array([np.quantile(terminal, 0.05), np.quantile(terminal, 0.01), np.median(terminal)])
        quantile_change = np.inf if previous_quantiles is None else float(np.max(np.abs(quantiles - previous_quantiles)))
        absolute_ok = bool(np.max(mcse[:3]) <= tolerance)
        rare_mask = probabilities[3:] > 0
        rare_ok = bool(rare_mask.all() and np.all(mcse[3:] / probabilities[3:] <= rare_relative))
        stable = len(terminal) >= minimum_paths and absolute_ok and rare_ok and quantile_change <= quantile_tolerance
        stable_batches = stable_batches + 1 if stable else 0
        history.append(
            {
                "batch": batch + 1,
                "paths": len(terminal),
                "probability_negative": probabilities[0],
                "probability_mdd_3": probabilities[1],
                "probability_mdd_5": probabilities[2],
                "probability_mdd_7": probabilities[3],
                "probability_mdd_10": probabilities[4],
                "max_probability_mcse": float(np.max(mcse)),
                "var_95": quantiles[0],
                "var_99": quantiles[1],
                "median_terminal_return": quantiles[2],
                "quantile_max_change": quantile_change,
                "stable": stable,
                "stable_batches": stable_batches,
            }
        )
        previous_quantiles = quantiles
        if stable_batches >= required_stable or len(terminal) >= maximum_paths:
            break
    price_paths = np.concatenate(price_batches)
    return_paths = np.concatenate(return_batches)
    regime_paths = np.concatenate(regime_batches)
    result = _simulation_result_from_paths(
        last_close,
        price_paths,
        return_paths,
        regime_paths,
        economic_labels,
        simulation_kwargs.get("method", "hybrid"),
        base_seed,
        simulation_kwargs.get("student_weight", 0.35),
    )
    result.summary["adaptive_stopping"] = True
    result.summary["stopping_reason"] = "stable" if stable_batches >= required_stable else "maximum_paths"
    return result, pd.DataFrame(history)


def _simulation_result_from_paths(
    last_close: float,
    price_paths: np.ndarray,
    return_paths: np.ndarray,
    regime_paths: np.ndarray,
    economic_labels: list[str],
    method: str,
    seed: int,
    student_weight: float,
) -> SimulationResult:
    horizon = price_paths.shape[1]
    paths = len(price_paths)
    terminal = return_paths.sum(axis=1)
    mdd = maximum_drawdown(np.column_stack([np.full(paths, last_close), price_paths]))
    quantiles = {level: np.quantile(price_paths, level, axis=0) for level in [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975]}
    forecast = pd.DataFrame(
        {
            "step": np.arange(1, horizon + 1),
            "mean": price_paths.mean(axis=0),
            "median": quantiles[0.50],
            "lower_50": quantiles[0.25],
            "upper_50": quantiles[0.75],
            "lower_80": quantiles[0.10],
            "upper_80": quantiles[0.90],
            "lower_90": quantiles[0.05],
            "upper_90": quantiles[0.95],
            "lower_95": quantiles[0.025],
            "upper_95": quantiles[0.975],
            "expected_volatility": np.std(return_paths, axis=0) * np.sqrt(np.arange(1, horizon + 1)),
        }
    )
    daily_regime = np.stack([(regime_paths == state).mean(axis=0) for state in range(max(regime_paths.max() + 1, len(economic_labels)))], axis=1)
    for class_name in CLASS_NAMES:
        matching = [state for state, label in enumerate(economic_labels) if label == class_name]
        forecast[f"probability_{class_name.lower()}"] = daily_regime[:, matching].sum(axis=1) if matching else 0.0
    assert_quantile_monotonicity(forecast)
    var95, var99 = float(np.quantile(terminal, 0.05)), float(np.quantile(terminal, 0.01))
    summary = {
        "expected_terminal_close": float(price_paths[:, -1].mean()),
        "median_terminal_close": float(np.median(price_paths[:, -1])),
        "expected_return": float(np.mean(np.exp(terminal) - 1)),
        "median_return": float(np.median(np.exp(terminal) - 1)),
        "probability_positive_return": float(np.mean(terminal > 0)),
        "probability_negative_return": float(np.mean(terminal < 0)),
        "var_95": var95,
        "var_99": var99,
        "expected_shortfall_95": float(terminal[terminal <= var95].mean()),
        "expected_shortfall_99": float(terminal[terminal <= var99].mean()),
        "expected_maximum_drawdown": float(mdd.mean()),
        "median_maximum_drawdown": float(np.median(mdd)),
        "maximum_drawdown_quantiles": {str(level): float(np.quantile(mdd, level)) for level in [0.05, 0.5, 0.95]},
        "drawdown_probabilities": {str(threshold): float(np.mean(mdd <= -threshold)) for threshold in [0.03, 0.05, 0.07, 0.10]},
        "simulation_method": method,
        "number_of_paths": paths,
        "random_seed": seed,
        "student_weight": student_weight,
        "regime_pool_fallback_count": 0,
    }
    return SimulationResult(price_paths, return_paths, regime_paths, forecast, summary)


def antithetic_variance_reduction(shocks: np.ndarray, statistic) -> dict[str, float | bool]:
    """Optional pilot experiment using paired Z and -Z."""
    shocks = np.asarray(shocks, dtype=float)
    plain = np.asarray(statistic(shocks), dtype=float)
    paired = 0.5 * (plain + np.asarray(statistic(-shocks), dtype=float))
    ratio = float(np.var(plain, ddof=1) / max(np.var(paired, ddof=1), 1e-15))
    return {"variance_reduction_ratio": ratio, "retain": ratio >= 1.10}


def control_variate_adjustment(target: np.ndarray, control: np.ndarray, expected_control: float) -> tuple[np.ndarray, float]:
    """Estimate beta on pilot paths and return an unbiased control-variate adjustment."""
    target, control = np.asarray(target, dtype=float), np.asarray(control, dtype=float)
    beta = float(np.cov(target, control, ddof=1)[0, 1] / max(np.var(control, ddof=1), 1e-15))
    return target - beta * (control - float(expected_control)), beta
