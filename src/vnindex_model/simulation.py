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
