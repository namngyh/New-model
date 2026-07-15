"""Likelihood-ratio importance sampling for conditional tail-risk estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .simulation import maximum_drawdown


@dataclass
class ImportanceSamplingResult:
    terminal_returns: np.ndarray
    maximum_drawdowns: np.ndarray
    normalized_weights: np.ndarray
    log_weights: np.ndarray
    diagnostics: dict[str, float | int]
    estimates: dict[str, float]


def proposal_transition_matrix(
    transition_matrix: np.ndarray,
    economic_labels: list[str],
    strength: float,
) -> np.ndarray:
    """Tilt destination columns toward Bear/Stress and normalize each row."""
    transition = np.asarray(transition_matrix, dtype=float)
    if transition.ndim != 2 or transition.shape[0] != transition.shape[1]:
        raise ValueError("Transition matrix phải vuông")
    multipliers = np.ones(transition.shape[1])
    for state, label in enumerate(economic_labels):
        if label == "Bear":
            multipliers[state] = np.exp(float(strength))
        elif label == "Stress":
            multipliers[state] = np.exp(1.5 * float(strength))
    proposal = np.maximum(transition, 1e-15) * multipliers[None, :]
    return proposal / proposal.sum(axis=1, keepdims=True)


def shock_proposal_probabilities(residuals: np.ndarray, strength: float) -> np.ndarray:
    """Exponentially tilt an empirical pool toward large negative residuals."""
    residuals = np.asarray(residuals, dtype=float)
    scale = max(float(np.std(residuals)), 1e-8)
    severity = np.maximum(-residuals / scale, 0.0)
    log_mass = np.clip(float(strength) * severity, -30.0, 30.0)
    mass = np.exp(log_mass - np.max(log_mass))
    return mass / mass.sum()


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    return float(total**2 / max(np.square(weights).sum(), 1e-300))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    cumulative = np.cumsum(weights) / max(weights.sum(), 1e-300)
    return float(values[min(np.searchsorted(cumulative, quantile, side="left"), len(values) - 1)])


def _weighted_probability_standard_error(indicator: np.ndarray, normalized_weights: np.ndarray) -> float:
    estimate = float(np.sum(normalized_weights * indicator))
    return float(np.sqrt(np.sum(np.square(normalized_weights) * np.square(indicator - estimate))))


def simulate_tail_importance(
    paths: int,
    horizon: int,
    daily_drift: float,
    daily_volatility: float,
    residuals: np.ndarray,
    transition_matrix: np.ndarray,
    current_regime_probability: np.ndarray,
    economic_labels: list[str],
    transition_strength: float = 0.25,
    shock_strength: float = 0.15,
    seed: int = 55,
) -> ImportanceSamplingResult:
    """Simulate under Q and recover P-expectations using log likelihood ratios."""
    rng = np.random.default_rng(seed)
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    lower, upper = np.quantile(residuals, [0.001, 0.999])
    residuals = np.clip(residuals, max(lower, -12.0), min(upper, 12.0))
    transition = np.asarray(transition_matrix, dtype=float)
    proposal = proposal_transition_matrix(transition, economic_labels, transition_strength)
    shock_q = shock_proposal_probabilities(residuals, shock_strength)
    shock_p = np.full(len(residuals), 1.0 / len(residuals))
    state_count = transition.shape[0]
    current = rng.choice(state_count, size=int(paths), p=np.asarray(current_regime_probability) / np.sum(current_regime_probability))
    log_weights = np.zeros(int(paths), dtype=float)
    returns = np.empty((int(paths), int(horizon)), dtype=float)
    for step in range(int(horizon)):
        next_state = np.empty(int(paths), dtype=int)
        for origin in range(state_count):
            mask = current == origin
            if not mask.any():
                continue
            destinations = rng.choice(state_count, size=int(mask.sum()), p=proposal[origin])
            next_state[mask] = destinations
            log_weights[mask] += np.log(np.maximum(transition[origin, destinations], 1e-300))
            log_weights[mask] -= np.log(np.maximum(proposal[origin, destinations], 1e-300))
        shock_index = rng.choice(len(residuals), size=int(paths), p=shock_q)
        shocks = residuals[shock_index]
        log_weights += np.log(shock_p[shock_index]) - np.log(np.maximum(shock_q[shock_index], 1e-300))
        returns[:, step] = float(daily_drift) + float(daily_volatility) * shocks
        current = next_state
    terminal = returns.sum(axis=1)
    price_paths = np.exp(np.cumsum(returns, axis=1))
    drawdowns = maximum_drawdown(np.column_stack([np.ones(int(paths)), price_paths]))
    shifted = log_weights - np.max(log_weights)
    raw_weights = np.exp(shifted)
    normalized = raw_weights / raw_weights.sum()
    ess = effective_sample_size(normalized)
    var95 = _weighted_quantile(terminal, normalized, 0.05)
    var99 = _weighted_quantile(terminal, normalized, 0.01)
    tail95 = terminal <= var95
    tail99 = terminal <= var99
    estimates = {
        "probability_negative_return": float(np.sum(normalized * (terminal < 0))),
        "probability_mdd_3": float(np.sum(normalized * (drawdowns <= -0.03))),
        "probability_mdd_5": float(np.sum(normalized * (drawdowns <= -0.05))),
        "probability_mdd_7": float(np.sum(normalized * (drawdowns <= -0.07))),
        "probability_mdd_10": float(np.sum(normalized * (drawdowns <= -0.10))),
        "var_95": var95,
        "var_99": var99,
        "expected_shortfall_95": float(np.sum(normalized[tail95] * terminal[tail95]) / normalized[tail95].sum()),
        "expected_shortfall_99": float(np.sum(normalized[tail99] * terminal[tail99]) / normalized[tail99].sum()),
    }
    event = drawdowns <= -0.07
    diagnostics = {
        "paths": int(paths),
        "ess": ess,
        "ess_ratio": ess / max(int(paths), 1),
        "maximum_normalized_weight": float(normalized.max()),
        "weight_coefficient_of_variation": float(np.std(raw_weights) / max(np.mean(raw_weights), 1e-300)),
        "log_weight_min": float(log_weights.min()),
        "log_weight_max": float(log_weights.max()),
        "tail_events_generated_mdd_7": int(event.sum()),
        "mcse_mdd_7": _weighted_probability_standard_error(event.astype(float), normalized),
        "transition_strength": float(transition_strength),
        "shock_strength": float(shock_strength),
    }
    return ImportanceSamplingResult(terminal, drawdowns, normalized, log_weights, diagnostics, estimates)


def importance_sensitivity_table(results: list[ImportanceSamplingResult]) -> pd.DataFrame:
    return pd.DataFrame([{**result.diagnostics, **result.estimates} for result in results])


def simulate_stratified_importance(
    paths: int,
    current_regime_probability: np.ndarray,
    minimum_paths_per_regime: int = 100,
    **kwargs,
) -> ImportanceSamplingResult:
    """Run importance sampling inside initial-state strata and combine by true state mass."""
    probability = np.asarray(current_regime_probability, dtype=float)
    probability /= probability.sum()
    remaining = int(paths) - len(probability) * int(minimum_paths_per_regime)
    if remaining < 0:
        raise ValueError("Không đủ paths cho stratified importance sampling")
    counts = np.full(len(probability), int(minimum_paths_per_regime), dtype=int)
    raw = remaining * probability
    counts += np.floor(raw).astype(int)
    for state in np.argsort(-(raw - np.floor(raw)))[: int(paths) - int(counts.sum())]:
        counts[state] += 1
    terminals, drawdowns, weights, log_weights = [], [], [], []
    generated_events = 0
    base_seed = int(kwargs.pop("seed", 55))
    for state, count in enumerate(counts):
        initial = np.zeros(len(probability))
        initial[state] = 1.0
        result = simulate_tail_importance(
            paths=int(count),
            current_regime_probability=initial,
            seed=base_seed + state,
            **kwargs,
        )
        terminals.append(result.terminal_returns)
        drawdowns.append(result.maximum_drawdowns)
        weights.append(probability[state] * result.normalized_weights)
        log_weights.append(result.log_weights)
        generated_events += int(result.diagnostics["tail_events_generated_mdd_7"])
    terminal = np.concatenate(terminals)
    mdd = np.concatenate(drawdowns)
    normalized = np.concatenate(weights)
    normalized /= normalized.sum()
    logs = np.concatenate(log_weights)
    ess = effective_sample_size(normalized)
    var95 = _weighted_quantile(terminal, normalized, 0.05)
    var99 = _weighted_quantile(terminal, normalized, 0.01)
    tail95, tail99 = terminal <= var95, terminal <= var99
    estimates = {
        "probability_negative_return": float(np.sum(normalized * (terminal < 0))),
        "probability_mdd_3": float(np.sum(normalized * (mdd <= -0.03))),
        "probability_mdd_5": float(np.sum(normalized * (mdd <= -0.05))),
        "probability_mdd_7": float(np.sum(normalized * (mdd <= -0.07))),
        "probability_mdd_10": float(np.sum(normalized * (mdd <= -0.10))),
        "var_95": var95,
        "var_99": var99,
        "expected_shortfall_95": float(np.sum(normalized[tail95] * terminal[tail95]) / normalized[tail95].sum()),
        "expected_shortfall_99": float(np.sum(normalized[tail99] * terminal[tail99]) / normalized[tail99].sum()),
    }
    diagnostics = {
        "paths": int(paths),
        "ess": ess,
        "ess_ratio": ess / int(paths),
        "maximum_normalized_weight": float(normalized.max()),
        "weight_coefficient_of_variation": float(np.std(normalized) / max(np.mean(normalized), 1e-300)),
        "log_weight_min": float(logs.min()),
        "log_weight_max": float(logs.max()),
        "tail_events_generated_mdd_7": generated_events,
        "mcse_mdd_7": _weighted_probability_standard_error((mdd <= -0.07).astype(float), normalized),
        "transition_strength": float(kwargs.get("transition_strength", 0.0)),
        "shock_strength": float(kwargs.get("shock_strength", 0.0)),
    }
    return ImportanceSamplingResult(terminal, mdd, normalized, logs, diagnostics, estimates)
