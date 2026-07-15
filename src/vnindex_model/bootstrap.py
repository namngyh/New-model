"""IID, moving, stationary, and regime-conditioned residual bootstrap."""

from __future__ import annotations

import numpy as np


def iid_bootstrap(residuals: np.ndarray, size: int | tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    pool = np.asarray(residuals)[np.isfinite(residuals)]
    if len(pool) == 0:
        raise ValueError("Residual pool rỗng")
    return rng.choice(pool, size=size, replace=True)


def moving_block_bootstrap(
    residuals: np.ndarray, sample_length: int, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    pool = np.asarray(residuals)[np.isfinite(residuals)]
    if block_length < 1 or block_length > len(pool):
        raise ValueError("block_length phải nằm trong [1, len(residuals)]")
    output: list[np.ndarray] = []
    while sum(len(block) for block in output) < sample_length:
        start = int(rng.integers(0, len(pool) - block_length + 1))
        output.append(pool[start : start + block_length])
    return np.concatenate(output)[:sample_length]


def stationary_bootstrap(
    residuals: np.ndarray, sample_length: int, mean_block_length: int, rng: np.random.Generator
) -> np.ndarray:
    pool = np.asarray(residuals)[np.isfinite(residuals)]
    if mean_block_length < 1:
        raise ValueError("mean_block_length phải dương")
    output = np.empty(sample_length)
    index = int(rng.integers(0, len(pool)))
    restart_probability = 1 / mean_block_length
    for position in range(sample_length):
        if position == 0 or rng.random() < restart_probability:
            index = int(rng.integers(0, len(pool)))
        else:
            index = (index + 1) % len(pool)
        output[position] = pool[index]
    return output


def regime_conditioned_sample(
    residuals: np.ndarray,
    probabilities: np.ndarray,
    state: int,
    size: int,
    rng: np.random.Generator,
    minimum_effective_size: float = 30,
) -> tuple[np.ndarray, bool]:
    residuals = np.asarray(residuals, dtype=float)
    weights = np.asarray(probabilities, dtype=float)[:, state]
    mask = np.isfinite(residuals) & np.isfinite(weights)
    residuals, weights = residuals[mask], weights[mask]
    effective = weights.sum() ** 2 / max(np.square(weights).sum(), 1e-12)
    fallback = effective < minimum_effective_size or weights.sum() <= 0
    if fallback:
        return rng.choice(residuals, size=size, replace=True), True
    weights = weights / weights.sum()
    return rng.choice(residuals, size=size, replace=True, p=weights), False


def choose_block_length(residuals: np.ndarray, candidates: list[int] | None = None) -> int:
    """Data-driven rule chosen without test labels: minimize lag-ACF reconstruction error."""
    pool = np.asarray(residuals, dtype=float)
    pool = pool[np.isfinite(pool)]
    candidates = candidates or [5, 10, 15, 20]
    lag_one = abs(np.corrcoef(pool[:-1], pool[1:])[0, 1]) if len(pool) > 2 else 0.0
    target = max(2, int(round(1 / max(1 - lag_one, 0.05))))
    return min(candidates, key=lambda value: abs(value - target))
