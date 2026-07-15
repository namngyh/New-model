"""Forecast comparison tests with HAC and block-resampling uncertainty."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_hac


def diebold_mariano(actual, primary, baseline, horizon: int = 1) -> dict[str, float]:
    differential = (np.asarray(actual) - np.asarray(primary)) ** 2 - (np.asarray(actual) - np.asarray(baseline)) ** 2
    model = OLS(differential, np.ones((len(differential), 1))).fit()
    covariance = cov_hac(model, nlags=max(horizon - 1, 0))
    standard_error = float(np.sqrt(max(covariance[0, 0], 1e-16)))
    statistic = float(model.params[0] / standard_error)
    return {
        "dm_statistic": statistic,
        "dm_pvalue": float(2 * (1 - norm.cdf(abs(statistic)))),
        "mean_loss_difference": float(differential.mean()),
    }


def moving_block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    blocks = []
    while sum(map(len, blocks)) < n:
        start = int(rng.integers(0, max(n - block_length + 1, 1)))
        blocks.append(np.arange(start, min(start + block_length, n)))
    return np.concatenate(blocks)[:n]


def block_bootstrap_difference(
    actual, primary, baseline, reps: int = 500, block_length: int = 10, seed: int = 55
) -> dict[str, float]:
    actual, primary, baseline = map(np.asarray, (actual, primary, baseline))
    rng = np.random.default_rng(seed)
    differences = np.empty(reps)
    for rep in range(reps):
        idx = moving_block_indices(len(actual), block_length, rng)
        differences[rep] = np.mean(np.abs(actual[idx] - primary[idx])) - np.mean(np.abs(actual[idx] - baseline[idx]))
    return {
        "bootstrap_difference_mean": float(differences.mean()),
        "bootstrap_ci_lower": float(np.quantile(differences, 0.025)),
        "bootstrap_ci_upper": float(np.quantile(differences, 0.975)),
    }
