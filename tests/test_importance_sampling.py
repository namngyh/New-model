import numpy as np

from vnindex_model.importance_sampling import (
    effective_sample_size,
    proposal_transition_matrix,
    simulate_tail_importance,
)
from vnindex_model.simulation import adaptive_simulate_paths, stratified_estimator


def test_proposal_transition_normalization_and_likelihood_ratio():
    transition = np.array([[0.8, 0.2], [0.3, 0.7]])
    proposal = proposal_transition_matrix(transition, ["Sideway", "Stress"], 0.5)
    assert np.allclose(proposal.sum(axis=1), 1)
    assert proposal[0, 1] > transition[0, 1]
    assert np.isclose(np.sum(proposal[0] * transition[0] / proposal[0]), 1.0)


def test_importance_estimator_on_known_empirical_distribution_and_seed():
    kwargs = {
        "paths": 40000,
        "horizon": 1,
        "daily_drift": 0.0,
        "daily_volatility": 1.0,
        "residuals": np.array([-1.0, 1.0]),
        "transition_matrix": np.ones((1, 1)),
        "current_regime_probability": np.ones(1),
        "economic_labels": ["Sideway"],
        "transition_strength": 0.0,
        "shock_strength": 1.0,
        "seed": 55,
    }
    one = simulate_tail_importance(**kwargs)
    two = simulate_tail_importance(**kwargs)
    assert abs(one.estimates["probability_negative_return"] - 0.5) < 0.015
    assert np.allclose(one.terminal_returns, two.terminal_returns)
    assert np.allclose(one.normalized_weights, two.normalized_weights)


def test_ess_and_stratified_weighting():
    assert effective_sample_size(np.ones(4)) == 4
    assert np.isclose(stratified_estimator(np.array([1.0, 3.0]), np.array([0.75, 0.25])), 1.5)


def test_adaptive_stopping_respects_path_budget():
    rng = np.random.default_rng(1)
    kwargs = {
        "last_close": 1000.0,
        "horizon": 5,
        "daily_drift": 0.0,
        "daily_volatility": 0.15,
        "degrees_of_freedom": 8.0,
        "residuals": rng.normal(size=300),
        "historical_regime_probabilities": np.ones((300, 1)),
        "transition_matrix": np.ones((1, 1)),
        "current_regime_probability": np.ones(1),
        "rf_class_probability": np.array([0.25, 0.25, 0.25, 0.25]),
        "economic_labels": ["Sideway"],
        "method": "student_t",
        "seed": 55,
    }
    stopping = {
        "batch_size": 100,
        "minimum_paths": 200,
        "maximum_paths": 400,
        "probability_mcse_tolerance": 1.0,
        "rare_probability_relative_mcse": 100.0,
        "quantile_stability_tolerance": 100.0,
        "consecutive_stable_batches": 1,
    }
    result, history = adaptive_simulate_paths(1000.0, ["Sideway"], kwargs, stopping)
    assert 200 <= len(result.return_paths) <= 400
    assert history["paths"].is_monotonic_increasing
