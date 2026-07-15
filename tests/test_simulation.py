import numpy as np

from vnindex_model.simulation import maximum_drawdown, simulate_paths
from vnindex_model.volatility import standardized_student_t


def _simulation(seed=55):
    rng = np.random.default_rng(1)
    residuals = rng.normal(size=500)
    probabilities = rng.dirichlet([2, 2, 2], size=500)
    return simulate_paths(
        1000,
        20,
        500,
        0.0002,
        0.01,
        8,
        residuals,
        probabilities,
        np.array([[0.9, 0.08, 0.02], [0.08, 0.84, 0.08], [0.03, 0.12, 0.85]]),
        np.array([0.5, 0.4, 0.1]),
        np.array([0.35, 0.35, 0.2, 0.1]),
        ["Bull", "Sideway", "Stress"],
        seed=seed,
    )


def test_student_t_standardization():
    draws = standardized_student_t(np.random.default_rng(55), 8, 200000)
    assert abs(draws.var() - 1) < 0.03


def test_simulation_shape_quantiles_and_risk():
    result = _simulation()
    assert result.price_paths.shape == (500, 20)
    ordered = result.forecast[
        ["lower_95", "lower_90", "lower_80", "lower_50", "median", "upper_50", "upper_80", "upper_90", "upper_95"]
    ].to_numpy()
    assert (np.diff(ordered, axis=1) >= 0).all()
    assert result.summary["expected_shortfall_95"] <= result.summary["var_95"]


def test_deterministic_seed_and_drawdown():
    one, two = _simulation(22), _simulation(22)
    assert np.allclose(one.price_paths, two.price_paths)
    paths = np.array([[100, 110, 90, 95]])
    assert np.isclose(maximum_drawdown(paths)[0], 90 / 110 - 1)
