import numpy as np

from vnindex_model.conformal import (
    eligible_score_indices,
    finite_sample_quantile,
    select_conformal_method,
    sequential_conformal,
)


def _sequential(evaluation_actual):
    n = 180
    calibration_actual = np.linspace(-0.03, 0.03, n)
    zeros = np.zeros(n)
    sigma = np.full(n, 0.01)
    regimes = np.tile([0, 1, 2], n // 3)
    bins = np.tile([0, 1, 2], n // 3)
    return sequential_conformal(
        calibration_actual,
        zeros,
        sigma,
        regimes,
        bins,
        np.asarray(evaluation_actual),
        np.zeros(len(evaluation_actual)),
        np.full(len(evaluation_actual), 0.01),
        np.zeros(len(evaluation_actual), dtype=int),
        np.zeros(len(evaluation_actual), dtype=int),
        20,
        [0.50, 0.20, 0.10, 0.05],
        "volatility_regime",
        None,
        80,
    )


def test_label_maturity_delay_for_all_horizons():
    origins = np.arange(100)
    for horizon in [1, 5, 10, 20, 40, 60]:
        target_end = origins + horizon
        eligible = eligible_score_indices(target_end, 70)
        assert np.all(target_end[eligible] <= 70)
        assert len(eligible) == max(71 - horizon, 0)


def test_finite_sample_quantile_is_conservative_higher():
    assert finite_sample_quantile(np.arange(1, 11), 0.05) == 10
    assert finite_sample_quantile(np.arange(1, 11), 0.20) == 9


def test_future_evaluation_score_cannot_change_past_interval():
    one_actual = np.linspace(-0.02, 0.02, 60)
    two_actual = one_actual.copy()
    two_actual[40:] = 1000
    one = _sequential(one_actual)
    two = _sequential(two_actual)
    assert np.allclose(one.loc[:39, ["lower_95", "upper_95"]], two.loc[:39, ["lower_95", "upper_95"]])


def test_stratum_fallback_and_interval_monotonicity():
    result = _sequential(np.linspace(-0.01, 0.01, 30))
    assert set(result["stratum_used"]).issubset({"regime_x_volatility", "volatility", "regime", "global"})
    ordered = result[["lower_95", "lower_90", "lower_80", "lower_50", "upper_50", "upper_80", "upper_90", "upper_95"]].to_numpy()
    assert (np.diff(ordered, axis=1) >= -1e-12).all()


def test_conformal_method_selection_has_no_test_argument():
    rng = np.random.default_rng(55)
    actual = rng.normal(0, 0.02, 500)
    center = np.zeros(500)
    sigma = np.full(500, 0.01)
    regimes = rng.integers(0, 3, 500)
    bins = rng.integers(0, 3, 500)
    one = select_conformal_method(actual, center, sigma, regimes, bins, 20, [0.5, 0.2, 0.1, 0.05], [250, None], 40)
    unrelated_test = rng.normal(100, 1, 500)
    unrelated_test[:] *= -5
    two = select_conformal_method(actual, center, sigma, regimes, bins, 20, [0.5, 0.2, 0.1, 0.05], [250, None], 40)
    assert (one.method, one.window) == (two.method, two.window)
