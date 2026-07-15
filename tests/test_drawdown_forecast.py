import json

import numpy as np

from vnindex_model.drawdown_forecast import (
    build_drawdown_forecast,
    compute_drawdown_paths,
    compute_running_maximum_drawdown,
    conditional_expected_drawdown,
    direct_drawdown_conformal_upper,
    drawdown_probability_intervals,
    first_passage_times,
    maximum_drawdown_at_risk,
    pointwise_drawdown_band,
    recovery_statistics,
    simultaneous_drawdown_band,
)


def _paths():
    return np.array([[90.0, 110.0, 100.0], [105.0, 95.0, 120.0]])


def test_origin_and_historical_peak_drawdown_and_nonnegative_severity():
    origin = compute_drawdown_paths(_paths(), 100.0, "origin_peak")
    historical = compute_drawdown_paths(_paths(), 100.0, "historical_peak", historical_peak=125.0)
    assert np.isclose(origin[0, 0], -0.10)
    assert np.isclose(historical[0, 0], 90 / 125 - 1)
    assert np.all(historical <= origin + 1e-12)
    result = build_drawdown_forecast(_paths(), 100.0, "historical_peak", historical_peak=125.0)
    assert np.all(result.severity_paths >= 0)


def test_running_mdd_and_breach_probabilities_are_monotone():
    drawdown = compute_drawdown_paths(_paths(), 100.0)
    running = compute_running_maximum_drawdown(drawdown)
    assert np.all(np.diff(running, axis=1) >= -1e-12)
    result = build_drawdown_forecast(_paths(), 100.0, "origin_peak")
    columns = [column for column in result.term_structure if column.startswith("probability_breach_")]
    for column in columns:
        assert result.term_structure[column].is_monotonic_increasing


def test_first_passage_and_recovery_right_censoring():
    severity = np.array([[0.01, 0.06, 0.08], [0.01, 0.02, 0.03]])
    passage = first_passage_times(severity, [0.05])[0.05]
    assert passage[0] == 2
    assert np.isnan(passage[1])
    recovery = recovery_statistics(_paths(), historical_peak=115.0)
    assert np.isnan(recovery["recovery_times"][0])
    assert recovery["right_censored"][0]
    assert recovery["recovery_times"][1] == 3


def test_mdar_and_ced_known_distribution():
    severity = np.arange(1, 101, dtype=float) / 100
    mdar = maximum_drawdown_at_risk(severity, [0.90, 0.95, 0.99])
    ced = conditional_expected_drawdown(severity, [0.90, 0.95, 0.99])
    assert np.isclose(mdar["mdar_90"], np.quantile(severity, 0.90))
    assert ced["ced_95"] >= mdar["mdar_95"]
    assert ced["ced_99"] >= ced["ced_95"]


def test_direct_conformal_uses_only_matured_drawdown_labels():
    actual = np.array([0.10, 0.20, 0.30, 100.0])
    upper = np.array([0.08, 0.15, 0.20, 0.0])
    result, correction = direct_drawdown_conformal_upper(
        actual,
        upper,
        np.array([0.25]),
        0.20,
        calibration_target_end=np.array([2, 3, 4, 20]),
        evaluation_origins=np.array([10]),
    )
    assert correction[0] < 1
    assert result[0] >= 0.25


def test_pointwise_and_simultaneous_bands_are_ordered():
    rng = np.random.default_rng(55)
    paths = np.abs(rng.normal(0.04, 0.01, size=(100, 5)))
    lower, upper = pointwise_drawdown_band(paths)
    assert np.all(lower <= upper)
    scale = np.full_like(paths, 0.01)
    simultaneous_lower, simultaneous_upper, multiplier = simultaneous_drawdown_band(
        paths,
        np.full_like(paths, 0.04),
        scale,
        np.full(5, 0.04),
        np.full(5, 0.01),
    )
    assert multiplier >= 0
    assert np.all(simultaneous_lower <= simultaneous_upper)


def test_probability_ci_weighted_mcse_and_deterministic_seed():
    events = {"probability_mdd_5": np.array([0, 0, 1, 1, 1], dtype=float)}
    weights = np.array([0.1, 0.1, 0.2, 0.3, 0.3])
    one = drawdown_probability_intervals(events, weights=weights, seed=55, bootstrap_replications=100)
    two = drawdown_probability_intervals(events, weights=weights, seed=55, bootstrap_replications=100)
    assert one.equals(two)
    assert one.loc[0, "mcse"] > 0
    assert 0 <= one.loc[0, "ci_95_lower"] <= one.loc[0, "ci_95_upper"] <= 1


def test_existing_summary_artifact_fields_remain_compatible(tmp_path):
    source = "artifacts/forecasts/latest_forecast_summary.json"
    with open(source, encoding="utf-8") as handle:
        before = json.load(handle)
    required = {"forecast_origin", "last_observed_close", "var_95", "expected_shortfall_95"}
    assert required.issubset(before)
