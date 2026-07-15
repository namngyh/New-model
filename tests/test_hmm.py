import numpy as np

from vnindex_model.features import build_features
from vnindex_model.hmm import fit_filtered_hmm, forward_filter


def test_filtered_probabilities_sum_to_one(synthetic_ohlcv):
    features = build_features(synthetic_ohlcv)
    result = fit_filtered_hmm(
        features, features["log_return"], features["current_drawdown"], np.arange(400), [2], [55], 50
    )
    probability = result.probabilities.filter(like="hmm_probability_").to_numpy()
    assert np.allclose(probability.sum(axis=1), 1)
    assert "not smoothed" in result.diagnostics["probability_type"]


def test_forward_filter_does_not_change_past(synthetic_ohlcv):
    features = build_features(synthetic_ohlcv)
    result = fit_filtered_hmm(
        features, features["log_return"], features["current_drawdown"], np.arange(400), [2], [55], 50
    )
    x = features[result.feature_names].ffill().fillna(0)
    scaled = result.scaler.transform(x)
    baseline = forward_filter(result.model, scaled)
    scaled[500:] += 100
    changed = forward_filter(result.model, scaled)
    assert np.allclose(baseline[:500], changed[:500])
