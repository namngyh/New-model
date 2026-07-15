import pandas as pd

from vnindex_model.features import build_features, select_train_features


def test_features_are_future_invariant(synthetic_ohlcv):
    baseline = build_features(synthetic_ohlcv).iloc[:400]
    changed = synthetic_ohlcv.copy()
    changed.loc[400:, ["open", "high", "low", "close", "volume"]] *= 5
    candidate = build_features(changed).iloc[:400]
    pd.testing.assert_frame_equal(baseline, candidate)


def test_no_backfill_at_start(synthetic_ohlcv):
    features = build_features(synthetic_ohlcv)
    assert pd.isna(features.loc[0, "return_lag_1"])
    assert pd.isna(features.loc[4, "distance_sma_20"])


def test_train_only_feature_selection(synthetic_ohlcv):
    features = build_features(synthetic_ohlcv)
    selected = select_train_features(features, list(range(300)))
    assert selected
    assert all(features.iloc[:300][name].nunique(dropna=True) > 1 for name in selected)
