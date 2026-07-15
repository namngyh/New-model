import numpy as np

from vnindex_model.features import build_features, select_train_features
from vnindex_model.hmm import fit_filtered_hmm
from vnindex_model.random_forest import fit_forest_bundle, predict_bundle
from vnindex_model.splits import purged_train_validation_test
from vnindex_model.targets import build_targets


def test_synthetic_end_to_end_smoke(synthetic_ohlcv):
    features = build_features(synthetic_ohlcv)
    targets = build_targets(synthetic_ohlcv, [20])
    split = purged_train_validation_test(synthetic_ohlcv["date"], targets["target_end_date_20"], embargo=20)
    finite = (
        targets[["forward_return_20", "normalized_return_20", "forward_max_drawdown_20"]].notna().all(axis=1).to_numpy()
    )
    train = split.train[finite[split.train]]
    test = split.test[finite[split.test]]
    selected = select_train_features(features, train)[:20]
    hmm = fit_filtered_hmm(features, features["log_return"], features["current_drawdown"], train, [2], [55], 30)
    assert len(hmm.probabilities) == len(synthetic_ohlcv)
    config = {"n_estimators": 8, "max_depth": 3, "min_samples_leaf": 5, "max_features": "sqrt", "max_samples": 0.8}
    bundle = fit_forest_bundle(
        features[selected].iloc[train],
        targets["regime_20"].astype(object).to_numpy()[train],
        targets["forward_return_20"].to_numpy()[train],
        targets["normalized_return_20"].to_numpy()[train],
        targets["forward_max_drawdown_20"].to_numpy()[train],
        selected,
        config,
        55,
    )
    prediction = predict_bundle(bundle, features[selected].iloc[test])
    assert prediction["probabilities"].shape == (len(test), 4)
    assert np.isfinite(prediction["return"]).all()
