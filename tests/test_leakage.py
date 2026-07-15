import numpy as np
import pandas as pd

from vnindex_model.targets import assign_regime_labels, build_targets, select_regime_thresholds
from vnindex_model.validation import assert_no_target_overlap


def test_target_alignment(synthetic_ohlcv):
    targets = build_targets(synthetic_ohlcv, [1, 20])
    expected = np.log(synthetic_ohlcv.loc[20, "close"] / synthetic_ohlcv.loc[0, "close"])
    assert np.isclose(targets.loc[0, "forward_return_20"], expected)
    assert targets.loc[0, "target_end_date_20"] == synthetic_ohlcv.loc[20, "date"]


def test_purge_contract_raises(synthetic_ohlcv):
    targets = build_targets(synthetic_ohlcv, [20])
    mask = pd.Series(False, index=synthetic_ohlcv.index)
    mask.iloc[:400] = True
    boundary = synthetic_ohlcv.loc[390, "date"]
    try:
        assert_no_target_overlap(targets["target_end_date_20"], mask, boundary)
    except AssertionError:
        return
    raise AssertionError("Expected overlapping target detection")


def test_forward_drawdown_never_positive(synthetic_ohlcv):
    targets = build_targets(synthetic_ohlcv, [20])
    assert (targets["forward_max_drawdown_20"].dropna() <= 1e-12).all()


def test_regime_thresholds_use_only_given_train_validation(synthetic_ohlcv):
    targets = build_targets(synthetic_ohlcv, [20])
    train = np.arange(30, 350)
    validation = np.arange(370, 500)
    selected, table = select_regime_thresholds(
        targets["forward_return_20"],
        targets["forward_max_drawdown_20"],
        targets["forecast_scale_20"],
        train,
        validation,
    )
    labels = assign_regime_labels(
        targets["forward_return_20"],
        targets["forward_max_drawdown_20"],
        targets["forecast_scale_20"],
        **selected,
    )
    assert table["selected"].sum() == 1
    assert set(labels.iloc[validation].dropna()).issubset({"Bull", "Sideway", "Bear", "Stress"})
