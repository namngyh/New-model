from vnindex_model.splits import PurgedWalkForwardSplit, purged_train_validation_test
from vnindex_model.targets import build_targets


def test_chronological_purged_split(synthetic_ohlcv):
    target = build_targets(synthetic_ohlcv, [20])
    split = purged_train_validation_test(synthetic_ohlcv["date"], target["target_end_date_20"], embargo=20)
    assert split.train.max() < split.validation.min() < split.test.min()
    assert (target.loc[split.train, "target_end_date_20"] < split.train_boundary).all()
    assert (target.loc[split.validation, "target_end_date_20"] < split.test_boundary).all()


def test_walk_forward_expands(synthetic_ohlcv):
    target = build_targets(synthetic_ohlcv, [10])
    folds = list(
        PurgedWalkForwardSplit(3, embargo=10, min_train=250).split(
            synthetic_ohlcv["date"], target["target_end_date_10"]
        )
    )
    assert len(folds) == 3
    assert len(folds[0][0]) < len(folds[-1][0])
    assert all(train.max() < valid.min() for train, valid in folds)
