import numpy as np

from vnindex_model.bootstrap import moving_block_bootstrap, regime_conditioned_sample, stationary_bootstrap


def test_moving_block_length_validation():
    rng = np.random.default_rng(1)
    try:
        moving_block_bootstrap(np.arange(5), 10, 6, rng)
    except ValueError:
        return
    raise AssertionError("Expected invalid block length")


def test_bootstrap_output_lengths():
    rng = np.random.default_rng(1)
    residuals = np.arange(100, dtype=float)
    assert len(moving_block_bootstrap(residuals, 250, 10, rng)) == 250
    assert len(stationary_bootstrap(residuals, 250, 10, rng)) == 250


def test_regime_conditioned_sampling():
    rng = np.random.default_rng(1)
    residuals = np.r_[np.full(100, -2.0), np.full(100, 2.0)]
    probabilities = np.column_stack([np.r_[np.ones(100), np.zeros(100)], np.r_[np.zeros(100), np.ones(100)]])
    sampled, fallback = regime_conditioned_sample(residuals, probabilities, 0, 500, rng)
    assert sampled.mean() < -1.5
    assert fallback is False
