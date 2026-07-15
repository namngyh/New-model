import numpy as np
import pandas as pd

from vnindex_model.bootstrap import outer_stationary_bootstrap_quick, stationary_bootstrap_indices
from vnindex_model.jackknife import delete_block_jackknife


def _records(n=240):
    rng = np.random.default_rng(55)
    actual = rng.normal(0, 0.03, n)
    old = actual + rng.normal(0, 0.02, n)
    improved = actual + rng.normal(0, 0.01, n)
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=n),
            "actual_return": actual,
            "old_center": old,
            "improved_center": improved,
            "old_lower_95": old - 0.04,
            "old_upper_95": old + 0.04,
            "lower_95": improved - 0.06,
            "upper_95": improved + 0.06,
            "conformal_var_95": improved - 0.05,
            "conformal_es_95": improved - 0.07,
            "brier_loss": rng.uniform(0, 1, n),
            "actual_drawdown": -rng.uniform(0, 0.10, n),
            "actual_regime": np.resize(np.array(["Bull", "Sideway", "Bear", "Stress"]), n),
            "predicted_regime": np.resize(np.array(["Bull", "Sideway", "Sideway", "Stress"]), n),
        }
    )


def test_stationary_bootstrap_retains_block_adjacency():
    indices = stationary_bootstrap_indices(1000, 20, np.random.default_rng(55))
    adjacency = indices[1:] == (indices[:-1] + 1) % 1000
    assert adjacency.mean() > 0.85


def test_outer_bootstrap_and_delete_block_outputs():
    records = _records()
    summary, replicates = outer_stationary_bootstrap_quick(records, 50, 20, 55)
    assert len(replicates) == 50
    assert "coverage_improvement" in set(summary["metric"])
    influence = delete_block_jackknife(records)
    assert {"month", "quarter", "regime_episode", "large_drawdown_block"}.issubset(
        set(influence["block_type"])
    )
    assert influence["absolute_influence"].ge(0).all()
