"""Explicit leakage contracts and data assertions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def assert_causal_feature_invariance(builder, frame: pd.DataFrame, cutoff: int) -> None:
    """Prove that perturbing future prices does not alter past features."""
    baseline = builder(frame.copy()).iloc[:cutoff]
    changed = frame.copy()
    numeric = changed.select_dtypes(include=[np.number]).columns
    changed.loc[changed.index[cutoff:], numeric] *= 7.0
    candidate = builder(changed).iloc[:cutoff]
    pd.testing.assert_frame_equal(baseline, candidate, check_exact=False, rtol=1e-10, atol=1e-12)


def assert_no_target_overlap(target_end_dates: pd.Series, train_mask: pd.Series, next_boundary: pd.Timestamp) -> None:
    observed = target_end_dates.loc[train_mask].dropna()
    if not observed.empty and not (observed < next_boundary).all():
        raise AssertionError("Phát hiện target train chồng lấn boundary kế tiếp")


def assert_probabilities(probabilities: np.ndarray, tolerance: float = 1e-7) -> None:
    if probabilities.ndim != 2 or np.any(probabilities < -tolerance):
        raise AssertionError("Xác suất HMM không hợp lệ")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=tolerance):
        raise AssertionError("Xác suất HMM không tổng bằng 1")


def assert_quantile_monotonicity(frame: pd.DataFrame) -> None:
    columns = ["lower_95", "lower_90", "lower_80", "lower_50", "median", "upper_50", "upper_80", "upper_90", "upper_95"]
    if not all(column in frame for column in columns):
        raise AssertionError("Thiếu cột quantile")
    values = frame[columns].to_numpy()
    if np.any(np.diff(values, axis=1) < -1e-8):
        raise AssertionError("Quantile không đơn điệu")
