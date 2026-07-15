"""Chronological split and purged expanding-window utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    train_boundary: pd.Timestamp
    test_boundary: pd.Timestamp


def purged_train_validation_test(
    dates: pd.Series,
    target_end_dates: pd.Series,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    embargo: int = 20,
) -> TemporalSplit:
    n = len(dates)
    train_end = int(n * train_fraction)
    valid_end = int(n * (train_fraction + validation_fraction))
    train_boundary = pd.Timestamp(dates.iloc[train_end])
    test_boundary = pd.Timestamp(dates.iloc[valid_end])
    valid_start = min(train_end + embargo, valid_end)
    test_start = min(valid_end + embargo, n)
    train = np.flatnonzero((dates < train_boundary) & (target_end_dates < train_boundary))
    validation = np.flatnonzero(
        (np.arange(n) >= valid_start) & (dates < test_boundary) & (target_end_dates < test_boundary)
    )
    test = np.flatnonzero((np.arange(n) >= test_start) & target_end_dates.notna())
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Split rỗng; hãy giảm embargo hoặc kiểm tra dữ liệu")
    return TemporalSplit(train, validation, test, train_boundary, test_boundary)


class PurgedWalkForwardSplit:
    """Expanding-window folds purged by label end date and row embargo."""

    def __init__(self, n_splits: int = 3, embargo: int = 20, min_train: int = 300):
        self.n_splits = n_splits
        self.embargo = embargo
        self.min_train = min_train

    def split(self, dates: pd.Series, target_end_dates: pd.Series):
        n = len(dates)
        available = n - self.min_train - self.embargo
        fold_size = max(1, available // self.n_splits)
        for fold in range(self.n_splits):
            valid_start = self.min_train + fold * fold_size
            valid_end = n if fold == self.n_splits - 1 else min(n, valid_start + fold_size)
            boundary = pd.Timestamp(dates.iloc[valid_start])
            train = np.flatnonzero((np.arange(n) < valid_start) & (target_end_dates < boundary))
            validation = np.arange(min(valid_start + self.embargo, valid_end), valid_end)
            if len(train) and len(validation):
                yield train, validation
