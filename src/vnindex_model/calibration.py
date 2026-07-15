"""Temporal holdout selection for multiclass probability calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .targets import CLASS_NAMES


def _multiclass_log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    one_hot = np.column_stack([(labels == name).astype(int) for name in CLASS_NAMES])
    return float(-np.mean(np.sum(one_hot * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1)))


@dataclass
class ProbabilityCalibrator:
    method: str
    models: list[object | None]

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)
        if self.method == "none":
            return probabilities / probabilities.sum(axis=1, keepdims=True)
        calibrated = np.zeros_like(probabilities)
        for column, model in enumerate(self.models):
            if model is None:
                calibrated[:, column] = probabilities[:, column]
            elif self.method == "sigmoid":
                logit = np.log(probabilities[:, column] / (1 - probabilities[:, column]))
                calibrated[:, column] = model.predict_proba(logit.reshape(-1, 1))[:, 1]
            else:
                calibrated[:, column] = model.predict(probabilities[:, column])
        return calibrated / np.maximum(calibrated.sum(axis=1, keepdims=True), 1e-12)


def fit_calibrator(probabilities: np.ndarray, labels: np.ndarray, method: str) -> ProbabilityCalibrator:
    models: list[object | None] = []
    for column, class_name in enumerate(CLASS_NAMES):
        target = (labels == class_name).astype(int)
        if np.unique(target).size < 2 or method == "none":
            models.append(None)
        elif method == "sigmoid":
            x = np.log(
                np.clip(probabilities[:, column], 1e-6, 1 - 1e-6) / np.clip(1 - probabilities[:, column], 1e-6, 1)
            )
            models.append(LogisticRegression(C=1.0, solver="lbfgs", random_state=0).fit(x.reshape(-1, 1), target))
        elif method == "isotonic":
            models.append(IsotonicRegression(out_of_bounds="clip").fit(probabilities[:, column], target))
        else:
            raise ValueError(f"Calibration method không hỗ trợ: {method}")
    return ProbabilityCalibrator(method, models)


def select_temporal_calibration(
    probabilities: np.ndarray, labels: np.ndarray
) -> tuple[ProbabilityCalibrator, list[dict[str, float | str]]]:
    """Select on the latter temporal slice, then refit the winner on all validation OOS records."""
    split = max(int(len(labels) * 0.6), 1)
    if split >= len(labels) or len(labels) < 40:
        calibrator = fit_calibrator(probabilities, labels, "none")
        return calibrator, [{"method": "none", "validation_log_loss": _multiclass_log_loss(labels, probabilities)}]
    comparison: list[dict[str, float | str]] = []
    for method in ["none", "sigmoid", "isotonic"]:
        candidate = fit_calibrator(probabilities[:split], labels[:split], method)
        transformed = candidate.transform(probabilities[split:])
        comparison.append({"method": method, "validation_log_loss": _multiclass_log_loss(labels[split:], transformed)})
    selected = min(comparison, key=lambda item: float(item["validation_log_loss"]))["method"]
    return fit_calibrator(probabilities, labels, str(selected)), comparison
