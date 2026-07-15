"""Point, classification, calibration, interval, and tail-risk metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_recall_fscore_support,
    r2_score,
)

from .targets import CLASS_NAMES


def point_metrics(
    actual_return, predicted_return, actual_price, predicted_price, insample_return=None
) -> dict[str, float]:
    actual_return = np.asarray(actual_return, dtype=float)
    predicted_return = np.asarray(predicted_return, dtype=float)
    actual_price = np.asarray(actual_price, dtype=float)
    predicted_price = np.asarray(predicted_price, dtype=float)
    errors = actual_return - predicted_return
    scale = np.mean(np.abs(np.diff(np.asarray(insample_return if insample_return is not None else actual_return))))
    return {
        "mae_return": float(mean_absolute_error(actual_return, predicted_return)),
        "rmse_return": float(np.sqrt(mean_squared_error(actual_return, predicted_return))),
        "median_absolute_error_return": float(median_absolute_error(actual_return, predicted_return)),
        "r2_return": float(r2_score(actual_return, predicted_return)),
        "directional_accuracy": float(np.mean(np.sign(actual_return) == np.sign(predicted_return))),
        "return_correlation": float(np.corrcoef(actual_return, predicted_return)[0, 1])
        if np.std(predicted_return) > 0
        else 0.0,
        "mean_signed_error": float(np.mean(predicted_return - actual_return)),
        "mae_price": float(mean_absolute_error(actual_price, predicted_price)),
        "rmse_price": float(np.sqrt(mean_squared_error(actual_price, predicted_price))),
        "smape_price": float(
            np.mean(
                2
                * np.abs(predicted_price - actual_price)
                / np.maximum(np.abs(predicted_price) + np.abs(actual_price), 1e-8)
            )
        ),
        "mase_return": float(np.mean(np.abs(errors)) / max(scale, 1e-12)),
    }


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    predicted = probabilities.argmax(axis=1)
    observed = np.array([CLASS_NAMES.index(str(label)) for label in labels])
    confidence = probabilities.max(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    error = 0.0
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence > left) & (confidence <= right)
        if mask.any():
            error += mask.mean() * abs(np.mean(predicted[mask] == observed[mask]) - confidence[mask].mean())
    return float(error)


def classification_metrics(labels, probabilities) -> tuple[dict[str, float], pd.DataFrame, np.ndarray]:
    labels = np.asarray(labels).astype(str)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = np.array(CLASS_NAMES)[probabilities.argmax(axis=1)]
    one_hot = np.column_stack([(labels == name).astype(int) for name in CLASS_NAMES])
    multiclass_log_loss = float(-np.mean(np.sum(one_hot * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1)))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=CLASS_NAMES, zero_division=0
    )
    false_risk_on = (
        float(np.mean(np.isin(predictions[np.isin(labels, ["Bear", "Stress"])], ["Bull", "Sideway"])))
        if np.isin(labels, ["Bear", "Stress"]).any()
        else np.nan
    )
    confidence = probabilities.max(axis=1).clip(1e-6, 1 - 1e-6)
    correctness = (predictions == labels).astype(int)
    if np.unique(correctness).size > 1:
        calibration_model = LogisticRegression(C=1e6, solver="lbfgs").fit(
            np.log(confidence / (1 - confidence)).reshape(-1, 1), correctness
        )
        calibration_slope = float(calibration_model.coef_[0, 0])
        calibration_intercept = float(calibration_model.intercept_[0])
    else:
        calibration_slope = np.nan
        calibration_intercept = np.nan
    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=CLASS_NAMES, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, labels=CLASS_NAMES, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "log_loss": multiclass_log_loss,
        "ece": expected_calibration_error(labels, probabilities),
        "recall_bear": float(recall[CLASS_NAMES.index("Bear")]),
        "recall_stress": float(recall[CLASS_NAMES.index("Stress")]),
        "false_risk_on": false_risk_on,
        "calibration_slope": calibration_slope,
        "calibration_intercept": calibration_intercept,
    }
    per_class = pd.DataFrame(
        {"class": CLASS_NAMES, "precision": precision, "recall": recall, "f1": f1, "support": support}
    )
    matrix = confusion_matrix(labels, predictions, labels=CLASS_NAMES)
    return metrics, per_class, matrix


def interval_metrics(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray, level: float) -> dict[str, float]:
    actual, lower, upper = map(lambda x: np.asarray(x, dtype=float), (actual, lower, upper))
    alpha = 1 - level
    coverage = np.mean((actual >= lower) & (actual <= upper))
    width = upper - lower
    score = width + 2 / alpha * (lower - actual) * (actual < lower) + 2 / alpha * (actual - upper) * (actual > upper)
    return {
        "level": level,
        "coverage": float(coverage),
        "coverage_error": float(coverage - level),
        "average_width": float(width.mean()),
        "normalized_width": float(width.mean() / max(np.mean(np.abs(actual)), 1e-8)),
        "interval_score": float(score.mean()),
        "winkler_score": float(score.mean()),
    }


def historical_var_tests(actual: np.ndarray, var_forecast: np.ndarray, alpha: float = 0.05) -> dict[str, float]:
    violations = np.asarray(actual) < np.asarray(var_forecast)
    n, x = len(violations), int(violations.sum())
    rate = x / max(n, 1)
    likelihood_null = max((1 - alpha) ** (n - x) * alpha**x, 1e-300)
    likelihood_alt = max((1 - rate) ** (n - x) * max(rate, 1e-12) ** x, 1e-300)
    statistic = -2 * np.log(likelihood_null / likelihood_alt)
    transitions = np.zeros((2, 2), dtype=int)
    for previous, current in zip(violations[:-1], violations[1:], strict=True):
        transitions[int(previous), int(current)] += 1
    p01 = transitions[0, 1] / max(transitions[0].sum(), 1)
    p11 = transitions[1, 1] / max(transitions[1].sum(), 1)
    transition_likelihood = max(
        (1 - p01) ** transitions[0, 0]
        * max(p01, 1e-12) ** transitions[0, 1]
        * (1 - p11) ** transitions[1, 0]
        * max(p11, 1e-12) ** transitions[1, 1],
        1e-300,
    )
    independent_likelihood = max(
        (1 - rate) ** transitions[:, 0].sum() * max(rate, 1e-12) ** transitions[:, 1].sum(),
        1e-300,
    )
    christoffersen = float(-2 * np.log(independent_likelihood / transition_likelihood))
    return {
        "var_exceedance_rate": rate,
        "kupiec_lr": float(statistic),
        "kupiec_pvalue": float(1 - chi2.cdf(statistic, 1)),
        "christoffersen_independence_lr": christoffersen,
        "christoffersen_independence_pvalue": float(1 - chi2.cdf(christoffersen, 1)),
    }
