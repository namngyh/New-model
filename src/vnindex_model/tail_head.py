"""Experimental hierarchical tail classifier with validation-only production gates."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def _select_threshold(labels: np.ndarray, probability: np.ndarray, minimum_precision: float) -> tuple[float, float, float]:
    precision, recall, thresholds = precision_recall_curve(labels, probability)
    candidates = np.flatnonzero(precision[:-1] >= float(minimum_precision))
    if len(candidates) == 0:
        return 1.0, 0.0, 0.0
    best = candidates[np.argmax(recall[candidates])]
    return float(thresholds[best]), float(precision[best]), float(recall[best])


def _fit_sigmoid(raw_probability: np.ndarray, labels: np.ndarray) -> LogisticRegression | None:
    if np.unique(labels).size < 2:
        return None
    clipped = np.clip(raw_probability, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return LogisticRegression(C=1.0, solver="lbfgs", random_state=0).fit(logit, labels)


def _calibrate(raw_probability: np.ndarray, calibrator: LogisticRegression | None) -> np.ndarray:
    if calibrator is None:
        return raw_probability
    clipped = np.clip(raw_probability, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return calibrator.predict_proba(logit)[:, 1]


def run_tail_head_experiment(
    x_train,
    labels_train: np.ndarray,
    x_validation,
    labels_validation: np.ndarray,
    x_test,
    labels_test: np.ndarray,
    seed: int = 55,
    n_estimators: int = 300,
    minimum_precision: float = 0.25,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, float | bool | str]]:
    """Try weighted temporal classifiers; return production probabilities only if gates pass."""
    imputer = SimpleImputer(strategy="median").fit(x_train)
    train = imputer.transform(x_train)
    validation = imputer.transform(x_validation)
    test = imputer.transform(x_test)
    y_train = np.isin(labels_train, ["Bear", "Stress"]).astype(int)
    y_validation = np.isin(labels_validation, ["Bear", "Stress"]).astype(int)
    y_test = np.isin(labels_test, ["Bear", "Stress"]).astype(int)
    candidates = {
        "class_weighted_rf": ("balanced", None),
        "balanced_block_weighting": (None, np.where(y_train == 1, max((y_train == 0).sum() / max(y_train.sum(), 1), 1), 1.0)),
        "regime_aware_weighting": (None, np.where(labels_train == "Stress", 5.0, np.where(labels_train == "Bear", 3.0, 1.0))),
    }
    rows: list[dict[str, float | str | bool]] = []
    models: dict[str, RandomForestClassifier] = {}
    calibrators: dict[str, LogisticRegression | None] = {}
    validation_split = max(int(len(y_validation) * 0.60), 1)
    for number, (name, (class_weight, sample_weight)) in enumerate(candidates.items()):
        model = RandomForestClassifier(
            n_estimators=int(n_estimators),
            max_depth=7,
            min_samples_leaf=20,
            max_features="sqrt",
            class_weight=class_weight,
            n_jobs=-1,
            random_state=int(seed) + number,
        ).fit(train, y_train, sample_weight=sample_weight)
        raw_probability = model.predict_proba(validation)[:, 1]
        calibrator = _fit_sigmoid(raw_probability[:validation_split], y_validation[:validation_split])
        probability = _calibrate(raw_probability, calibrator)
        selection_labels = y_validation[validation_split:]
        selection_probability = probability[validation_split:]
        threshold, precision, recall = _select_threshold(
            selection_labels, selection_probability, minimum_precision
        )
        auc = float(roc_auc_score(selection_labels, selection_probability))
        ap = float(average_precision_score(selection_labels, selection_probability))
        prevalence = float(selection_labels.mean())
        eligible = auc > 0.55 and ap > prevalence * 1.05 and precision >= minimum_precision
        rows.append(
            {
                "head": "tail_vs_non_tail",
                "candidate": name,
                "validation_roc_auc": auc,
                "validation_average_precision": ap,
                "validation_prevalence": prevalence,
                "threshold": threshold,
                "validation_precision": precision,
                "validation_recall": recall,
                "eligible": eligible,
                "probability_calibration": "sigmoid",
            }
        )
        models[name] = model
        calibrators[name] = calibrator
    table = pd.DataFrame(rows)
    eligible = table[(table["head"] == "tail_vs_non_tail") & table["eligible"]]
    if len(eligible):
        selected_row = eligible.sort_values(["validation_recall", "validation_average_precision"], ascending=False).iloc[0]
        selected_name = str(selected_row["candidate"])
        probability = _calibrate(models[selected_name].predict_proba(test)[:, 1], calibrators[selected_name])
        prediction = probability >= float(selected_row["threshold"])
        true_positive = np.sum(prediction & (y_test == 1))
        test_precision = float(true_positive / max(prediction.sum(), 1))
        test_recall = float(true_positive / max(y_test.sum(), 1))
        production = True
        reason = "validation gates passed"
    else:
        selected_row = table[table["head"] == "tail_vs_non_tail"].sort_values(
            "validation_average_precision", ascending=False
        ).iloc[0]
        selected_name = str(selected_row["candidate"])
        probability = _calibrate(models[selected_name].predict_proba(test)[:, 1], calibrators[selected_name])
        diagnostic_prediction = probability >= float(selected_row["threshold"])
        true_positive = np.sum(diagnostic_prediction & (y_test == 1))
        test_precision = float(true_positive / max(diagnostic_prediction.sum(), 1))
        test_recall = float(true_positive / max(y_test.sum(), 1))
        production = False
        reason = "tail head rejected by validation gates; existing multiclass model retained"
    table["selected"] = (table["head"] == "tail_vs_non_tail") & (table["candidate"] == selected_name)
    table["test_precision"] = np.nan
    table["test_recall"] = np.nan
    table.loc[table["selected"], "test_precision"] = test_precision
    table.loc[table["selected"], "test_recall"] = test_recall

    auxiliary_rows = []
    for head, subset, positive in [
        ("bear_vs_stress", ["Bear", "Stress"], "Stress"),
        ("bull_vs_sideway", ["Bull", "Sideway"], "Bull"),
    ]:
        train_mask = np.isin(labels_train, subset)
        validation_mask = np.isin(labels_validation, subset)
        test_mask = np.isin(labels_test, subset)
        auxiliary_train = (labels_train[train_mask] == positive).astype(int)
        auxiliary_validation = (labels_validation[validation_mask] == positive).astype(int)
        auxiliary_test = (labels_test[test_mask] == positive).astype(int)
        model = RandomForestClassifier(
            n_estimators=int(n_estimators),
            max_depth=7,
            min_samples_leaf=20,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=int(seed) + len(auxiliary_rows) + 20,
        ).fit(train[train_mask], auxiliary_train)
        raw_validation = model.predict_proba(validation[validation_mask])[:, 1]
        auxiliary_split = max(int(len(auxiliary_validation) * 0.60), 1)
        calibrator = _fit_sigmoid(raw_validation[:auxiliary_split], auxiliary_validation[:auxiliary_split])
        calibrated_validation = _calibrate(raw_validation, calibrator)
        selection_labels = auxiliary_validation[auxiliary_split:]
        selection_probability = calibrated_validation[auxiliary_split:]
        threshold, precision, recall = _select_threshold(selection_labels, selection_probability, 0.25)
        raw_test = model.predict_proba(test[test_mask])[:, 1]
        calibrated_test = _calibrate(raw_test, calibrator)
        test_prediction = calibrated_test >= threshold
        true_positive = np.sum(test_prediction & (auxiliary_test == 1))
        auxiliary_rows.append(
            {
                "head": head,
                "candidate": "class_weighted_rf",
                "validation_roc_auc": float(roc_auc_score(selection_labels, selection_probability)),
                "validation_average_precision": float(average_precision_score(selection_labels, selection_probability)),
                "validation_prevalence": float(selection_labels.mean()),
                "threshold": threshold,
                "validation_precision": precision,
                "validation_recall": recall,
                "eligible": False,
                "probability_calibration": "sigmoid",
                "selected": False,
                "test_precision": float(true_positive / max(test_prediction.sum(), 1)),
                "test_recall": float(true_positive / max(auxiliary_test.sum(), 1)),
            }
        )
    table = pd.concat([table, pd.DataFrame(auxiliary_rows)], ignore_index=True)
    summary = {
        "selected_candidate": selected_name,
        "production_enabled": production,
        "reason": reason,
        "test_tail_precision_at_locked_threshold": test_precision,
        "test_tail_recall_at_locked_threshold": test_recall,
    }
    return table, probability, summary
