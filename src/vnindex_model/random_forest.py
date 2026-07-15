"""Global, augmented, and soft-gated regime-aware random forests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer

from .targets import CLASS_NAMES


@dataclass
class ForestBundle:
    classifier: RandomForestClassifier
    return_regressor: RandomForestRegressor
    normalized_regressor: RandomForestRegressor
    drawdown_regressor: RandomForestRegressor
    imputer: SimpleImputer
    feature_names: list[str]


def forest_parameters(config: dict, seed: int) -> dict:
    return {
        "n_estimators": int(config["n_estimators"]),
        "max_depth": config.get("max_depth", 7),
        "min_samples_leaf": int(config.get("min_samples_leaf", 20)),
        "max_features": config.get("max_features", "sqrt"),
        "max_samples": config.get("max_samples", 0.8),
        "bootstrap": True,
        "n_jobs": -1,
        "random_state": int(seed),
    }


def fit_forest_bundle(
    x_train,
    y_class,
    y_return,
    y_normalized,
    y_drawdown,
    feature_names: list[str],
    config: dict,
    seed: int,
    sample_weight: np.ndarray | None = None,
) -> ForestBundle:
    imputer = SimpleImputer(strategy="median", add_indicator=False).fit(x_train)
    x = imputer.transform(x_train)
    parameters = forest_parameters(config, seed)
    classifier = RandomForestClassifier(**parameters, class_weight="balanced_subsample").fit(
        x, y_class, sample_weight=sample_weight
    )
    regressors = []
    for target in [y_return, y_normalized, y_drawdown]:
        regressors.append(
            RandomForestRegressor(**parameters, criterion="squared_error").fit(x, target, sample_weight=sample_weight)
        )
    return ForestBundle(classifier, *regressors, imputer, feature_names)


def aligned_probabilities(bundle: ForestBundle, x) -> np.ndarray:
    raw = bundle.classifier.predict_proba(bundle.imputer.transform(x))
    output = np.zeros((len(raw), len(CLASS_NAMES)))
    for position, label in enumerate(bundle.classifier.classes_):
        output[:, CLASS_NAMES.index(str(label))] = raw[:, position]
    return output / np.maximum(output.sum(axis=1, keepdims=True), 1e-12)


def predict_bundle(bundle: ForestBundle, x) -> dict[str, np.ndarray]:
    transformed = bundle.imputer.transform(x)
    return {
        "probabilities": aligned_probabilities(bundle, x),
        "return": bundle.return_regressor.predict(transformed),
        "normalized_return": bundle.normalized_regressor.predict(transformed),
        "drawdown": bundle.drawdown_regressor.predict(transformed),
    }


@dataclass
class SoftGatedForest:
    global_bundle: ForestBundle
    experts: list[ForestBundle | None]
    fallback_states: list[int]


def fit_soft_gated_forest(
    x_train,
    regime_probabilities: np.ndarray,
    y_class,
    y_return,
    y_normalized,
    y_drawdown,
    feature_names: list[str],
    config: dict,
    seed: int,
) -> SoftGatedForest:
    global_bundle = fit_forest_bundle(x_train, y_class, y_return, y_normalized, y_drawdown, feature_names, config, seed)
    experts: list[ForestBundle | None] = []
    fallback_states: list[int] = []
    for state in range(regime_probabilities.shape[1]):
        weights = regime_probabilities[:, state]
        effective = weights.sum() ** 2 / max(np.square(weights).sum(), 1e-12)
        if (
            effective < max(80, 4 * config.get("min_samples_leaf", 20))
            or np.unique(y_class[weights > np.quantile(weights, 0.5)]).size < 2
        ):
            experts.append(None)
            fallback_states.append(state)
            continue
        experts.append(
            fit_forest_bundle(
                x_train,
                y_class,
                y_return,
                y_normalized,
                y_drawdown,
                feature_names,
                config,
                seed + state + 1,
                sample_weight=weights,
            )
        )
    return SoftGatedForest(global_bundle, experts, fallback_states)


def predict_soft_gated(model: SoftGatedForest, x, regime_probabilities: np.ndarray) -> dict[str, np.ndarray]:
    global_prediction = predict_bundle(model.global_bundle, x)
    output = {key: np.zeros_like(value, dtype=float) for key, value in global_prediction.items()}
    for state, expert in enumerate(model.experts):
        prediction = global_prediction if expert is None else predict_bundle(expert, x)
        weight = regime_probabilities[:, state]
        for key in output:
            output[key] += prediction[key] * (weight[:, None] if output[key].ndim == 2 else weight)
    output["probabilities"] /= np.maximum(output["probabilities"].sum(axis=1, keepdims=True), 1e-12)
    return output
