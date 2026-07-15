import numpy as np

from vnindex_model.calibration import select_temporal_calibration
from vnindex_model.evaluation import classification_metrics, interval_metrics, point_metrics
from vnindex_model.targets import CLASS_NAMES


def test_point_metrics_perfect_prediction():
    actual = np.array([0.01, -0.02, 0.03])
    metrics = point_metrics(actual, actual, np.array([101, 98, 103]), np.array([101, 98, 103]))
    assert metrics["rmse_return"] == 0
    assert metrics["directional_accuracy"] == 1


def test_classification_metrics():
    labels = np.array(CLASS_NAMES * 5)
    probabilities = np.eye(4)[np.tile(np.arange(4), 5)] * 0.9 + 0.025
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    metrics, per_class, matrix = classification_metrics(labels, probabilities)
    assert metrics["macro_f1"] == 1
    assert matrix.trace() == len(labels)
    assert len(per_class) == 4


def test_interval_metrics_and_calibration():
    metric = interval_metrics(np.array([0, 1]), np.array([-1, 0]), np.array([1, 2]), 0.9)
    assert metric["coverage"] == 1
    labels = np.array(CLASS_NAMES * 20)
    probabilities = np.full((len(labels), 4), 0.25)
    calibrator, comparison = select_temporal_calibration(probabilities, labels)
    transformed = calibrator.transform(probabilities)
    assert np.allclose(transformed.sum(axis=1), 1)
    assert comparison
