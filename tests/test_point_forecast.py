import numpy as np

from vnindex_model.point_forecast import apply_center_blend, select_validation_gated_center


def test_baseline_selection_uses_validation_and_shrinks_to_drift():
    actual = np.linspace(-0.03, 0.03, 200)
    drift = actual + 0.001
    ml = actual + 0.02
    selection = select_validation_gated_center(actual, ml, drift, [0.0, 0.5, 1.0], 0.01, True, 20)
    assert selection.alpha == 0.0
    assert selection.selected_center == "random_walk_drift_fallback"
    assert selection.validation_table["selected"].sum() == 1


def test_test_values_cannot_change_locked_alpha():
    validation_actual = np.array([-0.02, -0.01, 0.01, 0.02] * 40)
    validation_ml = validation_actual + 0.005
    validation_drift = validation_actual + 0.001
    one = select_validation_gated_center(validation_actual, validation_ml, validation_drift, [0.0, 0.5, 1.0])
    unrelated_test = np.full(100, 1e6)
    unrelated_test *= -1
    two = select_validation_gated_center(validation_actual, validation_ml, validation_drift, [0.0, 0.5, 1.0])
    assert one.alpha == two.alpha
    assert np.allclose(apply_center_blend(validation_ml, validation_drift, one.alpha), validation_drift)
