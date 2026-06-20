import numpy as np

from linkage_compare.stats import mcnemar_test


def test_mcnemar_identical_predictions_gives_p_one():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 1, 1, 0])
    result = mcnemar_test(y_true, y_pred, y_pred)
    assert result.n01 == 0
    assert result.n10 == 0
    assert result.p_value == 1.0


def test_mcnemar_detects_one_sided_difference():
    y_true = np.array([1] * 20)
    y_pred_a = np.array([0] * 20)  # always wrong
    y_pred_b = np.array([1] * 20)  # always right
    result = mcnemar_test(y_true, y_pred_a, y_pred_b)
    assert result.n01 == 20
    assert result.n10 == 0
    assert result.p_value < 0.01
