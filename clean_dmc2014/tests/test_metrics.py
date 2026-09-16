import numpy as np

from dmc2014.metrics import best_threshold, dmc_points


def test_dmc_points_is_sum_absolute_error():
    y_true = np.array([0, 1, 1, 0], dtype=float)
    prediction = np.array([0.2, 0.7, 1.0, 0.4], dtype=float)
    assert np.isclose(dmc_points(y_true, prediction), 0.9)


def test_best_threshold_minimizes_hard_points():
    y_true = np.array([0, 0, 1, 1])
    probability = np.array([0.10, 0.44, 0.45, 0.90])
    threshold, points = best_threshold(
        y_true,
        probability,
        thresholds=[0.40, 0.45, 0.50],
    )
    assert threshold == 0.45
    assert points == 0.0


def test_best_threshold_prefers_value_closest_to_half_on_tie():
    y_true = np.array([0, 1])
    probability = np.array([0.2, 0.8])
    threshold, points = best_threshold(
        y_true,
        probability,
        thresholds=[0.4, 0.5, 0.6],
    )
    assert threshold == 0.5
    assert points == 0.0
