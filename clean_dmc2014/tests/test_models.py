import numpy as np
import pandas as pd

from dmc2014.features import FeatureSet
from dmc2014.models import fit_catboost, fit_lightgbm


def tiny_feature_sets():
    train_x = pd.DataFrame(
        {
            "price": [10, 11, 12, 13, 20, 21, 22, 23],
            "customerID": ["a", "a", "b", "b", "c", "c", "d", "d"],
            "hist_customerID_return_rate": [0.1, 0.2, 0.8, 0.7, 0.1, 0.2, 0.8, 0.7],
        }
    )
    train_y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    valid_x = pd.DataFrame(
        {
            "price": [10.5, 12.5, 20.5, 22.5],
            "customerID": ["a", "b", "c", "d"],
            "hist_customerID_return_rate": [0.15, 0.75, 0.15, 0.75],
        }
    )
    valid_y = np.array([0, 1, 0, 1])
    return (
        FeatureSet(train_x, train_y, ["customerID"]),
        FeatureSet(valid_x, valid_y, ["customerID"]),
    )


def assert_result(result):
    assert result.probabilities.shape == (4,)
    assert np.all((result.probabilities >= 0) & (result.probabilities <= 1))
    assert result.best_iteration >= 1
    assert result.runtime_seconds >= 0


def test_catboost_adapter_returns_probabilities():
    train, valid = tiny_feature_sets()
    result = fit_catboost(
        train,
        valid,
        {"iterations": 20, "depth": 3, "learning_rate": 0.1, "thread_count": 1},
    )
    assert_result(result)


def test_catboost_can_run_fixed_iterations_without_logloss_early_stop():
    train, valid = tiny_feature_sets()
    result = fit_catboost(
        train,
        valid,
        {
            "iterations": 7,
            "depth": 3,
            "learning_rate": 0.1,
            "thread_count": 1,
            "_early_stopping_rounds": None,
        },
    )
    assert_result(result)
    assert result.best_iteration == 7


def test_lightgbm_adapter_returns_probabilities():
    train, valid = tiny_feature_sets()
    result = fit_lightgbm(
        train,
        valid,
        {
            "n_estimators": 20,
            "num_leaves": 7,
            "learning_rate": 0.1,
            "min_child_samples": 1,
            "n_jobs": 1,
        },
    )
    assert_result(result)
