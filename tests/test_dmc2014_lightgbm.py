import numpy as np
import pandas as pd

from dmc2014_benchmark import fit_lightgbm_candidate, lightgbm_configs


def test_lightgbm_configs_are_deterministic_and_named():
    configs = lightgbm_configs()
    assert len(configs) >= 2
    assert len({config["name"] for config in configs}) == len(configs)
    assert all(config["params"]["random_state"] == 42 for config in configs)


def test_fit_lightgbm_candidate_handles_categorical_columns():
    train_x = pd.DataFrame(
        {
            "customerID": ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e", "f", "f"],
            "price": [10, 11, 20, 21, 30, 31, 40, 41, 50, 51, 60, 61],
        }
    )
    train_y = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1])
    valid_x = pd.DataFrame({"customerID": ["a", "b", "g", "f"], "price": [12, 22, 32, 62]})
    valid_y = np.array([0, 1, 0, 1])

    model, probability, best_iteration = fit_lightgbm_candidate(
        train_x,
        train_y,
        valid_x,
        valid_y,
        ["customerID"],
        {
            "n_estimators": 30,
            "learning_rate": 0.1,
            "num_leaves": 7,
            "random_state": 42,
            "n_jobs": 2,
            "verbosity": -1,
        },
    )

    assert model is not None
    assert probability.shape == (4,)
    assert np.all((probability >= 0) & (probability <= 1))
    assert best_iteration >= 1
