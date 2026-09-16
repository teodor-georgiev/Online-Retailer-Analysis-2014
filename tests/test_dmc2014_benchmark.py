import numpy as np
import pandas as pd

from dmc2014_benchmark import (
    add_history_features,
    choose_best_result,
    dmc_score,
    split_train_validation,
)


def test_dmc_score_is_sum_absolute_error():
    y = np.array([0.0, 1.0, 1.0])
    p = np.array([0.2, 0.7, 0.9])
    assert np.isclose(dmc_score(y, p), 0.6)


def test_split_uses_march_2013_as_validation_only():
    frame = pd.DataFrame(
        {
            "orderDate": ["2013-02-28", "2013-03-01", "2013-03-31"],
            "returnShipment": [0, 1, 0],
        }
    )
    train, valid = split_train_validation(frame)
    assert train["orderDate"].dt.strftime("%Y-%m-%d").tolist() == ["2013-02-28"]
    assert valid["orderDate"].dt.strftime("%Y-%m-%d").tolist() == ["2013-03-01", "2013-03-31"]


def test_history_features_use_history_only_and_fallback_to_prior():
    history = pd.DataFrame(
        {
            "customerID": [1, 1, 2],
            "returnShipment": [1, 0, 0],
        }
    )
    target = pd.DataFrame({"customerID": [1, 3]})
    out = add_history_features(history, target, [("customerID",)], smoothing=2.0)
    prior = history["returnShipment"].mean()
    expected_seen = (1 + 2.0 * prior) / (2 + 2.0)
    assert np.isclose(out.loc[0, "hist_customerID_return_rate"], expected_seen)
    assert np.isclose(out.loc[1, "hist_customerID_return_rate"], prior)
    assert out.loc[1, "hist_customerID_count"] == 0


def test_candidate_selection_prefers_lower_dmc_score():
    results = [
        {"name": "a", "validation_points": 10.0},
        {"name": "b", "validation_points": 8.0},
    ]
    assert choose_best_result(results)["name"] == "b"
