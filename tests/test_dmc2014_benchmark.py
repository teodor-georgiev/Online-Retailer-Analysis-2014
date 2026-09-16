import numpy as np
import pandas as pd

from dmc2014_benchmark import (
    add_history_features,
    build_row_features,
    candidate_configs,
    choose_best_result,
    dmc_score,
    fit_candidate,
    load_competition_data,
    load_realclass,
    prepare_prediction_features,
    prepare_training_features,
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


def test_build_row_features_derives_dates_and_normalizes_categories():
    frame = pd.DataFrame(
        {
            "orderItemID": [1],
            "orderDate": ["2013-03-10"],
            "deliveryDate": ["2013-03-13"],
            "itemID": [11],
            "size": ["M"],
            "color": ["blue"],
            "manufacturerID": [7],
            "price": [99.0],
            "customerID": [42],
            "salutation": ["Mrs"],
            "dateOfBirth": ["1983-03-10"],
            "state": ["NRW"],
            "creationDate": ["2012-03-10"],
        }
    )
    features, categorical = build_row_features(frame)
    assert features.loc[0, "delivery_delay_days"] == 3
    assert np.isclose(features.loc[0, "customer_age_years"], 30.0, atol=0.02)
    assert np.isclose(features.loc[0, "account_age_days"], 365.0, atol=1.0)
    assert features.loc[0, "order_month"] == 3
    assert features.loc[0, "customerID"] == "42"
    assert "customerID" in categorical
    assert "itemID" in categorical


def test_competition_loader_does_not_require_realclass(tmp_path):
    train = tmp_path / "orders_train.txt"
    test = tmp_path / "orders_class.txt"
    train.write_text("orderItemID;orderDate;returnShipment\n1;2013-02-01;0\n", encoding="utf-8")
    test.write_text("orderItemID;orderDate\n2;2013-04-01\n", encoding="utf-8")
    train_df, test_df = load_competition_data(tmp_path)
    assert train_df.shape == (1, 3)
    assert test_df.shape == (1, 2)


def test_realclass_loader_is_separate(tmp_path):
    real = tmp_path / "orders_realclass.txt"
    real.write_text("orderItemID;returnShipment\n2;1\n", encoding="utf-8")
    result = load_realclass(tmp_path)
    assert result.to_dict("records") == [{"orderItemID": 2, "returnShipment": 1}]


def test_training_history_expands_month_by_month_without_future_labels():
    frame = pd.DataFrame(
        {
            "orderItemID": [1, 2, 3],
            "orderDate": ["2012-04-10", "2012-05-10", "2012-06-10"],
            "customerID": [9, 9, 9],
            "returnShipment": [1, 0, 0],
        }
    )
    features, target, categorical = prepare_training_features(
        frame,
        group_specs=[("customerID",)],
        smoothing=0.0,
    )
    assert target.tolist() == [1, 0, 0]
    assert features["hist_customerID_count"].tolist() == [0, 1, 2]
    assert np.allclose(features["hist_customerID_return_rate"], [0.5, 1.0, 0.5])
    assert "customerID" in categorical


def test_prediction_features_use_all_supplied_history_but_no_target_label():
    history = pd.DataFrame(
        {
            "orderItemID": [1, 2],
            "orderDate": ["2012-04-10", "2012-05-10"],
            "customerID": [9, 9],
            "returnShipment": [1, 0],
        }
    )
    target = pd.DataFrame(
        {
            "orderItemID": [3],
            "orderDate": ["2012-06-10"],
            "customerID": [9],
        }
    )
    features, categorical = prepare_prediction_features(
        history,
        target,
        group_specs=[("customerID",)],
        smoothing=0.0,
    )
    assert features.loc[0, "hist_customerID_count"] == 2
    assert features.loc[0, "hist_customerID_return_rate"] == 0.5
    assert "returnShipment" not in features.columns
    assert "customerID" in categorical


def test_candidate_configs_are_named_and_unique():
    configs = candidate_configs()
    assert len(configs) >= 2
    assert len({config["name"] for config in configs}) == len(configs)
    assert all(config["params"]["random_seed"] == 42 for config in configs)


def test_fit_candidate_returns_probabilities_and_best_iteration():
    train_x = pd.DataFrame(
        {
            "customerID": ["a", "a", "b", "b", "c", "c", "d", "d"],
            "price": [10, 11, 20, 21, 30, 31, 40, 41],
        }
    )
    train_y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    valid_x = pd.DataFrame({"customerID": ["a", "b", "c", "d"], "price": [12, 22, 32, 42]})
    valid_y = np.array([0, 1, 1, 1])
    model, probability, best_iteration = fit_candidate(
        train_x,
        train_y,
        valid_x,
        valid_y,
        ["customerID"],
        {
            "iterations": 20,
            "depth": 3,
            "learning_rate": 0.1,
            "loss_function": "Logloss",
            "random_seed": 42,
            "allow_writing_files": False,
            "verbose": False,
        },
    )
    assert model is not None
    assert probability.shape == (4,)
    assert np.all((probability >= 0) & (probability <= 1))
    assert best_iteration >= 1
