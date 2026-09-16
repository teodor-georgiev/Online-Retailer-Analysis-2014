import numpy as np
import pandas as pd

from dmc2014.rolling import (
    build_training_rolling_profiles,
    build_validation_rolling_profiles,
)


def _row(row_id: int, date: str, price: float, target: int = 0) -> dict:
    return {
        "orderItemID": row_id,
        "orderDate": pd.Timestamp(date),
        "itemID": 10,
        "manufacturerID": 5,
        "customerID": 100,
        "price": price,
        "returnShipment": target,
    }


def test_training_rolling_profiles_are_same_day_isolated_and_have_expected_stats():
    frame = pd.DataFrame(
        [
            _row(1, "2013-01-01", 10.0, 0),
            _row(2, "2013-01-01", 20.0, 1),
            _row(3, "2013-01-05", 30.0, 0),
            _row(4, "2013-02-10", 40.0, 1),
        ]
    )

    result = build_training_rolling_profiles(frame)

    assert result.shape == (4, 78)
    assert result.loc[0, "user_roll_7d_count"] == 0.0
    assert result.loc[1, "user_roll_7d_count"] == 0.0
    assert result.loc[2, "user_roll_7d_count"] == 2.0
    assert result.loc[2, "user_roll_7d_price_mean"] == 15.0
    assert result.loc[2, "user_roll_7d_price_std"] == 5.0
    assert result.loc[2, "user_roll_7d_price_min"] == 10.0
    assert result.loc[2, "user_roll_7d_price_max"] == 20.0
    assert result.loc[3, "user_roll_30d_count"] == 0.0
    assert result.loc[3, "user_roll_90d_count"] == 3.0


def test_rolling_profiles_do_not_depend_on_target_values():
    frame = pd.DataFrame(
        [
            _row(1, "2013-01-01", 10.0, 0),
            _row(2, "2013-01-05", 20.0, 1),
            _row(3, "2013-01-10", 30.0, 0),
        ]
    )
    mutated = frame.copy()
    mutated["returnShipment"] = 1 - mutated["returnShipment"]

    left = build_training_rolling_profiles(frame)
    right = build_training_rolling_profiles(mutated)

    pd.testing.assert_frame_equal(left, right)


def test_validation_rolling_can_use_earlier_predictor_only_validation_dates_but_not_same_day():
    history = pd.DataFrame(
        [
            _row(1, "2013-01-01", 10.0, 0),
            _row(2, "2013-01-05", 20.0, 1),
        ]
    )
    validation = pd.DataFrame(
        [
            _row(3, "2013-01-10", 30.0, 0),
            _row(4, "2013-01-10", 40.0, 1),
            _row(5, "2013-01-20", 50.0, 0),
        ]
    )

    result = build_validation_rolling_profiles(history, validation)

    assert result.loc[0, "user_roll_7d_count"] == 1.0
    assert result.loc[1, "user_roll_7d_count"] == 1.0
    assert result.loc[2, "user_roll_30d_count"] == 4.0


def test_customer_last_order_and_gap_features_use_prior_completed_orders_only():
    frame = pd.DataFrame(
        [
            _row(1, "2013-01-01", 10.0),
            _row(2, "2013-01-01", 20.0),
            _row(3, "2013-01-05", 30.0),
            _row(4, "2013-01-10", 40.0),
            _row(5, "2013-01-20", 50.0),
        ]
    )

    result = build_training_rolling_profiles(frame)

    assert result.loc[2, "user_last3_basket_item_count_mean"] == 2.0
    assert result.loc[2, "user_last3_basket_value_mean"] == 30.0
    assert np.isnan(result.loc[2, "user_last3_gap_days_mean"])
    assert result.loc[3, "user_last3_gap_days_mean"] == 4.0
    assert result.loc[4, "user_last3_gap_days_mean"] == 4.5
