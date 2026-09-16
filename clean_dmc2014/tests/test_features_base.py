import numpy as np
import pandas as pd

from dmc2014.features import build_base_features


def sample_frame():
    return pd.DataFrame(
        {
            "orderItemID": [1, 2, 3],
            "orderDate": pd.to_datetime(["2013-01-10", "2013-01-10", "2013-01-20"]),
            "deliveryDate": pd.to_datetime(["2013-01-13", "2013-01-14", None]),
            "itemID": [10, 11, 10],
            "size": ["M", "L", "M"],
            "color": ["black", "red", "black"],
            "manufacturerID": [5, 5, 5],
            "price": [20.0, 30.0, 25.0],
            "customerID": [100, 100, 100],
            "salutation": ["Mrs", "Mrs", "Mrs"],
            "dateOfBirth": pd.to_datetime(["1980-01-10"] * 3),
            "state": ["NRW", "NRW", "NRW"],
            "creationDate": pd.to_datetime(["2012-01-10"] * 3),
            "returnShipment": [0, 1, 0],
        }
    )


def test_base_features_are_target_free_and_include_calendar_age_delivery():
    features, categorical = build_base_features(sample_frame())
    assert "returnShipment" not in features.columns
    assert "orderDate" not in features.columns
    assert "deliveryDate" not in features.columns
    assert features["order_month"].tolist() == [1, 1, 1]
    assert features["delivery_delay_days"].iloc[:2].tolist() == [3.0, 4.0]
    assert np.isnan(features["delivery_delay_days"].iloc[2])
    assert features["delivery_missing"].tolist() == [0, 0, 1]
    assert 32.9 < features["customer_age_years"].iloc[0] < 33.1
    assert features["account_age_days"].iloc[0] == 366.0
    assert "customerID" in categorical
    assert "itemID" in categorical


def test_base_features_compute_basket_structure_without_targets():
    features, _ = build_base_features(sample_frame())
    assert features["basket_item_count"].tolist() == [2.0, 2.0, 1.0]
    assert features["basket_total_price"].tolist() == [50.0, 50.0, 25.0]
    assert features["basket_mean_price"].tolist() == [25.0, 25.0, 25.0]
    assert features["basket_unique_items"].tolist() == [2.0, 2.0, 1.0]
    assert features["basket_unique_sizes"].tolist() == [2.0, 2.0, 1.0]
    assert features["price_minus_basket_mean"].tolist() == [-5.0, 5.0, 0.0]
