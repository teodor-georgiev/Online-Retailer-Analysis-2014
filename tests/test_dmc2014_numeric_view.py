import pandas as pd

from dmc2014_views import compact_history_view, numeric_history_view


def test_numeric_history_view_drops_raw_ids_and_categoricals():
    features = pd.DataFrame(
        {
            "orderItemID": [1, 2],
            "customerID": ["c1", "c2"],
            "itemID": ["i1", "i2"],
            "price": [10.0, 20.0],
            "hist_customerID_count": [2, 3],
            "hist_customerID_return_rate": [0.25, 0.75],
        }
    )
    output, categorical = numeric_history_view(features, ["customerID", "itemID"])
    assert categorical == []
    assert output.columns.tolist() == [
        "price",
        "hist_customerID_count",
        "hist_customerID_return_rate",
    ]


def test_compact_history_view_drops_high_card_ids_but_keeps_small_categories():
    features = pd.DataFrame(
        {
            "orderItemID": [1],
            "customerID": ["c1"],
            "itemID": ["i1"],
            "manufacturerID": ["m1"],
            "size": ["M"],
            "color": ["blue"],
            "salutation": ["Mrs"],
            "state": ["NRW"],
            "price": [10.0],
            "hist_customerID_return_rate": [0.25],
        }
    )
    output, categorical = compact_history_view(
        features,
        ["customerID", "itemID", "manufacturerID", "size", "color", "salutation", "state"],
    )
    assert "customerID" not in output.columns
    assert "itemID" not in output.columns
    assert "orderItemID" not in output.columns
    assert categorical == ["manufacturerID", "size", "color", "salutation", "state"]
    assert output["price"].tolist() == [10.0]
    assert output["hist_customerID_return_rate"].tolist() == [0.25]
