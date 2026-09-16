import pandas as pd

from dmc2014_benchmark import numeric_history_view


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
