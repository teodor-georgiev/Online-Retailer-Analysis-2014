import pandas as pd

from dmc2014.features import FeatureConfig, build_training_features, build_validation_features


def _row(row_id: int, date: str, target: int) -> dict:
    return {
        "orderItemID": row_id,
        "orderDate": pd.Timestamp(date),
        "deliveryDate": pd.Timestamp(date) + pd.Timedelta(days=2),
        "itemID": 10 + row_id % 2,
        "size": "M",
        "color": "black",
        "manufacturerID": 5,
        "price": 20.0 + row_id,
        "customerID": 100 + row_id % 2,
        "salutation": "Mrs",
        "dateOfBirth": pd.Timestamp("1980-01-01"),
        "state": "NRW",
        "creationDate": pd.Timestamp("2011-01-01"),
        "returnShipment": target,
    }


def test_feature_builder_adds_same_rolling_columns_to_train_and_validation():
    history = pd.DataFrame(
        [
            _row(1, "2012-12-01", 0),
            _row(2, "2012-12-10", 1),
            _row(3, "2012-12-20", 0),
        ]
    )
    validation = pd.DataFrame(
        [
            _row(4, "2013-01-02", 1),
            _row(5, "2013-01-05", 0),
        ]
    )
    config = FeatureConfig(
        history_groups=(),
        recency_groups=(),
        rolling_profiles=True,
    )

    train = build_training_features(history, config)
    valid = build_validation_features(history, validation, config)

    rolling_columns = [column for column in train.X.columns if "_roll_" in column or "user_last" in column]
    assert len(rolling_columns) == 78
    assert rolling_columns == [column for column in valid.X.columns if "_roll_" in column or "user_last" in column]
    assert list(train.X.columns) == list(valid.X.columns)
