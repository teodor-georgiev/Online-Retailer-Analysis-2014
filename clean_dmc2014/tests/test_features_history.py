import numpy as np
import pandas as pd

from dmc2014.features import (
    FeatureConfig,
    build_training_features,
    build_validation_features,
)


def frame_with_history():
    return pd.DataFrame(
        {
            "orderItemID": [1, 2, 3, 4],
            "orderDate": pd.to_datetime(
                ["2012-01-01", "2012-01-01", "2012-01-02", "2012-01-03"]
            ),
            "deliveryDate": pd.to_datetime(
                ["2012-01-03", "2012-01-03", "2012-01-04", "2012-01-05"]
            ),
            "itemID": [10, 11, 10, 10],
            "size": ["M", "M", "M", "M"],
            "color": ["black", "black", "black", "black"],
            "manufacturerID": [5, 5, 5, 5],
            "price": [20.0, 25.0, 22.0, 24.0],
            "customerID": [100, 100, 100, 100],
            "salutation": ["Mrs"] * 4,
            "dateOfBirth": pd.to_datetime(["1980-01-01"] * 4),
            "state": ["NRW"] * 4,
            "creationDate": pd.to_datetime(["2011-01-01"] * 4),
            "returnShipment": [1, 1, 0, 1],
        }
    )


def test_training_history_excludes_same_date_and_future_labels():
    config = FeatureConfig(
        history_groups=(("customerID",),),
        smoothing=0.0,
        recency_groups=(("customerID",),),
    )
    feature_set = build_training_features(frame_with_history(), config)
    x = feature_set.X

    assert x["hist_customerID_count"].tolist() == [0.0, 0.0, 2.0, 3.0]
    assert np.allclose(
        x["hist_customerID_return_rate"].to_numpy(),
        [0.5, 0.5, 1.0, 2 / 3],
    )
    assert x["prior_customerID_row_count"].tolist() == [0.0, 0.0, 2.0, 3.0]
    assert np.isnan(x["days_since_customerID_seen"].iloc[0])
    assert np.isnan(x["days_since_customerID_seen"].iloc[1])
    assert x["days_since_customerID_seen"].iloc[2] == 1.0
    assert x["days_since_customerID_seen"].iloc[3] == 1.0


def test_future_target_change_does_not_change_earlier_training_features():
    config = FeatureConfig(
        history_groups=(("customerID",),),
        smoothing=5.0,
        recency_groups=(),
    )
    original = frame_with_history()
    changed = original.copy()
    changed.loc[3, "returnShipment"] = 0
    a = build_training_features(original, config).X.iloc[:3]
    b = build_training_features(changed, config).X.iloc[:3]
    pd.testing.assert_frame_equal(a, b)


def test_validation_target_history_uses_only_prior_labeled_history():
    history = frame_with_history().iloc[:3].copy()
    valid = frame_with_history().iloc[3:].copy()
    config = FeatureConfig(
        history_groups=(("customerID",),),
        smoothing=0.0,
        recency_groups=(("customerID",),),
    )

    first = build_validation_features(history, valid, config)
    mutated = valid.copy()
    mutated["returnShipment"] = 0
    second = build_validation_features(history, mutated, config)

    pd.testing.assert_frame_equal(first.X, second.X)
    assert first.X["hist_customerID_count"].tolist() == [3.0]
    assert np.allclose(first.X["hist_customerID_return_rate"], [2 / 3])
    assert first.y.tolist() == [1]
    assert second.y.tolist() == [0]


def test_profile_flags_keep_training_and_validation_columns_aligned():
    frame = frame_with_history()
    history = frame.iloc[:3].copy()
    valid = frame.iloc[3:].copy()
    config = FeatureConfig(
        history_groups=(),
        recency_groups=(("customerID",),),
        user_profiles=True,
        product_profiles=True,
    )

    train_features = build_training_features(history, config)
    valid_features = build_validation_features(history, valid, config)

    assert train_features.X.columns.tolist() == valid_features.X.columns.tolist()
    assert "user_prior_cumulative_spend" in train_features.X.columns
    assert "item_prior_unique_customers" in train_features.X.columns
    assert "manufacturer_prior_unique_items" in train_features.X.columns


def test_validation_profile_features_ignore_validation_labels():
    frame = frame_with_history()
    history = frame.iloc[:2].copy()
    valid = frame.iloc[2:].copy().reset_index(drop=True)
    config = FeatureConfig(
        history_groups=(),
        recency_groups=(),
        user_profiles=True,
        product_profiles=True,
    )

    first = build_validation_features(history, valid, config)
    mutated = valid.copy()
    mutated["returnShipment"] = 1 - mutated["returnShipment"]
    second = build_validation_features(history, mutated, config)

    pd.testing.assert_frame_equal(first.X, second.X)
    assert first.y.tolist() != second.y.tolist()
