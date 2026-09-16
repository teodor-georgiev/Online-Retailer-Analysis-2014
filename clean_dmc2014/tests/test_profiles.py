import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from dmc2014.profiles import build_training_profiles, build_validation_profiles


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "orderDate": pd.to_datetime(
                ["2012-01-01", "2012-01-01", "2012-01-03", "2012-01-05"]
            ),
            "customerID": [1, 1, 1, 2],
            "itemID": [10, 11, 10, 10],
            "manufacturerID": [100, 100, 100, 100],
            "size": ["M", "L", "M", "M"],
            "color": ["red", "blue", "red", "red"],
            "state": ["A", "A", "A", "B"],
            "price": [10.0, 20.0, 12.0, 14.0],
            "returnShipment": [0, 1, 1, 0],
        }
    )


def test_profiles_are_target_invariant():
    frame = _frame()
    first = build_training_profiles(frame, user_profiles=True, product_profiles=True)
    mutated = frame.copy()
    mutated["returnShipment"] = 1 - mutated["returnShipment"]
    second = build_training_profiles(mutated, user_profiles=True, product_profiles=True)
    assert_frame_equal(first, second)


def test_same_date_rows_do_not_update_each_other_but_later_rows_see_both():
    profiles = build_training_profiles(_frame(), user_profiles=True, product_profiles=True)

    assert profiles.loc[0, "user_prior_row_count"] == 0
    assert profiles.loc[1, "user_prior_row_count"] == 0
    assert profiles.loc[2, "user_prior_row_count"] == 2

    assert profiles.loc[0, "manufacturer_prior_sales_count"] == 0
    assert profiles.loc[1, "manufacturer_prior_sales_count"] == 0
    assert profiles.loc[2, "manufacturer_prior_sales_count"] == 2

    assert profiles.loc[0, "item_prior_sales_count"] == 0
    assert profiles.loc[2, "item_prior_sales_count"] == 1


def test_user_profile_contains_causal_spend_diversity_and_familiarity():
    profiles = build_training_profiles(_frame(), user_profiles=True, product_profiles=False)
    later = profiles.loc[2]

    assert later["user_prior_order_count"] == 1
    assert later["user_prior_unique_items"] == 2
    assert later["user_prior_unique_manufacturers"] == 1
    assert later["user_prior_cumulative_spend"] == 30.0
    assert later["user_prior_price_mean"] == 15.0
    assert later["user_prior_item_count"] == 1
    assert later["user_prior_manufacturer_count"] == 2
    assert later["user_item_familiarity"] == 0.5
    assert later["user_manufacturer_familiarity"] == 1.0
    assert later["user_lifetime_days"] == 2.0
    assert later["user_days_since_previous_purchase"] == 2.0


def test_product_profiles_capture_prior_customer_and_price_history():
    profiles = build_training_profiles(_frame(), user_profiles=False, product_profiles=True)
    later = profiles.loc[2]

    assert later["item_prior_sales_count"] == 1
    assert later["item_prior_unique_customers"] == 1
    assert later["item_prior_unique_states"] == 1
    assert later["item_prior_cumulative_revenue"] == 10.0
    assert later["item_prior_price_mean"] == 10.0
    assert later["item_days_since_previous_sale"] == 2.0

    assert later["manufacturer_prior_sales_count"] == 2
    assert later["manufacturer_prior_unique_items"] == 2
    assert later["manufacturer_prior_unique_sizes"] == 2
    assert later["manufacturer_prior_unique_colors"] == 2
    assert later["manufacturer_prior_cumulative_revenue"] == 30.0


def test_product_profiles_measure_true_prior_repeat_purchases():
    profiles = build_training_profiles(_frame(), user_profiles=False, product_profiles=True)

    # At 2012-01-03, item 10 has one prior sale by one customer: no repeat yet.
    assert profiles.loc[2, "item_prior_repeat_purchase_count"] == 0.0
    assert profiles.loc[2, "item_prior_repeat_purchase_share"] == 0.0

    # At 2012-01-05, item 10 has two prior sales but only one unique buyer,
    # so one prior sale is a repeat purchase.
    assert profiles.loc[3, "item_prior_sales_count"] == 2.0
    assert profiles.loc[3, "item_prior_unique_customers"] == 1.0
    assert profiles.loc[3, "item_prior_repeat_purchase_count"] == 1.0
    assert profiles.loc[3, "item_prior_repeat_purchase_share"] == 0.5


def test_validation_profiles_use_history_and_earlier_validation_predictors_only():
    history = _frame().iloc[:2].copy()
    validation = _frame().iloc[2:].copy().reset_index(drop=True)

    first = build_validation_profiles(
        history, validation, user_profiles=True, product_profiles=True
    )
    mutated = validation.copy()
    mutated["returnShipment"] = 1 - mutated["returnShipment"]
    second = build_validation_profiles(
        history, mutated, user_profiles=True, product_profiles=True
    )
    assert_frame_equal(first, second)

    assert first.loc[0, "user_prior_row_count"] == 2
    assert first.loc[0, "item_prior_sales_count"] == 1
    # Customer 2 is new, but the product history includes the earlier validation row
    # from 2012-01-03 plus the original 2012-01-01 history row.
    assert first.loc[1, "item_prior_sales_count"] == 2
    assert first.loc[1, "item_days_since_previous_sale"] == 2.0


def test_disabled_profile_families_return_no_columns():
    frame = _frame()
    profiles = build_training_profiles(frame, user_profiles=False, product_profiles=False)
    assert profiles.shape == (len(frame), 0)
    assert np.array_equal(profiles.index.to_numpy(), np.arange(len(frame)))
