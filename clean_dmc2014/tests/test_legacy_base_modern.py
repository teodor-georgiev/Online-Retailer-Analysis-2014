import numpy as np
import pandas as pd

from dmc2014.legacy_base_modern import build_legacy_127_features


RAW_COLUMNS = [
    "order_item_id",
    "order_date",
    "delivery_date",
    "item_id",
    "item_size",
    "item_color",
    "brand_id",
    "item_price",
    "user_id",
    "user_title",
    "user_dob",
    "user_state",
    "user_reg_date",
    "return",
]


def _frames():
    rows = [
        [1, "2012-04-01", "2012-04-03", 10, "38", "black", 3, 29.9, 1, "Mrs", "1980-01-01", "North Rhine-Westphalia", "2011-01-01", 0],
        [2, "2012-04-01", "2012-04-04", 10, "M", "red", 3, 39.9, 1, "Mrs", "1980-01-01", "North Rhine-Westphalia", "2011-01-01", 1],
        [3, "2012-04-05", "2012-04-07", 11, "42", "blue", 4, 49.9, 1, "Mrs", "1980-01-01", "North Rhine-Westphalia", "2011-01-01", 0],
        [4, "2012-04-02", "?", 12, "unsized", "black", 5, 19.9, 2, "Mr", "1975-06-15", "Bavaria", "2010-02-01", 1],
        [5, "2012-04-06", "2012-04-09", 12, "XL", "green", 5, 59.9, 2, "Mr", "1975-06-15", "Bavaria", "2010-02-01", 0],
        [6, "2012-04-08", "2012-04-10", 13, "44", "green", 6, 69.9, 3, "Mrs", "?", "Hesse", "2012-01-15", 1],
    ]
    train = pd.DataFrame(rows, columns=RAW_COLUMNS)
    competition = pd.DataFrame(
        [
            [1, "2013-04-01", "2013-04-03", 10, "S", "black", 3, 34.9, 1, "Mrs", "1980-01-01", "North Rhine-Westphalia", "2011-01-01"],
            [2, "2013-04-02", "2013-04-05", 14, "40", "yellow", 7, 44.9, 4, "Mr", "1988-03-03", "Lower Saxony", "2012-10-01"],
        ],
        columns=RAW_COLUMNS[:-1],
    )
    return train, competition


def test_modern_legacy_base_reaches_127_columns():
    train, competition = _frames()
    output = build_legacy_127_features(train, competition)

    assert output.shape == (len(train) + len(competition), 127)
    assert output["return"].notna().sum() == len(train)
    assert output["return"].isna().sum() == len(competition)
    assert "mode_item_id" in output.columns
    assert "delivery_time_user_id_mad" in output.columns


def test_modern_legacy_base_predictors_do_not_depend_on_target_values():
    train, competition = _frames()
    changed = train.copy()
    changed["return"] = 1 - changed["return"]

    first = build_legacy_127_features(train, competition)
    second = build_legacy_127_features(changed, competition)

    pd.testing.assert_frame_equal(
        first.drop(columns=["return"]),
        second.drop(columns=["return"]),
        check_dtype=False,
    )
    np.testing.assert_array_equal(first["return"].dropna().to_numpy(), train["return"].to_numpy())
