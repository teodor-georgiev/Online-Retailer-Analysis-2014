import pandas as pd

from dmc2014.legacy_modern import build_legacy_modern_features, mean_abs_deviation


def _tiny_frame():
    rows = []
    for idx, (date, user, item, size, color, brand, returned) in enumerate(
        [
            ("2012-04-01", 1, 10, "s", "black", 3, 0),
            ("2012-04-02", 1, 10, "m", "black", 3, 1),
            ("2012-04-03", 1, 11, "m", "red", 4, 0),
            ("2012-04-01", 2, 11, "s", "red", 4, 1),
            ("2012-04-04", 2, 12, "l", "blue", 5, 0),
        ]
    ):
        rows.append(
            {
                "order_item_id": idx + 1,
                "order_date": date,
                "delivery_date": pd.Timestamp(date) + pd.Timedelta(days=2),
                "item_id": item,
                "size": size,
                "item_color": color,
                "brand_id": brand,
                "item_price": 20.0 + idx,
                "user_id": user,
                "user_title": "Mrs",
                "user_dob": "1980-01-01",
                "user_state": "NRW",
                "user_reg_date": "2012-01-01",
                "return": returned,
                "delivery_time": 2.0,
                "order_id": f"{date}_{user}",
                "user_age": 32,
                "user_reg_age": 90,
                "order_weekday": pd.Timestamp(date).dayofweek,
                "delivery_weekday": (pd.Timestamp(date) + pd.Timedelta(days=2)).dayofweek,
                "order_sum": 50.0 + idx,
                "mode_item_id": item,
                "mode_size": size,
                "mode_brand_id": brand,
                "mode_item_color": color,
            }
        )
    return pd.DataFrame(rows)


def test_mean_abs_deviation_matches_legacy_definition():
    series = pd.Series([1.0, 2.0, 5.0])
    expected = (series - series.mean()).abs().mean()
    assert mean_abs_deviation(series) == expected


def test_legacy_modern_features_do_not_depend_on_target_values():
    frame = _tiny_frame()
    changed = frame.copy()
    changed["return"] = 1 - changed["return"]

    first = build_legacy_modern_features(frame)
    second = build_legacy_modern_features(changed)

    pd.testing.assert_frame_equal(
        first.drop(columns=["return"]),
        second.drop(columns=["return"]),
        check_dtype=False,
    )


def test_legacy_modern_adds_340_columns():
    frame = _tiny_frame()
    output = build_legacy_modern_features(frame)
    assert output.shape[1] == frame.shape[1] + 340
