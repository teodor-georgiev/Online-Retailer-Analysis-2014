from __future__ import annotations

import numpy as np
import pandas as pd


CATEGORICAL_COLUMNS = [
    "itemID",
    "size",
    "color",
    "manufacturerID",
    "customerID",
    "salutation",
    "state",
]


def _datetime(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    return pd.to_datetime(frame[column], errors="coerce")


def build_base_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build target-free row, calendar, delivery, age, and basket features."""
    output = frame.drop(columns=["returnShipment"], errors="ignore").copy()

    order_date = _datetime(frame, "orderDate")
    delivery_date = _datetime(frame, "deliveryDate")
    birth_date = _datetime(frame, "dateOfBirth")
    creation_date = _datetime(frame, "creationDate")

    output["order_year"] = order_date.dt.year.astype(float)
    output["order_month"] = order_date.dt.month.astype(float)
    output["order_day"] = order_date.dt.day.astype(float)
    output["order_dayofweek"] = order_date.dt.dayofweek.astype(float)
    output["order_dayofyear"] = order_date.dt.dayofyear.astype(float)
    output["order_weekofyear"] = order_date.dt.isocalendar().week.astype(float)
    output["is_weekend"] = (order_date.dt.dayofweek >= 5).astype("int8")

    output["delivery_delay_days"] = (
        delivery_date - order_date
    ).dt.total_seconds() / 86400.0
    output["delivery_missing"] = delivery_date.isna().astype("int8")
    output["customer_age_years"] = (
        (order_date - birth_date).dt.total_seconds() / (86400.0 * 365.2425)
    )
    output["account_age_days"] = (
        (order_date - creation_date).dt.total_seconds() / 86400.0
    )

    price = pd.to_numeric(frame["price"], errors="coerce")
    output["price"] = price
    output["log1p_price"] = np.log1p(price.clip(lower=0))

    order_keys = [frame["customerID"], order_date]
    grouped_price = price.groupby(order_keys, dropna=False)
    output["basket_item_count"] = grouped_price.transform("size").astype(float)
    output["basket_total_price"] = grouped_price.transform("sum").astype(float)
    output["basket_mean_price"] = grouped_price.transform("mean").astype(float)
    output["basket_max_price"] = grouped_price.transform("max").astype(float)
    output["basket_min_price"] = grouped_price.transform("min").astype(float)

    for source, feature in [
        ("itemID", "basket_unique_items"),
        ("manufacturerID", "basket_unique_manufacturers"),
        ("size", "basket_unique_sizes"),
        ("color", "basket_unique_colors"),
    ]:
        output[feature] = frame[source].groupby(
            order_keys, dropna=False
        ).transform("nunique").astype(float)

    output["price_minus_basket_mean"] = price - output["basket_mean_price"]
    output["price_over_basket_mean"] = np.where(
        output["basket_mean_price"].abs() > 1e-12,
        price / output["basket_mean_price"],
        np.nan,
    )

    output = output.drop(
        columns=[
            "orderItemID",
            "orderDate",
            "deliveryDate",
            "dateOfBirth",
            "creationDate",
        ],
        errors="ignore",
    )

    categorical = [column for column in CATEGORICAL_COLUMNS if column in output]
    for column in categorical:
        values = output[column].astype("string").fillna("__MISSING__")
        output[column] = values.replace({"?": "__MISSING__"}).astype(str)

    return output, categorical
