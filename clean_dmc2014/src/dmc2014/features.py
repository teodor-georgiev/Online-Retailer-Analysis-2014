from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

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

DEFAULT_HISTORY_GROUPS = (
    ("customerID",),
    ("itemID",),
    ("manufacturerID",),
    ("size",),
    ("color",),
    ("state",),
    ("customerID", "manufacturerID"),
    ("customerID", "size"),
    ("itemID", "size"),
    ("manufacturerID", "itemID"),
)

DEFAULT_RECENCY_GROUPS = (
    ("customerID",),
    ("itemID",),
    ("manufacturerID",),
    ("customerID", "itemID"),
    ("customerID", "manufacturerID"),
)


@dataclass(frozen=True)
class FeatureConfig:
    history_groups: tuple[tuple[str, ...], ...] = DEFAULT_HISTORY_GROUPS
    smoothing: float = 20.0
    recency_groups: tuple[tuple[str, ...], ...] = DEFAULT_RECENCY_GROUPS


@dataclass
class FeatureSet:
    X: pd.DataFrame
    y: np.ndarray | None
    categorical: list[str]


def _datetime(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    return pd.to_datetime(frame[column], errors="coerce")


def _prefix(columns: Sequence[str]) -> str:
    return "_x_".join(columns)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"missing required columns: {missing}")


def _merge_values(
    target: pd.DataFrame,
    table: pd.DataFrame,
    keys: Sequence[str],
    value_columns: Sequence[str],
) -> pd.DataFrame:
    left = target[list(keys)].copy()
    left["__row_id__"] = np.arange(len(left))
    merged = left.merge(table, on=list(keys), how="left", sort=False)
    merged = merged.sort_values("__row_id__", kind="stable")
    return merged[list(value_columns)].reset_index(drop=True)


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


def _training_global_prior(data: pd.DataFrame) -> np.ndarray:
    daily = (
        data.groupby("orderDate", dropna=False, observed=True)["returnShipment"]
        .agg(["sum", "count"])
        .sort_index()
    )
    daily["prior_sum"] = daily["sum"].cumsum() - daily["sum"]
    daily["prior_count"] = daily["count"].cumsum() - daily["count"]
    daily["prior_rate"] = np.where(
        daily["prior_count"] > 0,
        daily["prior_sum"] / daily["prior_count"],
        0.5,
    )
    return data["orderDate"].map(daily["prior_rate"]).astype(float).to_numpy()


def _add_training_target_history(
    data: pd.DataFrame,
    output: pd.DataFrame,
    groups: Sequence[tuple[str, ...]],
    smoothing: float,
) -> None:
    if smoothing < 0:
        raise ValueError("smoothing must be non-negative")
    global_prior = _training_global_prior(data)

    for columns in groups:
        columns = tuple(columns)
        _require_columns(data, columns)
        prefix = _prefix(columns)
        keys = [*columns, "orderDate"]
        daily = (
            data.groupby(keys, dropna=False, observed=True)["returnShipment"]
            .agg(["sum", "count"])
            .reset_index()
            .sort_values(keys, kind="stable")
        )
        grouped = daily.groupby(list(columns), dropna=False, observed=True)
        daily["prior_sum"] = grouped["sum"].cumsum() - daily["sum"]
        daily["prior_count"] = grouped["count"].cumsum() - daily["count"]
        mapped = _merge_values(
            data,
            daily[keys + ["prior_sum", "prior_count"]],
            keys,
            ["prior_sum", "prior_count"],
        )
        prior_sum = mapped["prior_sum"].fillna(0.0).to_numpy(dtype=float)
        prior_count = mapped["prior_count"].fillna(0.0).to_numpy(dtype=float)
        denominator = prior_count + smoothing
        rate = np.where(
            denominator > 0,
            (prior_sum + smoothing * global_prior) / denominator,
            global_prior,
        )
        output[f"hist_{prefix}_count"] = prior_count
        output[f"hist_{prefix}_return_rate"] = rate


def _add_validation_target_history(
    history: pd.DataFrame,
    validation: pd.DataFrame,
    output: pd.DataFrame,
    groups: Sequence[tuple[str, ...]],
    smoothing: float,
) -> None:
    if smoothing < 0:
        raise ValueError("smoothing must be non-negative")
    prior = float(history["returnShipment"].mean()) if len(history) else 0.5

    for columns in groups:
        columns = tuple(columns)
        _require_columns(history, columns)
        _require_columns(validation, columns)
        prefix = _prefix(columns)
        stats = (
            history.groupby(list(columns), dropna=False, observed=True)["returnShipment"]
            .agg(["sum", "count"])
            .reset_index()
        )
        mapped = _merge_values(
            validation,
            stats,
            list(columns),
            ["sum", "count"],
        )
        prior_sum = mapped["sum"].fillna(0.0).to_numpy(dtype=float)
        prior_count = mapped["count"].fillna(0.0).to_numpy(dtype=float)
        denominator = prior_count + smoothing
        rate = np.where(
            denominator > 0,
            (prior_sum + smoothing * prior) / denominator,
            prior,
        )
        output[f"hist_{prefix}_count"] = prior_count
        output[f"hist_{prefix}_return_rate"] = rate


def _predictor_history_table(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    keys = [*columns, "orderDate"]
    daily = (
        frame.groupby(keys, dropna=False, observed=True)
        .size()
        .rename("row_count")
        .reset_index()
        .sort_values(keys, kind="stable")
    )
    grouped = daily.groupby(list(columns), dropna=False, observed=True)
    daily["prior_row_count"] = grouped["row_count"].cumsum() - daily["row_count"]
    daily["previous_seen_date"] = grouped["orderDate"].shift(1)
    return daily[keys + ["prior_row_count", "previous_seen_date"]]


def _add_predictor_history(
    source: pd.DataFrame,
    target: pd.DataFrame,
    output: pd.DataFrame,
    groups: Sequence[tuple[str, ...]],
) -> None:
    for columns in groups:
        columns = tuple(columns)
        _require_columns(source, columns)
        _require_columns(target, columns)
        prefix = _prefix(columns)
        table = _predictor_history_table(source, columns)
        keys = [*columns, "orderDate"]
        mapped = _merge_values(
            target,
            table,
            keys,
            ["prior_row_count", "previous_seen_date"],
        )
        count = mapped["prior_row_count"].fillna(0.0).to_numpy(dtype=float)
        previous = pd.to_datetime(mapped["previous_seen_date"], errors="coerce")
        current = pd.to_datetime(target["orderDate"], errors="coerce").reset_index(drop=True)
        days = (current - previous).dt.total_seconds() / 86400.0
        output[f"prior_{prefix}_row_count"] = count
        output[f"days_since_{prefix}_seen"] = days.to_numpy(dtype=float)


def build_training_features(
    train_frame: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> FeatureSet:
    config = config or FeatureConfig()
    data = train_frame.copy().reset_index(drop=True)
    if "returnShipment" not in data.columns:
        raise KeyError("training frame must contain returnShipment")
    data["orderDate"] = pd.to_datetime(data["orderDate"], errors="raise")

    output, categorical = build_base_features(data)
    _add_training_target_history(
        data,
        output,
        config.history_groups,
        config.smoothing,
    )
    _add_predictor_history(data, data, output, config.recency_groups)
    target = pd.to_numeric(data["returnShipment"], errors="raise").to_numpy(dtype=int)
    return FeatureSet(X=output.reset_index(drop=True), y=target, categorical=categorical)


def build_validation_features(
    history_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> FeatureSet:
    config = config or FeatureConfig()
    history = history_frame.copy().reset_index(drop=True)
    validation = validation_frame.copy().reset_index(drop=True)
    if "returnShipment" not in history.columns:
        raise KeyError("history frame must contain returnShipment")
    history["orderDate"] = pd.to_datetime(history["orderDate"], errors="raise")
    validation["orderDate"] = pd.to_datetime(validation["orderDate"], errors="raise")

    output, categorical = build_base_features(validation)
    _add_validation_target_history(
        history,
        validation,
        output,
        config.history_groups,
        config.smoothing,
    )

    predictor_source = pd.concat(
        [
            history.drop(columns=["returnShipment"], errors="ignore"),
            validation.drop(columns=["returnShipment"], errors="ignore"),
        ],
        ignore_index=True,
        sort=False,
    )
    _add_predictor_history(
        predictor_source,
        validation,
        output,
        config.recency_groups,
    )

    target = None
    if "returnShipment" in validation.columns:
        target = pd.to_numeric(
            validation["returnShipment"], errors="raise"
        ).to_numpy(dtype=int)
    return FeatureSet(X=output.reset_index(drop=True), y=target, categorical=categorical)
