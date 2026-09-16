from __future__ import annotations

from itertools import chain, combinations
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


LOW_CARDINALITY = ["size", "item_color", "user_title", "user_state"]
REDUCE_MEMORY = LOW_CARDINALITY + ["mode_size", "mode_item_color"]
ITEM_DESCRIPTIONS = ["item_id", "size", "item_color", "brand_id"]
INTERACTION_DESCRIPTIONS = ["item_id", "brand_id", "item_color", "size"]


def mean_abs_deviation(series: pd.Series) -> float:
    """Pandas-3-compatible replacement for the removed Series.mad()."""
    numeric = pd.to_numeric(series, errors="coerce")
    mean = numeric.mean()
    return float((numeric - mean).abs().mean())


mean_abs_deviation.__name__ = "mad"


def _powerset(values: Sequence[str], r1: int, r2_exclusive: int) -> Iterable[tuple[str, ...]]:
    return chain.from_iterable(combinations(values, r) for r in range(r1, r2_exclusive))


def _stat_name(stat: str | Callable) -> str:
    return stat if isinstance(stat, str) else stat.__name__


def _aggregate_stats(
    frame: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    stats: list[str | Callable],
    prefix: str,
) -> pd.DataFrame:
    grouped = (
        frame[group_cols + [value_col]]
        .groupby(group_cols, dropna=True, observed=False)[value_col]
        .agg(stats)
        .reset_index()
    )
    rename = {stat_col: f"{prefix}{_stat_name(stat_col)}" for stat_col in grouped.columns[len(group_cols):]}
    return grouped.rename(columns=rename).round(2).fillna(0)


def _encode_legacy_categories(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in REDUCE_MEMORY:
        if column not in output:
            continue
        encoder = LabelEncoder()
        values = output[column].astype("string").fillna("__MISSING__").astype(str)
        output[column] = encoder.fit_transform(values)
    return output


def _add_user_recency_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame
    order_group = (
        output[["order_date", "user_id"]]
        .drop_duplicates()
        .sort_values(by=["user_id", "order_date"])
    )
    order_group["order_date_shifted"] = order_group.groupby("user_id", sort=False)["order_date"].shift(1)
    order_group["days_since_last_order"] = (
        order_group["order_date"] - order_group["order_date_shifted"]
    ).dt.days
    stats = ["mean", "median", "max", "min", "std", mean_abs_deviation, "skew"]
    stat_frame = _aggregate_stats(
        order_group,
        ["user_id"],
        "days_since_last_order",
        stats,
        "days_since_last_order_user_id_",
    )
    order_group = order_group.merge(stat_frame, on="user_id", how="left")
    order_group = order_group.drop(columns=["order_date_shifted"]).fillna(0)
    output = output.merge(order_group, on=["user_id", "order_date"], how="left")

    for current in _powerset(ITEM_DESCRIPTIONS, 1, len(ITEM_DESCRIPTIONS) + 1):
        current = list(current)
        if "item_id" in current and "brand_id" in current:
            continue
        keys = ["user_id"] + current
        group = (
            output[["order_date", "user_id"] + current]
            .drop_duplicates()
            .sort_values(by=keys + ["order_date"])
        )
        shifted = group.groupby(keys, sort=False)["order_date"].shift(1)
        name = "days_since_last_order_same_" + "_".join(current)
        group[name] = (group["order_date"] - shifted).dt.days.fillna(0)
        output = output.merge(group, on=["user_id", "order_date"] + current, how="left")

    delivery_group = (
        output[["delivery_date", "user_id"]]
        .drop_duplicates()
        .sort_values(by=["user_id", "delivery_date"])
    )
    delivery_group["delivery_date_shifted"] = delivery_group.groupby("user_id", sort=False)["delivery_date"].shift(1)
    delivery_group["days_since_last_delivery"] = (
        delivery_group["delivery_date"] - delivery_group["delivery_date_shifted"]
    ).dt.days
    stat_frame = _aggregate_stats(
        delivery_group,
        ["user_id"],
        "days_since_last_delivery",
        stats,
        "days_since_last_delivery_user_id_",
    )
    delivery_group = delivery_group.merge(stat_frame, on="user_id", how="left")
    delivery_group = delivery_group.drop(columns=["delivery_date_shifted"]).fillna(0)
    output = output.merge(delivery_group, on=["user_id", "delivery_date"], how="left")

    for current in _powerset(ITEM_DESCRIPTIONS, 1, len(ITEM_DESCRIPTIONS) + 1):
        current = list(current)
        if "item_id" in current and "brand_id" in current:
            continue
        keys = ["user_id"] + current
        group = (
            output[["delivery_date", "user_id"] + current]
            .drop_duplicates()
            .sort_values(by=keys + ["delivery_date"])
        )
        shifted = group.groupby(keys, sort=False)["delivery_date"].shift(1)
        name = "days_since_last_delivery_same_" + "_".join(current)
        group[name] = (group["delivery_date"] - shifted).dt.days.fillna(0)
        output = output.merge(group, on=["user_id", "delivery_date"] + current, how="left")
    return output


def _add_delivery_distribution_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame
    order_sum_desc = output["order_sum"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    output["order_sum_bins"] = pd.cut(
        output["order_sum"],
        bins=order_sum_desc[["min", "10%", "25%", "50%", "75%", "90%", "max"]].to_numpy(),
        include_lowest=True,
        duplicates="drop",
    ).cat.codes

    price_desc = output["item_price"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.95])
    output["item_price_bins"] = pd.cut(
        output["item_price"],
        bins=price_desc[["min", "10%", "25%", "50%", "75%", "95%", "max"]].to_numpy(),
        include_lowest=True,
        duplicates="drop",
    ).cat.codes
    output["age_bins"] = pd.cut(
        output["user_age"], bins=[20, 24, 29, 34, 44, 55, 65, 85], include_lowest=True
    ).cat.codes
    output["reg_age_bins"] = pd.cut(
        output["user_reg_age"],
        bins=[-1, 0, 0.1, 7, 14, 30, 60, 90, 180, 365, 730, 805],
        include_lowest=True,
    ).cat.codes

    stats = ["mean", "std", "min", "max", "median", mean_abs_deviation, "skew"]
    bin_columns = ["order_sum_bins", "item_price_bins", "age_bins", "reg_age_bins"]
    for column in bin_columns + ["order_weekday", "delivery_weekday"]:
        group = _aggregate_stats(
            output,
            [column],
            "delivery_time",
            stats,
            f"delivery_time_{column}_",
        )
        output = output.merge(group, on=column, how="left")
    output = output.drop(columns=bin_columns)

    interaction_cols = ["item_id", "brand_id", "item_color", "size", "delivery_weekday", "user_id"]
    for current in _powerset(interaction_cols, 2, 3):
        current = list(current)
        if "item_id" in current and "brand_id" in current:
            continue
        group = _aggregate_stats(
            output,
            current,
            "delivery_time",
            stats,
            "delivery_time_" + "_".join(current) + "_",
        )
        output = output.merge(group, on=current, how="left")

    for current in _powerset(interaction_cols, 3, 4):
        current = list(current)
        if "item_id" in current and "brand_id" in current:
            continue
        group = _aggregate_stats(
            output,
            current,
            "delivery_time",
            stats,
            "delivery_time_" + "_".join(current) + "_",
        )
        output = output.merge(group, on=current, how="left")
    return output


def _add_order_count_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame
    output["user_id_total_items_ordered"] = output.groupby("user_id", dropna=True)["user_state"].transform("count")

    daily = (
        output[["user_id", "user_state", "order_date"]]
        .groupby(["user_id", "order_date"], dropna=True)["user_state"]
        .count()
        .rename("daily_count")
        .reset_index()
        .sort_values(["user_id", "order_date"])
    )
    daily["user_id_total_items_ordered_cumsum"] = daily.groupby("user_id", sort=False)["daily_count"].cumsum()
    output = output.merge(
        daily[["user_id", "order_date", "user_id_total_items_ordered_cumsum"]],
        on=["user_id", "order_date"],
        how="left",
    )

    interaction_sets: list[list[str]] = []
    for current in _powerset(INTERACTION_DESCRIPTIONS, 1, len(INTERACTION_DESCRIPTIONS) + 1):
        current = list(current)
        if "item_id" in current and "brand_id" in current:
            continue
        interaction_sets.append(current)
        name = "user_id_" + "_".join(current) + "_total_items_ordered"
        output[name] = output.groupby(["user_id"] + current, dropna=True)["user_state"].transform("count")

    for current in interaction_sets:
        keys = ["user_id"] + current
        daily = (
            output[keys + ["order_date", "user_state"]]
            .groupby(keys + ["order_date"], dropna=True)["user_state"]
            .count()
            .rename("daily_count")
            .reset_index()
            .sort_values(keys + ["order_date"])
        )
        name = "user_id_" + "_".join(current) + "_cumsum_items_ordered"
        daily[name] = daily.groupby(keys, sort=False)["daily_count"].cumsum()
        output = output.merge(daily[keys + ["order_date", name]], on=keys + ["order_date"], how="left")

    for current in _powerset(INTERACTION_DESCRIPTIONS, 0, len(INTERACTION_DESCRIPTIONS) - 1):
        current = list(current)
        if "item_id" in current and "brand_id" in current:
            continue
        keys = ["user_id"] + current
        targets = [column for column in INTERACTION_DESCRIPTIONS if column not in current]
        grouped = output[keys + targets].groupby(keys, dropna=True)[targets].nunique()
        rename = {
            column: "user_id_" + column + "_nunique_" + "_".join(current)
            for column in grouped.columns
        }
        grouped = grouped.rename(columns=rename)
        output = output.join(grouped, on=keys, how="left")
    return output


def build_legacy_modern_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the notebook's final 340 feature additions on modern pandas.

    This is an audit/compatibility path. It intentionally mirrors the old
    transductive predictor aggregation semantics and never reads the target.
    Use the clean-room causal feature builder for unbiased model selection.
    """
    output = frame.copy()
    output["order_date"] = pd.to_datetime(output["order_date"], errors="coerce")
    output["delivery_date"] = pd.to_datetime(output["delivery_date"], errors="coerce")
    output["user_reg_date"] = pd.to_datetime(output["user_reg_date"], errors="coerce")
    output["user_dob"] = pd.to_datetime(output["user_dob"], errors="coerce")
    output = _encode_legacy_categories(output)
    original_columns = output.shape[1]
    output = _add_user_recency_features(output)
    output = _add_delivery_distribution_features(output)
    output = _add_order_count_features(output)
    added = output.shape[1] - original_columns
    if added != 340:
        raise RuntimeError(f"legacy-modern feature count drift: expected 340 additions, got {added}")
    return output
