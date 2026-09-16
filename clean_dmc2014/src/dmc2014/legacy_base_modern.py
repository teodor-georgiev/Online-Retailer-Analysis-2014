from __future__ import annotations

from itertools import chain, combinations
from pathlib import Path
from typing import Iterable, Sequence
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .legacy_modern import mean_abs_deviation


CANONICAL_COLUMNS = [
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
ITEM_DESCRIPTIONS = ["item_id", "size", "item_color", "brand_id"]

# The notebook built this dictionary by sorting *non-returned* rows by state.
# It was only a one-to-one relabeling, so freeze the resulting mapping here
# instead of consulting the target during feature construction.
STATE_MAP = {
    "North Rhine-Westphalia": "NRW",
    "Lower Saxony": "Low-Saxony",
    "Bavaria": "Bayern",
    "Baden-Wuerttemberg": "Bad-Wue",
    "Hesse": "Hesse",
    "Rhineland-Palatinate": "S-Holstein",
    "Schleswig-Holstein": "Rhine-Pal",
    "Berlin": "Berlin",
    "Saxony": "Saxony",
    "Hamburg": "Hamburg",
    "Brandenburg": "Bburg",
    "Thuringia": "Thur",
    "Mecklenburg-Western Pomerania": "Mburg",
    "Bremen": "Bremen",
    "Saxony-Anhalt": "Saxony-A",
    "Saarland": "Saarland",
}


def _powerset(values: Sequence[str]) -> Iterable[tuple[str, ...]]:
    return chain.from_iterable(combinations(values, size) for size in range(1, len(values) + 1))


def load_legacy_raw_frames(zip_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load only train and competition predictors from the original DMC zip."""
    with ZipFile(zip_path) as archive:
        names = {Path(name).name.lower(): name for name in archive.namelist()}
        train = pd.read_csv(archive.open(names["orders_train.txt"]), sep=";")
        competition = pd.read_csv(archive.open(names["orders_class.txt"]), sep=";")
    train.columns = CANONICAL_COLUMNS
    competition.columns = CANONICAL_COLUMNS[:-1]
    return train, competition


def _normalize_size(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.rstrip("+").str.lower()
    named = {
        "xs": "xs",
        "s": "s",
        "m": "m",
        "l": "l",
        "xl": "xl",
        "xxl": "xxl",
        "xxxl": "xxxl",
        "xxxxl": "xxxl",
        "unsized": "unsized",
    }
    direct = text.map(named)
    numeric = pd.to_numeric(text, errors="coerce")
    buckets = pd.Series("unsized", index=series.index, dtype="string")
    buckets.loc[numeric.between(1, 36, inclusive="both")] = "xs"
    buckets.loc[numeric.gt(36) & numeric.le(38)] = "s"
    buckets.loc[numeric.gt(38) & numeric.le(40)] = "m"
    buckets.loc[numeric.gt(40) & numeric.le(42)] = "l"
    buckets.loc[numeric.gt(42) & numeric.le(44)] = "xl"
    buckets.loc[numeric.gt(44) & numeric.le(46)] = "xxl"
    buckets.loc[numeric.gt(46) & numeric.le(48)] = "xxxl"
    buckets.loc[numeric.gt(48)] = "unsized"
    return direct.fillna(buckets).astype(str)


def _prepare_combined(train: pd.DataFrame, competition: pd.DataFrame) -> pd.DataFrame:
    train = train.copy()
    competition = competition.copy()
    train.columns = CANONICAL_COLUMNS
    competition.columns = CANONICAL_COLUMNS[:-1]
    competition["return"] = np.nan
    frame = pd.concat([train, competition], ignore_index=True)

    frame["delivery_date"] = frame["delivery_date"].replace("?", pd.NA)
    frame["user_dob"] = frame["user_dob"].replace("?", pd.NA)
    frame["order_date"] = pd.to_datetime(frame["order_date"], format="%Y-%m-%d")
    frame["delivery_date"] = pd.to_datetime(frame["delivery_date"], format="%Y-%m-%d", errors="coerce")
    frame["user_dob"] = pd.to_datetime(frame["user_dob"], format="%Y-%m-%d", errors="coerce")
    frame["user_reg_date"] = pd.to_datetime(frame["user_reg_date"], format="%Y-%m-%d")

    filled_delivery = frame["delivery_date"].fillna(pd.Timestamp("2020-12-31"))
    delivery_time = (filled_delivery - frame["order_date"]).dt.days.astype(float)
    median_delivery = float(delivery_time.median())
    delivery_time = delivery_time.mask((delivery_time < 0) | (delivery_time > 1000), median_delivery)
    delivery_time = delivery_time.astype(int)
    frame["delivery_date"] = frame["order_date"] + pd.to_timedelta(delivery_time, unit="D")
    cap = int(delivery_time.quantile(0.98))
    frame["delivery_time"] = delivery_time.clip(upper=cap).astype(int)

    frame = frame.rename(columns={"item_size": "size"})
    frame["size"] = _normalize_size(frame["size"])
    frame["order_id"] = frame["order_date"].astype(str) + "_" + frame["user_id"].astype(str)

    age_years = (frame["order_date"] - frame["user_dob"]).dt.days / 365.2425
    median_age = float(age_years.median())
    age = age_years.fillna(median_age).round(0)
    age_median_rounded = float(age.median())
    age = age.mask((age < 20) | (age > 80), age_median_rounded)
    frame["user_age"] = age.astype(int)
    frame["user_state"] = frame["user_state"].map(STATE_MAP).fillna(frame["user_state"])
    frame["user_reg_age"] = (frame["order_date"] - frame["user_reg_date"]).dt.days.astype(int)
    return frame


def _add_date_and_order_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["order_weekday"] = output["order_date"].dt.weekday
    output["delivery_weekday"] = output["delivery_date"].dt.weekday
    output["order_month"] = output["order_date"].dt.month
    output["delivery_month"] = output["delivery_date"].dt.month
    output["order_day"] = output["order_date"].dt.day
    output["delivery_day"] = output["delivery_date"].dt.day
    output["order_week"] = output["order_date"].dt.isocalendar().week.astype(int)
    output["delivery_week"] = output["delivery_date"].dt.isocalendar().week.astype(int)

    output["order_item_count"] = output.groupby("order_id", dropna=False)["order_date"].transform("count")
    output["order_sum"] = output.groupby("order_id", dropna=False)["item_price"].transform("sum")
    output["average_item_price_order"] = (output["order_sum"] / output["order_item_count"]).round(2)

    for current in _powerset(ITEM_DESCRIPTIONS):
        current = list(current)
        if "item_id" in current and "brand_id" in current:
            continue
        same = "order_number_same_" + "_".join(current)
        different = "order_number_different_" + "_".join(current)
        keys = ["order_id"] + current
        output[same] = output.groupby(keys, dropna=False)["user_title"].transform("count")
        output[different] = output["order_item_count"] - output[same]
    return output


def _group_stats(
    frame: pd.DataFrame,
    key: str,
    value: str,
    stats: list,
    infix: str,
) -> pd.DataFrame:
    grouped = frame.groupby(key, dropna=False)[value].agg(stats).reset_index().round(2)
    rename = {}
    for column in grouped.columns[1:]:
        name = column if isinstance(column, str) else getattr(column, "__name__", str(column))
        rename[column] = f"{value}_{infix}_{name}"
    return grouped.rename(columns=rename).fillna(0)


def _add_encoding_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    order_unique = (
        output.groupby("order_id", dropna=False)
        .agg(
            order_item_id_nunique=("item_id", "nunique"),
            order_size_nunique=("size", "nunique"),
            order_brand_id_nunique=("brand_id", "nunique"),
            order_item_color_nunique=("item_color", "nunique"),
        )
        .reset_index()
    )
    output = output.merge(order_unique, on="order_id", how="left")

    by_item = (
        output.groupby(["order_id", "item_id"], dropna=False)
        .agg(
            order_item_id_color_nunique=("item_color", "nunique"),
            order_item_id_size_nunique=("size", "nunique"),
        )
        .reset_index()
    )
    output = output.merge(by_item, on=["order_id", "item_id"], how="left")

    by_brand = (
        output.groupby(["order_id", "brand_id"], dropna=False)
        .agg(
            order_brand_id_color_nunique=("item_color", "nunique"),
            order_brand_id_size_nunique=("size", "nunique"),
            order_brand_id_item_id_nunique=("item_id", "nunique"),
        )
        .reset_index()
    )
    output = output.merge(by_brand, on=["order_id", "brand_id"], how="left")

    price_stats = ["mean", "std", "min", "max", "sum", "count", "median", mean_abs_deviation]
    for key in ["item_id", "user_id", "brand_id"]:
        stats = _group_stats(output, key, "item_price", price_stats, key)
        output = output.merge(stats, on=key, how="left")

    for column in list(output.columns):
        if any(token in column for token in ["min", "max", "mean"]):
            if column.startswith("item_price_"):
                output["price-" + column] = output[column] - output["item_price"]

    mode_source = ["item_id", "size", "brand_id", "item_color"]
    modes = output.groupby("user_id", dropna=False)[mode_source].agg(lambda values: values.mode().iloc[0])
    modes = modes.add_prefix("mode_").reset_index()
    output = output.merge(modes, on="user_id", how="left")

    delivery_stats = ["mean", "std", "min", "max", "median", mean_abs_deviation]
    for key in ["item_id", "size", "brand_id", "item_color", "user_id"]:
        stats = _group_stats(output, key, "delivery_time", delivery_stats, key)
        output = output.merge(stats, on=key, how="left")
    return output


def build_legacy_127_features(train: pd.DataFrame, competition: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the notebook's 127-column checkpoint with modern pandas.

    This compatibility path intentionally preserves the notebook's transductive
    predictor aggregation semantics, but no feature is allowed to consult the
    return target. The causal clean-room pipeline remains the model-selection path.
    """
    output = _prepare_combined(train, competition)
    output = _add_date_and_order_features(output)
    if output.shape[1] != 51:
        raise RuntimeError(f"legacy processed feature count drift: expected 51, got {output.shape[1]}")
    output = _add_encoding_features(output)
    if output.shape[1] != 127:
        raise RuntimeError(f"legacy base feature count drift: expected 127, got {output.shape[1]}")
    return output


def build_legacy_127_from_zip(zip_path: str | Path) -> pd.DataFrame:
    train, competition = load_legacy_raw_frames(zip_path)
    return build_legacy_127_features(train, competition)
