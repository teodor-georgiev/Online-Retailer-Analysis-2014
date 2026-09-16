from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "orderDate",
    "customerID",
    "itemID",
    "manufacturerID",
    "price",
)
WINDOW_DAYS = (7, 30, 90)
ENTITY_SPECS = (
    ("customerID", "user"),
    ("itemID", "item"),
    ("manufacturerID", "manufacturer"),
)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"missing rolling columns: {missing}")


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, REQUIRED_COLUMNS)
    data = frame.drop(columns=["returnShipment"], errors="ignore").copy().reset_index(drop=True)
    data["orderDate"] = pd.to_datetime(data["orderDate"], errors="raise")
    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    data["__rolling_row__"] = np.arange(len(data), dtype=np.int64)
    return data


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> np.ndarray:
    num = pd.to_numeric(numerator, errors="coerce").to_numpy(dtype=float)
    den = pd.to_numeric(denominator, errors="coerce").to_numpy(dtype=float)
    return np.divide(
        num,
        den,
        out=np.full(len(num), np.nan, dtype=float),
        where=np.isfinite(den) & (np.abs(den) > 1e-12),
    )


def _daily_entity(source: pd.DataFrame, key: str) -> pd.DataFrame:
    working = source[[key, "orderDate", "price"]].copy()
    working["__price_sq__"] = working["price"] ** 2
    return (
        working.groupby([key, "orderDate"], dropna=False, observed=True)
        .agg(
            row_count=("price", "size"),
            price_count=("price", "count"),
            price_sum=("price", "sum"),
            price_sumsq=("__price_sq__", "sum"),
            price_min=("price", "min"),
            price_max=("price", "max"),
        )
        .reset_index()
        .sort_values([key, "orderDate"], kind="stable")
        .reset_index(drop=True)
    )


def _window_table(daily: pd.DataFrame, key: str, days: int) -> pd.DataFrame:
    indexed = daily.set_index("orderDate")
    sum_columns = ["row_count", "price_count", "price_sum", "price_sumsq"]
    summed = (
        indexed.groupby(key, dropna=False, observed=True, sort=False)[sum_columns]
        .rolling(f"{days}D", closed="left", min_periods=1)
        .sum()
        .reset_index()
    )
    bounded = (
        indexed.groupby(key, dropna=False, observed=True, sort=False)[
            ["price_min", "price_max"]
        ]
        .rolling(f"{days}D", closed="left", min_periods=1)
        .agg({"price_min": "min", "price_max": "max"})
        .reset_index()
    )
    table = summed.merge(bounded, on=[key, "orderDate"], how="left", sort=False)

    count = table["price_count"].to_numpy(dtype=float)
    total = table["price_sum"].to_numpy(dtype=float)
    sumsq = table["price_sumsq"].to_numpy(dtype=float)
    mean = np.divide(
        total,
        count,
        out=np.full(len(table), np.nan, dtype=float),
        where=count > 0,
    )
    variance = np.divide(
        sumsq,
        count,
        out=np.full(len(table), np.nan, dtype=float),
        where=count > 0,
    ) - np.square(mean)
    variance = np.where(np.isnan(variance), np.nan, np.maximum(variance, 0.0))

    table["count"] = table["row_count"].fillna(0.0).astype(float)
    table["mean"] = mean
    table["std"] = np.sqrt(variance)
    return table[
        [key, "orderDate", "count", "price_sum", "mean", "std", "price_min", "price_max"]
    ]


def _map_daily(
    source: pd.DataFrame,
    table: pd.DataFrame,
    key: str,
    value_columns: Sequence[str],
) -> pd.DataFrame:
    lookup = source[[key, "orderDate", "__rolling_row__"]].merge(
        table[[key, "orderDate", *value_columns]],
        on=[key, "orderDate"],
        how="left",
        sort=False,
    )
    lookup = lookup.sort_values("__rolling_row__", kind="stable")
    return lookup[list(value_columns)].reset_index(drop=True)


def _entity_features(source: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
    output = pd.DataFrame(index=np.arange(len(source)))
    daily = _daily_entity(source, key)
    current_price = source["price"].reset_index(drop=True)

    for days in WINDOW_DAYS:
        table = _window_table(daily, key, days)
        mapped = _map_daily(
            source,
            table,
            key,
            ["count", "price_sum", "mean", "std", "price_min", "price_max"],
        )
        stem = f"{prefix}_roll_{days}d"
        output[f"{stem}_count"] = mapped["count"].fillna(0.0)
        output[f"{stem}_price_sum"] = mapped["price_sum"].fillna(0.0)
        output[f"{stem}_price_mean"] = mapped["mean"]
        output[f"{stem}_price_std"] = mapped["std"]
        output[f"{stem}_price_min"] = mapped["price_min"]
        output[f"{stem}_price_max"] = mapped["price_max"]

    output[f"{prefix}_roll_7d_30d_count_ratio"] = _safe_ratio(
        output[f"{prefix}_roll_7d_count"], output[f"{prefix}_roll_30d_count"]
    )
    output[f"{prefix}_roll_30d_90d_count_ratio"] = _safe_ratio(
        output[f"{prefix}_roll_30d_count"], output[f"{prefix}_roll_90d_count"]
    )
    output[f"{prefix}_roll_30d_90d_price_mean_ratio"] = _safe_ratio(
        output[f"{prefix}_roll_30d_price_mean"],
        output[f"{prefix}_roll_90d_price_mean"],
    )
    std_30 = output[f"{prefix}_roll_30d_price_std"].to_numpy(dtype=float)
    mean_30 = output[f"{prefix}_roll_30d_price_mean"].to_numpy(dtype=float)
    output[f"{prefix}_roll_30d_price_zscore"] = np.divide(
        current_price.to_numpy(dtype=float) - mean_30,
        std_30,
        out=np.full(len(output), np.nan, dtype=float),
        where=np.isfinite(std_30) & (std_30 > 1e-12),
    )
    return output


def _prior_rolling_stat(
    values: pd.Series,
    groups: pd.Series,
    window: int,
    statistic: str,
) -> pd.Series:
    shifted = values.groupby(groups, dropna=False, sort=False).shift(1)
    rolling = shifted.groupby(groups, dropna=False, sort=False).rolling(
        window=window,
        min_periods=1,
    )
    if statistic == "mean":
        result = rolling.mean()
    elif statistic == "std":
        result = rolling.std(ddof=0)
    else:
        raise ValueError(f"unsupported rolling statistic: {statistic}")
    return result.reset_index(level=0, drop=True).reindex(values.index)


def _customer_order_features(source: pd.DataFrame) -> pd.DataFrame:
    daily = (
        source.groupby(["customerID", "orderDate"], dropna=False, observed=True)
        .agg(
            basket_item_count=("price", "size"),
            basket_value=("price", "sum"),
        )
        .reset_index()
        .sort_values(["customerID", "orderDate"], kind="stable")
        .reset_index(drop=True)
    )
    groups = daily["customerID"]
    daily["gap_days"] = (
        daily.groupby("customerID", dropna=False, observed=True, sort=False)["orderDate"]
        .diff()
        .dt.total_seconds()
        / 86400.0
    )

    value_columns: list[str] = []
    for window in (3, 5):
        for source_column, label in (
            ("basket_item_count", "basket_item_count"),
            ("basket_value", "basket_value"),
        ):
            for statistic in ("mean", "std"):
                name = f"user_last{window}_{label}_{statistic}"
                daily[name] = _prior_rolling_stat(
                    daily[source_column].astype(float), groups, window, statistic
                )
                value_columns.append(name)
        for statistic in ("mean", "std"):
            name = f"user_last{window}_gap_days_{statistic}"
            daily[name] = _prior_rolling_stat(
                daily["gap_days"].astype(float), groups, window, statistic
            )
            value_columns.append(name)

    return _map_daily(source, daily, "customerID", value_columns)


def _build(source: pd.DataFrame) -> pd.DataFrame:
    data = _prepare(source)
    blocks = [
        _entity_features(data, key, prefix)
        for key, prefix in ENTITY_SPECS
    ]
    blocks.append(_customer_order_features(data))
    result = pd.concat([block.reset_index(drop=True) for block in blocks], axis=1)
    if result.shape[1] != 78:
        raise RuntimeError(f"rolling feature contract changed: expected 78, got {result.shape[1]}")
    return result


def build_training_rolling_profiles(train_frame: pd.DataFrame) -> pd.DataFrame:
    """Build predictor-only rolling features using only dates before each training row."""
    return _build(train_frame)


def build_validation_rolling_profiles(
    history_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Build rolling validation features without reading validation targets.

    Earlier validation dates may contribute predictor-only information to later
    validation dates. Rows sharing the same date never contribute to each other.
    """
    history = history_frame.drop(columns=["returnShipment"], errors="ignore")
    validation = validation_frame.drop(columns=["returnShipment"], errors="ignore")
    combined = pd.concat([history, validation], ignore_index=True, sort=False)
    all_features = _build(combined)
    start = len(history)
    return all_features.iloc[start:].reset_index(drop=True)
