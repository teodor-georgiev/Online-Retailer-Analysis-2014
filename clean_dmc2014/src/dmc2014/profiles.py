from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


REQUIRED_PROFILE_COLUMNS = (
    "orderDate",
    "customerID",
    "itemID",
    "manufacturerID",
    "size",
    "color",
    "state",
    "price",
)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"missing profile columns: {missing}")


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, REQUIRED_PROFILE_COLUMNS)
    data = frame.drop(columns=["returnShipment"], errors="ignore").copy().reset_index(drop=True)
    data["orderDate"] = pd.to_datetime(data["orderDate"], errors="raise")
    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    data["__profile_row__"] = np.arange(len(data), dtype=np.int64)
    return data


def _map_daily(
    source: pd.DataFrame,
    table: pd.DataFrame,
    keys: Sequence[str],
    value_columns: Sequence[str],
) -> pd.DataFrame:
    lookup = source[[*keys, "orderDate", "__profile_row__"]].merge(
        table[[*keys, "orderDate", *value_columns]],
        on=[*keys, "orderDate"],
        how="left",
        sort=False,
    )
    lookup = lookup.sort_values("__profile_row__", kind="stable")
    return lookup[list(value_columns)].reset_index(drop=True)


def _entity_daily_stats(source: pd.DataFrame, keys: tuple[str, ...]) -> pd.DataFrame:
    working = source[[*keys, "orderDate", "price"]].copy()
    working["__price_sq__"] = working["price"] ** 2
    daily = (
        working.groupby([*keys, "orderDate"], dropna=False, observed=True)
        .agg(
            row_count=("price", "size"),
            price_count=("price", "count"),
            price_sum=("price", "sum"),
            price_sumsq=("__price_sq__", "sum"),
            daily_price_min=("price", "min"),
            daily_price_max=("price", "max"),
        )
        .reset_index()
        .sort_values([*keys, "orderDate"], kind="stable")
        .reset_index(drop=True)
    )
    grouped = daily.groupby(list(keys), dropna=False, observed=True, sort=False)

    daily["prior_row_count"] = grouped["row_count"].cumsum() - daily["row_count"]
    daily["prior_price_count"] = grouped["price_count"].cumsum() - daily["price_count"]
    daily["prior_price_sum"] = grouped["price_sum"].cumsum() - daily["price_sum"]
    daily["prior_price_sumsq"] = grouped["price_sumsq"].cumsum() - daily["price_sumsq"]
    daily["prior_order_count"] = grouped.cumcount().astype(float)
    daily["previous_date"] = grouped["orderDate"].shift(1)

    running_min = grouped["daily_price_min"].cummin()
    running_max = grouped["daily_price_max"].cummax()
    daily["__running_min__"] = running_min
    daily["__running_max__"] = running_max
    daily["prior_price_min"] = daily.groupby(
        list(keys), dropna=False, observed=True, sort=False
    )["__running_min__"].shift(1)
    daily["prior_price_max"] = daily.groupby(
        list(keys), dropna=False, observed=True, sort=False
    )["__running_max__"].shift(1)

    first_date = grouped["orderDate"].transform("min")
    daily["first_prior_date"] = first_date.where(daily["prior_row_count"] > 0)

    price_count = daily["prior_price_count"].to_numpy(dtype=float)
    price_sum = daily["prior_price_sum"].to_numpy(dtype=float)
    price_sumsq = daily["prior_price_sumsq"].to_numpy(dtype=float)
    mean = np.divide(
        price_sum,
        price_count,
        out=np.full(len(daily), np.nan, dtype=float),
        where=price_count > 0,
    )
    variance = np.divide(
        price_sumsq,
        price_count,
        out=np.full(len(daily), np.nan, dtype=float),
        where=price_count > 0,
    ) - np.square(mean)
    variance = np.where(np.isnan(variance), np.nan, np.maximum(variance, 0.0))
    daily["prior_price_mean"] = mean
    daily["prior_price_std"] = np.sqrt(variance)
    return daily.drop(columns=["__running_min__", "__running_max__"])


def _prior_distinct_count(
    source: pd.DataFrame,
    base_daily: pd.DataFrame,
    keys: tuple[str, ...],
    distinct_column: str,
) -> pd.Series:
    pair_first = (
        source.groupby([*keys, distinct_column], dropna=False, observed=True)["orderDate"]
        .min()
        .rename("orderDate")
        .reset_index()
    )
    new_by_date = (
        pair_first.groupby([*keys, "orderDate"], dropna=False, observed=True)
        .size()
        .rename("new_distinct")
        .reset_index()
    )
    dates = base_daily[[*keys, "orderDate"]].merge(
        new_by_date,
        on=[*keys, "orderDate"],
        how="left",
        sort=False,
    )
    dates["new_distinct"] = dates["new_distinct"].fillna(0).astype(float)
    grouped = dates.groupby(list(keys), dropna=False, observed=True, sort=False)
    dates["prior_distinct"] = grouped["new_distinct"].cumsum() - dates["new_distinct"]
    mapped = _map_daily(source, dates, keys, ["prior_distinct"])
    return mapped["prior_distinct"].fillna(0.0)


def _entity_mapped(source: pd.DataFrame, keys: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = _entity_daily_stats(source, keys)
    values = [
        "prior_row_count",
        "prior_price_count",
        "prior_price_sum",
        "prior_order_count",
        "previous_date",
        "first_prior_date",
        "prior_price_min",
        "prior_price_max",
        "prior_price_mean",
        "prior_price_std",
    ]
    return daily, _map_daily(source, daily, keys, values)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> np.ndarray:
    num = numerator.to_numpy(dtype=float)
    den = denominator.to_numpy(dtype=float)
    return np.divide(
        num,
        den,
        out=np.zeros(len(numerator), dtype=float),
        where=den > 0,
    )


def _user_gap_features(source: pd.DataFrame, user_daily: pd.DataFrame) -> pd.DataFrame:
    daily = user_daily[["customerID", "orderDate"]].copy()
    grouped = daily.groupby("customerID", dropna=False, observed=True, sort=False)
    daily["gap_days"] = grouped["orderDate"].diff().dt.total_seconds() / 86400.0
    daily["__gap_count__"] = daily["gap_days"].notna().astype(float)
    daily["__gap_value__"] = daily["gap_days"].fillna(0.0)
    grouped = daily.groupby("customerID", dropna=False, observed=True, sort=False)
    running_count = grouped["__gap_count__"].cumsum()
    running_sum = grouped["__gap_value__"].cumsum()
    running_min = grouped["gap_days"].cummin()
    running_max = grouped["gap_days"].cummax()
    daily["__running_count__"] = running_count
    daily["__running_sum__"] = running_sum
    daily["__running_min__"] = running_min
    daily["__running_max__"] = running_max
    grouped = daily.groupby("customerID", dropna=False, observed=True, sort=False)
    prior_count = grouped["__running_count__"].shift(1).fillna(0.0)
    prior_sum = grouped["__running_sum__"].shift(1).fillna(0.0)
    prior_min = grouped["__running_min__"].shift(1)
    prior_max = grouped["__running_max__"].shift(1)
    daily["prior_gap_mean"] = np.divide(
        prior_sum.to_numpy(dtype=float),
        prior_count.to_numpy(dtype=float),
        out=np.full(len(daily), np.nan, dtype=float),
        where=prior_count.to_numpy(dtype=float) > 0,
    )
    daily["prior_gap_min"] = prior_min
    daily["prior_gap_max"] = prior_max
    return _map_daily(
        source,
        daily,
        ("customerID",),
        ["prior_gap_mean", "prior_gap_min", "prior_gap_max"],
    )


def _build_user_profiles(source: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=np.arange(len(source)))
    daily, mapped = _entity_mapped(source, ("customerID",))
    current_date = source["orderDate"].reset_index(drop=True)
    previous_date = pd.to_datetime(mapped["previous_date"], errors="coerce")
    first_prior_date = pd.to_datetime(mapped["first_prior_date"], errors="coerce")

    output["user_prior_row_count"] = mapped["prior_row_count"].fillna(0.0)
    output["user_prior_order_count"] = mapped["prior_order_count"].fillna(0.0)
    output["user_prior_unique_items"] = _prior_distinct_count(
        source, daily, ("customerID",), "itemID"
    )
    output["user_prior_unique_manufacturers"] = _prior_distinct_count(
        source, daily, ("customerID",), "manufacturerID"
    )
    output["user_prior_unique_sizes"] = _prior_distinct_count(
        source, daily, ("customerID",), "size"
    )
    output["user_prior_unique_colors"] = _prior_distinct_count(
        source, daily, ("customerID",), "color"
    )
    output["user_prior_cumulative_spend"] = mapped["prior_price_sum"].fillna(0.0)
    output["user_prior_price_mean"] = mapped["prior_price_mean"]
    output["user_prior_price_std"] = mapped["prior_price_std"]
    output["user_prior_price_min"] = mapped["prior_price_min"]
    output["user_prior_price_max"] = mapped["prior_price_max"]
    output["user_prior_avg_basket_item_count"] = _safe_ratio(
        mapped["prior_row_count"], mapped["prior_order_count"]
    )
    output["user_prior_avg_basket_total_value"] = _safe_ratio(
        mapped["prior_price_sum"], mapped["prior_order_count"]
    )
    output["user_lifetime_days"] = (
        current_date - first_prior_date
    ).dt.total_seconds() / 86400.0
    output["user_days_since_previous_purchase"] = (
        current_date - previous_date
    ).dt.total_seconds() / 86400.0

    gap_features = _user_gap_features(source, daily)
    output["user_prior_gap_mean"] = gap_features["prior_gap_mean"]
    output["user_prior_gap_min"] = gap_features["prior_gap_min"]
    output["user_prior_gap_max"] = gap_features["prior_gap_max"]

    _interaction_daily, item_interaction = _entity_mapped(
        source, ("customerID", "itemID")
    )
    _manufacturer_daily, manufacturer_interaction = _entity_mapped(
        source, ("customerID", "manufacturerID")
    )
    output["user_prior_item_count"] = item_interaction["prior_row_count"].fillna(0.0)
    output["user_prior_manufacturer_count"] = manufacturer_interaction[
        "prior_row_count"
    ].fillna(0.0)
    output["user_item_familiarity"] = _safe_ratio(
        output["user_prior_item_count"], output["user_prior_row_count"]
    )
    output["user_manufacturer_familiarity"] = _safe_ratio(
        output["user_prior_manufacturer_count"], output["user_prior_row_count"]
    )
    return output


def _build_product_profiles(source: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=np.arange(len(source)))
    current_date = source["orderDate"].reset_index(drop=True)

    item_daily, item = _entity_mapped(source, ("itemID",))
    item_previous = pd.to_datetime(item["previous_date"], errors="coerce")
    item_first = pd.to_datetime(item["first_prior_date"], errors="coerce")
    output["item_prior_sales_count"] = item["prior_row_count"].fillna(0.0)
    output["item_prior_order_count"] = item["prior_order_count"].fillna(0.0)
    output["item_prior_unique_customers"] = _prior_distinct_count(
        source, item_daily, ("itemID",), "customerID"
    )
    output["item_prior_unique_states"] = _prior_distinct_count(
        source, item_daily, ("itemID",), "state"
    )
    output["item_prior_cumulative_revenue"] = item["prior_price_sum"].fillna(0.0)
    output["item_prior_price_mean"] = item["prior_price_mean"]
    output["item_prior_price_std"] = item["prior_price_std"]
    output["item_prior_price_min"] = item["prior_price_min"]
    output["item_prior_price_max"] = item["prior_price_max"]
    output["item_days_since_previous_sale"] = (
        current_date - item_previous
    ).dt.total_seconds() / 86400.0
    output["item_days_since_first_sale"] = (
        current_date - item_first
    ).dt.total_seconds() / 86400.0
    output["item_prior_repeat_purchase_count"] = (
        output["item_prior_sales_count"] - output["item_prior_unique_customers"]
    ).clip(lower=0.0)
    output["item_prior_repeat_purchase_share"] = _safe_ratio(
        output["item_prior_repeat_purchase_count"], output["item_prior_sales_count"]
    )

    manufacturer_daily, manufacturer = _entity_mapped(source, ("manufacturerID",))
    manufacturer_previous = pd.to_datetime(manufacturer["previous_date"], errors="coerce")
    manufacturer_first = pd.to_datetime(manufacturer["first_prior_date"], errors="coerce")
    output["manufacturer_prior_sales_count"] = manufacturer["prior_row_count"].fillna(0.0)
    output["manufacturer_prior_order_count"] = manufacturer["prior_order_count"].fillna(0.0)
    output["manufacturer_prior_unique_customers"] = _prior_distinct_count(
        source, manufacturer_daily, ("manufacturerID",), "customerID"
    )
    output["manufacturer_prior_unique_items"] = _prior_distinct_count(
        source, manufacturer_daily, ("manufacturerID",), "itemID"
    )
    output["manufacturer_prior_unique_sizes"] = _prior_distinct_count(
        source, manufacturer_daily, ("manufacturerID",), "size"
    )
    output["manufacturer_prior_unique_colors"] = _prior_distinct_count(
        source, manufacturer_daily, ("manufacturerID",), "color"
    )
    output["manufacturer_prior_cumulative_revenue"] = manufacturer[
        "prior_price_sum"
    ].fillna(0.0)
    output["manufacturer_prior_price_mean"] = manufacturer["prior_price_mean"]
    output["manufacturer_prior_price_std"] = manufacturer["prior_price_std"]
    output["manufacturer_days_since_previous_sale"] = (
        current_date - manufacturer_previous
    ).dt.total_seconds() / 86400.0
    output["manufacturer_days_since_first_sale"] = (
        current_date - manufacturer_first
    ).dt.total_seconds() / 86400.0
    return output


def _build_profiles(
    source: pd.DataFrame,
    *,
    user_profiles: bool,
    product_profiles: bool,
) -> pd.DataFrame:
    if not user_profiles and not product_profiles:
        return pd.DataFrame(index=np.arange(len(source)))
    pieces: list[pd.DataFrame] = []
    if user_profiles:
        pieces.append(_build_user_profiles(source))
    if product_profiles:
        pieces.append(_build_product_profiles(source))
    return pd.concat(pieces, axis=1)


def build_training_profiles(
    frame: pd.DataFrame,
    *,
    user_profiles: bool,
    product_profiles: bool,
) -> pd.DataFrame:
    """Build predictor-only profiles using only dates strictly before each row."""
    source = _prepare(frame)
    return _build_profiles(
        source,
        user_profiles=user_profiles,
        product_profiles=product_profiles,
    ).reset_index(drop=True)


def build_validation_profiles(
    history: pd.DataFrame,
    target: pd.DataFrame,
    *,
    user_profiles: bool,
    product_profiles: bool,
) -> pd.DataFrame:
    """Build target-row profiles from history plus earlier target predictor rows."""
    if not user_profiles and not product_profiles:
        return pd.DataFrame(index=np.arange(len(target)))

    history_prepared = _prepare(history)
    target_prepared = _prepare(target)
    history_prepared["__profile_target__"] = False
    history_prepared["__profile_target_order__"] = -1
    target_prepared["__profile_target__"] = True
    target_prepared["__profile_target_order__"] = np.arange(len(target_prepared), dtype=np.int64)

    source = pd.concat([history_prepared, target_prepared], ignore_index=True, sort=False)
    source["__profile_row__"] = np.arange(len(source), dtype=np.int64)
    profiles = _build_profiles(
        source,
        user_profiles=user_profiles,
        product_profiles=product_profiles,
    )
    target_mask = source["__profile_target__"].to_numpy(dtype=bool)
    selected = profiles.loc[target_mask].copy()
    selected["__profile_target_order__"] = source.loc[
        target_mask, "__profile_target_order__"
    ].to_numpy(dtype=np.int64)
    selected = selected.sort_values("__profile_target_order__", kind="stable")
    return selected.drop(columns=["__profile_target_order__"]).reset_index(drop=True)