from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

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


def dmc_score(y_true: Sequence[float], prediction: Sequence[float]) -> float:
    """Return the Data Mining Cup 2014 absolute-error point total."""
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    if truth.shape != pred.shape:
        raise ValueError(f"shape mismatch: {truth.shape} != {pred.shape}")
    return float(np.abs(truth - pred).sum())


def load_competition_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and competition predictors without touching released test labels."""
    root = Path(data_dir)
    train = pd.read_csv(root / "orders_train.txt", sep=";")
    test = pd.read_csv(root / "orders_class.txt", sep=";")
    return train, test


def load_realclass(data_dir: str | Path) -> pd.DataFrame:
    """Load released April labels; call only after model selection is frozen."""
    return pd.read_csv(Path(data_dir) / "orders_realclass.txt", sep=";")


def split_train_validation(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split original training rows into history through February and March validation."""
    data = frame.copy()
    data["orderDate"] = pd.to_datetime(data["orderDate"], errors="raise")
    validation_start = pd.Timestamp("2013-03-01")
    validation_end = pd.Timestamp("2013-04-01")
    valid_mask = data["orderDate"].between(validation_start, validation_end, inclusive="left")
    train = data.loc[data["orderDate"] < validation_start].copy()
    valid = data.loc[valid_mask].copy()
    return train, valid


def _as_datetime(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    values = frame[column].replace({"?": pd.NA, "": pd.NA})
    return pd.to_datetime(values, errors="coerce")


def build_row_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Create prediction-time row features without using return labels."""
    output = frame.drop(columns=["returnShipment"], errors="ignore").copy()

    order_date = _as_datetime(frame, "orderDate")
    delivery_date = _as_datetime(frame, "deliveryDate")
    birth_date = _as_datetime(frame, "dateOfBirth")
    creation_date = _as_datetime(frame, "creationDate")

    output["order_year"] = order_date.dt.year.astype("float64")
    output["order_month"] = order_date.dt.month.astype("float64")
    output["order_day"] = order_date.dt.day.astype("float64")
    output["order_dayofweek"] = order_date.dt.dayofweek.astype("float64")
    output["order_dayofyear"] = order_date.dt.dayofyear.astype("float64")
    output["delivery_year"] = delivery_date.dt.year.astype("float64")
    output["delivery_month"] = delivery_date.dt.month.astype("float64")
    output["delivery_dayofweek"] = delivery_date.dt.dayofweek.astype("float64")
    output["delivery_missing"] = delivery_date.isna().astype("int8")
    output["delivery_delay_days"] = (delivery_date - order_date).dt.total_seconds() / 86400.0
    output["customer_age_years"] = (order_date - birth_date).dt.total_seconds() / (86400.0 * 365.2425)
    output["account_age_days"] = (order_date - creation_date).dt.total_seconds() / 86400.0

    if "price" in frame.columns:
        price = pd.to_numeric(frame["price"], errors="coerce")
        output["price"] = price
        output["log1p_price"] = np.log1p(price.clip(lower=0))

    output = output.drop(columns=["orderDate", "deliveryDate", "dateOfBirth", "creationDate"], errors="ignore")

    categorical = [column for column in CATEGORICAL_COLUMNS if column in output.columns]
    for column in categorical:
        values = output[column].astype("string").fillna("__MISSING__")
        output[column] = values.replace({"?": "__MISSING__"}).astype(str)

    return output, categorical


def _history_prefix(columns: tuple[str, ...]) -> str:
    return "_x_".join(columns)


def add_history_features(
    history: pd.DataFrame,
    target: pd.DataFrame,
    group_specs: Iterable[tuple[str, ...]],
    smoothing: float = 20.0,
) -> pd.DataFrame:
    """Map history-only counts and smoothed return rates onto target rows."""
    if "returnShipment" not in history.columns:
        raise ValueError("history must contain returnShipment")
    if smoothing < 0:
        raise ValueError("smoothing must be non-negative")

    output = target.copy()
    prior = float(history["returnShipment"].mean())

    for columns in group_specs:
        columns = tuple(columns)
        if not columns:
            raise ValueError("group specs must not be empty")
        missing = [column for column in columns if column not in history.columns or column not in target.columns]
        if missing:
            raise KeyError(f"missing grouping columns: {missing}")

        stats = (
            history.groupby(list(columns), dropna=False, observed=True)["returnShipment"]
            .agg(["sum", "count"])
            .reset_index()
        )
        stats["return_rate"] = (stats["sum"] + smoothing * prior) / (stats["count"] + smoothing)

        prefix = _history_prefix(columns)
        count_name = f"hist_{prefix}_count"
        rate_name = f"hist_{prefix}_return_rate"
        mapped = target[list(columns)].merge(
            stats[list(columns) + ["count", "return_rate"]],
            how="left",
            on=list(columns),
            sort=False,
        )
        output[count_name] = mapped["count"].fillna(0).astype("int64").to_numpy()
        output[rate_name] = mapped["return_rate"].fillna(prior).astype(float).to_numpy()

    return output


def choose_best_result(results: Sequence[dict]) -> dict:
    """Return the candidate with the lowest validation DMC point total."""
    if not results:
        raise ValueError("results must not be empty")
    return min(results, key=lambda result: float(result["validation_points"]))
