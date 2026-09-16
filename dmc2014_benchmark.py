from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd


def dmc_score(y_true: Sequence[float], prediction: Sequence[float]) -> float:
    """Return the Data Mining Cup 2014 absolute-error point total."""
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    if truth.shape != pred.shape:
        raise ValueError(f"shape mismatch: {truth.shape} != {pred.shape}")
    return float(np.abs(truth - pred).sum())


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
