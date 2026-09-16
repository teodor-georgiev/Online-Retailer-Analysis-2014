from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from dmc2014_benchmark import DEFAULT_GROUP_SPECS, build_row_features


def _history_prefix(columns: tuple[str, ...]) -> str:
    return "_x_".join(columns)


def prepare_training_features_loo(
    frame: pd.DataFrame,
    group_specs: Iterable[tuple[str, ...]] = DEFAULT_GROUP_SPECS,
    smoothing: float = 20.0,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Build fast leave-one-out target-history features for model training.

    Every training row excludes its own target from each group statistic. The
    entire supplied training window may contribute to the remaining rows;
    validation/test rows are still encoded separately from labeled history.
    """
    if "returnShipment" not in frame.columns:
        raise ValueError("frame must contain returnShipment")
    if smoothing < 0:
        raise ValueError("smoothing must be non-negative")

    data = frame.copy().reset_index(drop=True)
    target = data["returnShipment"].to_numpy(dtype=int)
    prior = float(data["returnShipment"].mean()) if len(data) else 0.5

    for columns in group_specs:
        columns = tuple(columns)
        if not columns:
            raise ValueError("group specs must not be empty")
        missing = [column for column in columns if column not in data.columns]
        if missing:
            raise KeyError(f"missing grouping columns: {missing}")

        grouped = data.groupby(list(columns), dropna=False, observed=True)["returnShipment"]
        group_sum = grouped.transform("sum").astype(float)
        group_count = grouped.transform("count").astype(float)
        loo_count = group_count - 1.0
        numerator = group_sum - data["returnShipment"].astype(float) + smoothing * prior
        denominator = loo_count + smoothing
        rate = np.where(denominator > 0, numerator / denominator, prior)

        prefix = _history_prefix(columns)
        data[f"hist_{prefix}_count"] = loo_count.astype("int64")
        data[f"hist_{prefix}_return_rate"] = rate.astype(float)

    features, categorical = build_row_features(data)
    return features, target, categorical
