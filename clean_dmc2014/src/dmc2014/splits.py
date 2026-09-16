from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalFold:
    name: str
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp


def default_folds() -> list[TemporalFold]:
    return [
        TemporalFold(
            name="2013-01",
            train_end=pd.Timestamp("2013-01-01"),
            valid_start=pd.Timestamp("2013-01-01"),
            valid_end=pd.Timestamp("2013-02-01"),
        ),
        TemporalFold(
            name="2013-02",
            train_end=pd.Timestamp("2013-02-01"),
            valid_start=pd.Timestamp("2013-02-01"),
            valid_end=pd.Timestamp("2013-03-01"),
        ),
        TemporalFold(
            name="2013-03",
            train_end=pd.Timestamp("2013-03-01"),
            valid_start=pd.Timestamp("2013-03-01"),
            valid_end=pd.Timestamp("2013-04-01"),
        ),
    ]


def split_fold(
    frame: pd.DataFrame,
    fold: TemporalFold,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "orderDate" not in frame.columns:
        raise KeyError("orderDate is required for temporal splitting")
    order_date = pd.to_datetime(frame["orderDate"], errors="raise")
    train_mask = order_date < fold.train_end
    valid_mask = (order_date >= fold.valid_start) & (order_date < fold.valid_end)
    train = frame.loc[train_mask].copy()
    valid = frame.loc[valid_mask].copy()
    return train, valid
