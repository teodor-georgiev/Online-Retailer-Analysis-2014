from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


HIGH_CARDINALITY_COLUMNS = {"customerID", "itemID"}


def numeric_history_view(
    features: pd.DataFrame,
    categorical_columns: Sequence[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Drop raw identifiers/categoricals while retaining engineered numeric history features."""
    drop_columns = ["orderItemID", *categorical_columns]
    output = features.drop(columns=drop_columns, errors="ignore").copy()
    non_numeric = output.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    if non_numeric:
        output = output.drop(columns=non_numeric)
    return output, []


def compact_history_view(
    features: pd.DataFrame,
    categorical_columns: Sequence[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Drop expensive raw IDs while preserving low-cardinality categoricals and engineered features."""
    output = features.drop(columns=["orderItemID", *HIGH_CARDINALITY_COLUMNS], errors="ignore").copy()
    categorical = [
        column
        for column in categorical_columns
        if column in output.columns and column not in HIGH_CARDINALITY_COLUMNS
    ]
    return output, categorical
