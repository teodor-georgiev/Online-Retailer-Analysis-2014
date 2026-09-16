from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


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
