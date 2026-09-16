from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pandas as pd


TRAIN_COLUMNS = [
    "orderItemID",
    "orderDate",
    "deliveryDate",
    "itemID",
    "size",
    "color",
    "manufacturerID",
    "price",
    "customerID",
    "salutation",
    "dateOfBirth",
    "state",
    "creationDate",
    "returnShipment",
]
CLASS_COLUMNS = TRAIN_COLUMNS[:-1]
DATE_COLUMNS = ["orderDate", "deliveryDate", "dateOfBirth", "creationDate"]


def _read_member(archive: Path, member: str) -> pd.DataFrame:
    with ZipFile(archive) as zipped:
        try:
            with zipped.open(member) as handle:
                return pd.read_csv(handle, sep=";", na_values=["?", ""])
        except KeyError as exc:
            raise FileNotFoundError(f"{member} not found in {archive}") from exc


def _validate_columns(frame: pd.DataFrame, expected: list[str], name: str) -> None:
    actual = frame.columns.tolist()
    if actual != expected:
        raise ValueError(f"unexpected {name} schema: {actual}")


def _parse_dates(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in DATE_COLUMNS:
        output[column] = pd.to_datetime(output[column], errors="coerce")
    return output


def _validate_binary(values: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise")
    if numeric.isna().any():
        raise ValueError(f"{name} must be binary 0/1 and contain no missing values")
    invalid = ~numeric.isin([0, 1])
    if invalid.any():
        bad_values = sorted(numeric.loc[invalid].unique().tolist())
        raise ValueError(f"{name} must be binary 0/1; got {bad_values}")
    return numeric.astype("int8")


def load_train_and_class(zip_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load only model-selection-safe DMC files.

    This function deliberately never opens ``orders_realclass.txt``.
    """
    archive = Path(zip_path)
    train = _read_member(archive, "orders_train.txt")
    competition = _read_member(archive, "orders_class.txt")
    _validate_columns(train, TRAIN_COLUMNS, "training")
    _validate_columns(competition, CLASS_COLUMNS, "competition")

    train = _parse_dates(train)
    competition = _parse_dates(competition)
    train["returnShipment"] = _validate_binary(
        train["returnShipment"], "returnShipment"
    )
    return train, competition


def load_final_labels(zip_path: str | Path) -> pd.Series:
    """Explicitly load released April labels for final evaluation only."""
    archive = Path(zip_path)
    frame = _read_member(archive, "orders_realclass.txt")
    if "returnShipment" in frame.columns:
        values = frame["returnShipment"]
    elif frame.shape[1] == 1:
        values = frame.iloc[:, 0]
    else:
        raise ValueError(
            "orders_realclass.txt must contain returnShipment or one label column"
        )
    labels = _validate_binary(values, "final labels")
    labels.name = "returnShipment"
    return labels.reset_index(drop=True)
