from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from dmc2014.data import load_final_labels, load_train_and_class


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


def _row(target=None):
    values = [
        1,
        "2013-03-01",
        "2013-03-04",
        10,
        "M",
        "black",
        3,
        29.99,
        99,
        "Mrs",
        "1980-01-01",
        "Bavaria",
        "2012-01-01",
    ]
    if target is not None:
        values.append(target)
    return values


def _write_csv_member(zf: ZipFile, name: str, columns: list[str], row: list):
    text = ";".join(columns) + "\n" + ";".join(map(str, row)) + "\n"
    zf.writestr(name, text)


def make_zip(path: Path, malformed_final: bool = False):
    with ZipFile(path, "w") as zf:
        _write_csv_member(zf, "orders_train.txt", TRAIN_COLUMNS, _row(1))
        _write_csv_member(zf, "orders_class.txt", CLASS_COLUMNS, _row())
        if malformed_final:
            zf.writestr("orders_realclass.txt", "this;is;malformed\n")
        else:
            zf.writestr("orders_realclass.txt", "returnShipment\n1\n")


def test_normal_loader_reads_only_train_and_class(tmp_path):
    archive = tmp_path / "dmc.zip"
    make_zip(archive, malformed_final=True)
    train, competition = load_train_and_class(archive)
    assert train.columns.tolist() == TRAIN_COLUMNS
    assert competition.columns.tolist() == CLASS_COLUMNS
    assert train["returnShipment"].tolist() == [1]
    assert pd.api.types.is_datetime64_any_dtype(train["orderDate"])


def test_loader_rejects_non_binary_training_target(tmp_path):
    archive = tmp_path / "bad.zip"
    with ZipFile(archive, "w") as zf:
        _write_csv_member(zf, "orders_train.txt", TRAIN_COLUMNS, _row(2))
        _write_csv_member(zf, "orders_class.txt", CLASS_COLUMNS, _row())
    with pytest.raises(ValueError, match="binary"):
        load_train_and_class(archive)


def test_final_labels_are_loaded_only_explicitly(tmp_path):
    archive = tmp_path / "dmc.zip"
    make_zip(archive)
    labels = load_final_labels(archive)
    assert labels.tolist() == [1]
