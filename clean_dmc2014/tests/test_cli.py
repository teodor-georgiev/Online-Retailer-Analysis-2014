from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

import dmc2014.cli as cli


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


def _row(item_id, date, target=None):
    values = [
        item_id,
        date,
        date,
        10,
        "M",
        "black",
        5,
        20.0,
        100,
        "Mrs",
        "1980-01-01",
        "NRW",
        "2011-01-01",
    ]
    if target is not None:
        values.append(target)
    return values


def _csv(columns, rows):
    return ";".join(columns) + "\n" + "\n".join(
        ";".join(map(str, row)) for row in rows
    ) + "\n"


def make_archive(path: Path):
    train_rows = [
        _row(1, "2012-12-01", 0),
        _row(2, "2013-01-01", 1),
        _row(3, "2013-02-01", 0),
        _row(4, "2013-03-01", 1),
    ]
    with ZipFile(path, "w") as zf:
        zf.writestr("orders_train.txt", _csv(TRAIN_COLUMNS, train_rows))
        zf.writestr("orders_class.txt", _csv(CLASS_COLUMNS, [_row(5, "2013-04-01")]))
        zf.writestr("orders_realclass.txt", "returnShipment\n1\n")


def test_backtest_path_never_loads_final_labels(tmp_path, monkeypatch):
    archive = tmp_path / "dmc.zip"
    make_archive(archive)

    def forbidden(*args, **kwargs):
        raise AssertionError("backtest must not load final labels")

    monkeypatch.setattr(cli, "load_final_labels", forbidden)
    monkeypatch.setattr(
        cli,
        "run_backtest",
        lambda frame, **kwargs: {"rows": len(frame), "model": kwargs["model_name"]},
    )
    result = cli.backtest_from_zip(archive, "catboost", {}, cli.FeatureConfig())
    assert result == {"rows": 4, "model": "catboost"}


def test_final_evaluation_loads_labels_and_scores_explicitly(tmp_path, monkeypatch):
    archive = tmp_path / "dmc.zip"
    make_archive(archive)
    called = {"final": False}
    real_loader = cli.load_final_labels

    def tracked_loader(path):
        called["final"] = True
        return real_loader(path)

    monkeypatch.setattr(cli, "load_final_labels", tracked_loader)
    monkeypatch.setattr(
        cli,
        "fit_final_probabilities",
        lambda *args, **kwargs: np.array([0.9]),
    )
    config = {
        "model": "catboost",
        "params": {},
        "feature_config": {"history_groups": [], "recency_groups": [], "smoothing": 20.0},
        "threshold": 0.5,
        "recommended_iterations": 10,
    }
    result = cli.final_evaluate_from_zip(archive, config)
    assert called["final"] is True
    assert result["rows"] == 1
    assert result["points"] == 0.0
    assert result["accuracy"] == 1.0


def test_final_evaluation_rejects_label_alignment_mismatch(tmp_path, monkeypatch):
    archive = tmp_path / "dmc.zip"
    make_archive(archive)
    monkeypatch.setattr(cli, "load_final_labels", lambda path: pd.Series([1, 0]))
    config = {
        "model": "catboost",
        "params": {},
        "feature_config": {"history_groups": [], "recency_groups": [], "smoothing": 20.0},
        "threshold": 0.5,
        "recommended_iterations": 10,
    }
    with pytest.raises(ValueError, match="alignment"):
        cli.final_evaluate_from_zip(archive, config)
