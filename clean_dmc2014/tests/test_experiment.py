import numpy as np
import pandas as pd

import dmc2014.experiment as experiment
from dmc2014.features import FeatureConfig
from dmc2014.models import ModelResult


def row(item_id, date, target):
    return {
        "orderItemID": item_id,
        "orderDate": pd.Timestamp(date),
        "deliveryDate": pd.Timestamp(date) + pd.Timedelta(days=2),
        "itemID": 10 + item_id % 2,
        "size": "M",
        "color": "black",
        "manufacturerID": 5,
        "price": 20.0 + item_id,
        "customerID": 100 + item_id % 2,
        "salutation": "Mrs",
        "dateOfBirth": pd.Timestamp("1980-01-01"),
        "state": "NRW",
        "creationDate": pd.Timestamp("2011-01-01"),
        "returnShipment": target,
    }


def temporal_frame():
    rows = [
        row(1, "2012-12-01", 0),
        row(2, "2012-12-02", 0),
        row(3, "2013-01-01", 1),
        row(4, "2013-01-02", 0),
        row(5, "2013-02-01", 1),
        row(6, "2013-02-02", 1),
        row(7, "2013-02-03", 0),
        row(8, "2013-03-01", 0),
        row(9, "2013-03-02", 0),
        row(10, "2013-03-03", 0),
        row(11, "2013-03-04", 1),
    ]
    return pd.DataFrame(rows)


def test_backtest_runs_strictly_growing_folds_and_row_weighted_metrics(monkeypatch):
    seen = []

    def stub_fitter(train, valid, params):
        seen.append((len(train.y), len(valid.y)))
        return ModelResult(
            probabilities=np.zeros(len(valid.y), dtype=float),
            model=None,
            best_iteration=1,
            runtime_seconds=0.01,
        )

    monkeypatch.setitem(experiment.MODEL_FITTERS, "stub", stub_fitter)
    result = experiment.run_backtest(
        temporal_frame(),
        model_name="stub",
        params={},
        feature_config=FeatureConfig(history_groups=(), recency_groups=()),
    )

    assert seen == [(2, 2), (4, 3), (7, 4)]
    assert [fold["validation_rows"] for fold in result["folds"]] == [2, 3, 4]
    assert result["threshold"] == 0.5
    assert result["total_validation_rows"] == 9
    assert result["total_points"] == 4.0
    assert np.isclose(result["accuracy"], 5 / 9)
    assert np.isclose(result["point_rate"], 4 / 9)
