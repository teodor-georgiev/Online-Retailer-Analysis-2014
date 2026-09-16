import numpy as np
import pandas as pd

import dmc2014.experiment as experiment
from dmc2014.features import FeatureConfig
from dmc2014.models import ModelResult
from dmc2014.splits import TemporalFold


def _row(item_id: int, date: str, target: int) -> dict:
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


def _tiny_frame_and_fold():
    frame = pd.DataFrame(
        [
            _row(1, "2012-12-01", 0),
            _row(2, "2012-12-02", 1),
            _row(3, "2013-01-01", 0),
            _row(4, "2013-01-02", 1),
            _row(5, "2013-01-03", 1),
        ]
    )
    fold = TemporalFold(
        name="jan",
        train_end=pd.Timestamp("2013-01-01"),
        valid_start=pd.Timestamp("2013-01-01"),
        valid_end=pd.Timestamp("2013-02-01"),
    )
    return frame, fold


def test_select_blend_finds_unique_best_mixed_weight():
    y = np.array([0, 0, 1, 1, 1, 0], dtype=int)
    cat = np.array([0.56, 0.21, 0.62, 0.25, 0.32, 0.53], dtype=float)
    lgb = np.array([0.25, 0.73, 0.22, 0.84, 0.70, 0.13], dtype=float)

    result = experiment.select_blend(y, cat, lgb)

    assert result["weight_catboost"] == 0.60
    assert result["points"] == 0.0
    assert np.isclose(result["threshold"], 0.4575)


def test_select_blend_tie_breaks_toward_half_then_lower_catboost_weight():
    y = np.array([0, 1, 0, 1], dtype=int)
    probability = np.array([0.1, 0.9, 0.2, 0.8], dtype=float)

    centered = experiment.select_blend(
        y,
        probability,
        probability,
        weights=[0.0, 0.5, 1.0],
    )
    assert centered["weight_catboost"] == 0.5
    assert centered["threshold"] == 0.5

    lower = experiment.select_blend(
        y,
        probability,
        probability,
        weights=[0.45, 0.55],
    )
    assert lower["weight_catboost"] == 0.45


def test_ensemble_backtest_keeps_component_predictions_row_aligned(monkeypatch):
    frame, fold = _tiny_frame_and_fold()
    calls = []

    def fake_catboost(train, valid, params):
        calls.append(("catboost", len(train.y), len(valid.y)))
        return ModelResult(
            probabilities=np.array([0.20, 0.80, 0.75], dtype=float),
            model=None,
            best_iteration=11,
            runtime_seconds=0.01,
        )

    def fake_lightgbm(train, valid, params):
        calls.append(("lightgbm", len(train.y), len(valid.y)))
        return ModelResult(
            probabilities=np.array([0.25, 0.70, 0.85], dtype=float),
            model=None,
            best_iteration=13,
            runtime_seconds=0.02,
        )

    monkeypatch.setattr(experiment, "fit_catboost", fake_catboost)
    monkeypatch.setattr(experiment, "fit_lightgbm", fake_lightgbm)

    result = experiment.run_ensemble_backtest(
        frame,
        catboost_params={"iterations": 20},
        lightgbm_params={"n_estimators": 30},
        feature_config=FeatureConfig(history_groups=(), recency_groups=()),
        folds=[fold],
    )

    assert calls == [("catboost", 2, 3), ("lightgbm", 2, 3)]
    assert result["model"] == "ensemble"
    assert result["total_validation_rows"] == 3
    assert result["catboost"]["points"] == 0.0
    assert result["lightgbm"]["points"] == 0.0
    assert result["ensemble"]["points"] == 0.0
    assert result["folds"][0]["catboost_best_iteration"] == 11
    assert result["folds"][0]["lightgbm_best_iteration"] == 13


def test_ensemble_can_use_different_feature_configs_per_model(monkeypatch):
    frame, fold = _tiny_frame_and_fold()
    seen = {}

    def fake_catboost(train, valid, params):
        seen["catboost"] = (train.X.shape[1], valid.X.shape[1])
        return ModelResult(
            probabilities=np.array([0.20, 0.80, 0.75], dtype=float),
            model=None,
            best_iteration=11,
            runtime_seconds=0.01,
        )

    def fake_lightgbm(train, valid, params):
        seen["lightgbm"] = (train.X.shape[1], valid.X.shape[1])
        return ModelResult(
            probabilities=np.array([0.25, 0.70, 0.85], dtype=float),
            model=None,
            best_iteration=13,
            runtime_seconds=0.02,
        )

    monkeypatch.setattr(experiment, "fit_catboost", fake_catboost)
    monkeypatch.setattr(experiment, "fit_lightgbm", fake_lightgbm)

    lean = FeatureConfig(history_groups=(), recency_groups=())
    rich = FeatureConfig(
        history_groups=(),
        recency_groups=(),
        user_profiles=True,
        product_profiles=True,
    )
    result = experiment.run_ensemble_backtest(
        frame,
        catboost_params={},
        lightgbm_params={},
        feature_config=rich,
        catboost_feature_config=lean,
        lightgbm_feature_config=rich,
        folds=[fold],
    )

    assert seen["catboost"][0] < seen["lightgbm"][0]
    assert seen["catboost"][0] == seen["catboost"][1]
    assert seen["lightgbm"][0] == seen["lightgbm"][1]
    assert result["feature_config_catboost"]["user_profiles"] is False
    assert result["feature_config_catboost"]["product_profiles"] is False
    assert result["feature_config_lightgbm"]["user_profiles"] is True
    assert result["feature_config_lightgbm"]["product_profiles"] is True
