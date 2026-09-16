import threading

import numpy as np
import pandas as pd

import dmc2014.experiment as experiment
from dmc2014.features import FeatureConfig, FeatureSet
from dmc2014.models import ModelResult
from dmc2014.splits import TemporalFold


def _fold(name: str, month: int) -> TemporalFold:
    start = pd.Timestamp(2013, month, 1)
    end = start + pd.offsets.MonthBegin(1)
    return TemporalFold(name=name, train_end=start, valid_start=start, valid_end=end)


def _frame() -> pd.DataFrame:
    dates = pd.date_range("2012-12-01", "2013-03-03", freq="D")
    return pd.DataFrame(
        {
            "orderDate": dates,
            "returnShipment": (np.arange(len(dates)) % 2).astype(int),
        }
    )


def _features(frame: pd.DataFrame) -> FeatureSet:
    y = frame["returnShipment"].to_numpy(dtype=int)
    return FeatureSet(
        X=pd.DataFrame({"x": np.arange(len(frame), dtype=float)}),
        y=y,
        categorical=[],
    )


def test_shared_config_builds_features_once_per_fold_and_splits_16_workers(monkeypatch):
    frame = _frame()
    folds = [_fold("jan", 1), _fold("feb", 2), _fold("mar", 3)]
    build_calls = []
    worker_calls = []
    lock = threading.Lock()

    def fake_training(history, config):
        build_calls.append(("train", len(history)))
        return _features(history)

    def fake_validation(history, validation, config):
        build_calls.append(("valid", len(validation)))
        return _features(validation)

    def fake_catboost(train, valid, params):
        with lock:
            worker_calls.append(("catboost", int(params["thread_count"])))
        return ModelResult(
            probabilities=np.full(len(valid.y), 0.4),
            model=None,
            best_iteration=10,
            runtime_seconds=0.01,
        )

    def fake_lightgbm(train, valid, params):
        with lock:
            worker_calls.append(("lightgbm", int(params["n_jobs"])))
        return ModelResult(
            probabilities=np.full(len(valid.y), 0.6),
            model=None,
            best_iteration=12,
            runtime_seconds=0.01,
        )

    monkeypatch.setattr(experiment, "build_training_features", fake_training)
    monkeypatch.setattr(experiment, "build_validation_features", fake_validation)
    monkeypatch.setattr(experiment, "fit_catboost", fake_catboost)
    monkeypatch.setattr(experiment, "fit_lightgbm", fake_lightgbm)

    result = experiment.run_ensemble_backtest(
        frame,
        feature_config=FeatureConfig(history_groups=(), recency_groups=()),
        folds=folds,
        total_workers=16,
        parallel_folds=True,
    )

    assert len(build_calls) == 6
    assert sorted(value for model, value in worker_calls if model == "catboost") == [5, 5, 6]
    assert sorted(value for model, value in worker_calls if model == "lightgbm") == [5, 5, 6]
    assert [fold["fold_workers"] for fold in result["folds"]] == [6, 5, 5]
    assert result["total_workers"] == 16


def test_explicit_model_thread_setting_is_capped_by_fold_budget(monkeypatch):
    frame = _frame()
    fold = _fold("jan", 1)
    captured = {}

    monkeypatch.setattr(experiment, "build_training_features", lambda history, config: _features(history))
    monkeypatch.setattr(
        experiment,
        "build_validation_features",
        lambda history, validation, config: _features(validation),
    )

    def fake_catboost(train, valid, params):
        captured["cat"] = params["thread_count"]
        return ModelResult(np.full(len(valid.y), 0.4), None, 10, 0.01)

    def fake_lightgbm(train, valid, params):
        captured["lgb"] = params["n_jobs"]
        return ModelResult(np.full(len(valid.y), 0.6), None, 12, 0.01)

    monkeypatch.setattr(experiment, "fit_catboost", fake_catboost)
    monkeypatch.setattr(experiment, "fit_lightgbm", fake_lightgbm)

    experiment.run_ensemble_backtest(
        frame,
        catboost_params={"thread_count": 99},
        lightgbm_params={"n_jobs": 99},
        feature_config=FeatureConfig(history_groups=(), recency_groups=()),
        folds=[fold],
        total_workers=4,
    )

    assert captured == {"cat": 4, "lgb": 4}
