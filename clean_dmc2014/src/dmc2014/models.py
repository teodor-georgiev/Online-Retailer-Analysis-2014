from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from dmc2014.features import FeatureSet


@dataclass
class ModelResult:
    probabilities: np.ndarray
    model: Any
    best_iteration: int
    runtime_seconds: float


def _validate_feature_sets(train: FeatureSet, valid: FeatureSet) -> None:
    if train.y is None or valid.y is None:
        raise ValueError("train and validation targets are required")
    _validate_feature_columns(train, valid)


def _validate_feature_columns(train: FeatureSet, other: FeatureSet) -> None:
    if train.X.columns.tolist() != other.X.columns.tolist():
        raise ValueError("feature columns must match exactly")
    if train.categorical != other.categorical:
        raise ValueError("categorical columns must match")


def _catboost_defaults() -> dict:
    return {
        "iterations": 500,
        "depth": 8,
        "learning_rate": 0.05,
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "random_seed": 42,
        "l2_leaf_reg": 8.0,
        "random_strength": 0.5,
        "thread_count": 2,
        "allow_writing_files": False,
        "verbose": False,
    }


def _catboost_settings_and_controls(params: dict | None) -> tuple[dict, int | None]:
    options = dict(params or {})
    marker = object()
    requested = options.pop("_early_stopping_rounds", marker)
    settings = _catboost_defaults()
    settings.update(options)
    if requested is marker:
        early_stopping = min(80, max(10, int(settings["iterations"]) // 5))
    elif requested is None:
        early_stopping = None
    else:
        early_stopping = int(requested)
        if early_stopping < 1:
            raise ValueError("_early_stopping_rounds must be positive or null")
    return settings, early_stopping


def _lightgbm_defaults() -> dict:
    return {
        "objective": "binary",
        "n_estimators": 600,
        "learning_rate": 0.04,
        "num_leaves": 63,
        "min_child_samples": 80,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 3.0,
        "random_state": 42,
        "n_jobs": 2,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
    }


def fit_catboost(
    train: FeatureSet,
    valid: FeatureSet,
    params: dict | None = None,
) -> ModelResult:
    from catboost import CatBoostClassifier

    _validate_feature_sets(train, valid)
    settings, early_stopping = _catboost_settings_and_controls(params)

    started = perf_counter()
    model = CatBoostClassifier(**settings)
    if early_stopping is None:
        model.fit(
            train.X,
            np.asarray(train.y, dtype=int),
            cat_features=train.categorical,
            verbose=False,
        )
        best_iteration = int(settings["iterations"])
    else:
        model.fit(
            train.X,
            np.asarray(train.y, dtype=int),
            cat_features=train.categorical,
            eval_set=(valid.X, np.asarray(valid.y, dtype=int)),
            use_best_model=True,
            early_stopping_rounds=early_stopping,
            verbose=False,
        )
        zero_based = int(model.get_best_iteration())
        best_iteration = zero_based + 1 if zero_based >= 0 else int(model.tree_count_)

    probabilities = model.predict_proba(valid.X)[:, 1].astype(float)
    return ModelResult(
        probabilities=probabilities,
        model=model,
        best_iteration=max(1, best_iteration),
        runtime_seconds=perf_counter() - started,
    )


def _lightgbm_frames(
    train: FeatureSet,
    valid: FeatureSet,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_x = train.X.copy()
    valid_x = valid.X.copy()
    for column in train.categorical:
        train_values = train_x[column].astype("string").fillna("__MISSING__").astype(str)
        valid_values = valid_x[column].astype("string").fillna("__MISSING__").astype(str)
        levels = pd.Index(train_values.unique()).union(pd.Index(valid_values.unique()))
        dtype = pd.CategoricalDtype(categories=levels)
        train_x[column] = train_values.astype(dtype)
        valid_x[column] = valid_values.astype(dtype)
    return train_x, valid_x


def fit_lightgbm(
    train: FeatureSet,
    valid: FeatureSet,
    params: dict | None = None,
) -> ModelResult:
    import lightgbm as lgb

    _validate_feature_sets(train, valid)
    settings = _lightgbm_defaults()
    settings.update(params or {})
    train_x, valid_x = _lightgbm_frames(train, valid)

    started = perf_counter()
    model = lgb.LGBMClassifier(**settings)
    callbacks = [lgb.early_stopping(60, verbose=False)]
    model.fit(
        train_x,
        np.asarray(train.y, dtype=int),
        eval_X=valid_x,
        eval_y=np.asarray(valid.y, dtype=int),
        callbacks=callbacks,
        categorical_feature=train.categorical,
    )
    probabilities = model.predict_proba(valid_x)[:, 1].astype(float)
    best_iteration = int(getattr(model, "best_iteration_", 0) or settings["n_estimators"])
    return ModelResult(
        probabilities=probabilities,
        model=model,
        best_iteration=max(1, best_iteration),
        runtime_seconds=perf_counter() - started,
    )


def fit_final_probabilities(
    model_name: str,
    train: FeatureSet,
    competition: FeatureSet,
    params: dict | None,
    iterations: int,
) -> np.ndarray:
    """Fit on all labeled history with a fixed CV-selected iteration count."""
    if train.y is None:
        raise ValueError("training target is required")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    _validate_feature_columns(train, competition)

    if model_name == "catboost":
        from catboost import CatBoostClassifier

        settings, _early_stopping = _catboost_settings_and_controls(params)
        settings["iterations"] = int(iterations)
        model = CatBoostClassifier(**settings)
        model.fit(
            train.X,
            np.asarray(train.y, dtype=int),
            cat_features=train.categorical,
            verbose=False,
        )
        return model.predict_proba(competition.X)[:, 1].astype(float)

    if model_name == "lightgbm":
        import lightgbm as lgb

        settings = _lightgbm_defaults()
        settings.update(params or {})
        settings["n_estimators"] = int(iterations)
        train_x, competition_x = _lightgbm_frames(train, competition)
        model = lgb.LGBMClassifier(**settings)
        model.fit(
            train_x,
            np.asarray(train.y, dtype=int),
            categorical_feature=train.categorical,
        )
        return model.predict_proba(competition_x)[:, 1].astype(float)

    raise ValueError(f"unknown model: {model_name}")
