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
    if train.X.columns.tolist() != valid.X.columns.tolist():
        raise ValueError("train and validation feature columns must match exactly")
    if train.categorical != valid.categorical:
        raise ValueError("train and validation categorical columns must match")


def fit_catboost(
    train: FeatureSet,
    valid: FeatureSet,
    params: dict | None = None,
) -> ModelResult:
    from catboost import CatBoostClassifier

    _validate_feature_sets(train, valid)
    defaults = {
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
    defaults.update(params or {})

    started = perf_counter()
    model = CatBoostClassifier(**defaults)
    model.fit(
        train.X,
        np.asarray(train.y, dtype=int),
        cat_features=train.categorical,
        eval_set=(valid.X, np.asarray(valid.y, dtype=int)),
        use_best_model=True,
        early_stopping_rounds=min(80, max(10, int(defaults["iterations"]) // 5)),
        verbose=False,
    )
    probabilities = model.predict_proba(valid.X)[:, 1].astype(float)
    zero_based = int(model.get_best_iteration())
    best_iteration = zero_based + 1 if zero_based >= 0 else int(model.tree_count_)
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
    defaults = {
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
    defaults.update(params or {})
    train_x, valid_x = _lightgbm_frames(train, valid)

    started = perf_counter()
    model = lgb.LGBMClassifier(**defaults)
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
    best_iteration = int(getattr(model, "best_iteration_", 0) or defaults["n_estimators"])
    return ModelResult(
        probabilities=probabilities,
        model=model,
        best_iteration=max(1, best_iteration),
        runtime_seconds=perf_counter() - started,
    )
