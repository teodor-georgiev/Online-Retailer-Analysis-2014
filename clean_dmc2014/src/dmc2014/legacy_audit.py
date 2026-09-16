from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd


DROP_COLUMNS = [
    "order_date",
    "delivery_date",
    "user_dob",
    "user_reg_date",
    "order_id",
    "order_item_id",
]
CATEGORICAL_COLUMNS = [
    "size",
    "item_color",
    "user_title",
    "user_state",
    "item_id",
    "brand_id",
    "user_id",
    "mode_item_id",
    "mode_size",
    "mode_brand_id",
    "mode_item_color",
]


@dataclass
class LegacyAuditData:
    train_x: pd.DataFrame
    train_y: np.ndarray
    competition_x: pd.DataFrame
    categorical: list[str]


@dataclass
class LegacyAuditModelResult:
    probabilities: np.ndarray
    predictions: np.ndarray
    runtime_seconds: float
    model: Any


@dataclass
class LegacyAuditScore:
    points: int
    accuracy: float


def prepare_legacy_audit_data(frame: pd.DataFrame) -> LegacyAuditData:
    """Split the combined legacy feature matrix without loading April labels."""
    if "return" not in frame:
        raise ValueError("combined legacy frame must contain return")
    known_mask = frame["return"].notna()
    if not known_mask.any() or known_mask.all():
        raise ValueError("expected both known training rows and unlabeled competition rows")

    train = frame.loc[known_mask].copy()
    competition = frame.loc[~known_mask].copy()
    train_y = pd.to_numeric(train.pop("return"), errors="raise").astype(int).to_numpy()
    competition = competition.drop(columns=["return"])

    train_x = train.drop(columns=DROP_COLUMNS, errors="ignore")
    competition_x = competition.drop(columns=DROP_COLUMNS, errors="ignore")
    if train_x.columns.tolist() != competition_x.columns.tolist():
        raise ValueError("training and competition columns are not aligned")

    categorical = [column for column in CATEGORICAL_COLUMNS if column in train_x]
    for column in categorical:
        train_x[column] = train_x[column].astype("string").fillna("__MISSING__").astype(str)
        competition_x[column] = (
            competition_x[column].astype("string").fillna("__MISSING__").astype(str)
        )

    numeric_columns = [column for column in train_x.columns if column not in categorical]
    train_x[numeric_columns] = train_x[numeric_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
    competition_x[numeric_columns] = (
        competition_x[numeric_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
    )
    return LegacyAuditData(train_x, train_y, competition_x, categorical)


def fit_frozen_legacy_catboost(
    data: LegacyAuditData,
    params: dict | None = None,
) -> LegacyAuditModelResult:
    """Fit without eval_set or early stopping; April labels cannot affect training."""
    from catboost import CatBoostClassifier

    settings = {
        "iterations": 200,
        "learning_rate": 0.11,
        "depth": 10,
        "loss_function": "Logloss",
        "random_seed": 42,
        "l2_leaf_reg": 15,
        "max_ctr_complexity": 3,
        "thread_count": 8,
        "allow_writing_files": False,
        "verbose": False,
    }
    settings.update(params or {})
    started = perf_counter()
    model = CatBoostClassifier(**settings)
    model.fit(data.train_x, data.train_y, cat_features=data.categorical, verbose=False)
    probabilities = model.predict_proba(data.competition_x)[:, 1].astype(float)
    predictions = (probabilities >= 0.5).astype(np.int8)
    return LegacyAuditModelResult(
        probabilities=probabilities,
        predictions=predictions,
        runtime_seconds=perf_counter() - started,
        model=model,
    )


def save_predictions(path: str | Path, result: LegacyAuditModelResult) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "probability": result.probabilities,
            "prediction": result.predictions,
        }
    ).to_csv(destination, index=False)


def score_predictions(predictions: np.ndarray, labels: np.ndarray) -> LegacyAuditScore:
    predicted = np.asarray(predictions, dtype=int).reshape(-1)
    actual = np.asarray(labels, dtype=int).reshape(-1)
    if predicted.shape != actual.shape:
        raise ValueError(f"prediction/label length mismatch: {predicted.shape} vs {actual.shape}")
    points = int(np.abs(predicted - actual).sum())
    return LegacyAuditScore(points=points, accuracy=float(1.0 - points / len(actual)))
