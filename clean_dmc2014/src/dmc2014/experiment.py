from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from dmc2014.features import (
    FeatureConfig,
    build_training_features,
    build_validation_features,
)
from dmc2014.metrics import best_threshold, dmc_points
from dmc2014.models import fit_catboost, fit_lightgbm
from dmc2014.splits import TemporalFold, default_folds, split_fold


MODEL_FITTERS = {
    "catboost": fit_catboost,
    "lightgbm": fit_lightgbm,
}


def _config_record(config: FeatureConfig) -> dict:
    return {
        "history_groups": [list(group) for group in config.history_groups],
        "recency_groups": [list(group) for group in config.recency_groups],
        "smoothing": float(config.smoothing),
    }


def run_backtest(
    train_frame: pd.DataFrame,
    model_name: str,
    params: dict | None = None,
    feature_config: FeatureConfig | None = None,
    folds: Sequence[TemporalFold] | None = None,
) -> dict:
    """Run leakage-safe rolling validation and select one shared OOF threshold."""
    if model_name not in MODEL_FITTERS:
        raise ValueError(f"unknown model: {model_name}")
    config = feature_config or FeatureConfig()
    selected_folds = list(folds or default_folds())
    fitter = MODEL_FITTERS[model_name]

    pending: list[dict] = []
    all_y: list[np.ndarray] = []
    all_probability: list[np.ndarray] = []

    for fold in selected_folds:
        history, validation = split_fold(train_frame, fold)
        if history.empty:
            raise ValueError(f"fold {fold.name} has no training rows")
        if validation.empty:
            raise ValueError(f"fold {fold.name} has no validation rows")

        train_features = build_training_features(history, config)
        valid_features = build_validation_features(history, validation, config)
        if valid_features.y is None:
            raise ValueError(f"fold {fold.name} validation target is missing")

        model_result = fitter(train_features, valid_features, params or {})
        probability = np.asarray(model_result.probabilities, dtype=float)
        y_true = np.asarray(valid_features.y, dtype=int)
        if probability.shape != y_true.shape:
            raise ValueError(
                f"fold {fold.name} prediction shape mismatch: "
                f"{probability.shape} != {y_true.shape}"
            )

        pending.append(
            {
                "name": fold.name,
                "training_rows": int(len(train_features.y)),
                "validation_rows": int(len(y_true)),
                "best_iteration": int(model_result.best_iteration),
                "runtime_seconds": float(model_result.runtime_seconds),
                "soft_points": float(dmc_points(y_true, probability)),
                "y_true": y_true,
                "probability": probability,
            }
        )
        all_y.append(y_true)
        all_probability.append(probability)

    concatenated_y = np.concatenate(all_y)
    concatenated_probability = np.concatenate(all_probability)
    threshold, _ = best_threshold(concatenated_y, concatenated_probability)

    fold_records: list[dict] = []
    total_points = 0.0
    total_soft_points = 0.0
    total_rows = 0
    total_runtime = 0.0

    for item in pending:
        prediction = (item["probability"] >= threshold).astype(int)
        points = dmc_points(item["y_true"], prediction)
        rows = int(item["validation_rows"])
        fold_records.append(
            {
                "name": item["name"],
                "training_rows": item["training_rows"],
                "validation_rows": rows,
                "best_iteration": item["best_iteration"],
                "runtime_seconds": item["runtime_seconds"],
                "soft_points": item["soft_points"],
                "points": float(points),
                "accuracy": float(1.0 - points / rows),
            }
        )
        total_points += points
        total_soft_points += float(item["soft_points"])
        total_rows += rows
        total_runtime += float(item["runtime_seconds"])

    point_rate = float(total_points / total_rows)
    return {
        "model": model_name,
        "params": dict(params or {}),
        "feature_config": _config_record(config),
        "threshold": float(threshold),
        "folds": fold_records,
        "total_validation_rows": int(total_rows),
        "total_points": float(total_points),
        "point_rate": point_rate,
        "accuracy": float(1.0 - point_rate),
        "soft_points": float(total_soft_points),
        "soft_point_rate": float(total_soft_points / total_rows),
        "estimated_50078_points": float(point_rate * 50078.0),
        "runtime_seconds": float(total_runtime),
    }
