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
        "user_profiles": bool(config.user_profiles),
        "product_profiles": bool(config.product_profiles),
    }


def _component_summary(y_true: np.ndarray, probability: np.ndarray) -> dict:
    threshold, points = best_threshold(y_true, probability)
    rows = int(len(y_true))
    soft_points = float(dmc_points(y_true, probability))
    point_rate = float(points / rows)
    return {
        "threshold": float(threshold),
        "points": float(points),
        "point_rate": point_rate,
        "accuracy": float(1.0 - point_rate),
        "soft_points": soft_points,
        "soft_point_rate": float(soft_points / rows),
        "estimated_50078_points": float(point_rate * 50078.0),
    }


def select_blend(
    y_true: np.ndarray,
    cat_probability: np.ndarray,
    lgb_probability: np.ndarray,
    weights: Sequence[float] | None = None,
) -> dict:
    """Select a deterministic CatBoost/LightGBM OOF blend and hard threshold."""
    truth = np.asarray(y_true, dtype=int)
    cat = np.asarray(cat_probability, dtype=float)
    lgb = np.asarray(lgb_probability, dtype=float)
    if truth.shape != cat.shape or truth.shape != lgb.shape:
        raise ValueError(
            "ensemble shape mismatch: "
            f"truth={truth.shape} catboost={cat.shape} lightgbm={lgb.shape}"
        )

    selected_weights = (
        [float(value) for value in weights]
        if weights is not None
        else [float(value) for value in np.round(np.arange(0.0, 1.0001, 0.05), 2)]
    )
    if not selected_weights:
        raise ValueError("weights must not be empty")
    for weight in selected_weights:
        if not 0.0 <= weight <= 1.0:
            raise ValueError("blend weights must be between 0 and 1")

    candidates: list[dict] = []
    for weight in selected_weights:
        probability = weight * cat + (1.0 - weight) * lgb
        threshold, points = best_threshold(truth, probability)
        candidates.append(
            {
                "weight_catboost": float(weight),
                "weight_lightgbm": float(1.0 - weight),
                "threshold": float(threshold),
                "points": float(points),
            }
        )

    return min(
        candidates,
        key=lambda item: (
            item["points"],
            abs(item["weight_catboost"] - 0.5),
            abs(item["threshold"] - 0.5),
            item["weight_catboost"],
        ),
    )


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


def run_ensemble_backtest(
    train_frame: pd.DataFrame,
    catboost_params: dict | None = None,
    lightgbm_params: dict | None = None,
    feature_config: FeatureConfig | None = None,
    catboost_feature_config: FeatureConfig | None = None,
    lightgbm_feature_config: FeatureConfig | None = None,
    folds: Sequence[TemporalFold] | None = None,
    weights: Sequence[float] | None = None,
) -> dict:
    """Fit both models on shared folds and select a CatBoost/LightGBM OOF blend."""
    config = feature_config or FeatureConfig()
    cat_config = catboost_feature_config or config
    lgb_config = lightgbm_feature_config or config
    selected_folds = list(folds or default_folds())
    cat_params = dict(catboost_params or {})
    lgb_params = dict(lightgbm_params or {})

    pending: list[dict] = []
    all_y: list[np.ndarray] = []
    all_cat: list[np.ndarray] = []
    all_lgb: list[np.ndarray] = []

    for fold in selected_folds:
        history, validation = split_fold(train_frame, fold)
        if history.empty:
            raise ValueError(f"fold {fold.name} has no training rows")
        if validation.empty:
            raise ValueError(f"fold {fold.name} has no validation rows")

        cat_train_features = build_training_features(history, cat_config)
        cat_valid_features = build_validation_features(history, validation, cat_config)
        lgb_train_features = build_training_features(history, lgb_config)
        lgb_valid_features = build_validation_features(history, validation, lgb_config)
        if cat_valid_features.y is None or lgb_valid_features.y is None:
            raise ValueError(f"fold {fold.name} validation target is missing")
        y_true = np.asarray(cat_valid_features.y, dtype=int)
        lgb_y = np.asarray(lgb_valid_features.y, dtype=int)
        if not np.array_equal(y_true, lgb_y):
            raise ValueError(f"fold {fold.name} ensemble target alignment mismatch")

        cat_result = fit_catboost(cat_train_features, cat_valid_features, cat_params)
        lgb_result = fit_lightgbm(lgb_train_features, lgb_valid_features, lgb_params)
        cat_probability = np.asarray(cat_result.probabilities, dtype=float)
        lgb_probability = np.asarray(lgb_result.probabilities, dtype=float)
        if cat_probability.shape != y_true.shape or lgb_probability.shape != y_true.shape:
            raise ValueError(
                f"fold {fold.name} ensemble prediction shape mismatch: "
                f"truth={y_true.shape} catboost={cat_probability.shape} "
                f"lightgbm={lgb_probability.shape}"
            )

        pending.append(
            {
                "name": fold.name,
                "training_rows": int(len(cat_train_features.y)),
                "validation_rows": int(len(y_true)),
                "y_true": y_true,
                "cat_probability": cat_probability,
                "lgb_probability": lgb_probability,
                "catboost_best_iteration": int(cat_result.best_iteration),
                "lightgbm_best_iteration": int(lgb_result.best_iteration),
                "catboost_runtime_seconds": float(cat_result.runtime_seconds),
                "lightgbm_runtime_seconds": float(lgb_result.runtime_seconds),
            }
        )
        all_y.append(y_true)
        all_cat.append(cat_probability)
        all_lgb.append(lgb_probability)

    concatenated_y = np.concatenate(all_y)
    concatenated_cat = np.concatenate(all_cat)
    concatenated_lgb = np.concatenate(all_lgb)
    blend = select_blend(
        concatenated_y,
        concatenated_cat,
        concatenated_lgb,
        weights=weights,
    )
    weight_cat = float(blend["weight_catboost"])
    weight_lgb = float(blend["weight_lightgbm"])
    ensemble_probability = weight_cat * concatenated_cat + weight_lgb * concatenated_lgb

    cat_summary = _component_summary(concatenated_y, concatenated_cat)
    lgb_summary = _component_summary(concatenated_y, concatenated_lgb)
    ensemble_summary = _component_summary(concatenated_y, ensemble_probability)
    ensemble_summary["threshold"] = float(blend["threshold"])
    ensemble_summary["points"] = float(blend["points"])
    ensemble_summary["point_rate"] = float(blend["points"] / len(concatenated_y))
    ensemble_summary["accuracy"] = float(1.0 - ensemble_summary["point_rate"])
    ensemble_summary["estimated_50078_points"] = float(
        ensemble_summary["point_rate"] * 50078.0
    )
    ensemble_summary["weight_catboost"] = weight_cat
    ensemble_summary["weight_lightgbm"] = weight_lgb

    fold_records: list[dict] = []
    total_runtime = 0.0
    for item in pending:
        cat_prediction = (
            item["cat_probability"] >= cat_summary["threshold"]
        ).astype(int)
        lgb_prediction = (
            item["lgb_probability"] >= lgb_summary["threshold"]
        ).astype(int)
        fold_probability = (
            weight_cat * item["cat_probability"] + weight_lgb * item["lgb_probability"]
        )
        ensemble_prediction = (
            fold_probability >= ensemble_summary["threshold"]
        ).astype(int)
        rows = int(item["validation_rows"])
        cat_points = float(dmc_points(item["y_true"], cat_prediction))
        lgb_points = float(dmc_points(item["y_true"], lgb_prediction))
        ensemble_points = float(dmc_points(item["y_true"], ensemble_prediction))
        runtime = float(
            item["catboost_runtime_seconds"] + item["lightgbm_runtime_seconds"]
        )
        total_runtime += runtime
        fold_records.append(
            {
                "name": item["name"],
                "training_rows": item["training_rows"],
                "validation_rows": rows,
                "catboost_best_iteration": item["catboost_best_iteration"],
                "lightgbm_best_iteration": item["lightgbm_best_iteration"],
                "catboost_runtime_seconds": item["catboost_runtime_seconds"],
                "lightgbm_runtime_seconds": item["lightgbm_runtime_seconds"],
                "catboost_points": cat_points,
                "lightgbm_points": lgb_points,
                "ensemble_points": ensemble_points,
                "ensemble_accuracy": float(1.0 - ensemble_points / rows),
            }
        )

    return {
        "model": "ensemble",
        "params": {
            "catboost": cat_params,
            "lightgbm": lgb_params,
        },
        "feature_config": _config_record(config),
        "feature_config_catboost": _config_record(cat_config),
        "feature_config_lightgbm": _config_record(lgb_config),
        "weight_catboost": weight_cat,
        "weight_lightgbm": weight_lgb,
        "threshold": float(ensemble_summary["threshold"]),
        "folds": fold_records,
        "total_validation_rows": int(len(concatenated_y)),
        "catboost": cat_summary,
        "lightgbm": lgb_summary,
        "ensemble": ensemble_summary,
        "estimated_50078_points": float(ensemble_summary["estimated_50078_points"]),
        "runtime_seconds": float(total_runtime),
    }
