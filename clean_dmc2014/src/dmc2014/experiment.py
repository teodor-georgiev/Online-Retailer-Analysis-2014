from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from dmc2014.capacity import resolve_workers
from dmc2014.features import (
    FeatureConfig,
    FeatureSet,
    build_training_features,
    build_validation_features,
)
from dmc2014.metrics import best_threshold, dmc_points
from dmc2014.models import fit_catboost, fit_lightgbm
from dmc2014.speed import FeatureCache, feature_cache_key, frame_fingerprint, plan_fold_workers
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
        "rolling_profiles": bool(config.rolling_profiles),
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


def _prepare_feature_pair(
    history: pd.DataFrame,
    validation: pd.DataFrame,
    config: FeatureConfig,
    fold: TemporalFold,
    dataset_fingerprint: str | None,
    cache: FeatureCache | None,
) -> tuple[FeatureSet, FeatureSet, bool, float]:
    started = perf_counter()
    key = None
    if cache is not None:
        if dataset_fingerprint is None:
            raise ValueError("dataset fingerprint is required when feature cache is enabled")
        key = feature_cache_key(dataset_fingerprint, fold, config)
        restored = cache.load(key)
        if restored is not None:
            return restored[0], restored[1], True, perf_counter() - started

    train_features = build_training_features(history, config)
    valid_features = build_validation_features(history, validation, config)
    if cache is not None and key is not None:
        cache.store(key, train_features, valid_features)
    return train_features, valid_features, False, perf_counter() - started


def _bounded_model_workers(params: dict, key: str, fold_workers: int) -> dict:
    settings = dict(params)
    requested = settings.get(key)
    if requested is None:
        settings[key] = int(fold_workers)
        return settings
    try:
        requested_int = int(requested)
    except (TypeError, ValueError):
        settings[key] = int(fold_workers)
        return settings
    settings[key] = int(fold_workers) if requested_int <= 0 else min(requested_int, int(fold_workers))
    return settings


def _fit_ensemble_fold(
    prepared: dict,
    cat_params: dict,
    lgb_params: dict,
    fold_workers: int,
) -> dict:
    cat_settings = _bounded_model_workers(cat_params, "thread_count", fold_workers)
    lgb_settings = _bounded_model_workers(lgb_params, "n_jobs", fold_workers)
    cat_result = fit_catboost(
        prepared["cat_train_features"],
        prepared["cat_valid_features"],
        cat_settings,
    )
    lgb_result = fit_lightgbm(
        prepared["lgb_train_features"],
        prepared["lgb_valid_features"],
        lgb_settings,
    )
    y_true = prepared["y_true"]
    cat_probability = np.asarray(cat_result.probabilities, dtype=float)
    lgb_probability = np.asarray(lgb_result.probabilities, dtype=float)
    if cat_probability.shape != y_true.shape or lgb_probability.shape != y_true.shape:
        raise ValueError(
            f"fold {prepared['name']} ensemble prediction shape mismatch: "
            f"truth={y_true.shape} catboost={cat_probability.shape} "
            f"lightgbm={lgb_probability.shape}"
        )
    return {
        **prepared,
        "cat_probability": cat_probability,
        "lgb_probability": lgb_probability,
        "catboost_best_iteration": int(cat_result.best_iteration),
        "lightgbm_best_iteration": int(lgb_result.best_iteration),
        "catboost_runtime_seconds": float(cat_result.runtime_seconds),
        "lightgbm_runtime_seconds": float(lgb_result.runtime_seconds),
        "fold_workers": int(fold_workers),
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
    cache_dir: str | Path | None = None,
    parallel_folds: bool = True,
    total_workers: int | None = None,
) -> dict:
    """Fit both models on shared folds and select a CatBoost/LightGBM OOF blend.

    Feature preparation is intentionally sequential because pandas rolling/groupby
    work is memory-bandwidth heavy. Model fitting may then run folds concurrently
    under one bounded global CPU budget.
    """
    wall_started = perf_counter()
    config = feature_config or FeatureConfig()
    cat_config = catboost_feature_config or config
    lgb_config = lightgbm_feature_config or config
    selected_folds = list(folds or default_folds())
    if not selected_folds:
        raise ValueError("folds must not be empty")
    cat_params = dict(catboost_params or {})
    lgb_params = dict(lightgbm_params or {})
    cache = FeatureCache(cache_dir) if cache_dir is not None else None
    dataset_fingerprint = frame_fingerprint(train_frame) if cache is not None else None

    prepared_folds: list[dict] = []
    total_feature_runtime = 0.0
    cache_hits = 0
    cache_lookups = 0

    for fold in selected_folds:
        history, validation = split_fold(train_frame, fold)
        if history.empty:
            raise ValueError(f"fold {fold.name} has no training rows")
        if validation.empty:
            raise ValueError(f"fold {fold.name} has no validation rows")

        cat_train, cat_valid, cat_hit, cat_feature_runtime = _prepare_feature_pair(
            history,
            validation,
            cat_config,
            fold,
            dataset_fingerprint,
            cache,
        )
        total_feature_runtime += cat_feature_runtime
        if cache is not None:
            cache_lookups += 1
            cache_hits += int(cat_hit)

        if cat_config == lgb_config:
            lgb_train, lgb_valid = cat_train, cat_valid
            lgb_hit = cat_hit
            lgb_feature_runtime = 0.0
        else:
            lgb_train, lgb_valid, lgb_hit, lgb_feature_runtime = _prepare_feature_pair(
                history,
                validation,
                lgb_config,
                fold,
                dataset_fingerprint,
                cache,
            )
            total_feature_runtime += lgb_feature_runtime
            if cache is not None:
                cache_lookups += 1
                cache_hits += int(lgb_hit)

        if cat_valid.y is None or lgb_valid.y is None:
            raise ValueError(f"fold {fold.name} validation target is missing")
        y_true = np.asarray(cat_valid.y, dtype=int)
        lgb_y = np.asarray(lgb_valid.y, dtype=int)
        if not np.array_equal(y_true, lgb_y):
            raise ValueError(f"fold {fold.name} ensemble target alignment mismatch")

        prepared_folds.append(
            {
                "name": fold.name,
                "training_rows": int(len(cat_train.y)),
                "validation_rows": int(len(y_true)),
                "y_true": y_true,
                "cat_train_features": cat_train,
                "cat_valid_features": cat_valid,
                "lgb_train_features": lgb_train,
                "lgb_valid_features": lgb_valid,
                "feature_runtime_seconds": float(cat_feature_runtime + lgb_feature_runtime),
                "feature_cache_hit": bool(cat_hit and lgb_hit),
            }
        )

    worker_budget = int(total_workers or resolve_workers("batch"))
    if worker_budget < 1:
        raise ValueError("total_workers must be positive")
    worker_plan = plan_fold_workers(worker_budget, len(prepared_folds))

    model_started = perf_counter()
    fitted: list[dict | None] = [None] * len(prepared_folds)
    max_parallel = min(len(prepared_folds), worker_budget) if parallel_folds else 1
    if max_parallel > 1:
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = {
                executor.submit(
                    _fit_ensemble_fold,
                    prepared,
                    cat_params,
                    lgb_params,
                    worker_plan[index],
                ): index
                for index, prepared in enumerate(prepared_folds)
            }
            for future in as_completed(futures):
                fitted[futures[future]] = future.result()
    else:
        for index, prepared in enumerate(prepared_folds):
            fitted[index] = _fit_ensemble_fold(
                prepared,
                cat_params,
                lgb_params,
                worker_plan[index],
            )
    model_wall_seconds = perf_counter() - model_started
    pending = [item for item in fitted if item is not None]

    all_y = [item["y_true"] for item in pending]
    all_cat = [item["cat_probability"] for item in pending]
    all_lgb = [item["lgb_probability"] for item in pending]
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
    total_cat_runtime = 0.0
    total_lgb_runtime = 0.0
    for item in pending:
        cat_prediction = (item["cat_probability"] >= cat_summary["threshold"]).astype(int)
        lgb_prediction = (item["lgb_probability"] >= lgb_summary["threshold"]).astype(int)
        fold_probability = weight_cat * item["cat_probability"] + weight_lgb * item["lgb_probability"]
        ensemble_prediction = (fold_probability >= ensemble_summary["threshold"]).astype(int)
        rows = int(item["validation_rows"])
        cat_points = float(dmc_points(item["y_true"], cat_prediction))
        lgb_points = float(dmc_points(item["y_true"], lgb_prediction))
        ensemble_points = float(dmc_points(item["y_true"], ensemble_prediction))
        cat_runtime = float(item["catboost_runtime_seconds"])
        lgb_runtime = float(item["lightgbm_runtime_seconds"])
        runtime = cat_runtime + lgb_runtime
        total_runtime += runtime
        total_cat_runtime += cat_runtime
        total_lgb_runtime += lgb_runtime
        fold_records.append(
            {
                "name": item["name"],
                "training_rows": item["training_rows"],
                "validation_rows": rows,
                "fold_workers": item["fold_workers"],
                "feature_runtime_seconds": item["feature_runtime_seconds"],
                "feature_cache_hit": item["feature_cache_hit"],
                "catboost_best_iteration": item["catboost_best_iteration"],
                "lightgbm_best_iteration": item["lightgbm_best_iteration"],
                "catboost_runtime_seconds": cat_runtime,
                "lightgbm_runtime_seconds": lgb_runtime,
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
        "catboost_runtime_seconds": float(total_cat_runtime),
        "lightgbm_runtime_seconds": float(total_lgb_runtime),
        "feature_runtime_seconds": float(total_feature_runtime),
        "model_wall_seconds": float(model_wall_seconds),
        "wall_runtime_seconds": float(perf_counter() - wall_started),
        "feature_cache_hits": int(cache_hits),
        "feature_cache_lookups": int(cache_lookups),
        "parallel_folds": bool(parallel_folds),
        "total_workers": int(worker_budget),
    }
