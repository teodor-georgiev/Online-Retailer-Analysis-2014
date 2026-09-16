from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from dmc2014_benchmark import (
    best_threshold,
    build_row_features,
    candidate_configs,
    dmc_score,
    fit_candidate,
    fit_lightgbm_candidate,
    lightgbm_configs,
    load_competition_data,
    prepare_prediction_features,
    prepare_training_features,
    split_train_validation,
)
from dmc2014_fast_history import prepare_training_features_loo
from dmc2014_views import numeric_history_view


def ensure_data(repo_root: Path) -> Path:
    data_dir = repo_root / "Orders_data" / "extracted"
    train_path = data_dir / "orders_train.txt"
    class_path = data_dir / "orders_class.txt"
    if train_path.exists() and class_path.exists():
        return data_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    archive = repo_root / "Orders_data" / "Orders_train_test_class.zip"
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(data_dir)
    return data_dir


def prepare_mode(history, valid, feature_mode: str):
    if feature_mode == "raw":
        train_x, categorical = build_row_features(history)
        valid_x, valid_categorical = build_row_features(valid)
        train_y = history["returnShipment"].to_numpy(dtype=int)
    elif feature_mode in {"history", "history_numeric"}:
        train_x, train_y, categorical = prepare_training_features(history)
        valid_x, valid_categorical = prepare_prediction_features(history, valid)
        if feature_mode == "history_numeric":
            train_x, categorical = numeric_history_view(train_x, categorical)
            valid_x, valid_categorical = numeric_history_view(valid_x, valid_categorical)
    elif feature_mode == "history_loo_numeric":
        train_x, train_y, categorical = prepare_training_features_loo(history)
        valid_x, valid_categorical = prepare_prediction_features(history, valid)
        train_x, categorical = numeric_history_view(train_x, categorical)
        valid_x, valid_categorical = numeric_history_view(valid_x, valid_categorical)
    else:
        raise ValueError(f"unknown feature mode: {feature_mode}")

    if categorical != valid_categorical:
        raise ValueError("categorical columns differ between train and validation")

    train_x = train_x.drop(columns=["orderItemID"], errors="ignore")
    valid_x = valid_x.drop(columns=["orderItemID"], errors="ignore")
    categorical = [column for column in categorical if column in train_x.columns]
    return train_x, train_y, valid_x, valid["returnShipment"].to_numpy(dtype=int), categorical


def model_family(model_name: str):
    if model_name == "lightgbm":
        return lightgbm_configs(), fit_lightgbm_candidate
    if model_name == "catboost":
        return candidate_configs(), fit_candidate
    raise ValueError(f"unknown model: {model_name}")


def run_validation(
    repo_root: Path,
    feature_modes: list[str],
    full_sweep: bool,
    model_name: str,
) -> dict:
    data_dir = ensure_data(repo_root)
    train, _ = load_competition_data(data_dir)
    history, valid = split_train_validation(train)

    configs, fitter = model_family(model_name)
    if not full_sweep:
        configs = configs[:1]

    results = []
    for feature_mode in feature_modes:
        train_x, train_y, valid_x, valid_y, categorical = prepare_mode(history, valid, feature_mode)
        for config in configs:
            model, probability, best_iteration = fitter(
                train_x,
                train_y,
                valid_x,
                valid_y,
                categorical,
                config["params"],
            )
            threshold, hard_points = best_threshold(valid_y, probability)
            result = {
                "model": model_name,
                "feature_mode": feature_mode,
                "name": config["name"],
                "train_rows": int(len(train_y)),
                "validation_rows": int(len(valid_y)),
                "features": int(train_x.shape[1]),
                "best_iteration": int(best_iteration),
                "threshold": float(threshold),
                "validation_points": float(hard_points),
                "validation_accuracy": float(1.0 - hard_points / len(valid_y)),
                "validation_soft_points": float(dmc_score(valid_y, probability)),
                "estimated_50078_points": float(hard_points / len(valid_y) * 50078.0),
            }
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
            del model

    best = min(results, key=lambda item: item["validation_points"])
    summary = {
        "model": model_name,
        "history_rows": int(len(history)),
        "validation_rows": int(len(valid)),
        "results": results,
        "best": best,
    }
    print("DMC2014_VALIDATION_SUMMARY=" + json.dumps(summary, sort_keys=True), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parent))
    parser.add_argument(
        "--feature-mode",
        choices=["raw", "history", "history_numeric", "history_loo_numeric", "both"],
        default="history_loo_numeric",
    )
    parser.add_argument("--model", choices=["lightgbm", "catboost"], default="lightgbm")
    parser.add_argument("--full-sweep", action="store_true")
    args = parser.parse_args()

    modes = ["raw", "history_loo_numeric"] if args.feature_mode == "both" else [args.feature_mode]
    run_validation(Path(args.repo_root), modes, full_sweep=args.full_sweep, model_name=args.model)


if __name__ == "__main__":
    main()
