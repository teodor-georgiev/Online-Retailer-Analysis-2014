from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

import numpy as np

from dmc2014.data import load_final_labels, load_train_and_class
from dmc2014.experiment import run_backtest
from dmc2014.features import (
    DEFAULT_HISTORY_GROUPS,
    DEFAULT_RECENCY_GROUPS,
    FeatureConfig,
    build_training_features,
    build_validation_features,
)
from dmc2014.metrics import dmc_points
from dmc2014.models import fit_final_probabilities


def _feature_config_from_record(record: dict | None) -> FeatureConfig:
    record = record or {}
    history = record.get("history_groups", DEFAULT_HISTORY_GROUPS)
    recency = record.get("recency_groups", DEFAULT_RECENCY_GROUPS)
    return FeatureConfig(
        history_groups=tuple(tuple(group) for group in history),
        recency_groups=tuple(tuple(group) for group in recency),
        smoothing=float(record.get("smoothing", 20.0)),
    )


def backtest_from_zip(
    zip_path: str | Path,
    model_name: str,
    params: dict | None,
    feature_config: FeatureConfig,
) -> dict:
    train, _competition = load_train_and_class(zip_path)
    return run_backtest(
        train,
        model_name=model_name,
        params=params or {},
        feature_config=feature_config,
    )


def _recommended_iterations(config: dict) -> int:
    if "recommended_iterations" in config:
        return max(1, int(config["recommended_iterations"]))
    fold_iterations = [
        int(fold["best_iteration"])
        for fold in config.get("folds", [])
        if int(fold.get("best_iteration", 0)) > 0
    ]
    if fold_iterations:
        return max(1, int(round(median(fold_iterations))))
    params = config.get("params", {})
    if config.get("model") == "lightgbm":
        return max(1, int(params.get("n_estimators", 600)))
    return max(1, int(params.get("iterations", 500)))


def final_evaluate_from_zip(zip_path: str | Path, config: dict) -> dict:
    """Explicitly evaluate one frozen configuration on released April labels."""
    train, competition = load_train_and_class(zip_path)
    labels = load_final_labels(zip_path)
    if len(competition) != len(labels):
        raise ValueError(
            "final label alignment mismatch: "
            f"{len(competition)} competition rows != {len(labels)} labels"
        )

    model_name = str(config["model"])
    params = dict(config.get("params", {}))
    feature_config = _feature_config_from_record(config.get("feature_config"))
    threshold = float(config.get("threshold", 0.5))
    iterations = _recommended_iterations(config)

    train_features = build_training_features(train, feature_config)
    competition_features = build_validation_features(
        train,
        competition,
        feature_config,
    )
    probabilities = np.asarray(
        fit_final_probabilities(
            model_name,
            train_features,
            competition_features,
            params,
            iterations,
        ),
        dtype=float,
    )
    if probabilities.shape != (len(labels),):
        raise ValueError(
            "final prediction alignment mismatch: "
            f"{probabilities.shape} != {(len(labels),)}"
        )

    prediction = (probabilities >= threshold).astype(int)
    points = dmc_points(labels.to_numpy(dtype=int), prediction)
    soft_points = dmc_points(labels.to_numpy(dtype=int), probabilities)
    rows = int(len(labels))
    return {
        "model": model_name,
        "rows": rows,
        "threshold": threshold,
        "iterations": iterations,
        "points": float(points),
        "point_rate": float(points / rows),
        "accuracy": float(1.0 - points / rows),
        "soft_points": float(soft_points),
        "soft_point_rate": float(soft_points / rows),
    }


def _parse_json(value: str | None) -> dict:
    if not value:
        return {}
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text())
    return json.loads(value)


def _emit(result: dict, output: str | None) -> None:
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n")
    print(rendered)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dmc2014")
    subparsers = parser.add_subparsers(dest="command", required=True)

    backtest = subparsers.add_parser("backtest")
    backtest.add_argument("--zip", required=True, dest="zip_path")
    backtest.add_argument("--model", choices=["catboost", "lightgbm"], default="catboost")
    backtest.add_argument("--params-json", default=None)
    backtest.add_argument("--smoothing", type=float, default=20.0)
    backtest.add_argument("--no-history", action="store_true")
    backtest.add_argument("--no-recency", action="store_true")
    backtest.add_argument("--output", default=None)

    final = subparsers.add_parser("final-evaluate")
    final.add_argument("--zip", required=True, dest="zip_path")
    final.add_argument("--config-json", required=True)
    final.add_argument("--output", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "backtest":
        params = _parse_json(args.params_json)
        feature_config = FeatureConfig(
            history_groups=() if args.no_history else DEFAULT_HISTORY_GROUPS,
            recency_groups=() if args.no_recency else DEFAULT_RECENCY_GROUPS,
            smoothing=args.smoothing,
        )
        result = backtest_from_zip(
            args.zip_path,
            args.model,
            params,
            feature_config,
        )
        _emit(result, args.output)
        return

    config = _parse_json(args.config_json)
    result = final_evaluate_from_zip(args.zip_path, config)
    _emit(result, args.output)


if __name__ == "__main__":
    main()
