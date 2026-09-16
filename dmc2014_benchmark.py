from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd


CATEGORICAL_COLUMNS = [
    "itemID",
    "size",
    "color",
    "manufacturerID",
    "customerID",
    "salutation",
    "state",
]

DEFAULT_GROUP_SPECS = [
    ("customerID",),
    ("itemID",),
    ("manufacturerID",),
    ("size",),
    ("color",),
    ("state",),
    ("customerID", "manufacturerID"),
    ("customerID", "size"),
    ("manufacturerID", "itemID"),
    ("itemID", "size"),
]


def dmc_score(y_true: Sequence[float], prediction: Sequence[float]) -> float:
    """Return the Data Mining Cup 2014 absolute-error point total."""
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    if truth.shape != pred.shape:
        raise ValueError(f"shape mismatch: {truth.shape} != {pred.shape}")
    return float(np.abs(truth - pred).sum())


def best_threshold(
    y_true: Sequence[int],
    probability: Sequence[float],
    thresholds: Sequence[float] | None = None,
) -> tuple[float, float]:
    """Choose the probability threshold with the lowest hard DMC point total."""
    truth = np.asarray(y_true, dtype=int)
    prob = np.asarray(probability, dtype=float)
    if truth.shape != prob.shape:
        raise ValueError(f"shape mismatch: {truth.shape} != {prob.shape}")
    if thresholds is None:
        thresholds = np.linspace(0.30, 0.70, 161)
    candidates = [float(value) for value in thresholds]
    if not candidates:
        raise ValueError("thresholds must not be empty")

    scored = []
    for threshold in candidates:
        prediction = (prob >= threshold).astype(int)
        scored.append((dmc_score(truth, prediction), threshold))
    points, threshold = min(scored, key=lambda item: (item[0], abs(item[1] - 0.5), item[1]))
    return float(threshold), float(points)


def load_competition_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and competition predictors without touching released test labels."""
    root = Path(data_dir)
    train = pd.read_csv(root / "orders_train.txt", sep=";")
    test = pd.read_csv(root / "orders_class.txt", sep=";")
    return train, test


def load_realclass(data_dir: str | Path) -> pd.DataFrame:
    """Load released April labels; call only after model selection is frozen."""
    return pd.read_csv(Path(data_dir) / "orders_realclass.txt", sep=";")


def split_train_validation(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split original training rows into history through February and March validation."""
    data = frame.copy()
    data["orderDate"] = pd.to_datetime(data["orderDate"], errors="raise")
    validation_start = pd.Timestamp("2013-03-01")
    validation_end = pd.Timestamp("2013-04-01")
    valid_mask = data["orderDate"].between(validation_start, validation_end, inclusive="left")
    train = data.loc[data["orderDate"] < validation_start].copy()
    valid = data.loc[valid_mask].copy()
    return train, valid


def _as_datetime(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    values = frame[column].replace({"?": pd.NA, "": pd.NA})
    return pd.to_datetime(values, errors="coerce")


def build_row_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Create prediction-time row features without using return labels."""
    output = frame.drop(columns=["returnShipment"], errors="ignore").copy()

    order_date = _as_datetime(frame, "orderDate")
    delivery_date = _as_datetime(frame, "deliveryDate")
    birth_date = _as_datetime(frame, "dateOfBirth")
    creation_date = _as_datetime(frame, "creationDate")

    output["order_year"] = order_date.dt.year.astype("float64")
    output["order_month"] = order_date.dt.month.astype("float64")
    output["order_day"] = order_date.dt.day.astype("float64")
    output["order_dayofweek"] = order_date.dt.dayofweek.astype("float64")
    output["order_dayofyear"] = order_date.dt.dayofyear.astype("float64")
    output["delivery_year"] = delivery_date.dt.year.astype("float64")
    output["delivery_month"] = delivery_date.dt.month.astype("float64")
    output["delivery_dayofweek"] = delivery_date.dt.dayofweek.astype("float64")
    output["delivery_missing"] = delivery_date.isna().astype("int8")
    output["delivery_delay_days"] = (delivery_date - order_date).dt.total_seconds() / 86400.0
    output["customer_age_years"] = (order_date - birth_date).dt.total_seconds() / (86400.0 * 365.2425)
    output["account_age_days"] = (order_date - creation_date).dt.total_seconds() / 86400.0

    if "price" in frame.columns:
        price = pd.to_numeric(frame["price"], errors="coerce")
        output["price"] = price
        output["log1p_price"] = np.log1p(price.clip(lower=0))

    if {"customerID", "orderDate"}.issubset(frame.columns):
        order_key = [frame["customerID"], order_date]
        output["basket_item_count"] = frame.groupby(order_key, dropna=False)["orderItemID"].transform("size").astype(float)
        if "price" in frame.columns:
            price = pd.to_numeric(frame["price"], errors="coerce")
            output["basket_total_price"] = price.groupby(order_key, dropna=False).transform("sum")
            output["basket_mean_price"] = price.groupby(order_key, dropna=False).transform("mean")
        if "itemID" in frame.columns:
            output["basket_unique_items"] = frame.groupby(order_key, dropna=False)["itemID"].transform("nunique").astype(float)

    output = output.drop(columns=["orderDate", "deliveryDate", "dateOfBirth", "creationDate"], errors="ignore")

    categorical = [column for column in CATEGORICAL_COLUMNS if column in output.columns]
    for column in categorical:
        values = output[column].astype("string").fillna("__MISSING__")
        output[column] = values.replace({"?": "__MISSING__"}).astype(str)

    return output, categorical


def _history_prefix(columns: tuple[str, ...]) -> str:
    return "_x_".join(columns)


def add_history_features(
    history: pd.DataFrame,
    target: pd.DataFrame,
    group_specs: Iterable[tuple[str, ...]],
    smoothing: float = 20.0,
) -> pd.DataFrame:
    """Map history-only counts and smoothed return rates onto target rows."""
    if "returnShipment" not in history.columns:
        raise ValueError("history must contain returnShipment")
    if smoothing < 0:
        raise ValueError("smoothing must be non-negative")

    output = target.copy()
    prior = float(history["returnShipment"].mean()) if len(history) else 0.5

    for columns in group_specs:
        columns = tuple(columns)
        if not columns:
            raise ValueError("group specs must not be empty")
        missing = [column for column in columns if column not in history.columns or column not in target.columns]
        if missing:
            raise KeyError(f"missing grouping columns: {missing}")

        stats = (
            history.groupby(list(columns), dropna=False, observed=True)["returnShipment"]
            .agg(["sum", "count"])
            .reset_index()
        )
        if len(stats):
            denominator = stats["count"] + smoothing
            stats["return_rate"] = np.where(
                denominator > 0,
                (stats["sum"] + smoothing * prior) / denominator,
                prior,
            )
        else:
            stats["return_rate"] = pd.Series(dtype=float)

        prefix = _history_prefix(columns)
        count_name = f"hist_{prefix}_count"
        rate_name = f"hist_{prefix}_return_rate"
        mapped = target[list(columns)].merge(
            stats[list(columns) + ["count", "return_rate"]],
            how="left",
            on=list(columns),
            sort=False,
        )
        output[count_name] = mapped["count"].fillna(0).astype("int64").to_numpy()
        output[rate_name] = mapped["return_rate"].fillna(prior).astype(float).to_numpy()

    return output


def prepare_training_features(
    frame: pd.DataFrame,
    group_specs: Iterable[tuple[str, ...]] = DEFAULT_GROUP_SPECS,
    smoothing: float = 20.0,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Build month-expanding training features so a row never sees same/future labels."""
    data = frame.copy().reset_index(drop=True)
    data["orderDate"] = pd.to_datetime(data["orderDate"], errors="raise")
    month_key = data["orderDate"].dt.to_period("M")
    blocks: list[pd.DataFrame] = []
    categorical: list[str] = []

    for month in sorted(month_key.unique()):
        target_mask = month_key == month
        target_rows = data.loc[target_mask].copy()
        history_rows = data.loc[month_key < month].copy()
        enriched = add_history_features(history_rows, target_rows, group_specs, smoothing=smoothing)
        block, categorical = build_row_features(enriched)
        block["__row_order__"] = target_rows.index.to_numpy()
        blocks.append(block)

    features = pd.concat(blocks, ignore_index=True).sort_values("__row_order__")
    row_order = features.pop("__row_order__").astype(int).to_numpy()
    features = features.reset_index(drop=True)
    target = data.loc[row_order, "returnShipment"].to_numpy(dtype=int)
    return features, target, categorical


def prepare_prediction_features(
    history: pd.DataFrame,
    target: pd.DataFrame,
    group_specs: Iterable[tuple[str, ...]] = DEFAULT_GROUP_SPECS,
    smoothing: float = 20.0,
) -> tuple[pd.DataFrame, list[str]]:
    """Build features for a future window using only supplied labeled history."""
    enriched = add_history_features(history, target, group_specs, smoothing=smoothing)
    return build_row_features(enriched)


def candidate_configs() -> list[dict]:
    """Return a small deterministic CatBoost sweep for temporal validation."""
    common = {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "random_seed": 42,
        "allow_writing_files": False,
        "verbose": False,
        "thread_count": -1,
    }
    return [
        {
            "name": "catboost_d6_lr007",
            "params": {**common, "iterations": 800, "depth": 6, "learning_rate": 0.07, "l2_leaf_reg": 5.0},
        },
        {
            "name": "catboost_d7_lr005",
            "params": {**common, "iterations": 1100, "depth": 7, "learning_rate": 0.05, "l2_leaf_reg": 6.0},
        },
        {
            "name": "catboost_d8_lr0035",
            "params": {**common, "iterations": 1400, "depth": 8, "learning_rate": 0.035, "l2_leaf_reg": 8.0},
        },
    ]


def fit_candidate(
    train_x: pd.DataFrame,
    train_y: Sequence[int],
    valid_x: pd.DataFrame,
    valid_y: Sequence[int],
    categorical_columns: Sequence[str],
    params: dict,
):
    """Fit one CatBoost candidate and return validation probabilities."""
    from catboost import CatBoostClassifier

    missing = sorted(set(categorical_columns) - set(train_x.columns))
    if missing:
        raise KeyError(f"categorical columns missing from train_x: {missing}")
    if list(train_x.columns) != list(valid_x.columns):
        raise ValueError("train_x and valid_x must have identical columns in identical order")

    model = CatBoostClassifier(**params)
    model.fit(
        train_x,
        np.asarray(train_y, dtype=int),
        cat_features=list(categorical_columns),
        eval_set=(valid_x, np.asarray(valid_y, dtype=int)),
        use_best_model=True,
        early_stopping_rounds=80,
        verbose=False,
    )
    probability = model.predict_proba(valid_x)[:, 1].astype(float)
    zero_based_best = int(model.get_best_iteration())
    best_iteration = zero_based_best + 1 if zero_based_best >= 0 else int(model.tree_count_)
    return model, probability, best_iteration


def lightgbm_configs() -> list[dict]:
    """Return a deterministic LightGBM sweep sized for the CPU VPS."""
    common = {
        "objective": "binary",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
    }
    return [
        {
            "name": "lightgbm_l31_lr005",
            "params": {
                **common,
                "n_estimators": 800,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "min_child_samples": 100,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_lambda": 2.0,
                "cat_smooth": 20.0,
            },
        },
        {
            "name": "lightgbm_l63_lr0035",
            "params": {
                **common,
                "n_estimators": 1200,
                "learning_rate": 0.035,
                "num_leaves": 63,
                "min_child_samples": 80,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_lambda": 3.0,
                "cat_smooth": 25.0,
            },
        },
        {
            "name": "lightgbm_l127_lr003",
            "params": {
                **common,
                "n_estimators": 1400,
                "learning_rate": 0.03,
                "num_leaves": 127,
                "max_depth": 8,
                "min_child_samples": 120,
                "subsample": 0.9,
                "colsample_bytree": 0.85,
                "reg_lambda": 4.0,
                "cat_smooth": 30.0,
            },
        },
    ]


def _align_lightgbm_categories(
    train_x: pd.DataFrame,
    valid_x: pd.DataFrame,
    categorical_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train_x.copy()
    valid = valid_x.copy()
    for column in categorical_columns:
        if column not in train.columns or column not in valid.columns:
            raise KeyError(f"categorical column missing: {column}")
        train_values = train[column].astype("string").fillna("__MISSING__").astype(str)
        valid_values = valid[column].astype("string").fillna("__MISSING__").astype(str)
        levels = pd.Index(train_values.unique()).union(pd.Index(valid_values.unique()))
        dtype = pd.CategoricalDtype(categories=levels)
        train[column] = train_values.astype(dtype)
        valid[column] = valid_values.astype(dtype)
    return train, valid


def fit_lightgbm_candidate(
    train_x: pd.DataFrame,
    train_y: Sequence[int],
    valid_x: pd.DataFrame,
    valid_y: Sequence[int],
    categorical_columns: Sequence[str],
    params: dict,
):
    """Fit one LightGBM candidate with aligned pandas categorical columns."""
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation

    if list(train_x.columns) != list(valid_x.columns):
        raise ValueError("train_x and valid_x must have identical columns in identical order")
    train_aligned, valid_aligned = _align_lightgbm_categories(train_x, valid_x, categorical_columns)

    model = LGBMClassifier(**params)
    model.fit(
        train_aligned,
        np.asarray(train_y, dtype=int),
        eval_set=[(valid_aligned, np.asarray(valid_y, dtype=int))],
        eval_metric="binary_logloss",
        categorical_feature=list(categorical_columns),
        callbacks=[early_stopping(80, verbose=False), log_evaluation(0)],
    )
    probability = model.predict_proba(valid_aligned, num_iteration=model.best_iteration_)[:, 1].astype(float)
    best_iteration = int(model.best_iteration_ or params.get("n_estimators", 1))
    return model, probability, best_iteration


def choose_best_result(results: Sequence[dict]) -> dict:
    """Return the candidate with the lowest validation DMC point total."""
    if not results:
        raise ValueError("results must not be empty")
    return min(results, key=lambda result: float(result["validation_points"]))
