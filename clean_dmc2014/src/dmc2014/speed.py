from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickle
from typing import Any

import pandas as pd

from dmc2014.features import FeatureConfig, FeatureSet
from dmc2014.splits import TemporalFold


CACHE_SCHEMA_VERSION = 1


def plan_fold_workers(total_workers: int, fold_count: int) -> list[int]:
    """Split one CPU budget across folds without per-fold oversubscription."""
    if total_workers < 1:
        raise ValueError("total_workers must be positive")
    if fold_count < 1:
        raise ValueError("fold_count must be positive")
    if total_workers < fold_count:
        return [1] * fold_count
    quotient, remainder = divmod(total_workers, fold_count)
    return [quotient + (1 if index < remainder else 0) for index in range(fold_count)]


def frame_fingerprint(frame: pd.DataFrame) -> str:
    """Hash all input values, index, columns and dtypes used by feature building."""
    digest = hashlib.sha256()
    digest.update(json.dumps(list(frame.columns), separators=(",", ":")).encode("utf-8"))
    digest.update(json.dumps([str(dtype) for dtype in frame.dtypes], separators=(",", ":")).encode("utf-8"))
    hashed = pd.util.hash_pandas_object(frame, index=True, categorize=True)
    digest.update(hashed.to_numpy(dtype="uint64", copy=False).tobytes())
    return digest.hexdigest()


def _feature_code_fingerprint() -> str:
    digest = hashlib.sha256()
    directory = Path(__file__).resolve().parent
    for name in ("features.py", "profiles.py", "rolling.py"):
        path = directory / name
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _config_payload(config: FeatureConfig) -> dict[str, Any]:
    return {
        "history_groups": [list(group) for group in config.history_groups],
        "recency_groups": [list(group) for group in config.recency_groups],
        "smoothing": float(config.smoothing),
        "user_profiles": bool(config.user_profiles),
        "product_profiles": bool(config.product_profiles),
        "rolling_profiles": bool(config.rolling_profiles),
    }


def feature_cache_key(
    dataset_fingerprint: str,
    fold: TemporalFold,
    config: FeatureConfig,
    *,
    code_fingerprint: str | None = None,
) -> str:
    if len(dataset_fingerprint) != 64:
        raise ValueError("dataset_fingerprint must be a sha256 digest")
    payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "dataset": dataset_fingerprint,
        "feature_code": code_fingerprint or _feature_code_fingerprint(),
        "fold": {
            "name": fold.name,
            "train_end": pd.Timestamp(fold.train_end).isoformat(),
            "valid_start": pd.Timestamp(fold.valid_start).isoformat(),
            "valid_end": pd.Timestamp(fold.valid_end).isoformat(),
        },
        "config": _config_payload(config),
    }
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


class FeatureCache:
    """Small atomic disk cache for already-aligned train/validation FeatureSets."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser()

    def _path(self, key: str) -> Path:
        if len(key) != 64 or any(character not in "0123456789abcdef" for character in key):
            raise ValueError("cache key must be lowercase sha256 hex")
        return self.root / f"{key}.pkl"

    def load(self, key: str) -> tuple[FeatureSet, FeatureSet] | None:
        path = self._path(key)
        if not path.is_file() or path.is_symlink():
            return None
        try:
            with path.open("rb") as handle:
                payload = pickle.load(handle)
        except (OSError, EOFError, pickle.UnpicklingError):
            return None
        if not isinstance(payload, dict) or payload.get("schema") != CACHE_SCHEMA_VERSION:
            return None
        train = payload.get("train")
        valid = payload.get("valid")
        if not isinstance(train, FeatureSet) or not isinstance(valid, FeatureSet):
            return None
        return train, valid

    def store(self, key: str, train: FeatureSet, valid: FeatureSet) -> Path:
        path = self._path(key)
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        payload = {
            "schema": CACHE_SCHEMA_VERSION,
            "train": train,
            "valid": valid,
        }
        try:
            with temporary.open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        return path
