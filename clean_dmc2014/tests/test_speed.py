import numpy as np
import pandas as pd

from dmc2014.features import FeatureConfig, FeatureSet
from dmc2014.speed import (
    FeatureCache,
    feature_cache_key,
    frame_fingerprint,
    plan_fold_workers,
)
from dmc2014.splits import TemporalFold


def _feature_set() -> FeatureSet:
    return FeatureSet(
        X=pd.DataFrame({"x": [1.0, 2.0], "cat": ["a", "b"]}),
        y=np.array([0, 1], dtype=int),
        categorical=["cat"],
    )


def _fold() -> TemporalFold:
    return TemporalFold(
        name="jan",
        train_end=pd.Timestamp("2013-01-01"),
        valid_start=pd.Timestamp("2013-01-01"),
        valid_end=pd.Timestamp("2013-02-01"),
    )


def test_plan_fold_workers_uses_full_budget_without_oversubscription():
    assert plan_fold_workers(16, 3) == [6, 5, 5]
    assert plan_fold_workers(12, 3) == [4, 4, 4]
    assert plan_fold_workers(3, 3) == [1, 1, 1]
    assert plan_fold_workers(2, 3) == [1, 1, 1]


def test_frame_fingerprint_changes_when_input_changes():
    left = pd.DataFrame({"x": [1, 2], "returnShipment": [0, 1]})
    right = left.copy()
    right.loc[1, "returnShipment"] = 0
    assert frame_fingerprint(left) != frame_fingerprint(right)


def test_feature_cache_key_changes_with_feature_config():
    fingerprint = frame_fingerprint(pd.DataFrame({"x": [1, 2]}))
    lean = FeatureConfig(history_groups=(), recency_groups=())
    rich = FeatureConfig(history_groups=(), recency_groups=(), rolling_profiles=True)
    assert feature_cache_key(fingerprint, _fold(), lean) != feature_cache_key(
        fingerprint, _fold(), rich
    )


def test_feature_cache_round_trips_feature_sets(tmp_path):
    cache = FeatureCache(tmp_path)
    key = "a" * 64
    train = _feature_set()
    valid = _feature_set()

    assert cache.load(key) is None
    cache.store(key, train, valid)
    restored = cache.load(key)

    assert restored is not None
    restored_train, restored_valid = restored
    pd.testing.assert_frame_equal(restored_train.X, train.X)
    pd.testing.assert_frame_equal(restored_valid.X, valid.X)
    np.testing.assert_array_equal(restored_train.y, train.y)
    np.testing.assert_array_equal(restored_valid.y, valid.y)
    assert restored_train.categorical == train.categorical
