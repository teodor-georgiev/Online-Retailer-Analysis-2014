# DMC 2014 Modern Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible, leakage-safe modern CatBoost benchmark for DMC 2014 and measure whether it improves the repository's 14,893-point result and the historical 14,165 winning score.

**Architecture:** A focused `dmc2014_benchmark.py` module loads the original archive, creates the temporal March validation split, builds row-level and history-only aggregate features, trains a bounded CatBoost sweep, freezes the best March configuration, retrains on all labeled training rows, and computes the exact April score from `orders_realclass.txt`. Unit tests cover scoring, temporal splitting, and historical aggregate leakage boundaries without training a full model.

**Tech Stack:** Python 3.12, pandas, NumPy, scikit-learn, CatBoost.

**Spec:** `docs/superpowers/specs/2026-09-16-dmc2014-modern-benchmark-design.md`

## Global Constraints

- `orders_realclass.txt` is forbidden during model/parameter selection and may be read only by the final exact-scoring step.
- March 2013 is the only model-selection validation window.
- Historical target statistics for a prediction window use labels strictly before that prediction window.
- Source data is read-only.
- Work stays on `feat/dmc2014-modern-benchmark`; no merge or deployment.
- VPS execution uses an isolated temporary clone and isolated Python environment.

---

### Task 1: Score and temporal split contract

**Files:**
- Create: `tests/test_dmc2014_benchmark.py`
- Create: `dmc2014_benchmark.py`

**Interfaces:**
- Produces: `dmc_score(y_true, prediction) -> float`
- Produces: `split_train_validation(frame) -> tuple[pd.DataFrame, pd.DataFrame]`

- [ ] **Step 1: Write failing tests**

```python
import numpy as np
import pandas as pd

from dmc2014_benchmark import dmc_score, split_train_validation


def test_dmc_score_is_sum_absolute_error():
    y = np.array([0.0, 1.0, 1.0])
    p = np.array([0.2, 0.7, 0.9])
    assert dmc_score(y, p) == 0.6


def test_split_uses_march_2013_as_validation_only():
    frame = pd.DataFrame({
        "orderDate": ["2013-02-28", "2013-03-01", "2013-03-31"],
        "returnShipment": [0, 1, 0],
    })
    train, valid = split_train_validation(frame)
    assert train["orderDate"].dt.strftime("%Y-%m-%d").tolist() == ["2013-02-28"]
    assert valid["orderDate"].dt.strftime("%Y-%m-%d").tolist() == ["2013-03-01", "2013-03-31"]
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `pytest -q tests/test_dmc2014_benchmark.py`
Expected: import failure because `dmc2014_benchmark.py` does not exist yet.

- [ ] **Step 3: Implement the minimal functions**

Implement `dmc_score` with `np.abs(...).sum()` and normalize `orderDate` with `pd.to_datetime`; validation is `[2013-03-01, 2013-04-01)`.

- [ ] **Step 4: Re-run tests and confirm GREEN**

Run: `pytest -q tests/test_dmc2014_benchmark.py`
Expected: both tests pass.

### Task 2: Leakage-safe feature builder

**Files:**
- Modify: `tests/test_dmc2014_benchmark.py`
- Modify: `dmc2014_benchmark.py`

**Interfaces:**
- Produces: `build_row_features(frame) -> pd.DataFrame`
- Produces: `add_history_features(history, target, group_specs, smoothing=20.0) -> pd.DataFrame`
- Produces: `prepare_window(history, target) -> tuple[pd.DataFrame, list[str]]`

- [ ] **Step 1: Add failing tests for unseen-group fallback and no target leakage**

```python
def test_history_features_use_history_only_and_fallback_to_prior():
    history = pd.DataFrame({
        "customerID": [1, 1, 2],
        "returnShipment": [1, 0, 0],
    })
    target = pd.DataFrame({"customerID": [1, 3]})
    out = add_history_features(history, target, [("customerID",)], smoothing=2.0)
    prior = history["returnShipment"].mean()
    expected_seen = (1 + 2.0 * prior) / (2 + 2.0)
    assert np.isclose(out.loc[0, "hist_customerID_return_rate"], expected_seen)
    assert np.isclose(out.loc[1, "hist_customerID_return_rate"], prior)
    assert out.loc[1, "hist_customerID_count"] == 0
```

- [ ] **Step 2: Run the targeted test and confirm RED**

Run: `pytest -q tests/test_dmc2014_benchmark.py::test_history_features_use_history_only_and_fallback_to_prior`
Expected: import/name failure for `add_history_features`.

- [ ] **Step 3: Implement row and historical features**

Implement date-derived numeric features, delivery delay/missing indicator, customer age, account age, log price, categorical string normalization, and smoothed count/rate mappings for the groupings in the design. Do not use target labels from the target frame.

- [ ] **Step 4: Run all unit tests**

Run: `pytest -q tests/test_dmc2014_benchmark.py`
Expected: pass.

### Task 3: CatBoost training and bounded sweep

**Files:**
- Modify: `tests/test_dmc2014_benchmark.py`
- Modify: `dmc2014_benchmark.py`

**Interfaces:**
- Produces: `candidate_configs() -> list[dict]`
- Produces: `fit_candidate(train_x, train_y, valid_x, valid_y, cat_columns, params) -> tuple[object, np.ndarray]`
- Produces: `select_best_candidate(...) -> dict`

- [ ] **Step 1: Add a deterministic test for candidate selection**

```python
def test_candidate_selection_prefers_lower_dmc_score():
    results = [
        {"name": "a", "validation_points": 10.0},
        {"name": "b", "validation_points": 8.0},
    ]
    assert choose_best_result(results)["name"] == "b"
```

- [ ] **Step 2: Run and confirm RED**

Run: `pytest -q tests/test_dmc2014_benchmark.py::test_candidate_selection_prefers_lower_dmc_score`
Expected: import/name failure for `choose_best_result`.

- [ ] **Step 3: Implement the bounded sweep**

Use CatBoost binary logloss training with deterministic seed 42, CPU thread count bounded to available cores, early stopping on March, `allow_writing_files=False`, and a small fixed configuration list. Keep probability predictions continuous for DMC MAE.

- [ ] **Step 4: Run all unit tests**

Run: `pytest -q tests/test_dmc2014_benchmark.py`
Expected: pass.

### Task 4: Exact leaderboard evaluation CLI

**Files:**
- Modify: `tests/test_dmc2014_benchmark.py`
- Modify: `dmc2014_benchmark.py`

**Interfaces:**
- Produces CLI: `python dmc2014_benchmark.py --data-dir <dir> --output-json <path>`
- Produces JSON fields: `train_rows`, `test_rows`, `march_results`, `selected_config`, `march_mae`, `march_points`, `april_mae`, `april_points`, `beats_repo_14893`, `beats_winner_14165`, `runtime_seconds`.

- [ ] **Step 1: Add a test that real-class labels are loaded only in final-scoring helper**

Keep `load_competition_data()` limited to train/class; implement a separate `load_realclass()` helper and test both APIs independently with temporary CSV fixtures.

- [ ] **Step 2: Run and confirm RED**

Run: `pytest -q tests/test_dmc2014_benchmark.py`
Expected: failure until the separate loaders and CLI orchestration exist.

- [ ] **Step 3: Implement final retraining and exact April score**

Freeze the March-winning params, rebuild April features using all original training labels as history, train on all labeled rows, predict `orders_class.txt`, align with `orders_realclass.txt` by `orderItemID`, calculate exact DMC points, and write compact JSON.

- [ ] **Step 4: Verify the complete implementation**

Run: `pytest -q tests/test_dmc2014_benchmark.py`
Expected: pass.

Run: `python dmc2014_benchmark.py --data-dir data --output-json benchmark_result.json`
Expected: JSON result with exact April points and no source-data modification.

- [ ] **Step 5: Review diff and benchmark evidence**

Review only `dmc2014_benchmark.py`, `tests/test_dmc2014_benchmark.py`, and the two design/plan documents. Keep the branch unmerged and report the exact tested revision plus March and April scores.
