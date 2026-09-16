# DMC 2014 User/Product Profiles + Ensemble Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add leakage-safe user/product causal profiles and an OOF-selected CatBoost + LightGBM ensemble to the clean DMC 2014 benchmark.

**Architecture:** Keep `features.py` as orchestration, add a focused `profiles.py` module for causal predictor-only profile state, and add ensemble selection to `experiment.py`. Both models consume one aligned `FeatureSet`; ensemble weight and threshold are chosen only from Jan/Feb/Mar OOF predictions.

**Tech Stack:** Python 3.12+, pandas 3.x, NumPy, CatBoost, LightGBM, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-09-16-user-product-ensemble-design.md`

## Global Constraints

- Historical notebooks remain audit-only and are never imported by `clean_dmc2014`.
- Every profile feature for date `d` uses predictor information only from rows with `orderDate < d`; same-day rows never update each other.
- Profile features never read `returnShipment`.
- April labels remain sealed during feature generation, model fitting, blend selection, and threshold selection.
- CatBoost and LightGBM consume the same aligned feature columns.
- Heavy work uses the existing dynamic VPS batch-capacity policy and avoids nested oversubscription.

---

### Task 1: Causal user/product profile engine

**Files:**
- Create: `clean_dmc2014/src/dmc2014/profiles.py`
- Create: `clean_dmc2014/tests/test_profiles.py`

**Interfaces:**
- Consumes: raw/predictor DMC frames containing `orderDate`, `customerID`, `itemID`, `manufacturerID`, `size`, `color`, `state`, and `price`.
- Produces: `build_training_profiles(frame: pd.DataFrame, *, user_profiles: bool, product_profiles: bool) -> pd.DataFrame` and `build_validation_profiles(history: pd.DataFrame, target: pd.DataFrame, *, user_profiles: bool, product_profiles: bool) -> pd.DataFrame`.

- [ ] **Step 1: Write failing invariance and same-date tests**

Add tests that construct tiny dated frames and assert:

```python
base = build_training_profiles(frame, user_profiles=True, product_profiles=True)
mutated = frame.copy()
mutated["returnShipment"] = 1 - mutated["returnShipment"]
assert_frame_equal(base, build_training_profiles(mutated, user_profiles=True, product_profiles=True))
```

and two rows on the same date for one customer/item have identical pre-date counts, while a later-date row sees both earlier rows.

- [ ] **Step 2: Run focused tests and confirm red state**

Run:

```bash
uv run pytest tests/test_profiles.py -q
```

Expected: import/function failures because `dmc2014.profiles` does not exist.

- [ ] **Step 3: Implement date-batched causal profile state**

Implement helpers that sort by `orderDate`, emit features for a complete date batch from state accumulated on strictly earlier dates, then update state after the batch. Maintain focused state for customer, item, manufacturer, customer-item, and customer-manufacturer keys. Use numeric counters/sets/sums/sumsq/min/max/first/last dates; do not read `returnShipment`.

Required output columns include the feature families named in the design spec, including user counts/diversity/spend/price/basket/gaps/familiarity and item/manufacturer counts/diversity/revenue/price/recency.

- [ ] **Step 4: Run focused tests and confirm green**

Run:

```bash
uv run pytest tests/test_profiles.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add clean_dmc2014/src/dmc2014/profiles.py clean_dmc2014/tests/test_profiles.py
git commit -m "feat: add causal user and product profiles"
```

---

### Task 2: Integrate profile flags into the maintained feature pipeline

**Files:**
- Modify: `clean_dmc2014/src/dmc2014/features.py`
- Modify: `clean_dmc2014/tests/test_features_history.py`
- Modify: `clean_dmc2014/tests/test_features_base.py`

**Interfaces:**
- Consumes: Task 1 profile builders.
- Produces: `FeatureConfig.user_profiles: bool = False` and `FeatureConfig.product_profiles: bool = False`; training/validation feature builders append row-aligned profile columns when enabled.

- [ ] **Step 1: Write failing alignment/leakage tests**

Add a test enabling both flags and asserting training/validation column lists are identical. Add a validation-label mutation test:

```python
first = build_validation_features(history, validation, config).X
mutated = validation.copy()
mutated["returnShipment"] = 1 - mutated["returnShipment"]
second = build_validation_features(history, mutated, config).X
assert_frame_equal(first, second)
```

- [ ] **Step 2: Run focused tests and confirm red state**

```bash
uv run pytest tests/test_features_base.py tests/test_features_history.py -q
```

Expected: failure because `FeatureConfig` lacks profile flags/profile columns.

- [ ] **Step 3: Implement profile integration**

Extend `FeatureConfig` with:

```python
user_profiles: bool = False
product_profiles: bool = False
```

In `build_training_features`, append `build_training_profiles(...)`. In `build_validation_features`, append `build_validation_profiles(history, validation, ...)`. Reject duplicate column names before concatenation.

- [ ] **Step 4: Run focused tests**

```bash
uv run pytest tests/test_features_base.py tests/test_features_history.py tests/test_profiles.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add clean_dmc2014/src/dmc2014/features.py clean_dmc2014/tests/test_features_base.py clean_dmc2014/tests/test_features_history.py
git commit -m "feat: integrate causal profile features"
```

---

### Task 3: Add deterministic OOF ensemble selection

**Files:**
- Modify: `clean_dmc2014/src/dmc2014/experiment.py`
- Create: `clean_dmc2014/tests/test_ensemble.py`

**Interfaces:**
- Consumes: existing CatBoost/LightGBM fitters and profile-enabled `FeatureConfig`.
- Produces: `select_blend(y_true: np.ndarray, cat_probability: np.ndarray, lgb_probability: np.ndarray, weights: Sequence[float] | None = None) -> dict` and `run_ensemble_backtest(...) -> dict`.

- [ ] **Step 1: Write failing blend tests**

Create deterministic toy probabilities that make one blend weight uniquely best. Assert returned `weight`, `threshold`, and `points`. Add a tie case and assert tie-breaking uses closest-to-0.5 weight, then closest-to-0.5 threshold, then lower CatBoost weight.

- [ ] **Step 2: Run focused tests and confirm red state**

```bash
uv run pytest tests/test_ensemble.py -q
```

Expected: missing `select_blend`/`run_ensemble_backtest`.

- [ ] **Step 3: Implement OOF blend selection**

Use the deterministic grid:

```python
weights = np.round(np.arange(0.0, 1.0001, 0.05), 2)
```

For each weight, blend OOF probabilities and call existing `best_threshold`. Compare DMC points; implement tie-breakers exactly as specified. `run_ensemble_backtest` must build each fold's feature matrices once, fit both models, retain aligned probabilities, concatenate OOF arrays, call `select_blend`, and report component/ensemble metrics.

- [ ] **Step 4: Run focused tests**

```bash
uv run pytest tests/test_ensemble.py tests/test_experiment.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add clean_dmc2014/src/dmc2014/experiment.py clean_dmc2014/tests/test_ensemble.py
git commit -m "feat: add OOF CatBoost LightGBM ensemble"
```

---

### Task 4: Expose profile and ensemble controls in the CLI

**Files:**
- Modify: `clean_dmc2014/src/dmc2014/cli.py`
- Modify: `clean_dmc2014/tests/test_cli.py`

**Interfaces:**
- Consumes: profile-enabled `FeatureConfig` and `run_ensemble_backtest`.
- Produces: `--user-profiles`, `--product-profiles`, and `--model ensemble`; ensemble params JSON shape `{ "catboost": {...}, "lightgbm": {...} }`.

- [ ] **Step 1: Write failing CLI tests**

Assert parser accepts:

```text
backtest --model ensemble --user-profiles --product-profiles
```

and `_feature_config_from_record` round-trips the two booleans.

- [ ] **Step 2: Run focused tests and confirm red state**

```bash
uv run pytest tests/test_cli.py -q
```

Expected: parser/config failures.

- [ ] **Step 3: Implement CLI routing**

Add the two profile flags to `FeatureConfig` creation and record parsing. Route `model_name == "ensemble"` to `run_ensemble_backtest`; keep CatBoost/LightGBM behavior unchanged. Do not add final April ensemble evaluation yet; this task is model-selection only.

- [ ] **Step 4: Run focused tests**

```bash
uv run pytest tests/test_cli.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add clean_dmc2014/src/dmc2014/cli.py clean_dmc2014/tests/test_cli.py
git commit -m "feat: expose profile ensemble backtests"
```

---

### Task 5: Full verification and VPS ablations

**Files:**
- Modify only if verification exposes a defect; no planned production/default-branch edits.
- Persist result JSON under `/srv/chatgpt-results/dmc2014/user-product-ensemble/<run-id>/` when the reviewed result store is available.

**Interfaces:**
- Consumes: Tasks 1-4.
- Produces: comparable OOF result records for baseline, user-only, product-only, both, LightGBM both, and ensemble both.

- [ ] **Step 1: Run complete clean-room test suite**

```bash
uv run pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Run CatBoost profile ablations**

Run three rolling backtests with the existing best CatBoost parameters (`iterations=120`, `depth=7`, `learning_rate=0.07`, `max_ctr_complexity=2`, no early stopping), keeping target-history disabled and recency enabled:

1. user profiles only
2. product profiles only
3. both profile families

Use the dynamic batch worker policy before each independent run.

- [ ] **Step 3: Run LightGBM with both profiles**

Use a bounded baseline configuration (`n_estimators=600`, `learning_rate=0.04`, `num_leaves=63`, `min_child_samples=80`, existing regularization defaults), with both profile families and recency enabled.

- [ ] **Step 4: Run the ensemble backtest**

Run CatBoost + LightGBM on the same both-profile folds, select blend weight and threshold from concatenated OOF predictions only, and write the JSON result.

- [ ] **Step 5: Compare against current best**

Report projected 50,078-row DMC points for:

- existing best: ~15,944
- user-only CatBoost
- product-only CatBoost
- both-profile CatBoost
- both-profile LightGBM
- both-profile ensemble

Promote only a configuration with a lower OOF point estimate than the existing best. Do not open April labels.

- [ ] **Step 6: Review diff and exact branch state**

Inspect the complete branch diff versus `feat/dmc2014-dynamic-capacity`, rerun the full test suite if any verification fix was required, and leave the branch unmerged for user review.
