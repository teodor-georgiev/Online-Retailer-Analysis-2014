# DMC 2014 Clean-Room Benchmark Design

## Goal
Build a new, leakage-safe Python research project for the Data Mining Cup 2014 return-shipment task using only the raw competition files. The project should maximize honest out-of-time performance and record an exact final score against the released April labels only after model selection is frozen.

## Data boundary
The only source data are the original competition files in `Orders_data/Orders_train_test_class.zip`:
- `orders_train.txt` — labeled April 2012 through March 2013 history.
- `orders_class.txt` — April 2013 competition predictors.
- `orders_realclass.txt` — released April labels.

The notebooks, generated CSVs, and `Model_functions.py` are not imported or used for feature generation, model selection, or evaluation. They are historical reference only.

## Leakage policy
1. April 2013 labels are inaccessible to normal training, feature-building, cross-validation, and tuning code.
2. A separate final-evaluation command is the only code path allowed to load `orders_realclass.txt`.
3. Validation is chronological. Training rows always precede the validation month.
4. Target-derived history features for a validation month are computed only from labeled rows strictly before that validation month.
5. Training target-history features are expanding/shifted so a training row never sees its own label or any future label.
6. Predictor-only basket/order features may use all rows belonging to the same order because those predictors are known at prediction time.

## Validation design
Use rolling monthly holdouts for January, February, and March 2013. For each fold, train on all labeled rows before the validation month and evaluate that month. Primary model-selection metric is total DMC absolute error using hard 0/1 predictions. Report accuracy and probability MAE as diagnostics.

Thresholds are selected inside the temporal validation process. No threshold is selected from April labels.

## Feature families
Start simple and add features by ablation:
1. Raw row features: price, item/manufacturer/customer IDs, size, color, state, salutation.
2. Calendar and delivery features: order month/day/week, delivery lag, account age, customer age, missingness flags.
3. Basket features: basket size, total/mean/max price, unique item/manufacturer/size/color counts, relative price within basket.
4. Predictor-history features: prior order counts and recency for customer, item, manufacturer and selected interactions.
5. Target-history features: prior return counts and smoothed return rates for customer, item, manufacturer, size, color, state, customer×manufacturer, customer×size, item×size, and manufacturer×item.
6. Optional interaction and trend features only when rolling validation demonstrates improvement.

Every feature family must be independently switchable so ablations are measurable.

## Models
Primary models:
- CatBoost for raw/high-cardinality categorical handling.
- LightGBM on numeric/encoded feature views.
- XGBoost only if it adds measurable diversity or improves validation.

Ensembling is allowed only after individual models are validated. Blend weights are selected from rolling validation only.

## Project layout
`clean_dmc2014/` is standalone from the notebook code:

- `pyproject.toml` — package metadata and dependencies.
- `src/dmc2014/data.py` — raw ZIP loading and schema validation.
- `src/dmc2014/metrics.py` — DMC score and threshold search.
- `src/dmc2014/splits.py` — temporal fold generation.
- `src/dmc2014/features.py` — leakage-safe feature construction.
- `src/dmc2014/models.py` — model adapters/configurations.
- `src/dmc2014/experiment.py` — rolling evaluation and result records.
- `src/dmc2014/cli.py` — reproducible train/backtest/final-evaluate commands.
- `tests/` — scoring, split, leakage, feature and model smoke tests.
- `results/` — ignored runtime outputs; no generated model/data artifacts committed.

## Experiment discipline
Every run records model, parameters, enabled feature families, folds, per-fold scores, aggregate score, threshold, runtime, seed, and git SHA where available. Random seeds are fixed unless an explicit seed study is being run.

## Success criteria
1. Full test suite proves core scoring/split/leakage contracts.
2. One command reproduces rolling validation from raw ZIP data.
3. No notebook or processed CSV is required.
4. April labels are not loaded during model selection.
5. The first frozen model gets an exact April score recorded once.
6. Subsequent research aims to improve on 14,893 and then the historical winning 14,165 while retaining the clean validation protocol.

## Non-goals
- Reproducing the old notebook feature pipeline.
- Preserving notebook APIs.
- Production deployment.
- Modifying `master` directly.
