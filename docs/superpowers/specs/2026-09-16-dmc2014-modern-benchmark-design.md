# DMC 2014 Modern Benchmark Design

## Goal

Build a reproducible modern benchmark for the Data Mining Cup 2014 return-shipment task and try to improve the repository's documented 14,893-point CatBoost result below the historical winning score of 14,165.

## Source of truth

Use this repository's original competition archive at `Orders_data/Orders_train_test_class.zip`. It contains:

- `orders_train.txt`: 481,092 labeled rows from 2012-04-01 through 2013-03-31.
- `orders_class.txt`: 50,078 competition rows from 2013-04-01 through 2013-04-30.
- `orders_realclass.txt`: the 50,078 released competition labels used only for final scoring.

The private `teodor-georgiev/Online-retailer-analysis` repository remains the broader project reference, while this repository is the runnable VPS transport copy because the competition archive and prior 2014 modeling code are already present here.

## Evaluation contract

The competition score is `sum(abs(y_true - p_return))`. Lower is better.

Model selection must not use `orders_realclass.txt`. Use a chronological pseudo-competition split:

- training window: 2012-04-01 through 2013-02-28;
- validation window: 2013-03-01 through 2013-03-31.

After selecting a configuration using March only, retrain it on all of `orders_train.txt` and evaluate exactly once on April using `orders_realclass.txt`.

Report both mean absolute error and total DMC points. The main target is DMC points; probability predictions remain continuous rather than being rounded to 0/1 unless a separately validated transformation improves March MAE.

## Feature design

Start from the raw 13 competition predictors and add only features that can be constructed at prediction time.

### Row-level features

- order year/month/day/day-of-week/day-of-year;
- delivery year/month/day/day-of-week;
- delivery delay in days, plus an explicit missing-delivery indicator;
- customer age at order date;
- customer account age in days at order date;
- price as numeric plus log1p(price);
- raw high-cardinality identifiers retained as CatBoost categorical features.

All categorical values are normalized to strings and missing/unknown values use a stable sentinel.

### Leakage-safe historical features

For each prediction window, compute smoothed target statistics from its history window only. Candidate groupings:

- customerID;
- itemID;
- manufacturerID;
- size;
- color;
- customerID + manufacturerID;
- customerID + size;
- manufacturerID + itemID.

For each grouping, add observation count and a smoothed historical return rate using the history-wide return rate as prior. Unseen groups fall back to the global prior and zero count. March features are derived only from April-February labels; April features are derived only from the full original training labels.

Also add non-target history counts for customer, item and manufacturer to give the learner support/novelty information.

## Models

Primary model: CatBoostClassifier because the task contains several high-cardinality categorical identifiers and the repository's prior best result was CatBoost.

Run a small bounded configuration sweep rather than a broad expensive search. Candidate variants differ in depth, learning rate, L2 regularization, random strength and iterations. Early stopping uses the March validation set.

Secondary candidates may include LightGBM/XGBoost only after the CatBoost baseline is established. An ensemble is retained only if its March MAE improves on the best single model.

## Reproducibility

Add a Python benchmark module and unit tests. The benchmark writes a compact JSON result containing data hashes/row counts, selected configuration, March MAE/points, exact April MAE/points, and model runtime. It must never modify source data.

The VPS run uses an isolated temporary clone and isolated Python environment. No production services, databases, deployments or default branches are modified.
