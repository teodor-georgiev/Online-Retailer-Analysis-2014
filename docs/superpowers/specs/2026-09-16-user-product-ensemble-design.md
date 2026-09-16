# DMC 2014 User/Product Profiles + Ensemble Design

## Goal

Improve the leakage-safe DMC 2014 return-shipment benchmark by adding causal user-centric and product/manufacturer-centric predictor profiles and an out-of-fold CatBoost + LightGBM probability ensemble, while keeping April competition labels sealed during all model and blend selection.

## Scope

This design extends only the maintained `clean_dmc2014/` package. Historical notebooks remain audit-only and must not be imported or executed by the maintained pipeline.

The raw DMC files in `Orders_data/Orders_train_test_class.zip` remain the sole dataset source for the maintained benchmark.

## Leakage rules

1. Every profile feature for a row dated `d` may use predictor information only from rows with `orderDate < d`.
2. Same-day rows must not contribute to each other's historical profile values.
3. No profile feature may read `returnShipment`.
4. Target-history features remain optional and disabled for the leading experiment track unless explicitly requested by `FeatureConfig.history_groups`.
5. April labels from `orders_realclass.txt` must remain unavailable to feature generation, model fitting, model selection, ensemble-weight selection, and threshold selection.
6. Ensemble weight and classification threshold are selected from rolling Jan/Feb/Mar out-of-fold predictions only.

## User-centric feature family

For `customerID`, calculate causal values as of the current order date:

- prior row/item count
- prior distinct order-date count
- prior distinct item count
- prior distinct manufacturer count
- prior distinct size count
- prior distinct color count
- cumulative spend
- historical price mean, standard deviation, minimum, maximum
- average historical basket item count
- average historical basket total value
- customer lifetime in days since first prior purchase
- days since previous purchase
- mean, minimum, and maximum prior inter-order gap
- repeat-item interaction count for the current `customerID × itemID`
- repeat-manufacturer interaction count for the current `customerID × manufacturerID`
- user familiarity ratio for current item: prior customer-item count / max(1, prior customer row count)
- user familiarity ratio for current manufacturer: prior customer-manufacturer count / max(1, prior customer row count)

All customer profile state updates occur only after a complete date batch has been emitted.

## Product/manufacturer-centric feature family

For `itemID`, calculate causal values as of the current order date:

- prior sales row count
- prior distinct order-date count
- prior unique customer count
- prior unique state count
- cumulative revenue
- historical price mean, standard deviation, minimum, maximum
- days since previous sale
- days since first prior sale
- repeat-buyer count approximation based on customers previously seen for the item

For `manufacturerID`, calculate causal values as of the current order date:

- prior sales row count
- prior distinct order-date count
- prior unique customer count
- prior unique item count
- prior unique size count
- prior unique color count
- cumulative revenue
- historical price mean and standard deviation
- days since previous manufacturer sale
- days since first prior manufacturer sale

All item/manufacturer state updates occur only after a complete date batch has been emitted.

## Implementation architecture

Create `clean_dmc2014/src/dmc2014/profiles.py` as the only new feature-engineering module. It exposes profile builders that accept a historical source plus target rows and return a row-aligned numeric DataFrame. `features.py` remains the orchestrator and appends profile columns according to `FeatureConfig` flags.

`FeatureConfig` gains two booleans:

- `user_profiles: bool = False`
- `product_profiles: bool = False`

This keeps the current baseline reproducible and makes ablations explicit.

The training builder computes causal expanding profiles within the training frame. The validation builder computes profiles using history rows plus earlier validation predictor rows, but never validation labels and never same-day target rows.

## Model path

CatBoost and LightGBM continue to use the same aligned `FeatureSet`. No separate feature matrix is maintained per model.

CatBoost keeps raw categorical columns plus numeric profile columns.

LightGBM uses the existing categorical dtype conversion plus the same numeric profile columns.

Both model fits resolve their CPU budget through the dynamic batch-capacity policy before each fold/model stage.

## Ensemble path

Add an ensemble backtest function that runs both models on every temporal fold and stores aligned OOF probabilities.

For blend weight `w` in a deterministic grid from `0.0` to `1.0` inclusive in `0.05` increments:

`p_blend = w * p_catboost + (1 - w) * p_lightgbm`

For each blend weight, select the best hard classification threshold using all concatenated OOF rows only. Choose the `(weight, threshold)` pair with the fewest DMC points. Tie-break by:

1. weight closest to `0.5`
2. threshold closest to `0.5`
3. lower CatBoost weight

Return individual CatBoost and LightGBM OOF scores plus the ensemble score so the blend cannot hide a weaker component.

## Validation and experiment protocol

Use the existing chronological folds:

- January 2013 holdout
- February 2013 holdout
- March 2013 holdout

Primary comparison sequence:

1. current recency-only CatBoost baseline
2. CatBoost + user profiles
3. CatBoost + product profiles
4. CatBoost + both profile families
5. LightGBM + both profile families
6. CatBoost + LightGBM ensemble using both profile families

The winner is selected by estimated 50,078-row DMC points from concatenated OOF predictions. April is untouched.

## Resource policy

Heavy model fits use the approved utilization-first batch controller:

- fill measured idle CPU below the soft load brake
- trim at normalized load 1.5× effective CPUs
- hard brake at 2× effective CPUs
- emergency clamp at 3× effective CPUs
- memory pressure may reduce the allocation further

Avoid nested oversubscription when CatBoost and LightGBM are executed concurrently. If outer experiments run in parallel, divide the shared CPU budget across them rather than giving each the full recommendation.

## Tests

Add tests that prove:

- changing `returnShipment` values does not change user/product profiles
- same-day rows do not update each other's user/item/manufacturer profile state
- a later-date row sees earlier-date predictor history
- validation profile values do not change when validation labels are mutated
- training and validation feature columns stay aligned with profile flags enabled
- ensemble OOF probabilities are row-aligned across models
- blend selection never uses April labels
- deterministic blend tie-breaking works

## Success criteria

- all clean-room tests pass
- no April-label access is added to feature/model/ensemble selection code
- profile ablations are measured separately
- ensemble result is reported alongside both component scores
- current best projected score (~15,944) is retained or improved; only improvements are promoted as the leading configuration
