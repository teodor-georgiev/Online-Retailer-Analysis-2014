# DMC 2014 clean-room benchmark

Modern Python implementation of the Data Mining Cup 2014 return-shipment problem.

## Runtime

- Python 3.12+
- pandas 3.0+
- NumPy 2.x
- CatBoost 1.2.7+
- LightGBM 4.5+

## Two deliberately separate feature paths

### `dmc2014.features`

Leakage-safe feature engineering for real model selection. Temporal/history features are built so holdout labels and future labels are unavailable to the feature builder. Use this path for new benchmark work.

### `dmc2014.legacy_modern`

A pandas-3-compatible reconstruction of the final notebook feature ideas. It adds the same 340 logical feature columns to the historical 127-column checkpoint, yielding the historical 467-column layout. It replaces removed pandas APIs such as `Series.mad()` with explicit equivalents and avoids notebook state.

This path is **audit compatibility only**. It intentionally mirrors the old transductive predictor aggregation semantics, where statistics may be calculated using later predictor rows. It never reads the `return` target when building those 340 features, but it must not be used as evidence of leakage-free temporal performance.

## Sealed historical score audit

`dmc2014.legacy_audit` separates training from scoring:

1. `prepare_legacy_audit_data()` splits known rows from unlabeled April rows without loading April labels.
2. `fit_frozen_legacy_catboost()` fits the frozen 200-tree historical CatBoost recipe with no `eval_set` and no early stopping.
3. `save_predictions()` persists predictions before labels are opened.
4. `score_predictions()` can then score those frozen predictions in a separate step.

This separation prevents `orders_realclass.txt` from influencing model fitting or model selection.

## Verification

```bash
pytest -q
```

The compatibility tests lock the +340-column contract and verify that changing `return` values does not change any `legacy_modern` predictor feature.
