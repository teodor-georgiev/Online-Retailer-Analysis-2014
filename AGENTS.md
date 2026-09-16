# DMC 2014 agent defaults

These instructions apply to ChatGPT/Codex work in this repository.

## Source of truth

- Treat the raw DMC files in `Orders_data/Orders_train_test_class.zip` and the modern `clean_dmc2014/` package as the maintained research path.
- Do not use the historical notebooks as implementation source code. They may be inspected only for audit/reproduction context.
- Keep April competition labels sealed during feature/model selection. Load them only in an explicit final/audit scoring step.

## Dynamic multicore VPS default

- For CPU-heavy VPS research, use multicore execution by default; do not hard-code 1, 2, 8, 12, or 16 threads merely because the host has that many logical CPUs.
- Resolve the budget through `dmc2014.capacity.resolve_workers()`. It honors `DMC2014_WORKERS`, then `CHATGPT_WORKERS`, then the shared `chatgpt-compute/admin/vps_capacity.py` policy when available, and finally a conservative affinity-aware fallback.
- Use the resolved worker count as CatBoost `thread_count`, LightGBM/XGBoost/sklearn `n_jobs`, or the bound for independent joblib/multiprocessing work.
- Use the `batch` profile for model training, backtests, feature-generation batches, and other long CPU-heavy experiments. Re-evaluate the budget before a materially separate heavy stage when VPS load may have changed.
- Avoid nested oversubscription. If experiments are parallelized across outer processes, divide/reduce inner model and BLAS thread counts so aggregate runnable CPU demand stays within the resolved budget.
- Never kill unrelated jobs/services to obtain cores. When load or memory pressure is high, back off instead.

## Experiment discipline

- Use chronological/out-of-sample validation for model selection.
- Record model parameters, feature set, split, threshold, runtime, worker count, and score for reportable experiments.
- Persist important VPS results under `/srv/chatgpt-results/dmc2014/<experiment>/<run-id>/` when the reviewed result-store path is available.
- Do not merge, deploy, or alter production services unless explicitly requested.
