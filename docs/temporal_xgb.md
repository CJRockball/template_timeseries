# Temporal XGBoost 06 — candidate, daily development only

## Baseline and scope
Targets feature/base_model at caa5aa75fe6db77dbd3562b22101e90a781da001.
PR #4's test check was verified successful for that BASELINE, not this new code.
The installer also works after a merge commit descended from that checkpoint,
provided the audited dependency files are unchanged. No merge is performed here.

Adds sample materialisation, one global numeric XGBoost model PER development
origin, native save/load parity and a separate common-request comparison against
cached baselines. Reuses XGBRegressionAdapter and XGBRegressionParams. No duplicate
estimator implementation, shuffled splits, learned transforms or covariate encoder.
No GPU, sweeps, NN, final holdout prediction/evaluation or release promotion.

## Install and acceptance
Extract beside the repository. On a branch descended from the pushed baseline:

    python ../temporal_xgb_06/install.py --repo .
    python ../temporal_xgb_06/install.py --repo . --apply

This is add-only. Four critical old modules are checked by exact Git blob;
conflicting added files and symlinks are refused. Ordinary new-file write failure
rolls back files created by that attempt. No old-file backups are needed because
no old files are overwritten. Process termination is not a multi-file transaction;
inspect the receipt/payload before retrying. Keep the bundle outside your repo.
Installer does NOT install dependencies, run training/tests, switch branches,
commit, touch SQLite or change GitHub. Do not force a dependency mismatch.

    uv run --no-sync pytest -q tests/forecasting/test_temporal_xgb.py tests/forecasting/test_temporal_xgb_native.py
    uv run --no-sync pytest -q tests/forecasting tests/temporal_silver
    uv run ruff check
    uv run ruff format --check
    uv run --no-sync pytest

Native Arrow/XGBoost tests and Ruff were NOT executed by the build sandbox.
Core tests use explicit history/adapter test doubles; they prove orchestration,
not native model training. See VALIDATION.md. Missing Arrow/XGBoost fails native
collection rather than silently skipping. Format/review code BEFORE making new
sample/model artifacts: producer source hashes are intentionally frozen.

## 1. Make a LOCAL config from your working baseline config

    uv run --no-sync python -m ml_template.forecasting.temporal_xgb init --history-config configs/local_history_smoke.yaml --output configs/local_temporal_xgb_smoke.yaml

init refuses overwrite. It preserves the exact plan/Silver/project DB/recipe and
uses an xgb subdirectory under your prior history output root. No dataset-specific
YAML is added by the installer. Relative YAML paths resolve against that YAML's
parent; init writes absolute paths.
Defaults: seed 42, numeric last-value/lags/means from your recipe, strict complete
selected features, 50,000 candidate rows per partition, CPU reg:squarederror,
RMSE stopping, 400 maximum rounds, patience 20, max_depth 4, learning_rate .05.
Other regressor parameters use the existing XGBRegressionParams defaults. Model
and sample options are explicit in YAML. No automatic search or best-model choice.

## 2. Materialise historical samples

    uv run --no-sync python -m ml_template.forecasting.temporal_xgb materialize --config configs/local_temporal_xgb_smoke.yaml

Copy the printed artifact_path as SAMPLES. It contains samples.jsonl (historical
X/y), requests.jsonl (outer forecast X only, NEVER the requested future y),
eligibility.jsonl, report.json and manifest.json.

For every origin and plan-selected series:
- Fit labels obey fit_start/fit_end AND available<=fit_available_by.
- Stopping labels obey stopping_start/stopping_end AND available<=issued_at.
- Refit labels obey refit_start/refit_end AND available<=issued_at.
- Every sample's X comes from shared HistoryIndex.for_sample(series,target_date,
  recipe), NOT the outer origin's information set.
- Only explicit ordered numeric feature names are selected. IDs/diagnostic dates
  cannot become model inputs implicitly. Nullable selected values exclude that
  historical sample and are audited; no imputation is added.

The minimum complete fit/refit count is the plan's min_train_observations PER
selected series. Every stopping date must remain complete (stopping_periods rows
per series) or materialisation fails. Warm-up exclusions are allowed/reported, not
silently treated as successful rows. A failing build does not publish a partially
usable artifact; its exception names the first offending origin/phase/series.

Outer requests with incomplete selected features remain explicit unsupported rows,
provided another series at that origin is supported. An entirely unsupported
origin fails. No silent series drop or baseline substitution. This is stricter
than the baseline methods, which may need fewer features; comparison reports both
individual and common coverage.

The history index may include observations unavailable at an EARLIER origin but
available at a later development origin. Shared as-of queries isolate them. Future
feature/label data are never made available simply because the index contains them.
The declared daily delay/UTC observation availability is inherited, not relaxed.
There is no per-predictor availability/exogenous revision mechanism in this patch.

## 3. Fit, save, reload and generate development forecasts

    uv run --no-sync python -m ml_template.forecasting.temporal_xgb train --config configs/local_temporal_xgb_smoke.yaml --samples ACTUAL_SAMPLES_DIRECTORY

For EACH origin:
1. Initial fit on the materialised fit rows, stopping against chronological stopping
   rows using the existing adapter's fit_with_validation.
2. Record its selected ONE-BASED round count and stopping-history CSV.
3. Discard the stopping model and create a FRESH adapter.
4. Refit fixed rounds on the full eligible materialised refit rows.
5. Predict the stored outer request features, save native UBJ, reload via the existing
   adapter and require exact prediction parity and ordered feature names.
6. Freeze a model/prediction artifact; no outer outcome is loaded for round selection.

Copy its artifact_path as MODELS. It contains models/NNNN.ubj, stopping-history CSVs,
report.json, predictions.jsonl and manifest.json. Each request has method=xgb and
explicit ok/unsupported status. Saved models, selected rounds and feature order
are bound to the prediction artifact. This is NOT a single final deployment model.

The native candidate requires the repo's pinned xgboost==2.1.1 and CPU device.
Changing code/runtime/Silver/plan/feature options between materialisation and
training is refused. Changing only model parameters creates a new model artifact
from compatible samples (parameter values are bound to that artifact).

## 4. Compare frozen XGB with your frozen baseline predictions
Use the PREDICTION artifact from the baseline package, not its score directory:

    uv run --no-sync python -m ml_template.forecasting.temporal_xgb compare --config configs/local_temporal_xgb_smoke.yaml --models ACTUAL_MODELS_DIRECTORY --baseline ACTUAL_BASELINE_PREDICTION_DIRECTORY

This does NOT retrain or rewrite either artifact. Baseline, plan/Silver/recipe and
producer identity are checked, and every prediction row is aligned to the original
request ID/series/date/observation key BEFORE development outcomes are iterated.
Duplicates, missing rows (including omitted unsupported cases), unexpected methods,
invalid statuses and nonfinite values fail. Missing/duplicate/misaligned requested
outcomes fail too. A pre-format baseline with stale producer hashes must be
regenerated under frozen code, not resealed or edited to bypass the check.

The score artifact contains scores.jsonl and metrics.json. Both per-origin and
pooled MAE/RMSE are reported for each method's own supported set and for the
INTERSECTION supported by ALL compared methods. Counts and coverage are retained.
Pooled errors are request-weighted; no averaging of origin RMSEs, hierarchical M5
weights or claimed model superiority. Empty comparisons have null metrics, not 0.
Three smoke forecasts establish alignment, not forecasting skill.

## Verify

    uv run --no-sync python -m ml_template.forecasting.temporal_xgb verify --artifact ACTUAL_ARTIFACT_DIRECTORY

The registry table temporal_xgb_artifacts is separate from baseline/history and
old tabular split/run registries. Sample/model/score IDs, file inventory and source
bindings are immutable. Existing identical artifacts can be reused; moved/copied
paths do not silently replace immutable registry paths.

## Resource and correctness limits
History/request indexing remains disk-backed and batch-oriented. SAMPLE FILES are
incremental JSONL. XGBoost fitting itself is NOT out-of-core: candidate counts are
capped at 50,000 per origin/partition by default, and the learner materialises only
those bounded dense frames. Multiple frames/native buffers still coexist; the cap
is a row budget, not an RSS guarantee. Over-budget input fails, NEVER subsamples.
Full-M5 training will normally exceed this initial cap and is NOT certified here.
Use the smoke series then a fixed small panel. Do not raise the cap blindly.

Sample matrices repeat rows across fit/refit/origins by design; this is not a
single row-to-CV-fold assignment. The current loader rescans JSONL per partition,
so many origins increase I/O. Shared history creation may scan the whole bounded
Silver panel before model row-budget checks. Budget staging disk and elapsed time.
No full-M5 peak RAM/disk/throughput measurement has been made.

The reserved final holdout remains inaccessible to requests, materialisation and
comparison. Application scans are date-filtered, but Parquet row-group compression
and file integrity checks may read/decode broader bytes internally; this is not a
physical ACL or a claim that an administrator cannot inspect source files.

Source hashes/runtime versions are provenance, not a sealed production release.
Hashes cannot authenticate against an administrator rewriting all metadata.
Filesystem rename and SQLite registration are not jointly crash-atomic; power loss
or concurrent writers can leave orphans. Single-writer local runs are assumed.
No auto repair, retraining, promotion, calibration or holdout execution is provided.

## Next after native/smoke acceptance
Record the accepted sample/model/score IDs, inspect coverage and exclusion counts,
then use more predeclared development origins and a fixed small panel. Decide the
final evaluation issuance/refit strategy before EVER scoring the reserved holdout.
Keep local YAMLs, DBs, models and generated artifacts out of the code commit.
