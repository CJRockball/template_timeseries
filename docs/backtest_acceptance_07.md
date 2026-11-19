# Backtest acceptance 07 — descriptive audit, not model promotion

Targets feature/base_model at 5778eb5ecbfe8aa162385cb098e0fecbb1ace5d1.
Adds four modules and two test files. Does not edit old producer-hashed modules,
configuration identities, artifact inventories, algorithms or dataset snapshots.
Existing accepted sample/model/score artifacts remain report inputs. No need to
retrain them simply because these new reporting modules were installed.

## Install
Extract this bundle beside the repository, not inside it. From the repository:

    python ../backtest_acceptance_07/install.py --repo .
    python ../backtest_acceptance_07/install.py --repo . --apply

The installer requires the audited checkpoint as an ancestor and exact Git blobs
for six critical dependency files. Existing differing added files and symlinks
are refused; repeated identical installation is a no-op. No overwrite/force mode.
Ordinary write failures roll back files added in that attempt, not directories.
Power-loss atomicity and concurrent installations are not promised. Receipts stay
in the extracted bundle. It never installs dependencies, changes Git/SQLite,
runs models or creates a PR. A merge to main is not required.

## Local acceptance before use
This build environment has no Ruff, Python 3.11 interpreter, Arrow or XGBoost.
Run these LOCAL gates; do not mistake fixture tests for native training tests:

    uv run ruff format src/ml_template/forecasting/acceptance_cli.py src/ml_template/forecasting/acceptance_store.py src/ml_template/forecasting/backtest_acceptance.py src/ml_template/forecasting/backtest_resources.py tests/forecasting/test_backtest_acceptance.py tests/forecasting/test_backtest_acceptance_native.py
    uv run ruff check
    uv run --no-sync pytest -q tests/forecasting/test_backtest_acceptance.py tests/forecasting/test_backtest_acceptance_native.py
    uv run --no-sync pytest -q tests/forecasting tests/temporal_silver

All new Python is handwritten with NO f-strings and no AST-unparse generation.
Installer enforces that bundle policy and parses using the 3.11 grammar before
compiling. Actual execution on Python 3.11 and full native/Ruff gates remain local.
If a lint issue is reported, fix the NEW module before making report artifacts;
do not edit old model inputs or their frozen manifest hashes.

## Report existing accepted smoke artifacts
From the repository, substitute your REAL artifact directories and project DB:

    SAMPLES="/actual/sample/artifact"
    MODELS="/actual/model/artifact"
    SCORES="/actual/comparison/artifact"
    BASELINES="/actual/baseline/PREDICTION/artifact"
    DATABASE="/actual/project.sqlite"

    uv run --no-sync python -m ml_template.forecasting.acceptance_cli report --samples "$SAMPLES" --models "$MODELS" --scores "$SCORES" --baseline "$BASELINES" --database "$DATABASE" --output-root data/backtest_acceptance

Copy the printed artifact_path as REPORT:

    REPORT="/actual/printed/report/artifact_path"
    uv run --no-sync python -m ml_template.forecasting.acceptance_cli verify --artifact "$REPORT"
    cat "$REPORT/summary.md"

Outputs:
- summary.md: input bindings, pooled common-supported errors, warnings.
- report.json: pooled/per-origin/PER-SERIES own/common MAE/RMSE, coverage,
  unsupported reasons, sample eligibility, selected rounds and input payload sizes.
- metrics.csv: the same grouped metrics in a flat table (series keys are JSON).
- manifest.json: immutable identity/inventory and new report producer hashes.

Metrics are request-weighted, not M5 hierarchical/WRMSSE or equal-series weighting.
Common-supported is the intersection of ALL compared methods on each request.
Zero support has n=0 and null MAE/RMSE, never an invented zero error. Series/origin
own-support denominators remain the full planned request population for that group.
No confidence intervals or independence/significance claims are made.

The report verifies input inventories, manifest IDs, sample/model/score/baseline
bindings, request IDs and development bounds. It aligns frozen predictions with
score rows, checks truth consistency across methods and recomputes residuals,
common support and metrics. Cached pooled/origin metrics and reasons must agree.
Selected rounds, model-file inventory, feature order and counts are cross-checked.
It does NOT reload estimators or prove that a supplied UBJ contains a particular
number of trees; native training tests remain the model execution gate.

Only frozen artifact files are read. Training sample labels and source Silver
rows are not decoded for report computation; their file integrity can still be
hashed. y_true comes from the FROZEN DEVELOPMENT SCORE artifact, not a new raw
truth read. There is no new heldout evaluation. Hashes are not signatures against
an administrator forging every input and metadata field.

The report is descriptive evidence, NOT an automated acceptance decision,
configuration winner, benchmark certification, registry promotion or release.
Maximum selected rounds and incomplete coverage are warnings, not automatic
failure. Missing resource measurements remain explicitly UNMEASURED.

New registry: temporal_acceptance_artifacts in the supplied existing SQLite DB.
Sample/model/history registries and artifacts are not changed. File rename and
SQLite registration are not jointly crash-atomic. Assume a single local writer.
Outputs must be outside all immutable input artifact directories. Report grouping
has an explicit --max-groups budget (100000 default); working request/prediction
alignment uses a disposable disk-backed SQLite database.

## Optional Linux resource measurement
This is an EXPLICIT command runner, separate from read-only reporting. It executes
only the argv YOU provide after --, without a shell. Installation/report commands
do not automatically train. Do not pass secrets: argv and combined stdout/stderr
are retained in the measurement artifact.

Example for a FUTURE development materialisation:

    CONFIG=configs/local_temporal_xgb_smoke.yaml
    uv run --no-sync python -m ml_template.forecasting.acceptance_cli measure --label smoke-materialize --database "$DATABASE" --output-root data/backtest_resources --watch-dir data/forecast_history -- uv run --no-sync python -m ml_template.forecasting.temporal_xgb materialize --config "$CONFIG"

The wrapper prints the RESOURCE artifact_path, NOT the child model/sample path.
Read RESOURCE/stdout.log for the child's printed artifact_path. Repeat explicitly
around train or compare if desired. A failed child produces an exit-code record
and the CLI returns nonzero; failure is not reported as a successful model run.

Attach measurements when creating a report by repeating:

    --resource "/actual/resource/artifact"

Limitations:
- Linux wait4 child-reported ru_maxrss in KiB; not summed concurrent process-tree
  peak RSS, GPU memory or a full resource profiler.
- Directory peaks are SAMPLED observations and may miss short-lived files. Existing
  files are included; initial directory bytes are retained. Default interval .5s.
- Only explicitly watched directories are sampled; choose the real staging/output
  locations. Temporary SQLite databases can be under the OS temp directory; do not
  claim they were measured unless watched. Walking directories adds overhead.
- Resource records cannot retroactively measure an old run. Their association with
  report artifacts is a user-supplied label/command, not independently proven.
- No resource hard limits, full-M5 capacity claim or benchmark guarantee is added.
- The generic measurement wrapper is not a holdout security boundary. Never pass
  a forbidden command; the forecasting CLI retains its existing holdout guards.

## Broader development experiment (manual, no automatic selection)
Preserve the accepted smoke references, then predeclare more earlier development
origins with unchanged model settings. Next repeat on a deterministic fixed small
panel chosen WITHOUT holdout or comparative forecast errors. Use your existing
Silver/origin-plan builders and temporal_xgb CLI to create NEW bound artifacts.
This package does not choose series/dates for you or edit a prior immutable plan.

Before starting, estimate candidate rows per origin/partition against the existing
50000-row cap. Dense XGB training remains bounded in-memory, not external-memory.
Do not expand to full M5, sweeps, NN or multi-step forecasts yet. The final holdout
2016-04-25 through 2016-05-22 remains reserved. A report over old development scores
must never become an excuse to inspect or select using that holdout.
