# History features and development baselines 05

## Scope and baseline
Built against main 6c1176e26b7c6251021173cf8597da32b85e6023, after Silver/origin
plans were merged. Adds seven modules in forecasting, two test modules and this
guide. No existing code, YAML examples, source data or old artifact IDs are changed.

Implements disk-backed target-history features, last-available and strict weekly
naive forecasts, immutable prediction artifacts, and a SEPARATE development scorer.
Daily horizon 1 only. No XGBoost/NN, fitted transforms, external covariates, fitted
training matrices, tuning, final holdout execution or release promotion.
The historical-sample API replays each sample's issuance; a complete chronological
training executor still belongs to the next model package.

## Install: use ONE of installer or patch, not both
Extract the ZIP outside your repository. From the repository root on your next
working feature branch:

    python ../temporal_history_baselines_05/install.py --repo .
    python ../temporal_history_baselines_05/install.py --repo . --apply

Installer defaults to dry-run; add-only, refuses differing new files/symlinks,
checks audited ancestor and critical upstream source Git blobs, compiles new
Python and records an external receipt. It accepts the inspected Silver branch
checkpoint as well as merged main. No old-file backups are necessary because no
existing file is modified. Exclusive new-file creation and ordinary-error rollback
protect existing work. Abrupt process termination is not a multi-file transaction.
Directories can remain after rollback. Local examples, DB and outputs stay local.
Installer does not install dependencies, execute tests, commit, switch branches or
write to SQLite/GitHub. If a pinned old dependency differs, review rather than force.

Alternative from the same audited baseline:

    git apply --check ../temporal_history_baselines_05.patch
    git apply ../temporal_history_baselines_05.patch

The patch is a real add-only unified Git patch, not a JSON edit recipe. Unlike the
installer, git apply does not validate upstream dependency APIs or record receipts.
Review git diff and stage intended code/docs/tests only; never git add . here.

## Local acceptance before prediction

    uv run --no-sync pytest -q tests/forecasting/test_history.py tests/forecasting/test_history_native.py
    uv run --no-sync pytest -q tests/forecasting tests/temporal_silver
    uv run ruff check
    uv run ruff format --check
    uv run --no-sync pytest

PyArrow native tests and Ruff were not run in the build sandbox. See VALIDATION.md.
Missing Arrow fails collection rather than skipping the important native gate.
If formatting is needed, format/review the newly added files BEFORE generating
prediction artifacts: source hashes are part of producer identity. Do not edit
code between generation and scoring and expect old artifacts to remain compatible.

## 1. Initialise a LOCAL configuration
Use the exact Silver and plan paths printed by your accepted builds. The existing
project DB is the same file you used for those runs.

    uv run --no-sync python -m ml_template.forecasting.history_cli init --plan ACTUAL_PLAN_DIRECTORY --silver-dataset ACTUAL_SILVER_DIRECTORY --database ACTUAL_PROJECT_DB --output-root data/forecast_history --output configs/local_history_smoke.yaml

init refuses overwrite and writes absolute paths. Relative paths in hand-edited
YAML resolve against the YAML's own parent. No dataset-specific YAML is tracked.
Default recipe:

    recipe:
      lags: [1, 7, 14, 28]
      rolling:
        - {periods: 7, min_periods: 7}
        - {periods: 28, min_periods: 28}
      methods: [last_available, seasonal_naive_7]

Config also has plan, silver_dataset, database, output_root and batch_size (5000).
Unknown fields/methods, duplicate lags/windows and invalid minimum counts fail.
Weekly method requires explicit lag 7. Recipe is part of immutable input identity.
Source extra features are deliberately ignored; this package uses only target
history and identity/time/target-availability fields.

## 2. Generate and freeze features/predictions

    uv run --no-sync python -m ml_template.forecasting.history_cli predict --config configs/local_history_smoke.yaml

Copy the printed artifact_path for later scoring. The prediction directory holds:
- features.jsonl: one bound development request with features, unsupported feature
  reasons and history diagnostics (not its future target label).
- predictions.jsonl: one row per request AND configured method, even when unsupported.
- manifest.json: plan/Silver/recipe/producer hashes, file inventory and coverage.

Generation verifies the exact plan/Silver binding and every request's ID,
issuance, purpose, target date and observation alignment. It does not join scoring
outcomes or calculate accuracy metrics. Existing identical artifact IDs are reused.
With the accepted three-origin, single-series smoke plan, expect 3 feature rows
and 6 prediction rows. Coverage may be less than 3 per method if history is missing
or unavailable; do not equate emitted prediction rows with supported predictions.

## 3. Score in a separate operation

    uv run --no-sync python -m ml_template.forecasting.history_cli score --config configs/local_history_smoke.yaml --predictions ACTUAL_PREDICTION_ARTIFACT_DIRECTORY

All frozen artifact bytes, producer identity, request rows, methods, statuses and
prediction coverage are validated BEFORE outcome values are iterated. Then the
scorer joins development outcomes by series/date AND exact observation ID, not
row position. Duplicate/missing/unexpected/misaligned prediction rows fail. Missing
prediction support must be explicit, never a dropped row. Null/nonfinite predictions
cannot masquerade as supported. Missing/duplicate requested outcomes fail too.

The separate score artifact contains:
- scores.jsonl: actual/prediction/residual per request+method, including unsupported
  rows and an indicator for the common supported comparison set.
- metrics.json: per-origin and pooled MAE/RMSE, counts, coverage fractions and reasons.
- manifest.json: immutable bindings to plan, Silver and frozen prediction artifact.

Each method has own_supported metrics on its own valid predictions. The
common_supported comparison uses the INTERSECTION of supported request IDs across
ALL configured methods. Every method is compared on that same set. Both views are
reported: good error on a tiny easy subset must not conceal missing coverage.
MAE/RMSE are request-weighted when pooled, not averages of per-origin RMSEs. No
series weighting/hierarchical M5 metric is implemented. Zero supported counts
produce null metrics and zero coverage, NOT a misleading zero error.

Prediction files are never updated by scoring. Changed outcomes create another
score artifact (and normal immutable Silver changes require a newly bound plan
and predictions), not a rewritten old evaluation. Tests exercise core separation
with synthetic outcomes; production always re-verifies the bound Silver snapshot.

## 4. Verify and inspect

    uv run --no-sync python -m ml_template.forecasting.history_cli verify --artifact ACTUAL_PREDICTION_OR_SCORE_DIRECTORY

Inspect metrics.json inside the PRINTED score artifact directory. Three smoke
requests prove alignment and execution, not forecasting skill or general model
selection. Freeze additional origins/small panel before meaningful comparison.
Do not select dates after seeing which baseline wins.

## Exact feature semantics
For each request, the issuance must match its own target-date convention: target
minus one day at end-of-day, or target-day start. An outer issuance cannot be
substituted for an earlier training/stopping sample. The implementation reuses
policy.effective_availability: maximum of declared actual UTC timestamp and the
end of business day observation_date+target_delay_periods.

last_available is the most recent date BEFORE target_date with effective
availability <= issuance. It can be older than yesterday; diagnostics report its
date and age in calendar days. This is the baseline's definition, not a fallback.

lag_k is EXACTLY target_date-k calendar days and must be available by issuance.
No row-position shifts. Missing dates, policy-unavailable dates and late actual
availability are separately reported. seasonal_naive_7 uses only lag_7: it does not
silently search 14/21 days back when lag 7 is unavailable.

The trailing calendar window ends at the POLICY cutoff (origin minus target delay,
minus an extra day for start-of-day origins), not the most recent observed row.
It never shifts backward to gather enough records. Its fixed range is filtered by
actual as-of availability. Default min_periods equals width; an explicitly smaller
minimum permits a partial mean within the SAME dates, with count diagnostics.
Missing features do not automatically invalidate baselines that do not use them.
Last value and mean values remain finite continuous floats, including negative or
fractional values. There is no clipping, class-label conversion, imputation or
automatic category encoding. Nonfinite values/arithmetic fail rather than hide.

Historical API (not an entire training runner): HistoryIndex.for_sample(series,
target_date,recipe) derives that sample's own issuance; features(...) rejects a
mismatched issuance. Future fitted training code must separately obey plan fit/
stopping/refit label eligibility, then use own-sample feature replay. This package
does NOT materialise fitted matrices or certify a future training adapter.

## As-of and holdout boundaries
Source labels are scanned only through the maximum development-history cutoff for
prediction and through requested DEVELOPMENT target-date bounds for scoring.
Returned out-of-bound rows are rejected by the core. Every feature query also
filters by sample-specific availability and forbids same/future target dates.
The final holdout is blocked in request validation, history access and scoring.
For the accepted M5 smoke reservation this remains 2016-04-25 through 2016-05-22.
No holdout prediction, metric, label export or final strategy is generated.

The application only receives date-filtered Arrow batches. Parquet compression/
row-group boundaries and full-file checksums may physically read/decode broader
blocks internally. This is NOT a claim of byte-level holdout isolation or a secure
storage ACL. Returned holdout target values are never handed to the feature or
scoring core; internal library I/O is not used to calculate holdout metrics. The
historical source can contain observations unavailable to an EARLIER development
origin but available to a later one: the shared disk index keeps them separate by
as-of queries, not by assuming the whole dataset is known at every origin.

## Persistence, memory, provenance and limitations
SQLite table temporal_history_artifacts is separate from old split/run tables.
The row stores artifact kind, immutable manifest hash/path and parent plan ID.
Copied/moved artifact paths are not silently re-registered over an existing row.

History and request/prediction/outcome alignment use temporary SQLite files, not
whole-data pandas tables or in-memory dictionaries of the full panel. Reads and
writes are incremental; there is no per-file history reset. SQLite cache is about
32 MiB per active staging connection, NOT a total RSS cap. There can be multiple
connections plus Arrow buffers. Calendar windows/counts query the index for each
request: full-M5 performance and peak disk/RAM have NOT been benchmarked. Budget
staging disk in addition to source and final artifacts. No external covariates or
learned transformations are cached here. Arrow/source verification can reread all
file bytes, so narrow request scans do not imply narrow hash I/O.

Identity binds output bytes, recipe, plan/Silver snapshots, source hashes for all
new modules plus policy.py, Python and SQLite versions. Changing formatting or
runtime/code between predict and score intentionally causes a binding refusal;
freeze code then regenerate a new artifact instead of bypassing the check.
This is not a sealed dependency lock/release contract: native-library versions
and the full runtime are not certified. Hashes detect accidental mutation, not
an administrator rewriting manifests and data together.

Artifacts stage before publication and register after rename. Ordinary registration
failures remove newly published files. Filesystem and DB commits are not jointly
crash-atomic; power loss/concurrency can leave orphans. Single-writer operation is
assumed. No automatic repair, resealing, retraining, recalibration or promotion.
Keep all local configs, SQLite files and generated outputs out of commits.
