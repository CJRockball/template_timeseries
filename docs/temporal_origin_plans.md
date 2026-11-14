# Temporal origin plans 04 — development planning only

## Scope and status
This adds an opt-in forecasting package on top of the locally working Silver 03
layer. It does not replace any existing file or change old split/run identities.
It supports daily one-step DEVELOPMENT forecasts, expanding history, chronological
inner stopping, explicit availability eligibility and a reserved final calendar
window. Features, estimators, scores, holdout forecast execution and model releases
are intentionally absent. No automatic data-dependent origin selection is done.

The core planner and installer were tested here using real SQLite and metadata
records. Native PyArrow/Silver integration tests are supplied but NOT executed in
the build sandbox (PyArrow absent). See VALIDATION.md. Run native and repo gates
locally before merging. It is a candidate, not a fully validated release.

## Install (extract outside your repo)
From the repository root, on your working feature branch:

    python ../temporal_origin_plans_04/install.py --repo .
    python ../temporal_origin_plans_04/install.py --repo . --apply
    uv run --no-sync pytest -q tests/forecasting
    uv run --no-sync pytest -q tests/temporal_silver
    uv run ruff check
    uv run ruff format --check
    uv run --no-sync pytest

The installer requires audited commit 9be3c128b75db5f3632cbf9901384834e0014b56
as an ancestor plus the local Silver load/verify/schema APIs. Silver is not on
remote main at the inspected checkpoint; this is why the working local layer is
required. API checks are structural, not proof that an arbitrarily modified
Silver implementation behaves identically. A receipt records dependency hashes.

Installation is ADD-ONLY. Exact existing added files are reused, differing files
or symlink destinations cause refusal. New-file writes use exclusive creation;
ordinary failure removes files written by that attempt. Directories can remain.
No old code is overwritten, so no old-file backup is needed. An external receipt
lists added-file hashes. Abrupt termination is not a multi-file transaction:
inspect against the payload manifest before retrying. Installer does not commit,
switch branches, install dependencies, run tests or touch your project database.
Keep local examples/configurations/database/output artifacts untracked.

## Step 1: inspect dates, not outcomes
Use the actual dataset_path printed by your successful Silver build:

    uv run --no-sync python -m ml_template.forecasting.origins bounds --silver-dataset ACTUAL_SILVER_DIRECTORY

This reads only the date column and prints first/last dates, the final 28-calendar-
date reservation and latest allowed development origin. It does not choose your
origins or read target values. If the smoke dataset is too short, use a longer
fixed small dataset; do not shrink the protected window just to get a pass.
A valid origin needs enough history for BOTH initial fitting and stopping, plus
any availability embargo. Predeclare origins before examining their performance.

## Step 2: create a LOCAL policy
The following are placeholders; replace them with actual paths/date choices:

    uv run --no-sync python -m ml_template.forecasting.origins init --silver-dataset ACTUAL_SILVER_DIRECTORY --database ACTUAL_PROJECT_DB --output-root data/origin_plans --output configs/local_origin_plan.yaml --train-start YYYY-MM-DD --origin YYYY-MM-DD --origin YYYY-MM-DD --business-timezone YOUR_IANA_ZONE

init refuses overwrite and generates absolute local paths. Defaults: stopping 7
calendar dates, minimum 28 eligible fit observations, holdout 28 calendar dates,
horizon 1, expanding history. Optional init flags: --stopping-periods,
--min-train-observations and --holdout-periods. Explicit arguments are mandatory
for timezone, start and origins. Relative YAML paths resolve against the YAML's
own parent, not the current shell directory. Your existing SQLite DB must exist.

Choose business timezone for your date/issuance convention, not the terminal's
local timezone. UTC is an explicit possible modelling convention, not a claim
about M5 store clocks. zoneinfo needs system IANA data or tzdata on platforms that
lack it; no fallback to an unannounced timezone. Ambiguous/nonexistent midnight
boundaries fail closed. Daily dates are never shifted to another date axis.

Policy YAML fields:

    development_origins: [YYYY-MM-DD, YYYY-MM-DD]  # quote dates in hand-written YAML
    train_start: YYYY-MM-DD                      # also quote this date
    stopping_periods: 7
    min_train_observations: 28
    holdout_periods: 28
    business_timezone: YOUR_IANA_ZONE
    horizon: 1
    window: expanding
    series_selection: null

These fields live under policy in the generated config. init writes dates as
strings. No dataset-specific configuration is added by the installer. To restrict
to a fixed panel, set series_selection to explicit composite series lists in the
same order/types as Silver.schema.series, for example [["actual-series-key"]].
Integer keys must remain integers. Unknown/duplicate/malformed selections fail.
Initial bridge supports string/integer observation and series identities only.

## Step 3: build and verify

    uv run --no-sync python -m ml_template.forecasting.origins build --config configs/local_origin_plan.yaml

Copy the printed plan_path, then:

    uv run --no-sync python -m ml_template.forecasting.origins verify --plan ACTUAL_PLAN_DIRECTORY --silver-dataset ACTUAL_SILVER_DIRECTORY
    uv run --no-sync python -m ml_template.forecasting.origins summary --plan ACTUAL_PLAN_DIRECTORY

verify alone checks the plan's sealed inventory. --silver-dataset additionally
verifies that the current Silver snapshot and semantic contract match the binding.
Later executors MUST perform that binding check, not simply accept an arbitrary
Silver path next to valid request files. Registry rows are an index, not authority
to execute training. There are no training commands in this package.

## Time and availability rules
Origin dates use Silver's origin_policy and target_delay_periods. End-of-day origin
D means the last UTC microsecond before next local midnight; horizon 1 targets
D+1. Start-of-day origin D means local midnight at D; horizon 1 targets D itself.
This distinction is intentional, not an off-by-one error. IANA DST changes can
make a business day 23/25 hours; offsets are not assumed constant.

For a target observation dated d with delay L, the conservative policy floor is
the END of local business day d+L. If Silver declares an available_at field, the
row is eligible only after MAX(that floor, its actual aware UTC timestamp). Actual
availability never relaxes the configured daily assumption. That field is treated
as observation/target-row availability, NOT individual predictor availability.

At an outer origin, label-eligible date cutoff is D-L for end-of-day issuance,
or D-L-1 for start-of-day issuance. Stopping is the last configured number of
calendar dates ending at that cutoff. Refit uses the expanding range through it,
filtered by actual availability at the outer issuance time.

The INITIAL fit is stricter: its model must exist before the first stopping
prediction. If stopping begins at S, fit labels end at S-L-1 and must be available
at the issuance for S. The L-day gap is an availability embargo, not shuffled
validation. Late actual timestamps can exclude additional observations. Every
stopping date must have an eligible label by the outer origin; partial stopping
windows exclude that series/origin. No silent random or reduced-window fallback.

These windows do NOT authorise building historical features from all information
available at the outer origin. Each historical training/stopping sample's features
must replay its OWN issuance information set (previous-day end or target-day
start). The rule is persisted, but feature/executor enforcement belongs to the
next package; this planner creates no feature values or fitted models.

## Artifacts and audit
Each immutable plan directory contains:
- manifest.json: policy, full Silver semantic/hash binding, materialised UTC/date
  windows, projected index hash, reserved holdout, file hashes and validation.
- origins.jsonl: shared fit/stopping/refit windows and issuance boundaries.
- eligibility.jsonl: one series/origin audit with counts and exclusion reasons.
- requests.jsonl: eligible keyed DEVELOPMENT requests, horizon, dates, observation
  ID, composite series key and stable request_id within that plan.
- validation.json: totals, exclusions and bounded staging-batch evidence.

The plan stages only observation ID, series, business date and optional target-row
availability in temporary disk-backed SQLite. It never projects targets or
predictors or calculates their aggregates. Checksumming a whole Silver file is
still byte I/O, not target decoding/scoring; this report does not assert the source
outcomes were never exposed elsewhere in your project history.

The final holdout is a calendar-date interval ending at the MAX observed date in
this immutable Silver snapshot, not 28 records per series. Missing dates are not
filled. Shorter series can lack some holdout dates. Development requests and
history must end before the boundary. No holdout requests, labels, scores or
training assignments are generated. Final evaluation needs a separately specified
strategy (rolling one-step versus fixed-origin multi-step), not an implicit 28-step
executor smuggled into this development package.

Exclusions include insufficient_fit_history, incomplete_stopping_window and
missing_forecast_observation. A pair can have multiple reasons; reason counts
therefore need not sum to excluded-pair counts. Every configured origin must have
at least one eligible series or publication fails. Select a fixed panel and
inspect audit counts rather than silently changing the modelling population.

There is no global row-to-CV-fold assignment: one row can enter multiple historical
training windows. Windows plus explicit date/availability predicates define row
membership; the source binding and index digest make it reproducible. The next
executor must use these predicates, not all rows in a date range ignoring late
availability. No row-key list proportional to every fold x all training rows is
materialised.

## Registry, identity and resource limits
SQLite table temporal_origin_plans is separate from existing split/run tables.
Identity binds normalized policy, Silver hashes/schema, effective metadata index,
materialized windows, holdout and build batch size. Different batch sizes produce
different physical IDs but identical logical windows/eligibility. Old plans are
never overwritten. An existing identical plan is verified/reused; a conflicting
registry row is refused. Registry paths are immutable too, so relocation is not
an automatic registration repair.

Staging is on disk with bounded insert batches and a roughly 32-MiB SQL page cache.
The planner iterates series and writes JSONL incrementally. It performs repeated
indexed counts for each origin/series, so planning CPU/I/O grows with history,
series count and origins. No full-M5 performance/RSS claim is made. Stage disk can
be substantial; budget it in addition to source and final artifacts. Sorted index
hashing also scans metadata once. Future feature/model memory is not covered.

Files are staged and validated before rename and registry insertion. Ordinary
registry errors remove newly published files. Filesystem rename and SQLite commit
are NOT one crash-atomic transaction; power loss can leave orphan files/rows.
No concurrent build service, automatic orphan cleanup, registry repair, source
resealing or model execution is implemented. Metadata hashes detect accidental
changes, not an administrator rewriting both manifests and content. Retain the
original artifact and diagnose errors rather than editing hashes to force reuse.

## Next milestone
Availability-safe lag/trailing features across file/batch boundaries, development
last-value/weekly baselines, then chronological XGBoost stopping/refit using the
SAME request plan. Keep the final holdout reserved until its evaluation strategy
and frozen configuration are explicit.
