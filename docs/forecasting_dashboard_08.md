# Forecasting dashboard 08

Targets feature/base_model at 141dfa9676ad6c354de55df8ab4ead9969e9f025.
Adds Forecasting — Development to the existing Dash application. Single run stays
its default; all existing seven pages remain. There is no training or evaluator
button. Existing forecasting producers and artifact hashes are not modified.

## Install from repository root
Extract the ZIP beside the repo (not inside it):

    python ../forecasting_dashboard_08/install.py --repo .
    python ../forecasting_dashboard_08/install.py --repo . --apply

Review the preview. Three existing UI files receive exact hash-guarded edits:
app.py, report_client.py and report_export.py. Original UI files plus a receipt
are kept under the extracted bundle's backups directory. Two forecasting protocol
modules are verified but never edited. Other conflicting files and symlinks are
refused, with no force mode. Repeated identical installation is a no-op; original
pins are checked even when reverse-validating already patched UI files.
Ordinary write failures roll back old and newly added files. Process termination
is not multi-file atomic; assume a single installer/writer. The installer makes
no SQLite, model, dependency, Git or GitHub changes. Do not mix loose draft files
from earlier responses with this completed bundle.

## Local gates (before launch)

    uv run ruff format src/ml_template/dashboard/app.py src/ml_template/dashboard/forecasting_data.py src/ml_template/dashboard/forecasting_figs.py src/ml_template/dashboard/pages/forecasting_page.py tests/dashboard/forecasting_dashboard08_fixtures.py tests/dashboard/test_forecasting_dashboard08_data.py tests/dashboard/test_forecasting_dashboard08_ui.py
    uv run ruff check
    uv run --no-sync pytest -q tests/dashboard/test_forecasting_dashboard08_data.py tests/dashboard/test_forecasting_dashboard08_ui.py
    uv run --no-sync pytest -q tests/dashboard
    uv run --no-sync pytest -q

Dash/browser execution and Ruff were unavailable in the build environment. Read
VALIDATION.md: passing data/Plotly cases does NOT certify actual Dash callbacks.
The new payload has no f-strings, parses with 3.11 grammar, and is never AST-
unparsed. Pre-existing export f-strings are preserved; only their tab label changes.
Actual Python 3.11 execution remains a local gate. Formatting after installation
can change new source bytes; a later reinstall may correctly refuse those edits.

## Launch
Use the SAME existing SQLite database supplied to acceptance_cli report:

    uv run --no-sync python -m ml_template.dashboard.app --port 8050 --forecast-db /actual/project.sqlite

Open http://127.0.0.1:8050 and select Forecasting — Development.
--forecast-db controls this page only. Existing pages retain their old DB setup.
With no explicit path, a file-backed SQLite engine path can be used; otherwise the
page explains the missing configuration. Missing databases/tables are not created
by the forecasting adapter. No authentication/public deployment support is added.
Keep the existing localhost-only deployment posture.

## Use
1. Select one registered acceptance report. The selector does not auto-select or
   pretend that hash order is recency. Restart to refresh new registry entries.
2. Pooled bars, origin trends and a per-series heatmap use saved report metrics.
3. Select MAE/RMSE, visible methods, a baseline and common/own support.
4. Explicitly load linked DEVELOPMENT scores to filter series/origins and display
   bias, actual/predicted observations, residuals and largest errors.
5. Choose one plot series. Only evaluated target dates are shown as markers:
   sparse origins are NOT a continuous history or interpolated daily forecast.

Filtering uses all matched scored requests and aggregates their errors. It never
averages group RMSEs. Common support stays the original intersection of ALL report
methods; hiding a trace does not enlarge the comparison population. Own-support
baseline deltas are disabled. Zero baseline error gives an undefined percentage,
not infinity. Empty method selections, invalid controls, missing cohorts and
empty filter intersections fail closed. Filters with unchecked/unavailable scores
are refused rather than showing unfiltered numbers under filtered labels.

Unknown linked-score files do not block the unfiltered saved summary; diagnostics
show a warning. No fallback to raw Silver occurs. Frozen settings are read from a
bound model MANIFEST, not estimator weights. Nested configuration containers are
rendered as JSON rather than passed to scalar missing-value formatters.

The reader verifies report inventories and registered identities and the linked
score parent bindings, request IDs, actuals across methods, residual signs, row
populations, complete method coverage and common-support flags. It opens SQLite
mode=ro plus PRAGMA query_only and never calls report registration/build, model
training/prediction, evaluation or raw Silver scans. The legacy pages retain their
existing behavior; the new adapter does not migrate or reconfigure their DBs.

## Charts and scope
- Pooled MAE/RMSE bars with N/coverage.
- Error over evaluated origins.
- Per-series raw-error heatmap (first 100 series, labelled).
- Actual/predicted markers at evaluated dates only.
- Signed-error histograms (prediction minus actual).
- Largest-error table, explicitly limited to 20 rows.
- Selected rounds, exclusions, provenance, frozen parameters and saved resources.

All metrics use every filtered row within the detail budget. Charts can limit
residual traces/evaluated dates to 5000 points with explicit notices; this never
truncates the metric population. Raw errors are on the saved target scale and are
affected by demand level. No new MAPE/MASE/WRMSSE, confidence intervals, statistical
independence claims or hierarchical/scale-normalized rankings are introduced.
One cohort/report at a time: currently one XGB configuration with its baselines.
Simultaneous cross-report XGB ranking is intentionally out of scope; it would need
explicit compatible-cohort and joint-support rules, not naive concatenation.

Inventory is capped at 200 registered report IDs. Linked detail has a hard cap of
200000 score rows; over-budget data is refused, never partially scored. This is a
small-panel local UI, not full-M5 serving. Rehashing/decoding is on demand and can
be expensive. Row budgets are not an RSS limit. No low-latency/RAM claim is made.

The page verifies frozen evidence, not current model weights or raw ground truth.
Hashes are not signatures against someone resealing every artifact. Saved user-
associated resource records are displayed, not new measurements or certifications.
No final holdout evaluation/access, tuning, model selection, release or promotion
is performed. Keep 2016-04-25 through 2016-05-22 reserved and untouched.

## Exports
The existing top HTML/ZIP buttons now recognise the Forecasting tab. They export
the rendered view, not a functioning Dash application or independent model audit.
Tables retain displayed precision/limits, and figures carry their plotted data,
including legend-hidden traces. Review development labels, commands/parameters
and provenance before sharing. The CSV button exports full-precision current
comparison rows plus report ID, selected filters, baseline, support and counts.
It does not dump the complete underlying prediction file. No registry is written.
Existing Final evaluation opt-in/consent checks remain unchanged.

## Browser acceptance
- Eight tabs appear; Single run remains the default.
- Missing forecasting DB/registry gives a useful message; old tabs still work.
- A saved smoke report displays bars, origin trends and a heatmap.
- Linked detail enables both filters and selectable plot-series options.
- Filtered N, coverage and metrics agree with saved rows.
- Hiding a method does not change common support.
- Missing linked scores preserve the unfiltered summary but reject filters.
- Actual/prediction plots show only evaluated dates.
- CSV and rendered HTML/ZIP downloads work; HTML opens offline.
- No model, score, source dataset or registry content was modified.

Keep local paths, YAMLs, databases, snapshots and generated exports untracked.
