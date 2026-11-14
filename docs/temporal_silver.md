# TS Silver 03: scalable publication, loading and legacy training guards

## Status and scope
This candidate implements the next storage/integration milestone, not temporal
training. Read VALIDATION.md before installing: the native Parquet integration
suite is supplied but could not run in the build sandbox because PyArrow is not
installed there. Your repository already declares PyArrow; run the mandatory
native tests locally before committing or merging. Missing PyArrow deliberately
fails that suite rather than silently skipping it. Ruff/mypy and the whole repo
suite also remain local acceptance gates.

## Installation
Extract outside the repository. Start a working branch from updated main with
clean tracked files. Leave your untracked local examples alone.

    python ../temporal_silver_03/install.py --repo .
    python ../temporal_silver_03/install.py --repo . --apply

Default is dry-run. Installer expects audited main commit
9be3c128b75db5f3632cbf9901384834e0014b56 as an ancestor; it does not change branches.
It hashes payload files, verifies the streaming-store dependency and exact Git
blob hashes of five old modules, checks single-use patch anchors, compiles changed
Python code, rejects conflicting new files/symlinks, and backs up patched files
outside the repo under the extracted bundle before writing. Ordinary write
failures roll back completed changes. Process termination is not a multi-file
transaction: retain the backup receipt for manual recovery. No auto-install of
dependencies, no SQL, tests, commits, push or merge is performed by installation.
Do not force installation if a source hash differs: review the local changes and
request a rebased bundle. Formatting the patched files changes their hashes and
can make a later re-install refuse; do not use re-installation to overwrite edits.

## Changes
New modules under temporal_stream:
- silver_schema.py: explicit temporal roles, type/nullability contract, target
  delay and origin metadata; strict row parsing and logical hashing.
- silver.py: source/assembly bridge, disk-backed staging/validation/sort,
  batched multi-file Arrow publication, CLI and registry registration.
- silver_manifest.py: sealed JSON inventory, safe relative file paths, checksums.
- silver_io.py: verified handles, ordered iteration and optional dataset scanner.
- silver_guard.py: dependency-light legacy-workflow rejection helpers.
New db/temporal_silver_registry.py records multi-file temporal datasets separately
from the old single-file silver_datasets table. Existing dashboard/latest-Silver
queries do NOT automatically discover these new rows.

Five guarded updates:
- data/contract.py rejects explicitly temporal contracts in the tabular loader.
- data/silver_loader.py rejects temporal manifests and exposes an explicit
  load_temporal_silver_dataset(path) forwarding function.
- training/run_config.py rejects silver_ts_ IDs, including feature-set configs.
- training/split.py rejects temporal IDs/contracts/manifests before random splits.
- db/data_run_registry.py adds a lazy SQLite-engine registration forwarding API.
Old default config serialization, tabular hashes and old artifacts are not changed.
Low-level estimators accepting arbitrary raw arrays cannot infer temporal intent;
these guards protect supported configuration/loader entry points, not deliberate
manual metadata stripping. No chronological trainer is implemented yet.

## Mandatory acceptance before real data
    uv run --no-sync pytest -q tests/temporal_silver
    uv run ruff check
    uv run ruff format --check
    uv run --no-sync pytest

If formatting is needed, format the intended changed files and review the diff.
Run the existing binary/regression acceptance appropriate to your environment.
No all-repository compatibility pass has been claimed for this bundle.

## Generate YOUR LOCAL Silver config
Use the ingest YAML that successfully ingested the smoke or small source:

    uv run --no-sync python -m ml_template.temporal_stream.silver init --ingest-config examples/temporal_stream/m5_ingest_smoke.yaml --output configs/local_temporal_silver.yaml --target-delay-periods 0 --origin-policy end_of_business_day

The example flags declare the assumption that a day's total is usable at an
end-of-business-day origin without an additional daily delay. Choose your actual
policy; these are explicit assumptions, not timestamps evidenced by M5. init
requires one deterministic observation ID and source temporal target metadata.
It writes a NEW config, refuses overwrite, infers types/roles from the ingest
contract and sets features: [] and known_future: []. Edit those lists explicitly
before modelling. Generated paths are local and should remain untracked per your
preference. init never opens or changes the database.

Typical local YAML shape (illustrative; use generated source ID/path/version):

    database: /absolute/local/path/data/project.db
    input:
      source: {source_id: YOUR_SOURCE_ID, version: v1}
    schema:
      columns: {observation_id: string, series_id: string, date: date, sales: int}
      observation_id: observation_id
      series: [series_id]
      time: date
      target: sales
      features: []
      known_future: []
      frequency: D
      gaps: error
      target_delay_periods: 0
      origin_policy: end_of_business_day
    output_root: /absolute/local/path/data/silver_temporal
    batch_size: 5000
    max_rows_per_file: 250000
    compression: zstd

The actual columns mapping must match the complete ingested source, including
store/item/week fields. Do not copy the abbreviated mapping over a larger source.
A nullable field must be listed explicitly in schema.nullable. Daily dates are
local business dates, not converted to Singapore/UTC dates. Actual availability
instants, when supplied via available_at, use canonical UTC timestamp fields.
Business timezone/issuance timestamp construction belongs in the later origin
planner, which must preserve these semantics.

## Build and inspect
    uv run --no-sync python -m ml_template.temporal_stream.silver build --config configs/local_temporal_silver.yaml

Output prints dataset_path and silver_dataset_version_id. Use the printed path:

    uv run --no-sync python -m ml_template.temporal_stream.silver verify --dataset ACTUAL_PRINTED_DATASET_PATH

The directory contains data/part-*.parquet, validation.json and manifest.json.
Parquet dates are date32, UTC timestamps are timestamp[us, UTC], numeric columns
are int64/float64 and text is string. There is no fitted encoding, filling,
scaling, lag generation or target clipping. Category vocabularies/encoders remain
model-owned. Arrow write batches and file limits are independently configurable;
there is no date/store partition heuristic or one-file-per-series explosion.
Logical rows are sorted by declared series columns then date before writing.
A series may span files/batches without being split semantically.

## Reading without whole-data materialisation
    from ml_template.data.silver_loader import load_temporal_silver_dataset
    silver = load_temporal_silver_dataset("ACTUAL_PRINTED_DATASET_PATH")
    for batch in silver.scan_batches(
        columns=["observation_id", "series_id", "date", "sales"],
        start_date="2016-01-01", end_date="2016-01-31", batch_size=5000,
        ordered=True,
    ):
        process(batch)  # Your bounded consumer; do not concatenate all batches.

Date endpoints are inclusive. series_values={"series_id": ["actual-id"]} supports
explicit subsets. ordered=True traverses the manifest's file sequence and row
groups, preserving global series/date order; filtering is then applied per batch.
It prioritises deterministic history reads over query pushdown. Read-ahead options
are ignored in ordered mode. No lag buffer is created: future feature code must
carry per-series state across batches/files instead of resetting it.

ordered=False uses the Arrow dataset scanner with projection/predicate pushdown,
batch_readahead and fragment_readahead (defaults zero). Do not assume chronological
order in this mode. File discovery uses only verified manifest paths, not a glob.
Inventory hashes are checked when opening and again before each scan: this scans
all file bytes even for a narrow filter, an integrity/performance trade-off. File
row-group statistics can still reduce decode work for unordered filtered scans.

## External assembly inputs
To publish an already-enriched export, replace input.source with
input: {assembly: /path/to/registered/assembly/output}. Set schema.columns to the
full enriched columns mapping, choose feature/known_future roles explicitly,
and declare nullable joined predictors if allowed by your join contract.

The bridge checks assembly ID against temporal_assemblies_v2 in the selected DB,
checks raw CSV and manifest hashes and recomputes the input canonical data hash.
It does not rerun joins or add availability information that was never recorded.
Exact known-ahead calendar enrichment may be materialised; revised predictors
still need per-origin requests and point-in-time selection. A static Silver
snapshot is NOT authority to expose future realised features or target labels.

## Bounded memory and disk costs
Sources are read from a verified SQLite transaction in batches; assembly CSV is
streamed. Rows are validated into a temporary on-disk SQLite staging table. Its
UNIQUE constraints catch observation/series-date duplicates across all batches.
An indexed ordered pass checks gaps with one previous series/date in memory.
Sorted batches are written with ParquetWriter, then read back batchwise to verify
schema, order, row counts and the incremental logical hash. No entire pandas
DataFrame/Table or whole-dataset list is constructed.

Staging adds a full temporary database plus indexes. Budget disk for that AND
Parquet plus existing source snapshots. SQL cache is approximately 32 MiB per
staging connection, not a total RSS cap. Arrow compression/dictionaries,
metadata, OS caching and row width also affect memory. Batch count is bounded;
full-M5 resource use has NOT been benchmarked. The current temporal reader
avoids full DataFrames, but existing model training still does not become
out-of-core by itself.

File layout and Arrow version are included in the physical Silver version ID;
changing batch/file limits can create another version with the SAME logical data
hash. IDs differ from legacy Silver and old identities are never rewritten.
Manifests bind semantic schema, availability assumptions, input fingerprints,
layout, file checksums and validation report. They detect accidental tampering,
not an administrator editing both metadata and data. No cryptographic signing
or release/dependency sealing is implemented.

## Publication and recovery
Build stages in a sibling temporary directory and publishes only after round-trip
validation. Existing identical IDs are verified/reused; conflicting registrations
fail, no overwrite. SQLite registration uses temporal_silver_datasets to avoid
passing directory paths through legacy single-file registries.
Ordinary failure removes a newly published directory if registration fails.
Power loss can leave orphan directories or registry rows: filesystem publication
and database commits are not jointly crash-atomic. Preserve artifacts, inspect the
manifest and use full verify before an explicit registration repair; do not edit
hashes/reseal artifacts to bypass errors. This is a local single-writer workflow,
not a concurrent export service or automatic crash-recovery daemon.

## Next, not in this bundle
Immutable origin plans, chronological inner stopping/backtesting, target/feature
availability eligibility, history feature execution, model evaluation and sealed
forecast releases. Keep synthetic fixtures and generic code tracked; real M5
files, local configs, databases and output datasets remain outside commits.


A disposable native Arrow pilot is included (not executed in the build sandbox):

    uv run --no-sync python scripts/benchmark_temporal_silver.py --rows 50000 --batch-size 5000 --rows-per-file 25000

It creates/deletes synthetic sources, measures Silver publication and verified
scan separately, checks read batch bounds, and reports Parquet bytes. Linux RSS
includes source ingestion and is not a peak-temporary-disk measurement. Later
feature-generation/training benchmarks are deferred because those paths are not
implemented by this milestone.
