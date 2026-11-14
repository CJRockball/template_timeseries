# Streaming temporal ingestion 02

## Decision and scope
Use this successor before relying on full-M5 ingestion. It removes whole-file
bytes/DataFrames, list-of-all-rows insertion, whole-dataset JSON hashing and
fetchall loading from the new path. This is NOT a full-M5 performance certificate
or a forecasting training implementation.

The API lives in `ml_template.temporal_stream` alongside the first bundle's
`ml_template.temporal`. This intentionally leaves your local Ruff fixes and old
snapshots untouched. Stop using the old CLI for scalable imports. Do not install
the first bundle again and do not route v2 output through legacy random splits.
No changes are made to old classification/regression code, old manifests, Silver,
training, scoring or releases. Existing v1 tests remain runnable independently.

## Install
Extract outside the repository. From repository root (adjust bundle location):

    python ../temporal_ingestion_02/install.py --repo .
    python ../temporal_ingestion_02/install.py --repo . --apply
    uv run --no-sync pytest -q tests/temporal_stream/test_streaming.py
    uv run ruff check
    uv run ruff format --check src/ml_template/temporal_stream scripts/prepare_m5_stream.py scripts/benchmark_temporal_stream.py tests/temporal_stream

The first command is a dry-run. New files only: no overwrite, no auto-commit,
no dependency installation, no SQL. Identical existing files are safe to rerun;
differing destinations fail. SHA-256 payload checks and symlink checks precede
writes. Ordinary write failures remove newly written files (empty directories may
remain). Git HEAD must descend from the audited base
43a46fff36f46eea84566bbb6564341b9e32f9fc and tracked working files must be clean.
Commit or stash unrelated tracked changes first. Untracked first-bundle files
are preserved. Examples are outside configs/; inspect ignore rules before commit.
Do not run multiple installers or writers to the same export directory concurrently.

## Small M5 first
Use the notebook's history_long.csv as input, NOT concatenated with its train or
holdout exports. It contains the full observed history including evaluation labels.

    uv run --no-sync python scripts/prepare_m5_stream.py --input data/derived/m5_profiles/m5_small_v1/history_long.csv --output data/prepared/m5_small_stream.csv
    uv run --no-sync python -m ml_template.temporal_stream ingest --config examples/temporal_stream/m5_ingest.yaml
    uv run --no-sync python -m ml_template.temporal_stream assemble --config examples/temporal_stream/m5_assemble.yaml

Edit paths to match your machine. Record the preparation sidecar hashes in config
provenance before initial ingestion. All YAML paths are relative to that YAML.
Only stdlib plus PyYAML for CLI configuration is required; pandas is not imported
by this new implementation. Python 3.11+; tests use pytest.

## Full M5 preparation
This external M5-only script avoids pandas melt of the entire panel:

    uv run --no-sync python scripts/prepare_m5_stream.py --wide --input data/bronze/m5/sales_train_evaluation.csv --calendar data/bronze/m5/calendar.csv --output data/prepared/m5_full_stream.csv

It holds one wide sales row plus the small calendar mapping, writing long rows
incrementally. Item/store identities use an unambiguous JSON pair; observation
IDs hash [series_id,date]. Its series_id encoding differs from the notebook's
concatenated encoding: do not append one encoding into the other snapshot.
For full M5 copy the ingest config, change input/source_id/output configuration,
and keep explicit version strings. Do not combine overlapping validation and
evaluation sales files. Generic ingestion has no M5 column names hardcoded.
CSV expansion, database indexes, journalling and disk-backed joins need substantial
free disk; measure pilot sizes before expanding. The conversion can produce a
much larger file than the original wide sales CSV.

## Streaming invariants
- CSV parsing and inserts use bounded batches (`batch_size`, default 5000).
- Raw source hashing uses fixed-size reads. Parsed bytes are hashed during parsing
  and compared to the preliminary hash before commit, detecting source changes.
- Per-row strict types: date YYYY-MM-DD, explicit canonical UTC timestamps,
  signed 64-bit integers, finite floats and declared nullability. Empty CSV fields
  are null; literal NA is not. Sources require exact headers and row widths.
- SQLite UNIQUE constraints catch source-key and series/date collisions globally,
  including across batches. Temporal order is separate from ingestion order.
- Gaps are checked by an indexed ordered cursor holding only the previous
  series/date. Unequal start/end coverage across series is permitted. No filling.
- Immutable source/version reuse requires identical raw hash/contract/provenance;
  existing rows are verified before reuse. Changed data requires a new version.
- A complete source ingest, index creation, validation and registry entry is one
  transaction. Failure rolls it back. Long write locks are intentional for your
  occasional single-writer imports. There is no chunk-level resumability.
- Canonical hashes are incremental typed JSON-lines in deterministic key order.
  The explicit v2 hash format differs from v1 and includes a column header.
- `iter_source` yields bounded row batches after a streaming integrity pass.
  It holds a read transaction for iteration; exhaust or close the generator.
- SQLite cache is configured to approximately 32 MiB; temp_store=FILE lets SQL
  sorting/intermediate joins use disk. Python batch memory is bounded by batch
  size and row width, not total row count. SQLite/OS caches and disk allocation
  still affect measured process resources; there is no absolute RAM guarantee.

## External data and expansion
The same SQLite file can contain differently shaped sources, under
`temporal_sources_v2` and generated source tables. Old v1/legacy tables are not
modified or migrated. Reingest prepared input into v2; duplicate storage remains
until you deliberately retire an old database copy. The installer never deletes
existing databases or output files.

`external_recipes.yaml` documents exact calendar and point-in-time recipes; split
its sections into separate ingestion/assembly configs before use. Schema/recipe
shape follows bundle 01, so most YAMLs can be reused with the NEW module command;
change output paths to avoid collisions and note registries are separate.

Exact joins require a known_in_advance assertion and a unique right-side key.
Point-in-time joins select the latest available_at <= each row's origin among
exact applicable-period/location matches, using indexed correlated SQL lookups.
Ambiguous equal-availability revisions fail. SQL materialises each enrichment
stage in disk-backed TEMP tables, drops previous stages, checks row counts,
coverage/null limits and column collisions, then exports through fetchmany.

No Python dictionary of every external record, no all-source candidate scan per
request, and no full assembled DataFrame. SQL may require significant temporary
disk and time. Known-in-advance remains a contract author's assertion, not proof
that a vendor actually published that value. Do not mark realised weather/prices
as known simply because the competition supplied them. No forward filling,
range joins, vintage reconstruction, API downloads or automatic target lags.

Monthly updates are complete new snapshots, not delta/upsert corrections.
Prepare/merge snapshots externally with explicit lineage. Additional series are
validated with the same key/temporal contract. History ingestion does not decide
forecast-origin label eligibility; that must be implemented in temporal training.

## Artifacts and recovery
Assembly output contains data.csv and manifest.json. Manifest pins source versions,
fingerprints, recipes, canonical data hash, CSV hash and coverage reports; it is
explicitly not eligible for legacy training. Export stages into a sibling temporary
directory, renames it into place, then commits assembly identity registration.
Ordinary failures roll back and remove output created by the failed operation.
Process termination/power loss at the filesystem/database boundary can leave an
orphan output directory: inspect manifest/registry and choose a new output path;
do not silently overwrite or treat an orphan as registered. Atomic transactions
do not make filesystem renames and SQLite commits jointly crash-atomic.

No code/dependency sealing or release recipe replay is claimed. The hash registry
detects accidental source changes, not an administrator rewriting both data and
metadata. Back up databases using a consistent SQLite backup, not by copying an
actively written file. No network-shared active database is intended.

## Validation and scaling gate
See VALIDATION.md for executed evidence. Synthetic bounded batches and tests are
not substitutes for full-M5 runtime/memory/disk benchmarking. A convenience pilot:

    uv run --no-sync python scripts/benchmark_temporal_stream.py --rows 1000000 --batch-size 2000

This creates/deletes synthetic data in a disposable directory and reports times,
database bytes, batch bounds and platform-specific peak RSS. It is NOT a full-M5
extrapolation and does not benchmark external joins unless added separately.
Start with the small M5 data, then a larger pilot. Run full M5 only after checking
free disk and measured resource use. Native Silver integration, global training
guards, chronological splits, lag buffers, monitoring and release/scoring replay
remain the next milestones. SQLite remains the selected backend.
