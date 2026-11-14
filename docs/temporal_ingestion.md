# TS ingestion 01: opt-in source snapshots and external enrichment

## Scope and compatibility
This is an additive first milestone, NOT the entire forecasting lifecycle.
It adds `ml_template.temporal` without modifying legacy ingestion, Silver, split,
training, release or scoring modules. Existing binary/regression identifiers and
artifacts are not rewritten. New outputs are NOT registered as legacy Silver;
`eligible_for_legacy_training: false` is metadata, not a global trainer guard.
Do not manually route them into the old shuffled training path.

The namespace intentionally consolidates the previously proposed source registry,
contracts and enrichment modules, rather than introducing partially connected
changes in the old modules. Connecting this to the existing Silver loader,
central SQLAlchemy factory, temporal split guards and release recipe replay is a
subsequent integration milestone. The small store uses Python sqlite3 with the
same foreign-key/busy-timeout policy as the inspected project factory. It can
share `data/project.db`, using only its own registry and hashed source tables.

## Storage and invariants
- One table per immutable source/version supports independently shaped sources.
- Explicit schema, source byte hash, provenance, canonical data hash and row count
  are recorded in `temporal_sources`. Canonical columns are alphabetically ordered.
- Exact same file/contract/provenance reuses a version after integrity verification.
  Conflicting reuse fails; no delete-and-replace or implicit latest version.
- Transactions cover source DDL, rows and registration. Failed insertion rolls back.
- All loaded data is revalidated and rehashed. This detects accidental row changes,
  not a malicious actor modifying both data and its registry hashes.
- UTF-8 CSV input only; empty fields are null. Text such as NA remains literal.
- Initial regular temporal axis: daily local business dates in YYYY-MM-DD.
  Gaps are error/allow; unequal series start/end coverage is permitted and no
  synthetic dates/targets are filled. Optional target must be finite numeric.
  Nonnegative/integer-only demand is NOT imposed on generic float targets.
- UTC availability/origin timestamps must be explicit Z-form strings with seconds.
  Offset timestamps are deliberately rejected: normalise externally. Daily M5
  dates are not converted to Singapore dates. UTC is the internal comparison
  representation; user-facing operational times can be displayed in Singapore.

## Installation
Extract the ZIP outside the repository. With a clean tracked working tree at
43a46fff36f46eea84566bbb6564341b9e32f9fc:

    python /path/to/bundle/install.py --repo /path/to/template_timeseries
    python /path/to/bundle/install.py --repo /path/to/template_timeseries --apply

Default is dry-run. The installer verifies HEAD and payload hashes, rejects
symlinks and differing existing destination files, installs new files only and
rolls back created files on ordinary copy failure. Re-running identical files is
safe. It does not create a branch, commit, update dependencies, run SQL or ingest
anything. Run it from a new working branch if preferred; branch HEAD must match.
It does not offer an unsafe force/overwrite mode.

## Tests and first M5 import
Run from repository root (dependencies: pandas, PyYAML, pytest, already part of the
project/development setup). Use `uv run --no-sync` if the environment is synced.

    uv run --no-sync pytest -q tests/temporal/test_ingestion_bundle.py
    uv run --no-sync python scripts/prepare_m5_temporal.py --history data/derived/m5_profiles/m5_small_v1/history_long.csv --output data/prepared/m5_small.csv
    uv run --no-sync python -m ml_template.temporal ingest --config examples/temporal_ingestion/m5_ingest.yaml
    uv run --no-sync python -m ml_template.temporal assemble --config examples/temporal_ingestion/m5_assemble.yaml

Use your actual notebook output path. The external preparation script projects
only generic observation fields and constructs a SHA-256 observation ID from the
unambiguous JSON pair [series_id,date]. Record its printed source checksum in the
config provenance BEFORE ingestion. Do not concatenate notebook history/train/
holdout files. The full history includes future evaluation outcomes; ingestion
is not authority to train on all rows. Create temporal partitions later.

For smoke, copy the ingestion config to another file in the same example folder,
change source_id to m5_smoke, input to a separately prepared smoke file, and use
that ID in its assembly config. The smoke series overlaps the development profile;
independent snapshot import duplicates it intentionally for this first milestone.
Profile membership deduplication is deferred.

Paths in YAML are relative to the YAML file, NOT the process working directory.
Examples live outside configs/ to avoid the existing config ignore policy; inspect
`git status --ignored` before committing. The installer does not change .gitignore.

## External data
`calendar_ingest.yaml` and `enriched_assemble.yaml` demonstrate separate source
storage and a many-to-one known-in-advance calendar join. Prepare the two-column
calendar_selected.csv externally. No new M5-specific logic is in the package.

Exact enrichment requires an explicit known_in_advance assertion. This is a
semantic promise by the contract author, not proof from data. Do NOT use it to
bypass publication delays for prices or realised weather. Even item metadata
may need versioned availability if it changes historically.

`point_in_time_recipe.yaml` illustrates request rows with origin timestamps and
external forecasts/revisions. Copy its component contracts into ingest configs
and its assembly mapping into an assemble config; it is not a direct CLI config.
The join selects the latest available_at <= origin among exact key matches.
Include applicable date/period/location in those keys. Duplicate availability
instants per key fail rather than selecting an arbitrary revision. No automatic
range joins, carry-forward, monthly-to-daily resampling, API download, vintage
recovery or target-based aggregation is implemented. Structure those externally
first, preserving availability timestamps and original source lineage.

The recipe pins source versions and hashes and checks row counts, output-column
collisions, cardinality and unmatched/null-feature limits. Assembly manifests live
in SQLite and CSV/manifest exports. A failed export can leave a registered,
deterministically reproducible assembly; it cannot publish a partial output
folder during ordinary errors. Use a different output directory to export again.
Assembly provenance changes when a source version changes, even when selected
feature values are identical. Verification of source data is not cryptographic
signing or protection against database administrators.

## Monthly updates / expansion
This release ingests full snapshots under NEW explicit version IDs. It does not
append deltas or merge corrections automatically. Construct the new complete
snapshot externally, then ingest it. Calendar/prices can be separately versioned.
Existing versions and old assemblies remain unchanged.

## Performance and remaining integration
Designed for the ~50k-row development fixture, not full-M5 performance. Parsing,
validation, hashing and enrichment are in-memory; point-in-time selection scans
candidates within key groups. SQL query pushdown, chunked ingestion, profile
membership tables and bulk analytics are deferred. There is no feature fitting,
lag generation, chronological backtesting, forecast strategy, retraining, model
execution, release integration, native Silver publication or production approval.
Never compare this subset to the full hierarchical M5 leaderboard.

Next step: wire this opt-in source/assembly boundary into versioned Silver and a
chronological training guard, then implement temporal splits and history-aware
features without changing tabular defaults.
