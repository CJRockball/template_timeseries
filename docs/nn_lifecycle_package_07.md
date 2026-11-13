# Package 07 — final synthetic NN acceptance and closure

Baseline feature/update_nn: 57fa05eeb216212c7bf356ce1a138a4d2ff7010a.
This adds infrastructure/docs/tests ONLY. No src/ml_template files, pyproject.toml, uv.lock,
existing workflow, model behavior, numerical tolerances or frozen artifacts are changed.
A delivered ZIP is NOT acceptance evidence. User-reported Package 06 tests are not recast as
an independently verified native run. Package 07 closes only after the new gates execute.

## Install / local commands
Extract ZIP outside repo. From editable repository root, BUNDLE is nn_lifecycle_package_07:

```bash
uv run python "$BUNDLE/install_nn_lifecycle_07.py" --repo . --check
uv run python "$BUNDLE/install_nn_lifecycle_07.py" --repo . --apply
uv run --no-sync pytest -v -rs tests/runtime/test_nn_lifecycle_07_acceptance.py
uv run --no-sync ruff check .
uv run --no-sync python scripts/nn_lifecycle_acceptance.py --check
uv run --no-sync python scripts/nn_lifecycle_acceptance.py --execute --timeout-s 1800
```

No --execute is implied by --check. Check reads plan/test availability, dependency specs and
installed distribution metadata, and git HEAD; no model imports, test execution or output writes.
It returns blocked/nonzero if a native/UI dependency is absent. Missing libs must NOT be silently
skipped. Existing development Torch CUDA installation can run CPU-configured synthetic fixtures;
there is no instruction to replace your installed/pinned runtime with the CI CPU overlay.

Ruff must pass independently. If fixes are needed, review/format NEW infrastructure files only,
not original pinned runtime snapshots. Do not let formatting alter accepted runtime sources.

## Exact acceptance contract
Plan: configs/acceptance/nn_package_07.json. Nine sequential subprocess phases:
1. Package 07 self-test (40 cases when unmodified).
2-7. Packages 01-06, pure + actual native CPU tests, including nested input-format regression.
8. Monitoring/usage construction and selected-export callback tests.
9. Entire tests tree with `-m "not local_data"` compatibility policy.

Strict phases 1-8 fail on ANY skip (including collection/importorskip), xfail/XPASS,
deselection, collection error, runtime failure, process nonzero, absent event/JUnit file,
missing required test file, no passes, duplicate pass record or collected/pass count mismatch.
Missing Torch, Arrow, SQLAlchemy, Dash/AG Grid is a failed prerequisite, not a green native gate.
No minimum aggregate pass-count shortcut. Counts are recorded per file and node ID.

Compatibility phase permits/records optional-library skips, expected failures and explicit
local_data deselections; it cannot substitute for a critical native phase. The full compatibility
process must still return zero with at least one executed pass. All skip/xfail/deselection reasons
are retained rather than silently removed from the evidence. Default pytest addopts/environment
selection are cleared; only the explicit compatibility marker can exclude local data.

A phase failure stops the run and exits nonzero. Linux timeout kills the whole pytest process
group, including worker descendants; no model subprocess is left deliberately running. These are
synthetic CPU tests: they create their own training/calibration/evaluation/release/scoring/monitor
fixtures. They do not authorize reopening Concrete holdout, training on real data or generating
new independent model-quality evidence. Do not add real-data tasks to this plan.

## Evidence
A fresh artifacts/nn_acceptance/nnaccept_... directory contains:
- acceptance.json: status, exact commands, per-file collection/pass counts, node-level failures,
  skips and xfails, timings, environment/package versions, source/plan/test/lock hashes and git HEAD.
- Each phase: pytest.log, events.json and junit.xml.
- acceptance_manifest.json: final file hashes/bytes (integrity inventory, NOT a signature).

Runner status is running until all required phases pass; any failure persists failed status.
Tests, package runtime sources, acceptance scripts, plan and lock are checked unchanged at end.
No `healthy`, production-approved or quality-improved conclusion is produced. The report explicitly
sets production_approval=false and fresh_model_quality_evidence=false.

Each subprocess uses a fresh pytest basetemp inside its NEW phase directory, disables pytest cache
and bytecode output. Temporary synthetic fixture trees are removed after each completed phase;
logs/JUnit/events are retained. This prevents CI artifacts from collecting hundreds of temporary
checkpoint/source copies. For detailed failed fixture inspection rerun the recorded pytest command
manually with your own fresh --basetemp; never use a real release folder. Runner refuses output
inside src/data/.git/.venv or any sealed artifact ancestor. Never reseal or modify an old release.
No selected real label source/registry/input path is accepted by this CLI.

## New CI lane
.github/workflows/nn-lifecycle-acceptance.yml runs on main and feature/update_nn pushes, PRs and
manual workflow_dispatch, separately from unchanged .github/workflows/ci.yml. Python 3.11/3.12 CPU
matrix. Existing CI's optional-NN skips are no longer your only evidence.

Project currently pins optional torch==2.9.1+cu126. The new lane creates a FRESH CI venv from
uv sync --locked --group dev, then explicitly adds torch==2.9.1+cpu using the CPU wheel index.
This overlay intentionally is not a new CUDA project lock; pyproject/uv.lock remain untouched.
All subsequent commands use uv run --no-sync to prevent resolver synchronization undoing the
CPU overlay. Workflow records actual installed versions. Do not copy this overlay into old pinned
inference/training environments. CPU acceptance does not certify CUDA equivalence or performance.

Ruff/preflight/strict+compatibility steps must all pass. Upload uses always(), even on failure;
no evidence artifact may mean dependency/install/lint failed BEFORE acceptance started. Inspect
job logs; absence is not success. It does not run the disabled old SECOM operational replay,
download user datasets, promote releases or modify repository settings. Requiring both matrix jobs
as branch protection is an operator/admin action, not performed by this ZIP.

## Closure matrix (evidence, not invented success)
- P01: native predictor/GOLD/no-refit/key/hash semantics -> strict package_01 evidence.
- P02: correction/interval calibration -> strict package_02 evidence.
- P03: explicitly authorized synthetic final evaluation -> strict package_03 evidence.
- P04: sealed release/offline parity/tamper checks -> strict package_04 evidence.
- P05: pinned batch pipeline and ordered export -> strict package_05 evidence.
- P06: units, pipeline binding, maturity/support, saved read-only UI/report -> package_06 + UI evidence.
- P07: gate policies/plugin/no-skip + full compatibility and Ruff -> new workflow matrix.

Mark lifecycle implementation accepted ONLY when strict phases, full compatibility and Ruff pass
on intended supported environments, and inspect any optional compatibility skips. This is not
business sign-off, a new score-quality claim, measured-outcome monitoring approval or an automatic
production release. Existing NN OOF diagnostics and exposed Concrete holdout remain research-only.

## Recovery
Installer makes no code/environment/data side effects beyond NEW files listed in its manifest.
It verifies known baseline dependency byte hashes; no ast.dump fingerprints. On mismatch stop
and reconcile, do not force/reset or rewrite manifest hashes. Supplied patch OR installer, not both.
Installer backups/record directory .nn_lifecycle_07_installation is local/untracked. Do not commit
bundle ZIP, backup folder, generated acceptance data, model checkpoints or downloaded packages.
A rollback only removes installed new files if their SHA256 still matches; later edits are refused.

No GitHub writes/settings changes or real dataset access were performed during authoring.
Authoring policy/plugin self-tests passed, but complete native matrix, full project suite, Ruff
and hosted Actions must execute locally/CI before claiming Package 07 is accepted.
