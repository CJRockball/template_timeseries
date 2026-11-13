# Package 08 — observable, bounded acceptance runner

Baseline: feature/update_nn @ 4371557fe2038893ddaad7b7a262dbca29d58bfb.
Scope approved: progress/heartbeat, phase diagnostics, bounded cancellation, manual-first CI.
NOT another NN capability, new calibration, model-quality evidence or completed hosted matrix.
Only existing file changed: commented-out .github/workflows/nn-lifecycle-acceptance.yml ->
manual-only workflow. No automatic push/PR trigger. No GitHub write/dispatch/admin action performed.
Package 07 runner/plugin/plan/tests and all src/ml_template model/runtime code remain unchanged.

## Install and smoke test
Extract outside repo; BUNDLE points to nn_lifecycle_package_08. From editable repo root:

```bash
uv run python "$BUNDLE/install_nn_lifecycle_08.py" --repo . --check
uv run python "$BUNDLE/install_nn_lifecycle_08.py" --repo . --apply
uv run --no-sync pytest -v -rs tests/runtime/test_nn_lifecycle_08_acceptance.py
uv run --no-sync pytest -v -rs tests/runtime/test_nn_lifecycle_07_acceptance.py
uv run --no-sync ruff check .
```

22 new cases when unchanged (one POSIX process-group case skips on non-POSIX; Linux is the
certified cancellation platform). Ruff/full original-project compatibility must run locally.
Installer uses exact baseline Git-byte hashes and payload SHA256, Python 3.11 grammar/compile,
backups and guarded rollback. No ast.dump fingerprints. Stop on source mismatch; don't force,
reset work or edit manifest hashes. Installer OR patch, not both. Backup directory
.nn_lifecycle_08_installation stays untracked. --rollback restores original DISABLED workflow
and removes unchanged new files; later edits/formatting are refused.

## Start with a quick diagnostic, not the full matrix

```bash
uv run --no-sync python -u scripts/nn_acceptance_live.py \
  --phase gate_selftest --check
uv run --no-sync python -u scripts/nn_acceptance_live.py \
  --phase gate_selftest --execute \
  --phase-timeout-s 120 --total-timeout-s 180 --heartbeat-s 10
```

This runs the existing 40 Package 07 policy/plugin cases. Only pytest is required for this
one selected phase. All real lifecycle/UI/full requests retain existing native prerequisites.
No model is imported by --check; it does not write acceptance artifacts or run tests.

For a specific native phase:

```bash
uv run --no-sync python -u scripts/nn_acceptance_live.py \
  --phase package_06 --execute \
  --phase-timeout-s 900 --total-timeout-s 1200 --heartbeat-s 15
```

Repeat --phase to select several (plan order preserved). Duplicate/unknown names are errors.
Valid: gate_selftest, package_01..package_06, ui_compatibility, compatibility. Explicit selections
ALWAYS yield diagnostic_passed, full_acceptance=false, even if all names were listed manually.
They must NEVER substitute for lifecycle acceptance. A green diagnostic Actions job is only that.

## Full acceptance — only when ready

```bash
uv run --no-sync python -u scripts/nn_acceptance_live.py \
  --phase all --check
uv run --no-sync python -u scripts/nn_acceptance_live.py \
  --phase all --execute \
  --phase-timeout-s 900 --total-timeout-s 3600 --heartbeat-s 15
```

Omitting --phase also requests the full plan. Full scope requires the EXACT approved Package 07
plan blob b7e1fd4c595fec68cb3a84d9b489adb02b31ec12. Editing it down cannot earn full acceptance.
The same core.assess logic handles all final evidence; progress JSON is never used as pass proof.
All original no-skip/xfail/deselection/count/identity checks remain; optional compatibility skips
are recorded only under the existing full-suite policy. No changed prediction tolerance or model
code, no real input/label registry argument, no holdout/calibration/release replay on user data.
Existing synthetic native fixtures still deliberately train/evaluate/build temporary models.
This is synthetic implementation acceptance, NOT new Concrete generalization evidence.

The full run still deliberately repeats some native tests in the final compatibility suite.
Patch 08 makes that visible/bounded; it does not delete tests or skip native cases to appear fast.
Full acceptance requires all nine Package 07 phases; Ruff and Package 08 smoke tests are separate
required checks. Hosted 3.11/3.12 jobs remain pending until BOTH full runs actually pass.

## What you will see
- START: current phase and available time budget.
- Live prefixed pytest output (full unmodified output is in pytest.log).
- TEST: current collected test node ID as progress arrives.
- Heartbeat: elapsed time, stage, last node, collected/completed/failed/skipped counters.
- END: timing, pass/skip/failure counts, failed-node IDs and evidence-policy result.

Progress is observational: no fresh output does not prove a hang. Heartbeat elapsed/counts are
not training metrics. pytest.log + Package 07 events.json/JUnit remain authoritative. New atomic
progress.json and acceptance.json.active_phase help locate a slow collection/setup/call/teardown.
Collection/import time is explicitly included. Per-phase limit and overall deadline both use a
monotonic clock; the earlier wins. Positive finite budgets required. Defaults: 900s phase, 3600s
overall, 15s heartbeat. Overall budget covers the acceptance run, not environment installation or
separate Ruff/smoke steps; filesystem cleanup/sealing adds small I/O time, not a hard real-time SLA.

## Cancellation/failure evidence
SIGINT/SIGTERM on the wrapper are caught; current pytest process group is killed/reaped on POSIX,
including ordinary descendants, and the report becomes cancelled (exit 130). Timeouts yield
nonzero and timed_out. Failed evidence yields failed. No cancelled/timed-out/diagnostic run can
set full_acceptance=true. Current/last phase and progress, existing stdout/events/JUnit are retained.
No restart/resume of half-written model artifacts is attempted; reruns create a NEW acceptance ID.

Hard SIGKILL, runner loss/power loss, double signal during finalization, or GitHub forced job kill
cannot guarantee finalization/upload. A leftover running report is NOT success. Review partial
logs and rerun. Process-group cleanup guarantees apply to Linux/POSIX, not detached external jobs
or deliberately re-sessioned children; no such user external jobs are launched by this runner.

No old/frozen artifacts are changed. Reports live in a new artifacts/nn_acceptance/nnaccept08_...
folder. The schema-2 report has explicit selected/unselected phases, budgets, scope, environment,
input hashes and production_approval=false. The manifest explicitly covers REPORT files only,
excluding pytest tmp fixture trees. Normal completed-phase temps are cleaned; interrupted temps
may remain for inspection and are excluded from CI upload. All temporary data is synthetic.

## Manual CI
The replacement workflow has workflow_dispatch ONLY, defaults to gate_selftest and Python 3.11.
Choose 3.12 or both explicitly. Full scope is the all option. Correct CPU/PyPI index priority is
retained, using the fresh CI Torch CPU overlay and --no-sync afterward. pyproject/uv.lock unchanged.
CI accepts phase/overall budgets up to 3600s with an 80-minute outer job limit for install/tests/
finalization. No automatic four-job push/PR duplication. Concurrency cancels only a previous manual
run of the same ref, phase and Python selection. It does NOT set branch protection or promotion.

IMPORTANT GitHub requirement: workflow_dispatch requires a workflow on the DEFAULT branch.
Your feature-only workflow may not show Run workflow yet. Use the local CLI meanwhile, or add
only a reviewed workflow definition to the default branch separately; do NOT merge unaccepted
model changes just to obtain a button. No branch/default/settings change is performed here.
Once available, select the intended feature ref, phase and Python in Actions. If you disabled the
workflow via GitHub settings as well as comments, explicitly re-enable it when ready. Never infer
native acceptance from a green diagnostic. Partial/failure upload is best effort (always step).

## Authoring validation
22 new tests passed in an isolated harness with the actual new wrapper/progress code, inspected
Package 07 API protocols and byte-verified original authoritative plugin. Includes real pytest
subprocesses, strict skip rejection, diagnostic/full separation, changed-plan rejection, heartbeat,
phase/overall timeouts, and a REAL SIGTERM with an ordinary descendant stopped and cancelled report.
These are infrastructure/private fixtures, not actual native NN acceptance. Python 3.11 syntax
and compilation, installer fixture, YAML structure, patch and ZIP hashes/CRC verified separately.
Full existing 07/08 compatibility, native NN suites, Ruff, hosted 3.11/3.12 matrix and CPU installation
are NOT claimed executed. Preserve original pinned model environments; source/runtime untouched.
