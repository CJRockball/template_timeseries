# Calibrated threshold and policy-selection update

Prepared for CJRockball/tabular-ml-template main inspected 2026-10-19.
This is a guarded source-edit package, not a GitHub commit or a database migration.
It contains one new policy-selection module, exact updates for ten existing
modules, three new repository test files, and standalone installer tests.

The files/ tree holds new repository files. module_updates.json holds all edits
for existing modules. The installer uses your existing complete source files,
verifies their Git blob hashes and exact anchors, and produces the complete
updated modules. It does not download code or replace unrelated module contents.
Use --export to obtain/review every complete updated source file before applying.
Do not copy files/ alone: that would omit the required existing-module edits.

## Installation

Extract this ZIP into a sibling directory called calibrated_policy_update.
From the tabular-ml-template repository root, on a working branch:

```bash
uv run python ../calibrated_policy_update/apply_update.py --repo . --check
uv run python ../calibrated_policy_update/apply_update.py --repo . --export ../policy_update_preview
```

Review the exported modules and tests. The export does not edit the repository.
Then install:

```bash
uv run python ../calibrated_policy_update/apply_update.py --repo . --apply
uv run ruff check src tests
uv run pytest tests/evaluation/test_policy_selection.py tests/evaluation/test_policy_selection_preflight.py tests/dashboard/test_calibrated_thresholds.py -q
make check
make dashboard
```

The installer validates every target before writing, creates backups under
.policy_update_backups/, and prints the exact backup location. --restore with
that location reverses only this update, refusing to overwrite later edits:

```bash
uv run python ../calibrated_policy_update/apply_update.py --repo . --restore <printed-backup-directory>
```

If the baseline differs, stop and inspect the local diff. --allow-drift is only
for reviewed formatting/local differences; exact anchors must still occur once.
Do not use it to bypass an unreviewed semantic change. Do not commit the backup
folder or generated policy artifacts inadvertently. Keep local data/model/split
artifacts intact; this installer does not modify them.

No dependency changes, new DB tables, training reruns, or calibration refits are
needed. New source/tests belong in version control; generated artifacts follow
your existing local-artifact retention rules.

## Updated modules

- dashboard/calibration_data.py: score-variant adapter, canonical key ordering,
  raw/calibrated-cross-fit labels and parent/calibration metadata.
- dashboard/threshold_panels.py: optional variant metadata; old callers continue
  working; source metadata is included in recommendations/folds/budget tables.
- dashboard/threshold_figs.py: score-aware development cutoff axis label.
- dashboard/pages/calibration_page.py: variant selector, exploratory cutoff and
  budget controls, shared F1/F2/workload/fold/tie panels. Never freezes a policy.
- evaluation/evaluation_config.py: optional strict PolicySelectionRef and
  release-scale/predictor consistency checks; manual legacy policies still work.
- evaluation/evaluation_inputs.py: selection hash/source validation during
  preflight; selection provenance/hashes enter evaluation identity before labels.
- evaluation/evaluate_run.py: snapshot selection record in report data and
  include source/batch definition in frozen policy results; no holdout search.
- evaluation/decision_policy.py: F1/F2 metrics, with undefined-positive support
  handled consistently. Existing statistics/bootstrap loops forward these keys.
- evaluation/final_report.py: selection provenance in HTML and Markdown.
- dashboard/pages/evaluation_page.py: visible selection-provenance table,
  backward compatible with earlier report data lacking that optional field.

New module: evaluation/policy_selection.py. Training/calibration/model modules,
calibration registries, app.py, and existing calibration artifacts are unchanged.

## New tests

- tests/evaluation/test_policy_selection.py
- tests/evaluation/test_policy_selection_preflight.py
- tests/dashboard/test_calibrated_thresholds.py

The preflight tests include failure-before-holdout protection. The dashboard
checks cover canonical sorting, score variants, p_cal_fit exclusion, and metadata.
Standalone installer checks are in package_tests/test_installer.py; these remain
in this package, not the repository's tests directory.

## Using the calibration page

Choose your existing calibration, then Raw OOF or Calibrated cross-fit in the
new threshold section. The numeric input is an exploratory cutoff on that selected
score scale, not a deployment recommendation. Changing variant does not mean the
same number represents an equivalent policy. F1/F2 maxima do not move the input.
The budget tables report exact top-k and whether one scalar cutoff can reproduce
a boundary tie. Sorting uses ascending entity key for deterministic budget ties.
No callbacks read holdout data, write policies, or start evaluation.

## Policy-selection CLI: inspect first

Example against the saved test parent/calibration, explicitly selecting F2:

```bash
uv run python -m ml_template.evaluation.policy_selection \
  --parent-run-id run_20261016_121745_958bec \
  --calibration-id cal_20261017_063551_260e14 \
  --score-variant calibrated_crossfit --objective f2 --check
```

This prints an exploratory candidate, not a validated optimum. No policy files
are written. Existing registry initialization may still be performed by standard
repository access helpers.

After deciding that this objective and workload are appropriate, explicitly save:

```bash
uv run python -m ml_template.evaluation.policy_selection \
  --parent-run-id run_20261016_121745_958bec \
  --calibration-id cal_20261017_063551_260e14 \
  --score-variant calibrated_crossfit --objective f2 \
  --acknowledge-transfer
```

The acknowledgement concerns the approximate transfer from several cross-fit
mappings to the one final calibrator. It is not a statement that independent
validation has occurred. F1/F2 remain development selection-biased estimates.

Other choices:
- --objective f1: estimated F1 maximizer.
- --objective manual --value 0.08: explicit probability cutoff, no search.
- --objective alert_rate --value 0.05: frozen 5% batch budget, not a scalar cutoff.
- --score-variant raw: uses raw OOF scores; no calibrated-transfer acknowledgement.

All selections in this version require a calibration ID, even raw variants, so
source predictions come from one verified calibration artifact contract. Raw-only
projects without calibration can continue using existing manually specified
EvaluationConfig policies; a raw-only selector is out of scope for this change.

--output-root overrides artifacts/policies. --batch-definition documents the
actual operational grouping (for example one daily batch). The default is the
entire scoring batch; evaluation applies the policy globally to its evaluation
population, not separately to subgroups. Do not imply that this evaluates a
multi-day daily-budget policy unless your evaluation population is that batch.

Optionally use --evaluation-template <existing-evaluation-YAML>. It must already
identify the same parent/calibration. Intended use, criteria, uncertainty settings,
slices and limitations are retained; the selected policy and release variant are
set explicitly. Existing policy selection, if any, is replaced by the new explicit
selection. Review the resulting spec.

## Saved artifacts and review

Each command writes a new policy_<timestamp>_<id>/ directory, never changing a
completed calibration folder. It contains:
- selection.json: full-precision policy, source hashes, objective, predictor/data
  identity, diagnostics, tie rules, batch definition and transfer caveat.
- evaluation_draft.yaml: strict EvaluationConfig with the selection-file hash.
- README.txt: limitations and reminder that no holdout evaluation was executed.

The command prints their locations. Preserve selection.json unchanged. The draft
has an absolute artifact_path; after relocating a checkout, update that path in
the draft to the same preserved selection file, not the file's contents. Such an
edited evaluation specification has its own identity. Hashes detect accidental
changes, not maliciously rewritten artifacts and manifests.

Review intended use, objective, workload, limitations and predeclared acceptance
criteria before evaluation. A default draft has no deployment acceptance criteria.
Policy source files/config/calibrator/dataset/split must still match at preflight.
No threshold is rounded to the dashboard's table precision during export.

Then run the existing evaluator with the printed draft path:

```bash
uv run python -m ml_template.evaluation.evaluate_run --config <printed-evaluation-draft> --check
```

Do not run --execute until the policy and entire specification are frozen and any
prior holdout exposure is truthfully documented. The selection CLI never calls
--execute. Prior evaluation histories are retained. Adding optional schema fields,
policy metrics and new code hashes can change cache identities; old reports are
not overwritten or silently declared fresh evaluations.

## Statistical boundaries

- Neither calibration nor the final model is refitted by these changes.
- p_cal_fit never enters threshold assessment or selection.
- Each original fold's selected threshold is descriptive sensitivity, not held-out
  validation of the threshold-selection procedure.
- Cross-fit cutoffs are not mathematically converted through the final sigmoid.
- Selecting/reporting an optimum on the same development pairs is optimistic.
- The holdout assesses the frozen complete predictor-policy combination, not a
  newly optimized policy. No threshold search is added to holdout evaluation.
- A top-k budget splitting tied scores is not silently converted into p>=boundary.
- F2 weights recall more heavily; it is not an inferred business cost optimum.

## Validation and limitations

Portable checks here: 34 passed, 4 skipped. The skips cover the dashboard module
(including several tests), actual-project source-validation wrapper, evaluator
integration requiring SQLAlchemy, and final-metrics/bootstrap forwarding requiring
full repository modules. See VALIDATION.md for exact scope. New selector numerical,
provenance, export roundtrip and installer safety tests ran here.

The authoring sandbox has no full checkout, SQLAlchemy, PyArrow, Dash or Ruff.
Actual source-file/prediction I/O, all ten existing-module edit applications,
Dash rendering, full-project regressions and Ruff must be verified locally.
The installer performs all-target hashes/anchors/AST checks before source edits;
it refuses mismatches rather than assuming untested edits are safe.

No actual threshold value for your real run is included or claimed. The example
commands choose an objective explicitly for inspection; they do not select F2
on your behalf or authorize deployment.
