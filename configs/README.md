# Template config examples

Prepared against main 1d35b184ba83576c37bb1f2c241d1b05e2edff02.
These are tabular binary/regression examples; time-series support is NOT implemented here.

## Copying
1. Work on a maintenance branch, not an original pinned release/runtime checkout.
2. Archive the current configs directory outside Git (including ignored/local files) before replacement.
3. Copy this bundle's configs/ directory to the repository root. No installer, patch or source-code edits.
4. Replace the root config-ignore block with GITIGNORE_CONFIGS.txt. The supplied per-directory
   .gitignore files intentionally contain only comments so stale nested wildcard rules do not hide examples.
5. Do not leave a merged mixture of old named operator configs unintentionally. They are backed up,
   not migrated, by this manual procedure. Old names/paths referenced by callers need review.
6. From the extracted bundle, run validate_examples.py --repo /PATH/TO/REPO.
   It only reads files. Add --schemas for the documented subset of real repository config validators.
   Run audit_config_references.py --repo /PATH/TO/REPO --strict before committing a wholesale deletion.
   Literal-reference scanning is not a full dependency proof and may include documentation references.
7. Review git diff and run native compatibility tests before merging. No GitHub files were changed by
   producing this bundle and the full native suite was NOT run by its author.

## Naming and supported scope
Most reusable YAMLs end in .example.yaml. Fragments end in .fragment.yaml and must be merged into
an experiment, not passed as standalone experiment requests. Compatibility filenames and the exact
acceptance plan are intentional exceptions; see manifest.json and GITIGNORE_CONFIGS.txt.
Binary XGB/NN, regression XGB/NN and optional LGBM/CatBoost binary model presets are included.
No LGBM/CatBoost regression, multiclass or sequence-model support is asserted.
There are several files where a folder supports distinct tasks/operations: a monitoring import,
reference and assessment cannot be collapsed into one schema-valid request file.

## First use: preserve templates, edit ignored copies
Example:
  cp configs/experiment/xgb_regression.example.yaml configs/experiment/local_xgb.yaml
Edit local_xgb.yaml and its dataset contract/model references. In particular:
- Provide an actual CSV and truthful source row/column counts; the illustrative 256-row contract
  is not supplied data. id/x1/x2/target are generic column names, not measurements.
- Build/verify Silver and use its returned immutable ID. Do not invent Silver or fs_ identities.
- The root bronze_silver_contract.yaml is the generic regression compatibility contract.
  Binary experiments use binary_contract.example.yaml.
- The generic interactions example uses product/difference, avoiding hidden denominator/log domains.
- No imputation is applied globally at Silver. The fitted fragment is opt-in, XGB-only here;
  NN numeric inputs are handled by its own saved preprocessing. Do not mix external fitted FE into NN.
- Small model/sweep budgets demonstrate wiring, not tuned performance.

## Existing model / release requests
Replace every REPLACE token with a value verified from local registries and saved artifacts.
Also replace the illustrative as_of 2000-01-01T00:00:00Z with the actual timezone-aware reporting
cutoff. target_unit is operator-declared and must agree with frozen evidence; no unit is inferred.
The NN manifest-SHA placeholder is intentionally invalid until replaced by its real 64-hex hash.
Parsing YAML successfully does NOT mean a request is ready, that data exist or that labels may be read.
All research/exposure/scoring acknowledgements remain false. Do not enable them automatically.
Prior exposure cannot be erased by choosing new config names or reserving new partitions.

Correction/interval workflow:
1. Declare new calibration/recalibration partitions prospectively, before fitting. Do not relabel
   old splits. The split fragment reserves both; set recalibration_fraction:0 if correction:none.
2. Freeze a predictor. Fit only authorized calibration/recalibration roles.
3. For correction:none use release_variant:raw; for bias/affine use recalibrated. Set actual regcal_
   ID in the evaluation, match saved calibrated coverages, then bind release to that evaluation.
4. The release examples default calibration_id:null. Set the actual matching ID when packaging
   a calibrated/interval evaluation. Never reuse an evaluation for a different predictor/variant.

Release-first NN scoring does not train, calibrate, evaluate or construct a release.
Use the original pinned interpreter AND source checkout for that release; never switch a live
release runtime to the current editable checkout merely to pass integrity checks.
Input contains source features/keys, not target truth. Export reference supplies only keys/order.
Monitoring uses genuine measured outcomes, truthful sampling and eligible_at/observed_at as-of rules.
No-label performance is unknown. Warning limits stay null until a unit-aware protocol is declared.
Saved-report config uses monitoring_id (not monitor_id). Regression explanation is native XGB tree
SHAP, not a generic NN explainer, not causal effects and not interval-bound attributions.

## Compatibility filenames
- experiment/xgb_baseline.yaml: generic binary baseline.
- model/xgb.yaml and model/nn_mlp.yaml: copies of the corresponding binary examples.
- preprocessing/nn_preprocessing_example.yaml: compatibility copy of nn.example.yaml.
- explainability/example.yaml: generic binary explanation request, no personal paths.
- smoke/nn_training.yaml and smoke/nn_preprocessing.yaml: synthetic smoke schemas.
- smoke/nn_training_development.yaml: separately bound development-data smoke schema.
- sweep/xgb_grid.yaml: 12-run binary compatibility grid; xgb.example.yaml is four-run regression.
Do not treat this short compatibility list as proof that all old custom config names remain valid.
The old patch_suite/00-08 configs and dataset-named request files are intentionally not replicated.

## Acceptance policy
acceptance/nn_package_07.json is byte-for-byte the original plan (1928 bytes), original baseline
57fa05eeb216212c7bf356ce1a138a4d2ff7010a, Git blob b7e1fd4c595fec68cb3a84d9b489adb02b31ec12.
It is maintained infrastructure, not an operator demo. Never delete, reduce or rewrite its phases
to make a config cleanup pass. No existing source/weights/dependencies/lockfiles or frozen artifacts
are included in or modified by this bundle.

## Local schema checks (read-only, no training/data access)
  uv run --no-sync python /PATH/TO/EXTRACTED/validate_examples.py --repo . --schemas
This covers real RunConfig/model, sweep, dataset-role, NN preprocessing, fitted/split fragment,
regression evaluation and report schemas. Other requests are template-only in that helper, explicitly
reported as not schema-checked. Missing dependencies fail rather than being silently skipped.
It does NOT call operational preflight, load datasets/models/labels, access a database or fit anything.

Run the project's normal tests and strict NN acceptance separately when appropriate. A cancelled
or selected diagnostic is not full native acceptance; this bundle does not resolve the outstanding
hosted Python3.11/3.12 full-matrix evidence requirement.

## Synthetic smoke commands, after copying
  uv run --no-sync python -m ml_template.preprocessing.smoke_nn --config configs/smoke/nn_preprocessing.yaml
  uv run --no-sync python -m ml_template.training.nn_smoke --config configs/smoke/nn_training.yaml
The training smoke is the existing binary smoke runner. Its synthetic columns are num_0/num_1/cat_0,
so that request explicitly overrides preprocessing to infer branches rather than use x1/x2.
This does not change the feature-column policy of actual experiment examples. Neither smoke produces
performance/release evidence. The development-data smoke also uses a binary experiment, not regression.
