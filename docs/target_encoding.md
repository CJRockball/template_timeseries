# Cross-fitted target encoding for ml-template-kaggle

## Scope, compatibility and deliverables

This is a local patch bundle prepared against repository commit
`2d063eb52ab0b35f47edeb4852dc80af4ebfd6d3`. No GitHub, database, dataset or model has been modified.
The modified source snapshot was verified against Git blob
`289ae53700b05062e5f3296253b56ec3508c0083` before producing the patch.

Modified:
- `src/ml_template/features/fitted_features.py`

New:
- `src/ml_template/features/target_encoding.py`
- `tests/features/test_target_encoding.py`
- `examples/experiments/airline_target_encoding_overlay.yaml`
- `docs/target_encoding.md` (this guide)

The existing fitted_features schema validation and model factory discover the new registered
step through build_steps; neither module needs changing. No dependency is added. Existing
row-wise nullable FE, group statistics, permutation metric edits and source contracts are not
replaced by this patch. It is independent of your previously applied nullable-FE patch.

## Safe application

Extract the bundle OUTSIDE your repository. Start with a clean/checkpointed working tree.
From the repository root:

```bash
git status --short
git switch -c feat/crossfit-target-encoding
```

Set PATCH to the local path of `target_encoding.patch` inside the extracted bundle, then:

```bash
git apply --check "$PATCH"
git apply "$PATCH"
git diff --check
git diff -- src/ml_template/features/fitted_features.py
git status --short
```

The patch modifies one existing file and creates the four new paths listed above. If the
check fails, stop: review local source differences rather than forcing the replacement.
New files remain untracked until explicitly staged. The src tree in the bundle is also
included for review; do not replace entire repository folders with it.

## Why the wrapper also changes

Previously FittedFeatureTransformer.fit accepted X only and TransformingAdapter fitted steps
then transformed the same training frame. That is valid for unsupervised group statistics but
would let supervised target means include each training row's label.

The new protocol is:

1. A supervised step declares `requires_y = True`.
2. FittedFeatureTransformer.fit_transform(X, y) obtains cross-fitted training outputs from
   supervised steps. Unsupervised steps retain their previous fit(X)/transform(X) behaviour.
3. TransformingAdapter uses that training representation in fit_with_validation, refit and
   learning_curve. Validation/prediction/margin/explanation use frozen transform(X) only.
4. Existing raw columns are retained and new columns appended; duplicates fail explicitly.
5. Every step still reads the common original input frame, not outputs from preceding steps.
   This preserves the existing independent-step semantics.

No general-purpose imputer or in-place column replacement is introduced.

## Smoothing and nested leakage boundary

For group g in the permitted fitting rows:

\[
TE(g) = \frac{\sum_{i\in g} y_i + \alpha\,\bar y_{fit}}{n_g + \alpha}.
\]

Here alpha is smoothing (initially 20), and the prior is always from the same permitted
fitting subset. Groups below min_group_size and groups not seen during fitting receive that
subset's prior. A missing input category is learned as a distinct key, not replaced with a
literal sentinel label. Composite keys use a MultiIndex plus explicit null flags, not string
concatenation, so delimiters and literal `nan`/`<NA>`/empty strings cannot collide with nulls.

For TRAINING inputs, shuffled KFold makes 5 inner partitions. Each held-out partition receives
mappings and a prior learned only from the other inner partitions. KFold intentionally does
not inspect labels; changing labels within one partition cannot affect its own encodings.
This is an IID row-based encoder, not stratified/grouped/temporal target encoding.

After generating those training values, the encoder retains mappings fitted on all of the
supplied fitting rows for future transform(X). It does not retain the raw target vector,
training matrix or OOF training matrix in the saved transformer. Maps still contain learned
target aggregates and should be treated as model artifacts, not non-sensitive raw data.

The model-training and prediction encodings intentionally differ:
- fit_transform(X, y): inner-OOF training representation, plus saved full-fitting maps.
- fit(X, y).transform(X): full-fitting map representation, NOT a safe training substitute.
- transform(X_new): frozen mappings, without accepting labels or fitting anything.

In this repository there are three relevant levels:

```text
Outer development-CV training rows
  -> Inner fit/early-stopping split
       -> Target encoder cross-fits only the inner FIT rows
       -> Early-stopping rows use inner-FIT maps, never their labels
  -> Outer-fold refit
       -> Target encoder cross-fits outer TRAIN rows
       -> Outer validation uses outer-TRAIN maps, never validation labels
Final refit
  -> Cross-fit all development training rows; holdout remains excluded
  -> Saved maps on those development rows serve later holdout/test predictions
```

The outer CV runner is unchanged; its existing train/refit boundaries supply the permitted
rows. Inner encoder folds are determined by row order and encoder seed. Keep training ordering
stable for exact reproducibility. Prediction remains invariant to batching and row order.

## YAML usage

Merge the provided overlay into a NEW experiment YAML, not a feature/block-parameter YAML.
It adds 4 numeric columns for flight_distance alone and its combinations with class, travel
type and age. Keep original numeric distance and categorical features alongside the encodings.
No new Silver dataset or stateless feature_set_id is required. The fitted_features change
will participate in the experiment's training config identity.

For an existing fitted_features list, append steps under the one existing key, or replace
that list deliberately to isolate this experiment. Never add duplicate YAML mapping keys.
The supplied values are starting settings: smoothing=20, n_splits=5, min_group_size=1.
Smaller groups are shrunk rather than discarded; raise min_group_size in a separate test
if noisy rare combinations are a concern. Original-source mappings/teacher predictions are
NOT implemented here and no external dataset is opened.

Set EXP to the local path of the copied, merged experiment and run:

```bash
uv run python -m ml_template.training.run_config --config "$EXP"
make train EXP="$EXP"
```

Do not pass the overlay alone to training: it is not a complete RunConfig. Keep your current
data/model/split/training/metrics fields. Compare raw baseline + TE and/or current FE + TE
on the same stored folds, stopping policy, model parameters and model seed.

## Validation commands

```bash
uv run pytest -q tests/features/test_target_encoding.py tests/features/test_fitted_features.py
uv run pytest -q tests/features tests/scoring
uv run ruff check src/ml_template/features/target_encoding.py \
  src/ml_template/features/fitted_features.py tests/features/test_target_encoding.py
uv run pytest -q
```

Executed here: 53 isolated regression cases passed: 47 new target-encoding/wrapper cases and
6 unchanged existing fitted-feature tests. They cover the smoothing formula, inner-prior
exclusion, changing a held-out fold's labels, unique/singleton categories, unseen/missing keys,
composite-key collisions, dtype handling, invalid labels/configs, early-stopping/refit/learning
curve paths, mixed old/new steps, joblib save/load, singleton/shuffle inference parity and a
real scikit-learn decision-tree smoke test through the adapter wrapper.

The isolated package used the new encoder and patched fitted_features module, a hash-verified
copy of GroupStatisticsStep and the unchanged PowerTransformStep class extracted from the
existing generic_interactions module. It was NOT the complete installed repository. Spies
in wrapper unit tests intentionally record training/validation inputs; the sklearn smoke
case uses a real estimator. No production GBDT, real airline data, database/registry,
release packaging, Ruff or complete repository test suite was executed here.

## Persistence and releases

The existing TransformingAdapter.save/load stores the fitted transformer beside each model
as its `.transformer.joblib` companion. Keep that companion for fold and final models; model
weights alone are insufficient. Existing prediction, permutation and explanation paths use
those frozen maps. Input feature counts/features.json may describe pre-transform inputs,
so inspect estimator_feature_names or the saved transformer for the additional columns.

Source changes intentionally invalidate runtime fingerprints of existing sealed releases.
Do not edit old release manifests or bypass verify_runtime. For a model newly trained with
TE, run the normal evaluation/release path and verify preparation + scoring parity locally.
Do not overwrite old model artifacts or attach an encoder from a different training fold/run.
Existing saved unsupervised transformer objects remain structurally compatible with the
wrapper code, but sealed release runtime checks still apply.

## Limits

- Binary canonical targets 0/1 only; no implicit True/False string conversion.
- Requires at least n_splits rows for a cross-fitted training call. Tiny folds fail explicitly.
- Single-class inner fitting subsets use their legitimate 0/1 prior rather than target leakage.
- Cross-fitting does not fix duplicate entities, original/synthetic overlap, time leakage or
  grouping mistakes in outer splits. It does not create grouped or temporal inner folds.
- If repeated entities must stay together, implement group-aware folds before using this step.
- No public-test, holdout or validation labels are accepted by transform().
- No sample weighting, multiclass encoding, target selection, original-data-only mapping,
  noise injection or performance improvement guarantee is supplied.
- TE adds cross-fitting work per model fit/refit and can increase compute/memory use.

After reviewing diffs and passing your local tests, stage the new files and modified module
on your feature branch. Nothing in this bundle commits, pushes, retrains or submits automatically.
