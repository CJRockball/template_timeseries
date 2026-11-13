# NN integration

This is the remaining NN model layer, not a replacement for the previous NN
preprocessing bundle. Keep its nn_config.py, nn_preprocessor.py and contracts.py
installed; they are currently local-only at inspected GitHub head fecf473.
The installer checks the prerequisite API and never overwrites those modules.

## Supported scope

Binary classification with an embedding MLP. Numeric projection, SiLU, dropout
and hidden LayerNorm follow the supplied nn.txt approach while handling singleton
batches. No GPU auto-selection. Both train and inference default to CPU. Explicit
CUDA is opt-in and untested here; deterministic CUDA requires
CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python. No mixed precision,
background data-loader workers or distributed training is introduced.
Multiclass/regression and transformer backbones are deliberately rejected until
repository-wide task support and the new backbones are implemented.

## Fit and prediction boundaries

The existing fitted-feature wrapper still generates features first, preserving
its cross-fitted target-encoding path. The NN adapter fits its own saved Gold
preprocessor on the actual training slice, never validation/test. Fresh outer
fold/final refits get fresh preprocessing. A five-fold run uses 11 training fits:
five inner early-stop fits, five outer-fold refits and one final development fit.
The protocol's round/best_round values mean epochs for this adapter.

Model checkpoints contain CPU tensor weights, frozen preprocessing, configuration,
seed, epoch count, training history and execution metadata. Existing .ubj paths
are retained for compatibility, but NN contents are torch checkpoints rather than
UBJSON. Loading is CPU-mapped, weights-only, configuration/seed checked and uses
strict parameter keys. No extra Gold sidecar is introduced. Existing fitted-FE
transformer.joblib sidecars remain unchanged. Final/cv_mean scoring reuses the
factory; each fold independently loads its own embedded Gold state.

## Input roles

In model params.preprocessing: numeric/categorical null/null means string,
object and pandas-category columns are cats and the remainder nums. Integer
ratings remain numeric unless explicitly listed as categorical. One explicit
list and one null infer the remainder. numeric: [] means all source inputs are
categorical; categorical: [] means all source inputs must be numerical. Two
explicit lists must cover all post-fitted-FE columns. Every input keeps a missing
flag, so categorical-only source inputs still have a numeric indicator channel.
Never use inherited pandas category levels as a fold-fitted vocabulary.

The prior preprocessing choices apply: observed-value scaling, median default,
reserved category IDs and fixed indicator schema. Missing embedding ID 0 starts
at zero and is trainable, not padding. Unknown ID 1 is refreshed after each epoch
to the mean observed-category embedding (zero if no observed categories). This
is an explicit fallback, not evidence of learned unknown-category performance.

## Optimisation and diagnostics

BCEWithLogitsLoss, row-weighted batch losses, shuffled indices including every
remainder row, finite losses/gradients/weights/logits, CPU best-state clones,
fixed-epoch refitting, and train/validation learning curves are implemented.
Adam defaults; AdamW, cosine scheduling, clipping and training-only automatic
positive weights are optional. The cosine horizon stays max_epochs during both
early selection and refits. pos_weight defaults to 1. monitor controls early
stopping (logloss=min, auc/aucpr=max), independently of metrics.primary.
LayerNorm avoids BatchNorm's singleton/remainder and batch-statistics issues.
Models return to CPU after fitting/inference, so saved CV members do not retain
all GPU models. Torch RNG/default dtype/thread count/determinism settings are
scoped and restored. Launch GPU jobs from terminals, not interactive kernels;
process isolation cannot protect against driver/system-level failures.

## Install and local acceptance

Extract outside the repo. Set BUNDLE to this extracted bundle directory, then
from repository root run the README's guarded installer commands. --check is
read-only unless --patch-output is explicitly requested. The installer preserves
local edits outside its unique anchors and refuses conflicts. It writes backups
and a complete native Git patch under .nn_model_backup/<id>/applied.patch.
Do not stage that directory. A repeat install of identical unformatted sources
is a no-op; local reformatting can require manual review of anchors.
uv.lock is not fabricated offline: uv sync --extra nn resolves it locally.

Run the supplied tests, then the synthetic smoke CLI. The smoke is a separate
schema/process with a timeout and checks actual backward, inner validation,
fresh refit, embedded checkpoint reload, finite probabilities, frozen Gold and
singleton/chunk/shuffle parity. It registers no normal run or release evidence.
Use fresh output directories. Its sampled-experiment variant filters saved dev
keys before reading labels and currently requires feature_set_id: null; normal
training supports existing feature sets and fitted-feature transformations.
A sampled smoke is not a predictive-quality estimate.

Copy a working experiment, retain exact data/split references, remove any inline
tree model override, and set model_file: configs/model/nn_mlp.yaml. The supplied
experiment has explicit placeholders; fill them before training. Keep stored
split settings unchanged. Start with expensive permutation importance and
learning curves disabled; both are supported through adapter probabilities and
independent diagnostic fits. TreeSHAP and tree gain importance are not supplied
for NNs. Run one model before the five-seed sweep (55 fits).

After code/environment are final, use the existing evaluate/release/prep/score/
export pipeline. The NN release example explicitly requests parity_atol=1e-6 for
float32 batching roundoff within existing permitted limits; global defaults are
not weakened. Version/provenance patches record torch and bind preprocessing
code to evaluation/release evidence. New code invalidates runtime-pinned old
packages: retain old checkouts for old releases and never edit sealed files or
bypass holdout/integrity checks.

## Validation limitation

Torch and Ruff are unavailable in the authoring environment. Supporting tests,
syntax/YAML/installer/native-patch checks are reported in validation/. Real
forward/backward, full CV, wrapper composition and release parity must pass
locally with the nn extra installed. A skipped torch suite is NOT NN validation.
