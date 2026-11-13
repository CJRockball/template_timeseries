# Nullable stateless feature engineering

## Scope and provenance

Prepared against CJRockball/ml-template-kaggle at commit `2d063eb52ab0b35f47edeb4852dc80af4ebfd6d3`.
The two input source files were reconstructed from connector-returned contents and verified
byte-for-byte against their Git blob SHA-1 identifiers before patch generation:

- `src/ml_template/features/block_registry.py`: `1d332eece1ca6cedf753b75a79a97a5ec85f2c23`
- `src/ml_template/features/blocks/generic_interactions.py`: `42f1e9801cc7550576b6b0ac0fb6a4409d3aa98f`

This is a local, downloadable patch. No GitHub branch, commit, pull request, database,
Silver artifact, or release has been modified.

## Apply safely

Start from a clean working tree or commit/stash your own work first. Create a branch:

```bash
git status --short
git switch -c feat/nullable-feature-blocks
```

Set the shell variable PATCH to the local path of the downloaded patch, then run:

```bash
git apply --check "$PATCH"
git apply "$PATCH"
git diff --check
git diff --stat
git status --short
```

If the check fails, stop and inspect differences; do not force or blindly overwrite files.
New files appear as untracked until staged. `git diff --stat` alone does not list those.

## Files

Modified:

- `src/ml_template/features/block_registry.py`
- `src/ml_template/features/blocks/generic_interactions.py`

Added:

- `tests/features/test_nullable_features.py` (56 pure regression cases)
- `tests/features/test_nullable_feature_store.py` (2 full-repository persistence cases)
- `examples/features/airline_nullable.yaml`
- `docs/nullable_feature_engineering.md` (this guide)

## Behaviour

- Strict by default: a block may emit missing values only in explicitly named outputs.
- New decorator argument `nullable_outputs` accepts an exact sequence of output names,
  or a callable taking resolved block parameters and returning that sequence.
- `Block.missing_columns(params)` validates the declaration. Duplicate names, malformed
  declarations and names not actually emitted by the block are rejected.
- Numeric output validation ignores permitted null entries but still rejects both infinities.
  Pandas nullable integer, floating, Boolean and string outputs retain their null masks/dtypes.
- Input-column requirements, output index equality, name collisions and duplicate checks remain.
- Generic numeric operations propagate missing values; their domain checks apply to observed
  values. Logarithms of invalid observed values and zero denominators still raise.
- Generic numeric inputs containing infinity are rejected before computation, so transforms
  cannot hide an infinite input (for example through a reciprocal).
- Category crosses use nullable strings, preserving nulls instead of creating the literal
  labels `nan`, `None` or `<NA>`. Genuine input strings with these spellings remain strings.
- Numeric/category gates return null if either operand is missing; an observed nonmatching
  category still yields zero. This intentionally avoids losing a missing numeric value
  through a nonmatching gate.
- No imputation, learned statistics, row dropping, target access or test-data fitting is added.
- Custom blocks remain responsible for mathematical domain validity and any intentionally
  created nulls. An allowlist permits missing outputs; it does not prove their provenance.

## Build the first airline feature set

The example produces arrival-minus-departure delay and log1p arrival delay. Confirm that
observed arrival delays exceed -1 before choosing log1p; otherwise omit that transform.
Choose the actual airline experiment YAML rather than assuming the filename below exists:

```bash
EXP=configs/experiment/YOUR_AIRLINE_EXPERIMENT.yaml
uv run python -m ml_template.features.build_features \
  --config "$EXP" \
  --blocks generic_interactions \
  --block-params examples/features/airline_nullable.yaml \
  --description "Nullable delay features; imputation deferred"
```

Put the printed feature-set ID in `data.feature_set_id` in a copied FE experiment YAML.
Reuse the baseline's immutable split for comparisons; do not recreate the split or inspect
holdout outcomes while selecting features. Do not overwrite your baseline configuration.
The build keeps raw feature columns and appends the new ones. Null permissions apply to
outputs, not to the raw Silver input contract, which remains independently enforced.

For another generic operation, list its exact generated output name in `nullable_outputs`.
Omitting the list preserves strict behaviour; misspelling an output is an error, not ignored.
The policy is carried by resolved parameters in the existing manifest/recipe mechanism.

## Custom blocks

```python
@register_block(
    "delay_difference",
    requires=("arrival_delay", "departure_delay"),
    description="Arrival minus departure; retain missing source values",
    nullable_outputs=("delay_difference",),
)
def delay_difference(df, params):
    return pd.DataFrame(
        {"delay_difference": df["arrival_delay"] - df["departure_delay"]},
        index=df.index,
    )
```

Use the standard pandas and register_block imports in the block module. No central registry
list needs editing. Callable policies must be deterministic from parameters, never from a
particular batch's values or missingness rate.

## Imputation and model boundary

This patch enables nullable FE only. It does not implement Gold imputation. A learned imputer
must be fitted on each training fold only, reused for that fold's validation data, and saved
with the final fitted model for inference. Do not build one globally imputed Gold table before
cross-validation. The current fitted-feature transformer appends columns; implementing an
imputer requires explicit replacement/selection semantics in a subsequent change.

## Scoring contracts and immutable artifacts

Block execution is not release schema validation. If a future release consumes nullable
engineered columns, its model-input missingness policy must explicitly permit them. Where
your release integration supports `silver_eda_contract.model_missingness`, supply a complete
policy including raw `arrival_delay` and the derived columns. Do not silently infer rates
from competition test data or globally disable schema checks.

The dataset YAML's hash is tied to its Silver manifest. Do not edit an old training contract
in place just to package a release: make a deliberate new version/build when changing the
contract, or separately design/version release policy handling. This patch does not touch
that YAML or bypass hash checks; existing nullable-aware release integration remains a
separate prerequisite for end-to-end scoring.

Shared validator source now participates in each block's code hash; resolved nullable
output declarations also participate in block IDs. Thus all newly built block IDs change,
including strict blocks. Old engineered feature recipes will correctly reject changed code.
Build fresh blocks/feature sets for new runs and package fresh releases when needed. Never
rewrite stored manifests or disable frozen-recipe/runtime verification. Existing Silver data
can be reused for FE if its canonicalisation/contract is unchanged.

## Validation

Run from the actual repository root after applying:

```bash
uv run pytest -q tests/features/test_nullable_features.py
uv run pytest -q tests/features/test_nullable_feature_store.py
uv run pytest -q tests/features tests/scoring
uv run ruff check src/ml_template/features/block_registry.py \
  src/ml_template/features/blocks/generic_interactions.py \
  tests/features/test_nullable_features.py tests/features/test_nullable_feature_store.py
uv run pytest -q
```

Executed here: 56 new pure-function regression cases passed using the patched modules and
an exact hash-verified snapshot of the existing feature_recipe.py. The isolated package had
only the two FE modules and recipe module; it did not substitute mocks for their execution.
Coverage includes strict defaults, nullable extension types, malformed declarations,
infinity/domain failures, category nulls, unchanged inputs, frozen recipe hash rejection,
new row IDs, shuffled input and singleton chunks.

Not executed here: the 2 Parquet/store tests, the original repository suite, Ruff,
registries, actual airline files, model training, Gold imputation or real release scoring.
The execution environment has no PyArrow/SQLAlchemy/Ruff. Those checks must run locally.

## Review and stage

After tests and diff review, stage the two modified modules plus the four new files.
Commit on your feature branch only after reviewing the results; this bundle does not
perform Git writes or submit anything automatically.
