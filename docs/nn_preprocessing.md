# NN Gold preprocessing

## Scope

This implements NN input imputation/encoding/scaling, not a neural-network model.
It was built against repository commit fecf47325941da4634170d867dc75cfef539cb7b.
Existing tree training, Silver, fitted feature generation, releases, experiment
schemas and model factory are deliberately not changed. No NN adapter currently
exists in that baseline. This code is called by that adapter when it is added;
installing the bundle does NOT automatically impute existing training runs.

## API and defaults

- `NNPreprocessorConfig`: explicit feature-only numeric/categorical lists. These
  are semantic choices: integer ratings may be categorical. Both lists may not
  be empty. Every supplied column must be assigned; labels/IDs are never accepted
  as undeclared extras. No inferred vocabulary from test data or category metadata.
- `NNPreprocessor.fit(X)`: learn only from the caller's training slice.
- `transform(X)`: fixed-order DataFrame; original numerical columns become
  float32, categorical columns become int64 IDs, and every source column gains
  a float32 `<column>__is_missing` flag. Observed=0, missing=1. Input is not mutated.
- `transform_arrays(X)`: contiguous `NNInputs.numeric` float32 and `.categorical`
  int64 matrices, plus names/cardinalities. Flags join the numerical channel and
  are NOT standardised. A categorical-only source therefore still has a numeric
  indicator channel; a numerical-only source returns an `(n_rows, 0)` cat array.
- Default numerical replacement: training observed median. Scaling: population
  mean/std of observed training values BEFORE filling. Mean, median and constant
  replacement, and standard/no scaling, are configurable. Median replacement is
  not generally zero after mean-centred scaling; mean replacement is zero.
- Constant scale = 1. Entirely missing numerical fit: configured fallback
  (default 0), centre=0, scale=1; recorded `n_observed=0`. This preserves columns
  but cannot learn the effects of values absent during fitting; inspect the
  smoke report's all-missing feature list.
- Categories: 0 missing, 1 unknown, observed categories 2 onwards. Actual labels
  called 'missing', '__UNKNOWN__', 'nan' or the empty string are ordinary values.
  String '1', numeric 1 and boolean True remain distinct; numeric 1 and 1.0 map
  identically to tolerate integer categories made float by missing values.
- Missing category IDs are not automatically embedding padding. Let the NN learn
  a missing embedding; do not silently freeze it with `padding_idx=0`.
- Unknown embeddings are not guaranteed to have learned weights when no unknown
  examples occur during training. The future model must select a reproducible
  initialisation/fallback, or evaluate category dropout. `unknown_policy: error`
  is available. The preprocessor's reserved bucket does not relax upstream
  allowed-category validation in a release.
- NaNs are permitted before Gold, but NaN/inf/float32 overflow are rejected after
  Gold. Categorical index bounds are checked. Numeric strings, complex numbers
  and non-scalar categorical objects are rejected, not silently coerced.

A constant replacement does not preserve the missing-data distribution. Missing
flags identify absence; they do not reconstruct the original values or remove
missingness bias. Evaluate alternatives on the same stored folds.

## Dedicated YAML (not an experiment or sweep)

`configs/preprocessing/nn_preprocessing_example.yaml` is independently loaded by
`load_nn_preprocessor_config()`. Replace its example feature names with the exact
inputs after existing fitted features. Do not paste these keys at the top level
of an existing RunConfig: that strict experiment schema is intentionally unchanged.
The preprocessor contains no PyTorch dependency and cannot select CUDA.

## Future NN adapter boundary (runnable preprocessing example)

```python
from ml_template.preprocessing.nn_config import NNPreprocessorConfig
from ml_template.preprocessing.nn_preprocessor import NNPreprocessor

# X_fit/X_stop contain features ONLY, after optional existing fitted FE.
pre = NNPreprocessor(
    NNPreprocessorConfig(
        numeric=("measurement", "delay"),
        categorical=("category", "rating"),
    )
)
pre.fit(X_fit)
fit_inputs = pre.transform_arrays(X_fit)
stop_inputs = pre.transform_arrays(X_stop)
# Future PyTorch adapter: torch.from_numpy(fit_inputs.numeric/categorical).
```

Lifecycle requirements:

1. Existing TransformingAdapter first fits/generates its fitted features. Preserve
   its new cross-fitted `fit_transform(X, y)` path for target encoding.
2. NNAdapter then fits NNPreprocessor on the representation it actually receives.
   During early stopping this is INNER FIT only. Transform inner-stop unchanged.
3. Outer-fold refit gets a fresh NN and fresh preprocessor fitted on that outer
   training fold. Final refit gets another fresh pair on development rows only.
4. Prediction calls the saved preprocessor exactly once, then the network.
5. Persist `pre.state_dict()` INSIDE the NN checkpoint alongside architecture
   metadata and network weights. Load with `NNPreprocessor.from_state_dict(...)`.
   Only primitive JSON-safe data are used. When PyTorch is added, use its
   state_dict/weights-only loading conventions rather than pickling this class.
6. This embedded-state approach avoids adding a new preprocessing sidecar to the
   repository's hard-coded release/fold artifact lists. Standalone `.save/.load`
   JSON methods are provided for diagnostics; an arbitrary new sidecar is NOT
   automatically copied or verified by existing release code.
7. For cv_mean, every member retains its own checkpoint/preprocessor; average
   predictions, not encoded arrays or preprocessing statistics.
8. Never insert NNPreprocessor as an existing append-only fitted feature step.
   It intentionally replaces columns. Never pre-impute all of Silver across CV.

The existing post-Gold frame validator remains unchanged. A new
`validate_nn_arrays` additionally validates the final shapes/dtypes/finiteness and
embedding ranges. It does not import or initialise CUDA.

## Smoke test

From repository root after installation:

```bash
uv run python -m ml_template.preprocessing.smoke_nn \
  --config configs/smoke/nn_preprocessing.yaml \
  --output-dir artifacts/smoke/nn_preprocessing_v1
```

The supplied configuration is synthetic and immediately runnable. It samples a
small dataset and splits fit/validation rows. Checks include finite arrays,
embedding bounds, fixed schema, frozen state, JSON reload and
singleton/chunk/shuffle parity. Output directories must not already exist.
No model is trained, no prediction quality is measured, no registry is touched,
and no final-evaluation/release evidence is created. This is NOT the later
forward/backward training smoke test; PyTorch is not required.

For a small test of real canonical data, copy the smoke YAML and use:

```yaml
preprocessor: configs/preprocessing/your_resolved_nn_columns.yaml
source:
  kind: development
  data: data/silver/DATASET/SILVER_ID/data.parquet
  assignments: data/splits/SPLIT_ID/assignments.parquet
  key: id
sample_rows: 4096
seed: 42
validation_fraction: 0.25
```

Paths resolve from the process working directory. Only configured features and
key are read, never the target. Source rows are filtered using stored `dev` keys
before sampling/fitting. The reader may scan feature columns for all source rows;
it does not load labels. Use canonical Parquet to preserve string key types and
nullable dtypes. CSV is supported but key inference must match assignments.
This bypasses production feature-recipe replay; for engineered/fitted features,
prepare the proper development feature frame first or call the API directly.

## Tests

```bash
uv run pytest -q tests/preprocessing
uv run ruff check --fix src/ml_template/preprocessing tests/preprocessing
uv run ruff format src/ml_template/preprocessing tests/preprocessing
uv run pytest -q
```

Test invalid/missing schema columns, future missingness, string/int categories,
unused full-frame category levels, all-missing columns, flags, dtype overflow,
independent fits, no input/state mutation, empty branches and checkpoint integrity.
A real-Parquet selection case requires PyArrow; a separate CSV case uses an
explicit assignment-reader fixture and does not need PyArrow. A full repository
checkout/test run and real model release parity must still be run locally.

## Release caution

New source files change runtime code hashes. Do not edit old sealed release
packages or bypass runtime checks. Package a new release according to the
repository's existing evidence/runtime requirements after local verification.
