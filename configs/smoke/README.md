# configs/smoke

Examples for this capability. Copy a template to an ignored local filename before editing.

- `nn_preprocessing.yaml` — preprocessing_smoke. 
- `nn_training.yaml` — training_smoke. Isolated synthetic smoke. Compatibility filename; no Silver or holdout needed.
- `nn_training_development.yaml` — training_smoke. Requires a bound experiment/Silver ID; use synthetic smoke first.

These are requests or fragments, not interchangeable experiment YAMLs. Artifact-bound requests
contain deliberate REPLACE placeholders and are not operationally ready. See configs/README.md.

Synthetic NN training overrides preprocessing column selection: source-generated num_0/num_1/cat_0 differ from the dataset example x1/x2. Keep that override when using the synthetic source.
