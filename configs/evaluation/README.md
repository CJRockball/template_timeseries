# configs/evaluation

Examples for this capability. Copy a template to an ignored local filename before editing.

- `binary.example.yaml` — binary_evaluation. 
- `intervals.example.yaml` — regression_evaluation. correction:none -> raw. bias/affine -> recalibrated. Match saved calibration and levels exactly.
- `regression.example.yaml` — regression_evaluation. Supported XGB or NN predictor; freeze final/CV-mean before holdout access.

These are requests or fragments, not interchangeable experiment YAMLs. Artifact-bound requests
contain deliberate REPLACE placeholders and are not operationally ready. See configs/README.md.
