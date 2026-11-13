# configs/calibration

Examples for this capability. Copy a template to an ignored local filename before editing.

- `binary.example.yaml` — binary_calibration. Binary probability calibration, never regression.
- `nn.example.yaml` — nn_calibration. 
- `regression.example.yaml` — regression_calibration. 
- `split.fragment.yaml` — split_fragment. Prospective new split only; never relabel old partitions. Set recalibration_fraction:0 before splitting for correction:none.

These are requests or fragments, not interchangeable experiment YAMLs. Artifact-bound requests
contain deliberate REPLACE placeholders and are not operationally ready. See configs/README.md.
