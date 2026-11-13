# configs/monitoring

Examples for this capability. Copy a template to an ignored local filename before editing.

- `nn_package_06_labelled.example.yaml` — nn_monitor. --operation monitor. Warning limits require a target-unit protocol and adequate labels.
- `nn_package_06_labels.example.yaml` — nn_monitor. --operation labels. Only genuine known outcomes; declare sampling truthfully.
- `nn_package_06_reference.example.yaml` — nn_monitor. --operation reference. A tiny demo is plumbing, not a supported reference population.
- `nn_package_06_report.example.yaml` — nn_report. --operation report. The key is monitoring_id, NOT monitor_id; reads saved evidence only.
- `nn_package_06_template.example.yaml` — nn_monitor. --operation template: creates blank measured-outcome fields, not synthetic labels. Bind actual SHA256 first.
- `nn_package_06_unlabelled.example.yaml` — nn_monitor. --operation monitor. REPLACE illustrative as_of cutoff. Performance unknown without outcomes.
- `regression_labels.example.yaml` — labels. 
- `regression_monitor.example.yaml` — monitor. REPLACE illustrative as_of. Bind real mature labels before evaluating performance.
- `regression_reference.example.yaml` — reference. 

These are requests or fragments, not interchangeable experiment YAMLs. Artifact-bound requests
contain deliberate REPLACE placeholders and are not operationally ready. See configs/README.md.

NN commands: `python -m ml_template.monitoring.nn_monitor --operation reference|labels|monitor|template|report --config PATH --check`.
Use one operation name, not the literal pipe-separated expression. Database/output root are CLI arguments.
The SHA256 placeholder is deliberately not a fabricated valid hash; replace it with actual evidence before `--check`.
