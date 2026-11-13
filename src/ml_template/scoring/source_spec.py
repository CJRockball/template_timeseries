"""Freeze raw-to-canonical preparation against the exact training Silver contract."""

import json

import yaml

from ml_template.scoring.release_artifacts import sha256
from ml_template.training.run_config import resolve_path


def freeze_source_spec(prepared, source):
    cfg = prepared.run_cfg
    path = resolve_path(cfg.data.dataset_config)
    silver = (
        prepared.roots["silver_root"] / cfg.data.dataset_name / cfg.data.silver_dataset_version_id
    )
    manifest = json.loads((silver / "manifest.json").read_text())
    expected = manifest.get("config_sha256")
    if not expected or sha256(path) != expected:
        raise ValueError("dataset YAML differs from Silver build; restore the training contract")
    raw = yaml.safe_load(path.read_text())
    if sha256(path) != expected:
        raise ValueError("dataset YAML changed while reading")
    spec = dict(raw["silver_eda_contract"])
    key = prepared.key_col
    columns = [key, *source.columns]
    if any(col not in spec["canonical_dtypes"] for col in columns):
        raise ValueError("canonical dtype missing for scoring source")
    target = raw["data"]["target"]
    target = spec["canonical_names"].get(target, target)
    # JSON object keys are strings; record pairs preserve typed mapping keys.
    spec["value_mappings"] = {
        col: [{"source": a, "value": b} for a, b in mapping.items()]
        for col, mapping in spec.get("value_mappings", {}).items()
    }
    spec["required_columns"] = columns
    spec["forbidden_columns"] = [target, "y_true"]
    spec["training_contract_sha256"] = expected
    # Optional explicit policy for derived model features; raw-feature overrides persist.
    model_policy = spec.get("model_missingness", spec.get("missingness", {}))
    return spec, spec.get("missingness", {}), model_policy
