"""Verified NN continuous predictors; no training or source-label access."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

SERIALIZATION = {
    "format": "torch",
    "weights_only": True,
    "checkpoint_version": 1,
    "preprocessing": "embedded",
}


def validate_descriptor(descriptor, legacy_validator):
    fields = {
        "schema_version",
        "task",
        "prediction_kind",
        "model_type",
        "serialization",
        "kind",
        "weighting",
        "fitted_features",
        "members",
        "files_sha256",
        "training_rows",
    }
    if (
        not isinstance(descriptor, dict)
        or set(descriptor) != fields
        or type(descriptor["schema_version"]) is not int
        or descriptor["schema_version"] != 3
        or descriptor["task"] != "regression"
        or descriptor["prediction_kind"] != "continuous"
        or descriptor["model_type"] != "nn"
        or descriptor["fitted_features"] is not False
        or not isinstance(descriptor["files_sha256"], dict)
        or descriptor["serialization"] != SERIALIZATION
        or type(descriptor["serialization"].get("checkpoint_version")) is not int
        or descriptor["serialization"].get("weights_only") is not True
    ):
        raise ValueError("Invalid NN regression descriptor semantics")
    legacy = {
        name: copy.deepcopy(descriptor[name])
        for name in ("kind", "weighting", "fitted_features", "members", "files_sha256")
    }
    legacy["schema_version"] = 1
    for name in ("config.json", "features.json"):
        value = legacy["files_sha256"].pop(name, None)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("Missing/invalid NN metadata hash: " + name)
    names = legacy_validator(legacy)
    models = {member["model_path"] for member in descriptor["members"]}
    counts = descriptor["training_rows"]
    if (
        not isinstance(counts, dict)
        or set(counts) != models
        or any(type(n) is not int or n < 1 for n in counts.values())
    ):
        raise ValueError("NN training-row inventory differs")
    return sorted([*names, "config.json", "features.json"])


def validate_config(cfg, descriptor):
    from ml_template.models import predictor_selection as selection

    validate_descriptor(descriptor, selection._legacy_required_predictor_files)
    if cfg.task != "regression" or cfg.model.type != "nn" or cfg.fitted_features:
        raise ValueError("NN predictor requires regression and model-owned preprocessing")
    if descriptor["kind"] == "cv_mean":
        ids = [member["fold_id"] for member in descriptor["members"]]
        if ids != list(range(cfg.split.n_folds)):
            raise ValueError("NN predictor must contain every configured fold")


def _json(path):
    def pairs(items):
        result = {}
        for name, value in items:
            if name in result:
                raise ValueError("Duplicate NN metadata key")
            result[name] = value
        return result

    def invalid(value):
        raise ValueError("Nonfinite NN metadata constant: " + value)

    return json.loads(
        Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=invalid
    )


def metadata_frame(info):
    names = info["features"]
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(c, str) or not c for c in names)
        or len(names) != len(set(names))
        or not isinstance(info.get("dtypes"), dict)
        or set(info["dtypes"]) != set(names)
    ):
        raise ValueError("Invalid saved NN input inventory")
    return pd.DataFrame({name: pd.Series([], dtype=info["dtypes"][name]) for name in names})


def bound_metadata(cfg, root, descriptor):
    from ml_template.models import predictor_selection as selection

    validate_config(cfg, descriptor)
    selection.verify_predictor_files(root, descriptor)
    from ml_template.training.run_config import RunConfig, config_hash

    archived = RunConfig.model_validate(_json(selection.safe_file(root, "config.json")))
    if config_hash(archived) != config_hash(cfg):
        raise ValueError("NN predictor/config snapshot differs")
    info = _json(selection.safe_file(root, "features.json"))
    metadata_frame(info)
    if (
        info["silver_dataset_version_id"] != cfg.data.silver_dataset_version_id
        or (info.get("fitted_features") or []) != []
    ):
        raise ValueError("NN feature metadata/config differs")
    return info


def describe(cfg, root, kind, *, fold_ids=None, verify_artifact=None, training_rows=None):
    from ml_template.models import predictor_selection as selection
    from ml_template.tracking.run_artifacts import verify_artifact as original_verify

    if cfg.task != "regression" or cfg.model.type != "nn" or cfg.fitted_features:
        raise ValueError("Expected NN regression with embedded preprocessing")
    verifier = original_verify if verify_artifact is None else verify_artifact
    # Original immutable artifact names remain unchanged, including legacy .ubj names.
    legacy = selection._legacy_describe_predictor(
        cfg, root, kind, fold_ids=fold_ids, verify_artifact=verifier
    )
    hashes = dict(legacy["files_sha256"])
    for name in ("config.json", "features.json"):
        verifier(root, name)
        hashes[name] = selection.file_sha256(selection.safe_file(root, name))
    descriptor = {
        **legacy,
        "schema_version": 3,
        "task": "regression",
        "prediction_kind": "continuous",
        "model_type": "nn",
        "serialization": dict(SERIALIZATION),
        "files_sha256": hashes,
        "training_rows": copy.deepcopy(training_rows),
    }
    validate_descriptor(descriptor, selection._legacy_required_predictor_files)
    bound_metadata(cfg, root, descriptor)
    return descriptor


def validate_gold(cfg, info, descriptor, member, adapter):
    from ml_template.models.nn.data import resolve_preprocessing

    pre = adapter.preprocessor
    expected = resolve_preprocessing(cfg.model.params, metadata_frame(info)).to_dict()
    if pre.config.to_dict() != expected:
        raise ValueError("Embedded NN preprocessing/config/input roles differ")
    if pre.state_dict()["fit_rows"] != descriptor["training_rows"][member["model_path"]]:
        raise ValueError("Embedded NN preprocessing fit-row count differs from verified partition")


class NNRegressionPredictor:
    def __init__(self, cfg, root, descriptor, *, adapter_loader=None):
        self.cfg = cfg
        self.root = Path(root)
        self.descriptor = copy.deepcopy(descriptor)
        self.info = bound_metadata(cfg, self.root, self.descriptor)
        self.adapter_loader = adapter_loader

    def _load_member(self, member):
        from ml_template.models import predictor_selection as selection

        selection.verify_predictor_files(self.root, self.descriptor)
        loader = self.adapter_loader
        if loader is None:
            from ml_template.models.model_factory import load_adapter

            loader = load_adapter
        try:
            adapter = loader(self.cfg, selection.safe_file(self.root, member["model_path"]))
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                raise RuntimeError("PyTorch required: install the project's pytorch extra") from exc
            raise
        validate_gold(self.cfg, self.info, self.descriptor, member, adapter)
        selection.verify_predictor_files(self.root, self.descriptor)
        return adapter

    def verify_checkpoints(self):
        # Explicit operation: deserialize all bound checkpoints but never load features/labels or predict.
        self.info = bound_metadata(self.cfg, self.root, self.descriptor)
        for member in self.descriptor["members"]:
            adapter = self._load_member(member)
            del adapter
        return {
            "checkpoints_verified": len(self.descriptor["members"]),
            "models_loaded": True,
            "predictions_computed": False,
            "labels_loaded": False,
        }

    def predict(self, X):
        from ml_template.models import predictor_selection as selection
        from ml_template.models.regression_predictor import continuous

        self.info = bound_metadata(self.cfg, self.root, self.descriptor)
        if (
            not isinstance(X, pd.DataFrame)
            or X.empty
            or not X.index.is_unique
            or X.index.hasnans
            or not X.columns.is_unique
            or set(X.columns) != set(self.info["features"])
        ):
            raise ValueError(
                "NN prediction requires a nonempty uniquely keyed estimator-input frame"
            )
        total = np.zeros(len(X), dtype=np.float64)
        for member in self.descriptor["members"]:
            adapter = self._load_member(member)
            try:
                before = adapter.preprocessor.state_dict()
                values = continuous(adapter.predict(X.copy(deep=True)), len(X))
                if adapter.preprocessor.state_dict() != before:
                    raise ValueError("NN preprocessing changed during prediction")
                total += values * member["weight"]
                selection.verify_predictor_files(self.root, self.descriptor)
            finally:
                del adapter
        return continuous(total, len(X))


def load(cfg, root, descriptor, *, adapter_loader=None):
    return NNRegressionPredictor(cfg, root, descriptor, adapter_loader=adapter_loader)
