"""Task-bound continuous predictors. Reuses the existing artifact verifier."""

import copy
from pathlib import Path

import numpy as np
import pandas as pd


def continuous(values, size):
    values = np.asarray(values)
    if values.dtype.kind not in "iuf" or values.shape != (size,):
        raise ValueError("Expected a numeric regression vector, not labels/objects/matrices")
    values = values.astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Regression predictions must be finite")
    return values


def validate_descriptor(descriptor, legacy_validator):
    if (
        descriptor.get("schema_version") != 2
        or descriptor.get("task") != "regression"
        or descriptor.get("prediction_kind") != "continuous"
    ):
        raise ValueError("Invalid regression predictor semantics")
    expected = {
        "schema_version",
        "task",
        "prediction_kind",
        "kind",
        "weighting",
        "fitted_features",
        "members",
        "files_sha256",
    }
    if set(descriptor) != expected:
        raise ValueError("Unexpected regression descriptor fields")
    legacy = {k: v for k, v in descriptor.items() if k not in {"task", "prediction_kind"}}
    legacy["schema_version"] = 1
    return legacy_validator(legacy)


def describe(cfg, root, kind, *, fold_ids=None, verify_artifact=None):
    from ml_template.models import predictor_selection as selection

    if getattr(cfg, "task", None) != "regression":
        raise ValueError("Expected a regression config")
    from ml_template.preprocessing.numeric_preprocessing import validate_regression_specs

    validate_regression_specs(cfg.fitted_features)
    descriptor = selection._legacy_describe_predictor(
        cfg, root, kind, fold_ids=fold_ids, verify_artifact=verify_artifact
    )
    descriptor = {
        **descriptor,
        "schema_version": 2,
        "task": "regression",
        "prediction_kind": "continuous",
    }
    selection.required_predictor_files(descriptor)
    return descriptor


class FinalRegressionPredictor:
    def __init__(self, cfg, root, descriptor, loader):
        self.cfg, self.root, self.descriptor = cfg, Path(root), copy.deepcopy(descriptor)
        self.loader = loader

    def predict(self, X):
        from ml_template.models import predictor_selection as selection

        _frame(X)
        selection._validate_against_config(self.cfg, self.descriptor)
        selection.verify_predictor_files(self.root, self.descriptor)
        adapter = self.loader(
            self.cfg, selection.safe_file(self.root, self.descriptor["members"][0]["model_path"])
        )
        try:
            return continuous(adapter.predict(X.copy(deep=True)), len(X))
        finally:
            del adapter


def _frame(X):
    if not isinstance(X, pd.DataFrame) or X.empty or not X.index.is_unique or X.index.hasnans:
        raise ValueError("Expected a nonempty uniquely keyed feature frame")


class RegressionFoldMeanPredictor(FinalRegressionPredictor):
    """Load one member at a time; never fit or silently omit a failed member."""

    def predict(self, X):
        from ml_template.models import predictor_selection as selection

        _frame(X)
        selection._validate_against_config(self.cfg, self.descriptor)
        selection.verify_predictor_files(self.root, self.descriptor)
        total = np.zeros(len(X), dtype=np.float64)
        for member in self.descriptor["members"]:
            names = [member["model_path"]]
            if member["transformer_path"]:
                names.append(member["transformer_path"])
            selection.verify_predictor_files(self.root, self.descriptor, names=names)
            adapter = self.loader(self.cfg, selection.safe_file(self.root, member["model_path"]))
            try:
                total += continuous(adapter.predict(X.copy(deep=True)), len(X)) / len(
                    self.descriptor["members"]
                )
            finally:
                del adapter
        return continuous(total, len(X))


def load(cfg, root, descriptor, *, adapter_loader=None):
    from ml_template.models import predictor_selection as selection

    if descriptor is None:
        raise ValueError("Regression requires a task-bound version-2 predictor descriptor")
    selection._validate_against_config(cfg, descriptor)
    selection.verify_predictor_files(root, descriptor)
    if adapter_loader is None:
        from ml_template.models.model_factory import load_adapter

        adapter_loader = load_adapter
    cls = FinalRegressionPredictor if descriptor["kind"] == "final" else RegressionFoldMeanPredictor
    return cls(cfg, root, descriptor, adapter_loader)
