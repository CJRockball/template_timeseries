"""Describe, verify and load one final model or the complete saved CV-fold mean.

The shared stateless recipe runs upstream. Every fold adapter owns its own fitted
Gold transformer. This module never calls fit/refit and never learns test statistics.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_file(root, name):
    if not isinstance(name, str):
        raise ValueError("artifact name must be a relative string")
    root = Path(root).resolve()
    part = PurePosixPath(name)
    if (
        not name
        or part.is_absolute()
        or ".." in part.parts
        or "\\" in name
        or part.as_posix() != name
    ):
        raise ValueError("unsafe predictor artifact path")
    path = root.joinpath(*part.parts)
    if not path.resolve().is_relative_to(root):
        raise ValueError("predictor artifact escapes its root")
    if path.is_symlink() or any(
        p.is_symlink() for p in path.parents if p != root and p.is_relative_to(root)
    ):
        raise ValueError("predictor artifact symlinks are unsupported")
    return path


def _legacy_required_predictor_files(descriptor):
    """Validate descriptor structure before any model/transformer deserialisation."""
    if descriptor.get("schema_version") != 1 or descriptor.get("kind") not in {"final", "cv_mean"}:
        raise ValueError("unsupported predictor descriptor")
    if descriptor.get("weighting") != "equal" or not isinstance(
        descriptor.get("fitted_features"), bool
    ):
        raise ValueError("unsupported weighting or fitted-feature declaration")
    members = descriptor.get("members")
    if not isinstance(members, list) or not members:
        raise ValueError("predictor members are missing")
    kind, fitted = (descriptor["kind"], descriptor["fitted_features"])
    if kind == "final" and len(members) != 1:
        raise ValueError("final predictor requires one member")
    if kind == "cv_mean" and len(members) < 2:
        raise ValueError("CV mean requires at least two members")
    expected, folds = (set(), [])
    for member in members:
        if set(member) != {"fold_id", "model_path", "transformer_path", "weight"}:
            raise ValueError("invalid predictor member fields")
        fold = member["fold_id"]
        if kind == "cv_mean":
            if not isinstance(fold, int) or isinstance(fold, bool) or fold < 0:
                raise ValueError("invalid fold ID")
            folds.append(fold)
            model_name = f"fold_models/fold_{fold}.ubj"
        else:
            if fold is not None:
                raise ValueError("final member must not have a fold ID")
            model_name = "model.ubj"
        transformer = model_name.removesuffix(".ubj") + ".transformer.joblib" if fitted else None
        if member["model_path"] != model_name or member["transformer_path"] != transformer:
            raise ValueError("model/transformer member paths disagree")
        weight = member["weight"]
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (float, int))
            or (not np.isfinite(weight))
        ):
            raise ValueError("invalid member weight")
        if not np.isclose(weight, 1.0 / len(members), atol=1e-15, rtol=0):
            raise ValueError("only fixed equal weights are supported")
        expected.add(model_name)
        if transformer:
            expected.add(transformer)
    if kind == "cv_mean":
        if folds != sorted(set(folds)):
            raise ValueError("fold members must be unique and sorted")
        expected.add("cv_folds.csv")
    hashes = descriptor.get("files_sha256")
    if not isinstance(hashes, dict) or set(hashes) != expected:
        raise ValueError("predictor artifact hash coverage differs from members")
    if any(
        not isinstance(h, str) or re.fullmatch("[0-9a-f]{64}", h) is None for h in hashes.values()
    ):
        raise ValueError("invalid predictor artifact hash")
    return sorted(expected)


def verify_predictor_files(root, descriptor, *, names=None):
    declared = required_predictor_files(descriptor)
    selected = declared if names is None else list(names)
    if not set(selected) <= set(declared):
        raise ValueError("attempted undeclared predictor artifact load")
    for name in selected:
        path = safe_file(root, name)
        if not path.is_file() or file_sha256(path) != descriptor["files_sha256"][name]:
            raise ValueError(f"predictor artifact missing or changed: {name}")


def _legacy_describe_predictor(cfg, run_dir, kind="final", *, fold_ids=None, verify_artifact=None):
    """Bind selected predictor artifacts to the ORIGINAL training artifact manifest."""
    if kind not in {"final", "cv_mean"}:
        raise ValueError("unknown predictor kind")
    if verify_artifact is None:
        from ml_template.tracking.run_artifacts import verify_artifact
    root = Path(run_dir)
    fitted = bool(cfg.fitted_features)
    names, members = ([], [])
    if kind == "cv_mean":
        if fold_ids is None:
            raise ValueError("CV mean requires fold IDs from the verified stored split")
        ids = list(fold_ids)
        if any(
            not isinstance(k, int) or isinstance(k, bool) or k < 0 for k in ids
        ) or ids != sorted(set(ids)):
            raise ValueError("invalid stored split fold IDs")
        if len(ids) != cfg.split.n_folds or len(ids) < 2:
            raise ValueError("stored split fold count differs from run configuration")
        verify_artifact(root, "cv_folds.csv")
        table = pd.read_csv(root / "cv_folds.csv")
        if "fold" not in table or table["fold"].isna().any() or table["fold"].duplicated().any():
            raise ValueError("invalid saved CV fold inventory")
        numeric = pd.to_numeric(table["fold"], errors="raise")
        if (numeric % 1 != 0).any() or sorted(int(k) for k in numeric) != ids:
            raise ValueError("saved fold inventory differs from stored split")
        names.append("cv_folds.csv")
    else:
        ids = [None]
    for fold in ids:
        model_name = "model.ubj" if fold is None else f"fold_models/fold_{fold}.ubj"
        transformer = model_name.removesuffix(".ubj") + ".transformer.joblib" if fitted else None
        for name in [model_name, *([transformer] if transformer else [])]:
            verify_artifact(root, name)
            names.append(name)
        members.append(
            {
                "fold_id": fold,
                "model_path": model_name,
                "transformer_path": transformer,
                "weight": 1.0 / len(ids),
            }
        )
    descriptor = {
        "schema_version": 1,
        "kind": kind,
        "weighting": "equal",
        "fitted_features": fitted,
        "members": members,
        "files_sha256": {name: file_sha256(safe_file(root, name)) for name in names},
    }
    required_predictor_files(descriptor)
    for name in names:
        verify_artifact(root, name)
    verify_predictor_files(root, descriptor)
    return descriptor


def _legacy_validate_against_config(cfg, descriptor):
    required_predictor_files(descriptor)
    if bool(cfg.fitted_features) != descriptor["fitted_features"]:
        raise ValueError("saved predictor fitted-feature identity differs from config")
    if descriptor["kind"] == "cv_mean" and len(descriptor["members"]) != cfg.split.n_folds:
        raise ValueError("saved predictor fold count differs from config")


class FoldMeanPredictor:
    """Sequential fold loading: at most one member plus its Gold state in memory.

    Fold adapters receive independent copies of the same pre-Gold feature frame.
    Loading for each chunk prioritises bounded model memory over maximum throughput.
    """

    def __init__(self, cfg, root, descriptor, *, adapter_loader=None):
        if getattr(cfg, "task", "binary_classification") == "regression":
            raise ValueError("Use RegressionFoldMeanPredictor for regression")
        import copy

        self.cfg = cfg
        self.root = Path(root)
        self.descriptor = copy.deepcopy(descriptor)
        _validate_against_config(cfg, self.descriptor)
        if self.descriptor["kind"] != "cv_mean":
            raise ValueError("FoldMeanPredictor requires cv_mean")
        verify_predictor_files(self.root, self.descriptor)
        if adapter_loader is None:
            from ml_template.models.model_factory import load_adapter

            adapter_loader = load_adapter
        self.adapter_loader = adapter_loader

    def predict_proba(self, X):
        if not isinstance(X, pd.DataFrame) or X.empty or not X.index.is_unique:
            raise ValueError("fold prediction requires a nonempty, uniquely keyed feature frame")
        total = np.zeros(len(X), dtype=np.float64)
        for member in self.descriptor["members"]:
            names = [member["model_path"]]
            if member["transformer_path"]:
                names.append(member["transformer_path"])
            verify_predictor_files(self.root, self.descriptor, names=names)
            adapter = self.adapter_loader(self.cfg, safe_file(self.root, member["model_path"]))
            values = np.asarray(adapter.predict_proba(X.copy(deep=True)))
            if values.dtype.kind == "b" or values.shape != (len(X),):
                raise ValueError("fold returned labels or unexpected prediction shape")
            values = values.astype(np.float64)
            if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError("fold returned invalid probabilities")
            total += values
            del adapter
        return total / len(self.descriptor["members"])


def _legacy_load_selected_predictor(cfg, root, descriptor=None, *, adapter_loader=None):
    if adapter_loader is None:
        from ml_template.models.model_factory import load_adapter

        adapter_loader = load_adapter
    if descriptor is None:
        return adapter_loader(cfg, Path(root) / "model.ubj")
    _validate_against_config(cfg, descriptor)
    verify_predictor_files(root, descriptor)
    if descriptor["kind"] == "final":
        return adapter_loader(cfg, safe_file(root, descriptor["members"][0]["model_path"]))
    return FoldMeanPredictor(cfg, root, descriptor, adapter_loader=adapter_loader)


def load_evaluation_predictor(prepared, *, adapter_loader=None):
    descriptor = prepared.provenance.get("predictor")
    kind = getattr(getattr(getattr(prepared, "spec", None), "predictor", None), "kind", "final")
    if descriptor is None and kind != "final":
        raise ValueError("CV evaluation is missing its bound predictor descriptor")
    return load_selected_predictor(
        prepared.run_cfg, prepared.run_dir, descriptor, adapter_loader=adapter_loader
    )


def required_predictor_files(descriptor):
    if descriptor.get("schema_version") == 3:
        from ml_template.models.nn_regression_predictor import validate_descriptor

        return validate_descriptor(descriptor, _legacy_required_predictor_files)
    if descriptor.get("schema_version") == 2:
        from ml_template.models.regression_predictor import validate_descriptor

        return validate_descriptor(descriptor, _legacy_required_predictor_files)
    if descriptor.get("task") == "regression":
        raise ValueError("Regression requires a version-2 predictor descriptor")
    return _legacy_required_predictor_files(descriptor)


def _is_nn_regression(cfg):
    return (
        getattr(cfg, "task", None) == "regression"
        and getattr(getattr(cfg, "model", None), "type", None) == "nn"
    )


def describe_predictor(
    cfg, run_dir, kind="final", *, fold_ids=None, verify_artifact=None, training_rows=None
):
    if _is_nn_regression(cfg):
        from ml_template.models.nn_regression_predictor import describe

        return describe(
            cfg,
            run_dir,
            kind,
            fold_ids=fold_ids,
            verify_artifact=verify_artifact,
            training_rows=training_rows,
        )
    if getattr(cfg, "task", "binary_classification") == "regression":
        from ml_template.models.regression_predictor import describe

        return describe(cfg, run_dir, kind, fold_ids=fold_ids, verify_artifact=verify_artifact)
    return _legacy_describe_predictor(
        cfg, run_dir, kind, fold_ids=fold_ids, verify_artifact=verify_artifact
    )


def _validate_against_config(cfg, descriptor):
    neural = _is_nn_regression(cfg)
    if neural or descriptor.get("schema_version") == 3:
        from ml_template.models.nn_regression_predictor import validate_config

        return validate_config(cfg, descriptor)
    regression = getattr(cfg, "task", "binary_classification") == "regression"
    if regression != (descriptor.get("schema_version") == 2):
        raise ValueError("Predictor task/schema disagrees with run configuration")
    return _legacy_validate_against_config(cfg, descriptor)


def load_selected_predictor(cfg, root, descriptor=None, *, adapter_loader=None):
    if _is_nn_regression(cfg):
        from ml_template.models.nn_regression_predictor import load

        return load(cfg, root, descriptor, adapter_loader=adapter_loader)
    if getattr(cfg, "task", "binary_classification") == "regression":
        from ml_template.models.regression_predictor import load

        return load(cfg, root, descriptor, adapter_loader=adapter_loader)
    return _legacy_load_selected_predictor(cfg, root, descriptor, adapter_loader=adapter_loader)
