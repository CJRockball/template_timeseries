"""Torch-free descriptor, binding and pre-deserialization safety tests."""

import copy
import hashlib
from types import SimpleNamespace

import pandas as pd
import pytest

from ml_template.evaluation.nn_predictor_preflight import PredictorRequest
from ml_template.models import predictor_selection as selection
from ml_template.models.nn_regression_predictor import (
    SERIALIZATION,
    NNRegressionPredictor,
    metadata_frame,
    validate_config,
    validate_descriptor,
)


def descriptor(kind="final"):
    members = [
        {"fold_id": None, "model_path": "model.ubj", "transformer_path": None, "weight": 1.0}
    ]
    if kind == "cv_mean":
        members = [
            {
                "fold_id": k,
                "model_path": f"fold_models/fold_{k}.ubj",
                "transformer_path": None,
                "weight": 0.5,
            }
            for k in range(2)
        ]
    names = ["config.json", "features.json", *[m["model_path"] for m in members]]
    if kind == "cv_mean":
        names += ["cv_folds.csv"]
    return {
        "schema_version": 3,
        "task": "regression",
        "prediction_kind": "continuous",
        "model_type": "nn",
        "serialization": dict(SERIALIZATION),
        "kind": kind,
        "weighting": "equal",
        "fitted_features": False,
        "members": members,
        "files_sha256": {n: "0" * 64 for n in names},
        "training_rows": {m["model_path"]: 10 for m in members},
    }


@pytest.mark.parametrize("kind", ["final", "cv_mean"])
def test_valid_explicit_nn_descriptor(kind):
    d = descriptor(kind)
    assert set(validate_descriptor(d, selection._legacy_required_predictor_files)) == set(
        d["files_sha256"]
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("task", "binary_classification"),
        ("prediction_kind", "probability"),
        ("model_type", "xgb"),
        ("serialization", {}),
        ("fitted_features", True),
        ("training_rows", None),
        ("training_rows", {"model.ubj": True}),
        ("training_rows", {"model.ubj": 0}),
        ("files_sha256", None),
        ("members", []),
        ("weighting", "learned"),
    ],
)
def test_bad_descriptor_fields(field, value):
    d = descriptor()
    d[field] = value
    with pytest.raises(ValueError):
        validate_descriptor(d, selection._legacy_required_predictor_files)


@pytest.mark.parametrize(
    "change",
    ["extra", "metadata_missing", "hash", "path", "sidecar", "weight", "format_version_bool"],
)
def test_descriptor_integrity(change):
    d = descriptor()
    if change == "extra":
        d["extra"] = True
    elif change == "metadata_missing":
        del d["files_sha256"]["features.json"]
    elif change == "hash":
        d["files_sha256"]["config.json"] = "z" * 64
    elif change == "path":
        d["members"][0]["model_path"] = "../escape.ubj"
    elif change == "sidecar":
        d["members"][0]["transformer_path"] = "model.transformer.joblib"
    elif change == "weight":
        d["members"][0]["weight"] = 0.9
    else:
        d["serialization"]["checkpoint_version"] = True
    with pytest.raises(ValueError):
        validate_descriptor(d, selection._legacy_required_predictor_files)


@pytest.mark.parametrize("change", ["fold", "reordered", "weight", "duplicate", "missing_count"])
def test_incomplete_cv_inventory(change):
    d = descriptor("cv_mean")
    if change == "fold":
        d["members"][0]["fold_id"] = True
    elif change == "reordered":
        d["members"].reverse()
    elif change == "weight":
        d["members"][0]["weight"] = 0.6
    elif change == "duplicate":
        d["members"][1] = copy.deepcopy(d["members"][0])
    else:
        d["training_rows"].pop("fold_models/fold_0.ubj")
    with pytest.raises(ValueError):
        validate_descriptor(d, selection._legacy_required_predictor_files)


@pytest.mark.parametrize(
    "task,model,fitted,folds",
    [
        ("binary_classification", "nn", None, 2),
        ("regression", "xgb", None, 2),
        ("regression", "nn", [{"step": "bad"}], 2),
        ("regression", "nn", None, 3),
    ],
)
def test_config_task_owner_and_fold_binding(task, model, fitted, folds):
    cfg = SimpleNamespace(
        task=task,
        model=SimpleNamespace(type=model),
        fitted_features=fitted,
        split=SimpleNamespace(n_folds=folds),
    )
    with pytest.raises(ValueError):
        validate_config(cfg, descriptor("cv_mean"))


@pytest.mark.parametrize("kind", ["final", "cv_mean"])
def test_config_accepts_full_inventory(kind):
    cfg = SimpleNamespace(
        task="regression",
        model=SimpleNamespace(type="nn"),
        fitted_features=None,
        split=SimpleNamespace(n_folds=2),
    )
    validate_config(cfg, descriptor(kind))


def test_tampering_rejected_before_adapter_loader(tmp_path):
    d = descriptor()
    for name in d["files_sha256"]:
        (tmp_path / name).write_bytes(b"original")
        d["files_sha256"][name] = hashlib.sha256(b"original").hexdigest()
    (tmp_path / "model.ubj").write_bytes(b"tampered")
    obj = object.__new__(NNRegressionPredictor)
    obj.root = tmp_path
    obj.descriptor = d
    obj.cfg = None
    obj.info = {}
    obj.adapter_loader = lambda *a: pytest.fail("tampered checkpoint reached loader")
    with pytest.raises(ValueError, match="changed"):
        obj._load_member(d["members"][0])


@pytest.mark.parametrize(
    "info",
    [
        {"features": ["a", "a"], "dtypes": {"a": "float64"}},
        {"features": ["a"], "dtypes": {}},
        {"features": [], "dtypes": {}},
        {"features": ["a"], "dtypes": None},
    ],
)
def test_bad_feature_inventory(info):
    with pytest.raises(ValueError):
        metadata_frame(info)


def test_feature_roles_can_be_resolved_without_rows():
    frame = metadata_frame({"features": ["a", "k"], "dtypes": {"a": "Float64", "k": "category"}})
    assert frame.empty and list(frame) == ["a", "k"]
    assert isinstance(frame.k.dtype, pd.CategoricalDtype)


@pytest.mark.parametrize("value", ["bad", "probability"])
def test_request_predictor_kind_strict(value):
    with pytest.raises(ValueError):
        PredictorRequest(parent_run_id="run_test", predictor_kind=value)


def test_request_does_not_authorise_evaluation():
    with pytest.raises(ValueError):
        PredictorRequest(parent_run_id="run_test", execute=True)
