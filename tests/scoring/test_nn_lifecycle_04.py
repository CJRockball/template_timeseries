"""Pure NN release metadata and strict keyed parity tests."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ml_template.scoring.nn_release import compare_reference, neural_config, validate_packaged_nn


def world():
    cfg = SimpleNamespace(task="regression", model=SimpleNamespace(type="nn"), fitted_features=None)
    summary = {"model_type": "nn", "nn_evaluation_schema": 1, "target_unit": "MPa"}
    info = {
        "model_type": "nn",
        "research_only": True,
        "automatic_promotion": False,
        "target_unit": "MPa",
        "evaluation_summary": summary.copy(),
    }
    contract = {
        "schema_version": 3,
        "model_schema_stage": "pre_gold",
        "gold_preprocessing": "owned_by_saved_model_adapter",
        "estimator_validation": "saved_nn_adapter_finite_arrays",
        "unseen_category_policy": "error",
    }
    descriptor = {
        "schema_version": 3,
        "model_type": "nn",
        "serialization": {
            "format": "torch",
            "weights_only": True,
            "checkpoint_version": 1,
            "preprocessing": "embedded",
        },
    }
    evidence = {"evaluation_spec": {"target_unit": "MPa"}, "evaluation_summary": summary}
    return (cfg, info, contract, descriptor, evidence)


def test_valid_packaged_metadata():
    validate_packaged_nn(*world())


@pytest.mark.parametrize(
    "kind",
    [
        "model",
        "units",
        "summary",
        "promotion",
        "research",
        "owner",
        "stage",
        "finite",
        "category",
        "descriptor",
        "external_fitted",
    ],
)
def test_bad_packaged_metadata(kind):
    cfg, info, contract, d, e = world()
    if kind == "model":
        info["model_type"] = "xgb"
    elif kind == "units":
        info["target_unit"] = "wrong"
    elif kind == "summary":
        info["evaluation_summary"] = {"wrong": True}
    elif kind == "promotion":
        info["automatic_promotion"] = True
    elif kind == "research":
        info["research_only"] = False
    elif kind == "owner":
        contract["gold_preprocessing"] = "global"
    elif kind == "stage":
        contract["model_schema_stage"] = "post_gold"
    elif kind == "finite":
        contract["estimator_validation"] = "not wired"
    elif kind == "category":
        contract["unseen_category_policy"] = "allow_all"
    elif kind == "descriptor":
        d["schema_version"] = 2
    else:
        cfg.fitted_features = [{"step": "other"}]
    with pytest.raises(ValueError):
        validate_packaged_nn(cfg, info, contract, d, e)


def test_legacy_xgb_metadata_unchanged():
    cfg, info, contract, d, e = world()
    cfg.model.type = "xgb"
    info = {}
    d = {}
    validate_packaged_nn(cfg, info, contract, d, e)


@pytest.mark.parametrize(
    "cfg",
    [
        None,
        SimpleNamespace(task="regression"),
        SimpleNamespace(task="binary_classification", model=SimpleNamespace(type="nn")),
    ],
)
def test_lightweight_legacy_config_not_misrouted(cfg):
    assert not neural_config(cfg)


def parity():
    reference = pd.DataFrame(
        {
            "id": [3, 1, 2],
            "raw": [-1.0, 2.0, 3.0],
            "point": [0.0, 3.0, 4.0],
            "lower": [-2.0, 1.0, 2.0],
            "upper": [2.0, 5.0, 6.0],
        }
    )
    whole = reference.rename(columns={"raw": "y_pred_raw", "point": "y_pred_release"}).copy()
    return (
        reference,
        whole,
        whole.iloc[::-1].copy(),
        whole.iloc[:1].copy(),
        {"raw": "y_pred_raw", "point": "y_pred_release", "lower": "lower", "upper": "upper"},
    )


def test_keyed_shuffled_and_negative_parity():
    ref, whole, chunk, single, columns = parity()
    assert all(
        v == 0 for v in compare_reference(ref, whole, chunk, single, "id", columns, 1e-06).values()
    )


@pytest.mark.parametrize("tol", [True, 0.0, -1.0, np.inf, np.nan, 0.0001])
def test_invalid_tolerances(tol):
    a, b, c, d, cols = parity()
    with pytest.raises(ValueError):
        compare_reference(a, b, c, d, "id", cols, tol)


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "duplicate",
        "null_key",
        "float_key",
        "raw",
        "point",
        "interval",
        "chunk",
        "singleton",
        "singleton_key",
        "nonfinite",
        "boolean",
    ],
)
def test_parity_failures(kind):
    ref, whole, chunk, single, columns = parity()
    if kind == "missing":
        whole = whole.iloc[:-1]
    elif kind == "duplicate":
        whole.loc[1, "id"] = 3
    elif kind == "null_key":
        whole.loc[0, "id"] = np.nan
    elif kind == "float_key":
        whole["id"] = whole.id.astype(float)
    elif kind == "raw":
        whole.loc[0, "y_pred_raw"] += 0.1
    elif kind == "point":
        whole.loc[0, "y_pred_release"] += 0.1
    elif kind == "interval":
        whole.loc[0, "lower"] -= 0.1
    elif kind == "chunk":
        chunk.loc[0, "y_pred_release"] += 0.1
    elif kind == "singleton":
        single.loc[single.index[0], "y_pred_release"] += 0.1
    elif kind == "singleton_key":
        single.loc[single.index[0], "id"] = 99
    elif kind == "nonfinite":
        whole.loc[0, "y_pred_raw"] = np.inf
    else:
        whole["y_pred_raw"] = True
    with pytest.raises(ValueError):
        compare_reference(ref, whole, chunk, single, "id", columns, 1e-06)


def test_explicit_small_float32_roundoff_allowed():
    a, b, c, d, cols = parity()
    b["y_pred_release"] += 1e-07
    c["y_pred_release"] += 1e-07
    d["y_pred_release"] += 1e-07
    compare_reference(a, b, c, d, "id", cols, 1e-06)


def test_inputs_not_mutated_by_parity():
    a, b, c, d, cols = parity()
    before = [frame.copy(deep=True) for frame in [a, b, c, d]]
    compare_reference(a, b, c, d, "id", cols, 1e-06)
    for original, frame in zip(before, [a, b, c, d], strict=True):
        pd.testing.assert_frame_equal(original, frame)


@pytest.mark.parametrize("kind", ["valid", "tamper_before", "tamper_during"])
def test_reference_projection_and_tamper_guards(tmp_path, monkeypatch, kind):
    import hashlib
    import sys
    import types

    from ml_template.scoring.nn_release import check_parity

    ref, whole, _, _, _ = parity()
    ref = ref.rename(columns={"raw": "y_pred_raw", "point": "y_pred"})
    path = tmp_path / "holdout_predictions.parquet"
    path.write_bytes(b"reference fixture")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    module = types.ModuleType("ml_template.scoring.release_artifacts")
    module.sha256 = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    monkeypatch.setitem(sys.modules, "ml_template.scoring.release_artifacts", module)
    post = types.ModuleType("ml_template.evaluation.regression_postprocessing")
    post.reference_columns = lambda material: ["y_pred_raw", "lower", "upper"]
    monkeypatch.setitem(sys.modules, "ml_template.evaluation.regression_postprocessing", post)
    reads = []
    predicted = []

    def read(candidate, columns):
        assert candidate == path and "y_true" not in columns and ("y_pred_raw" in columns)
        reads.append(columns)
        return ref[columns].copy()

    monkeypatch.setattr(pd, "read_parquet", read)

    class Predictor:
        def predict(self, batch, **kwargs):
            predicted.append(batch.id.tolist())
            if kind == "tamper_during":
                path.write_bytes(b"changed during parity")
            return whole.set_index("id").loc[batch.id].reset_index()

    source = pd.DataFrame({"id": ref.id, "x": 0.0}).set_index("id")
    material = {
        "prepared": SimpleNamespace(key_col="id", hold_keys=ref.id.tolist()),
        "evaluation_dir": tmp_path,
        "reference_sha256": expected,
        "source": source,
    }
    if kind == "tamper_before":
        path.write_bytes(b"changed before parity")
    if kind == "valid":
        result = check_parity(Predictor(), material, 1e-06)
        assert (
            result["max_abs_diff"] == 0
            and result["labels_loaded"] is False
            and (result["metrics_recomputed"] is False)
        )
        assert len(predicted) == 3
    else:
        with pytest.raises(ValueError):
            check_parity(Predictor(), material, 1e-06)
        if kind == "tamper_before":
            assert not reads and (not predicted)
