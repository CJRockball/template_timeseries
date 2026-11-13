"""Real CPU NN frozen evaluation; reuses accepted Package 01 synthetic fixture."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_path = Path(__file__).parents[1] / "models" / "test_nn_lifecycle_01_native.py"
_spec = importlib.util.spec_from_file_location("nn_package_03_training_fixture", _path)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
trained = _module.trained


def request_for(trained, **extra):
    from ml_template.evaluation.regression_config import RegressionEvaluationConfig

    return RegressionEvaluationConfig(
        **{
            "parent_run_id": trained["rid"],
            "target_unit": "synthetic_units",
            "intended_use": "Synthetic NN evaluation only",
            **extra,
        }
    )


def in_memory_holdout(p, trained):
    from ml_template.features.prepared_recipe import prepare_frame

    plan = p.provenance["feature_preparation"]
    f = trained["df"].set_index("id").loc[p.hold_keys]
    return (
        prepare_frame(f[plan["recipe"]["raw_columns"]], plan["recipe"], plan["model_schema"]),
        f.y.to_numpy(float),
    )


@pytest.mark.parametrize("kind", ["final", "cv_mean"])
@pytest.mark.parametrize(
    "variant", ["raw", "intervals", "bias_intervals", "affine_intervals", "bias_only"]
)
def test_native_frozen_nn(trained, tmp_path, monkeypatch, kind, variant):
    from ml_template.db import evaluation_registry as registry
    from ml_template.evaluation import nn_evaluate as nn
    from ml_template.evaluation.evaluate_run import evaluate_spec
    from ml_template.models.nn_regressor import NNRegressionAdapter
    from ml_template.models.predictor_selection import load_evaluation_predictor
    from ml_template.models.regression_postprocessing import apply
    from ml_template.preprocessing.nn_preprocessor import NNPreprocessor
    from ml_template.training.regression_calibration import CalibrationConfig, calibrate

    correction = "none" if variant in ["raw", "intervals"] else variant.split("_")[0]
    coverages = [] if variant in ["raw", "bias_only"] else [0.8]
    stage = {"value": "calibrate"}
    reads = []
    allowed = {
        role: set(trained["assignment"].loc[trained["assignment"].partition.eq(role), "id"])
        for role in ["dev", "holdout", "calibration", "recalibration"]
    }
    old_read = pd.read_parquet

    def spy(path, *args, **kwargs):
        if Path(path) == trained["parquet"]:
            keys = set(kwargs["filters"][0][2])
            if stage["value"] == "calibrate":
                assert keys in [allowed["calibration"], allowed["recalibration"]]
            else:
                assert stage["value"] == "evaluate" and keys == allowed["holdout"]
                assert not keys & (
                    allowed["dev"] | allowed["calibration"] | allowed["recalibration"]
                )
                record = nn.rows(
                    trained["engine"],
                    "final_evaluations",
                    "SELECT * FROM final_evaluations WHERE status='running'",
                )
                assert len(record) == 1 and record[0]["accessed_at"]
                assert nn.rows(
                    trained["engine"],
                    "final_evaluation_locks",
                    "SELECT * FROM final_evaluation_locks",
                )
                reads.append(keys)
        return old_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", spy)
    monkeypatch.setattr(
        NNPreprocessor, "fit", lambda *a, **k: pytest.fail("fitted Gold during evaluation")
    )
    monkeypatch.setattr(
        NNRegressionAdapter, "_train", lambda *a, **k: pytest.fail("trained NN during evaluation")
    )
    calibration_id = None
    if variant != "raw":
        fitted = calibrate(
            trained["engine"],
            CalibrationConfig(
                model_type="nn",
                parent_run_id=trained["rid"],
                predictor_kind=kind,
                correction=correction,
                coverages=coverages,
                target_unit="synthetic_units",
                min_calibration_rows=5,
                min_recalibration_rows=3,
                intended_use="Synthetic only",
            ),
            check_only=False,
            output_root=tmp_path / "calibrations",
            **trained["roots"],
        )
        calibration_id = fitted["calibration_id"]
    spec = request_for(
        trained,
        predictor={"kind": kind},
        calibration_id=calibration_id,
        release_variant="raw" if correction == "none" else "recalibrated",
        uncertainty={"enabled": bool(coverages), "coverages": coverages},
        criteria=[{"metric": "rmse", "operator": "<=", "value": 1e100}],
    )
    stage["value"] = "check"
    old_load = trained["torch"].load
    monkeypatch.setattr(
        trained["torch"], "load", lambda *a, **k: pytest.fail("check deserialized a checkpoint")
    )
    output = tmp_path / "nn_evaluations"
    before = trained["sha256"](tmp_path / "project.db")
    checked = evaluate_spec(
        trained["engine"], spec, check_only=True, output_root=output, **trained["roots"]
    )
    assert (
        checked["ready"]
        and (not checked["models_loaded"])
        and (not checked["holdout_labels_loaded"])
    )
    assert not checked["files_written"] and (not output.exists()) and (not reads)
    assert before == trained["sha256"](tmp_path / "project.db")
    monkeypatch.setattr(trained["torch"], "load", old_load)
    stage["value"] = "evaluate"
    result = evaluate_spec(trained["engine"], spec, output_root=output, **trained["roots"])
    assert result["holdout_labels_loaded"] and (not result["release_enabled"]) and (len(reads) == 1)
    assert (
        result["summary"]["criteria_status"] == "passed"
        and result["summary"]["automatic_promotion"] is False
    )
    p = nn.prepare_nn_evaluation(trained["engine"], spec, **trained["roots"])
    X, y = in_memory_holdout(p, trained)
    raw = load_evaluation_predictor(p).predict(X)
    point, bounds = (raw, {})
    if calibration_id:
        point, bounds = apply(raw, p.provenance["postprocessing"]["state"]["parameters"], coverages)
    assert result["summary"]["primary_value"] == pytest.approx(
        float(np.sqrt(np.mean((y - point) ** 2)))
    )
    frame = old_read(Path(result["artifact_dir"]) / "holdout_predictions.parquet")
    assert set(frame.id) == allowed["holdout"]
    np.testing.assert_allclose(frame.y_pred_raw, raw, rtol=0, atol=1e-06)
    np.testing.assert_allclose(frame.y_pred, point, rtol=0, atol=1e-06)
    for name, values in bounds.items():
        np.testing.assert_allclose(frame[name], values, rtol=0, atol=1e-06)
    stage["value"] = "saved"
    monkeypatch.setattr(
        trained["torch"], "load", lambda *a, **k: pytest.fail("cache/UI loaded a model")
    )
    cached = evaluate_spec(trained["engine"], spec, output_root=output, **trained["roots"])
    assert (
        cached["cached"]
        and cached["evaluation_id"] == result["evaluation_id"]
        and (not cached["holdout_labels_loaded"])
        and (len(reads) == 1)
    )
    from ml_template.dashboard.regression_multi_data import load_regression_evaluation

    metadata = load_regression_evaluation(
        trained["engine"], result["evaluation_id"], predictions=False
    )
    assert metadata["state"] == "verified", metadata
    assert metadata["predictions"] is None
    saved = load_regression_evaluation(trained["engine"], result["evaluation_id"], predictions=True)
    assert saved["state"] == "verified", saved
    assert len(saved["predictions"]) == len(allowed["holdout"]) and len(reads) == 1
    assert not registry.query(trained["engine"], "SELECT * FROM final_evaluation_locks")
    revised = spec.model_copy(update={"intended_use": "Revised synthetic protocol"})
    assert not evaluate_spec(trained["engine"], revised, check_only=True, **trained["roots"])[
        "ready"
    ]
    with pytest.raises(ValueError, match="previously exposed"):
        evaluate_spec(trained["engine"], revised, output_root=output, **trained["roots"])
    from ml_template.evaluation.regression_inputs import prepare_regression

    with pytest.raises(ValueError, match="Expected XGB"):
        prepare_regression(trained["engine"], spec, **trained["roots"])
    (Path(result["artifact_dir"]) / "metrics.csv").write_text("tampered")
    with pytest.raises(ValueError):
        evaluate_spec(trained["engine"], spec, **trained["roots"])


def test_native_failed_load_keeps_access(trained, tmp_path):
    from ml_template.db import evaluation_registry as registry
    from ml_template.evaluation.evaluate_run import evaluate_spec

    def bad(*args, **kwargs):
        raise RuntimeError("synthetic NN load failure")

    with pytest.raises(RuntimeError, match="synthetic NN load failure"):
        evaluate_spec(
            trained["engine"],
            request_for(trained),
            output_root=tmp_path / "failed",
            adapter_loader=bad,
            **trained["roots"],
        )
    rows = registry.query(
        trained["engine"], "SELECT status,accessed_at,active_identity FROM final_evaluations"
    )
    assert (
        len(rows) == 1
        and rows[0]["status"] == "failed"
        and rows[0]["accessed_at"]
        and (rows[0]["active_identity"] is None)
    )
    assert not registry.query(trained["engine"], "SELECT * FROM final_evaluation_locks")


def test_native_bad_request_no_targets(trained, monkeypatch):
    from ml_template.evaluation.evaluate_run import evaluate_spec

    old = pd.read_parquet

    def spy(path, *args, **kwargs):
        if Path(path) == trained["parquet"]:
            pytest.fail("invalid request read source values")
        return old(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", spy)
    with pytest.raises(ValueError, match="target units"):
        evaluate_spec(
            trained["engine"],
            request_for(trained, target_unit=None),
            check_only=True,
            **trained["roots"],
        )
    with pytest.raises(ValueError, match="inside source/run"):
        evaluate_spec(
            trained["engine"], request_for(trained), output_root=trained["root"], **trained["roots"]
        )
