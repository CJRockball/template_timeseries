"""Real synthetic CPU NN calibration. Reuses the accepted Package 01 training fixture."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_fixture_path = Path(__file__).parents[1] / "models" / "test_nn_lifecycle_01_native.py"
_fixture_spec = importlib.util.spec_from_file_location(
    "nn_package_02_training_fixture", _fixture_path
)
_fixture_module = importlib.util.module_from_spec(_fixture_spec)
sys.modules[_fixture_spec.name] = _fixture_module
_fixture_spec.loader.exec_module(_fixture_module)
trained = _fixture_module.trained


def request_for(trained, **extra):
    from ml_template.training.regression_calibration import CalibrationConfig

    return CalibrationConfig(
        **{
            "model_type": "nn",
            "parent_run_id": trained["rid"],
            "predictor_kind": "final",
            "correction": "none",
            "coverages": [0.5, 0.8],
            "target_unit": "synthetic_units",
            "min_calibration_rows": 5,
            "min_recalibration_rows": 3,
            "intended_use": "Synthetic acceptance only",
            **extra,
        }
    )


def prepare(trained, kind):
    return _fixture_module.prepare(trained, kind)


def in_memory_role(prepared, trained, role):
    from ml_template.features.prepared_recipe import prepare_frame

    plan = prepared.provenance["feature_preparation"]
    keys = trained["assignment"].loc[trained["assignment"].partition.eq(role), "id"].tolist()
    frame = trained["df"].set_index("id").loc[keys]
    X = prepare_frame(frame[plan["recipe"]["raw_columns"]], plan["recipe"], plan["model_schema"])
    return X, frame.y.to_numpy(float)


@pytest.mark.parametrize("predictor_kind", ["final", "cv_mean"])
@pytest.mark.parametrize(
    "variant", ["intervals", "bias_intervals", "affine_intervals", "bias_only", "affine_only"]
)
def test_real_nn_reserved_postprocessing(trained, tmp_path, monkeypatch, predictor_kind, variant):
    from ml_template.models.nn_regressor import NNRegressionAdapter
    from ml_template.models.predictor_selection import load_evaluation_predictor
    from ml_template.models.regression_postprocessing import (
        apply,
        correct,
        fit_correction,
        fit_quantiles,
    )
    from ml_template.preprocessing.nn_preprocessor import NNPreprocessor
    from ml_template.training import regression_calibration as calibration

    method = "none" if variant == "intervals" else variant.split("_")[0]
    coverages = [] if variant.endswith("_only") else [0.5, 0.8]
    config = request_for(
        trained, predictor_kind=predictor_kind, correction=method, coverages=coverages
    )
    inventory = {
        role: set(trained["assignment"].loc[trained["assignment"].partition.eq(role), "id"])
        for role in ("dev", "holdout", "calibration", "recalibration")
    }
    phase = {"value": "check"}
    seen = []
    original_read = pd.read_parquet

    def spy(path, *args, **kwargs):
        if Path(path) == trained["parquet"]:
            assert phase["value"] == "fit", "preflight read source values"
            assert "y" in kwargs.get("columns", [])
            keys = set(kwargs["filters"][0][2])
            assert not keys & (inventory["dev"] | inventory["holdout"])
            role = next(
                (name for name in ("calibration", "recalibration") if keys == inventory[name]), None
            )
            assert role is not None, "read crossed reserved-role boundary"
            seen.append(role)
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", spy)
    monkeypatch.setattr(
        NNPreprocessor, "fit", lambda *a, **k: pytest.fail("calibration fitted Gold")
    )
    monkeypatch.setattr(
        NNRegressionAdapter, "_train", lambda *a, **k: pytest.fail("calibration trained NN")
    )
    old_load = trained["torch"].load
    monkeypatch.setattr(
        trained["torch"], "load", lambda *a, **k: pytest.fail("preflight loaded checkpoint")
    )
    output = tmp_path / "nn_calibrations"
    database = tmp_path / "project.db"
    before = trained["sha256"](database)
    checked = calibration.calibrate(
        trained["engine"], config, check_only=True, output_root=output, **trained["roots"]
    )
    assert checked["ready"] and not checked["models_loaded"] and not checked["predictions_computed"]
    assert not checked["calibration_labels_loaded"] and not checked["recalibration_labels_loaded"]
    assert not seen and not output.exists() and before == trained["sha256"](database)
    monkeypatch.setattr(trained["torch"], "load", old_load)
    phase["value"] = "fit"
    result = calibration.calibrate(
        trained["engine"], config, check_only=False, output_root=output, **trained["roots"]
    )
    expected_reads = (["recalibration"] if method != "none" else []) + (
        ["calibration"] if coverages else []
    )
    assert seen == expected_reads
    assert result["status"] == "succeeded" and not result["holdout_labels_loaded"]
    assert result["models_loaded"] and result["predictions_computed"]
    assert not result["evaluation_enabled"] and not result["release_enabled"]
    phase["value"] = "binding"
    prepared = prepare(trained, predictor_kind)
    artifact = calibration.load_calibration(
        trained["engine"],
        result["calibration_id"],
        trained["rid"],
        prepared.provenance,
        config.target_unit,
    )
    parameters = artifact["state"]["parameters"]
    assert artifact["state"]["config"]["model_type"] == "nn"
    assert parameters["n_calibration"] == (len(inventory["calibration"]) if coverages else 0)
    predictor = load_evaluation_predictor(prepared)
    expected_correction = {"method": "none", "slope": 1.0, "intercept": 0.0}
    if method != "none":
        X, y = in_memory_role(prepared, trained, "recalibration")
        expected_correction = fit_correction(predictor.predict(X), y, method)
    assert parameters["correction"]["slope"] == pytest.approx(expected_correction["slope"])
    assert parameters["correction"]["intercept"] == pytest.approx(expected_correction["intercept"])
    if coverages:
        X, y = in_memory_role(prepared, trained, "calibration")
        expected_quantiles = fit_quantiles(
            correct(predictor.predict(X), expected_correction), y, coverages
        )
        for tag, item in expected_quantiles.items():
            assert parameters["quantiles"][tag]["rank"] == item["rank"]
            assert parameters["quantiles"][tag]["half_width"] == pytest.approx(item["half_width"])
    else:
        assert parameters["quantiles"] == {}
    X, _ = in_memory_role(prepared, trained, "calibration")
    raw = predictor.predict(X)
    point, bounds = apply(raw, parameters, coverages)
    np.testing.assert_allclose(point, correct(raw, expected_correction), atol=1e-6, rtol=0)
    for tag in parameters["quantiles"]:
        np.testing.assert_allclose(
            (bounds["pi_lower_" + tag] + bounds["pi_upper_" + tag]) / 2, point, atol=1e-6, rtol=0
        )
    # A final calibration cannot be rebound to CV-mean, or vice versa.
    other = prepare(trained, "cv_mean" if predictor_kind == "final" else "final")
    with pytest.raises(ValueError, match="binding differs"):
        calibration.load_calibration(
            trained["engine"],
            result["calibration_id"],
            trained["rid"],
            other.provenance,
            config.target_unit,
        )
    with pytest.raises(ValueError, match="binding differs"):
        calibration.load_calibration(
            trained["engine"],
            result["calibration_id"],
            trained["rid"],
            prepared.provenance,
            "wrong_units",
        )
    from ml_template.evaluation.regression_config import RegressionEvaluationConfig
    from ml_template.evaluation.regression_inputs import prepare_regression

    with pytest.raises(ValueError, match="Expected XGB"):
        prepare_regression(
            trained["engine"],
            RegressionEvaluationConfig(parent_run_id=trained["rid"]),
            **trained["roots"],
        )
    # New calibration evidence is immutable after completion.
    state_path = Path(artifact["artifact_dir"]) / "calibration.json"
    state_path.write_text(state_path.read_text() + " ")
    with pytest.raises(ValueError):
        calibration.load_calibration(
            trained["engine"],
            result["calibration_id"],
            trained["rid"],
            prepared.provenance,
            config.target_unit,
        )


@pytest.mark.parametrize("case", ["minimum", "finite_rank"])
def test_support_failure_has_no_reads_or_outputs(trained, tmp_path, monkeypatch, case):
    from ml_template.training import regression_calibration as calibration

    original_read = pd.read_parquet

    def spy(path, *args, **kwargs):
        if Path(path) == trained["parquet"]:
            pytest.fail("support failure read source values")
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", spy)
    monkeypatch.setattr(
        trained["torch"], "load", lambda *a, **k: pytest.fail("support failure loaded model")
    )
    extra = {"min_calibration_rows": 100} if case == "minimum" else {"coverages": [0.95]}
    config = request_for(trained, **extra)
    output = tmp_path / "no_output"
    with pytest.raises(ValueError):
        calibration.calibrate(trained["engine"], config, output_root=output, **trained["roots"])
    assert not output.exists()


def test_prior_exposure_requires_acknowledgement(trained, tmp_path, monkeypatch):
    from ml_template.training import regression_calibration as calibration

    monkeypatch.setattr(
        calibration,
        "prior_exposures",
        lambda *a: [{"evaluation_id": "synthetic_prior", "accessed_at": "synthetic"}],
    )
    checked = calibration.calibrate(trained["engine"], request_for(trained), **trained["roots"])
    assert not checked["ready"]
    output = tmp_path / "no_output"
    with pytest.raises(ValueError, match="explicit research acknowledgement"):
        calibration.calibrate(
            trained["engine"],
            request_for(trained),
            check_only=False,
            output_root=output,
            **trained["roots"],
        )
    assert not output.exists()
    request = request_for(
        trained, acknowledge_prior_exposure=True, prior_exposure_note="Synthetic history only"
    )
    assert calibration.calibrate(trained["engine"], request, **trained["roots"])["ready"]


def test_failure_keeps_audit_and_releases_lock(trained, tmp_path, monkeypatch):
    from ml_template.db.evaluation_registry import query
    from ml_template.training import regression_calibration as calibration

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic quantile failure")

    monkeypatch.setattr(calibration, "fit_quantiles", fail)
    with pytest.raises(RuntimeError, match="synthetic quantile failure"):
        calibration.calibrate(
            trained["engine"],
            request_for(trained),
            check_only=False,
            output_root=tmp_path / "failed",
            **trained["roots"],
        )
    records = query(trained["engine"], "SELECT status,accessed_at FROM regression_calibrations")
    assert (
        len(records) == 1
        and records[0]["status"] == "failed"
        and records[0]["accessed_at"] is not None
    )
    assert not query(trained["engine"], "SELECT * FROM final_evaluation_locks")
