"""Native release-first subprocess scoring/export; no new operational evaluation."""

import copy
import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_path = Path(__file__).parents[1] / "models" / "test_nn_lifecycle_01_native.py"
_spec = importlib.util.spec_from_file_location("nn_package_05_training_fixture", _path)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
trained = _module.trained


@pytest.fixture(autouse=True)
def _microbatch_before_synthetic_training(monkeypatch):
    from ml_template.training.run_config import RunConfig

    original = RunConfig.model_validate

    def fixed(cls, raw, **kwargs):
        if isinstance(raw, dict) and raw.get("experiment_name") == "nn_package_01":
            raw = copy.deepcopy(raw)
            raw["model"]["params"]["prediction_batch_size"] = 1
        return original(raw, **kwargs)

    monkeypatch.setattr(RunConfig, "model_validate", classmethod(fixed))


def build(trained, tmp_path, kind="final", corrected=False):
    import ml_template
    from ml_template.evaluation.evaluate_run import evaluate_spec
    from ml_template.evaluation.regression_config import RegressionEvaluationConfig
    from ml_template.scoring.regression_release_config import RegressionReleaseConfig
    from ml_template.scoring.release_builder import build_release
    from ml_template.training.regression_calibration import CalibrationConfig, calibrate

    calibration = None
    if corrected:
        calibration = calibrate(
            trained["engine"],
            CalibrationConfig(
                model_type="nn",
                parent_run_id=trained["rid"],
                predictor_kind=kind,
                correction="bias",
                coverages=[0.5, 0.8],
                target_unit="synthetic_units",
                min_calibration_rows=5,
                min_recalibration_rows=3,
                intended_use="Synthetic only",
            ),
            check_only=False,
            output_root=tmp_path / "calibrations",
            **trained["roots"],
        )["calibration_id"]
    spec = RegressionEvaluationConfig(
        parent_run_id=trained["rid"],
        predictor={"kind": kind},
        calibration_id=calibration,
        target_unit="synthetic_units",
        release_variant="recalibrated" if corrected else "raw",
        uncertainty={"enabled": corrected, "coverages": [0.8] if corrected else []},
    )
    evaluation = evaluate_spec(
        trained["engine"], spec, output_root=tmp_path / "evaluations", **trained["roots"]
    )
    release = build_release(
        trained["engine"],
        RegressionReleaseConfig(
            name="nn_package05_synthetic",
            parent_run_id=trained["rid"],
            evaluation_id=evaluation["evaluation_id"],
            calibration_id=calibration,
            intended_use="Synthetic research",
            acknowledge_research_release=True,
            parity_atol=1e-06,
        ),
        output_root=tmp_path / "releases",
    )
    root = tmp_path / "pinned/src/ml_template"
    root.parent.mkdir(parents=True)
    shutil.copytree(
        Path(ml_template.__file__).resolve().parent,
        root,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    return (release, root)


@pytest.mark.parametrize("kind", ["final", "cv_mean"])
@pytest.mark.parametrize("input_format", ["raw", "canonical"])
@pytest.mark.parametrize("corrected", [False, True], ids=["raw_point", "bias_intervals"])
def test_native_batch_pipeline(trained, tmp_path, kind, input_format, corrected):
    from ml_template.db import evaluation_registry as registry
    from ml_template.scoring.nn_batch_pipeline import BatchRequest, ExportRequest, run_pipeline

    release, root = build(trained, tmp_path, kind, corrected)
    frame = pd.DataFrame(
        {
            "id": [9001, 9002, 9003, 9004],
            "a": [1.1, 2.2, np.nan, 3.3],
            "b": [2.0, np.nan, 3.0, 1.0],
            "k": ["u", "v", None, "u"],
        }
    )
    source = tmp_path / "unlabelled.csv"
    frame.to_csv(source, index=False)
    reference = tmp_path / "reference.csv"
    pd.DataFrame({"id": frame.id.iloc[::-1]}).to_csv(reference, index=False)
    columns = {"y_pred_raw": "raw", "y_pred_release": "prediction"}
    if corrected:
        columns.update({"pi_lower_c800000": "lower80", "pi_upper_c800000": "upper80"})
    request = BatchRequest(
        release_dir=release["artifact_dir"],
        expected_release_id=release["release_id"],
        input_path=str(source),
        input_format=input_format,
        runtime_source_root=str(root),
        runtime_python=sys.executable,
        database=str(tmp_path / "project.db"),
        output_root=str(tmp_path / "jobs"),
        chunk_size=1,
        acknowledge_research_scoring=True,
        export=ExportRequest(reference=str(reference), columns=columns),
        timeout_s=180,
    )
    before = registry.query(trained["engine"], "SELECT evaluation_id FROM final_evaluations")
    checksum = trained["sha256"](tmp_path / "project.db")
    checked = run_pipeline(request, check_only=True)
    assert (
        checked["ready"]
        and (not checked["predictions_computed"])
        and (not checked["files_written"])
    )
    assert not (tmp_path / "jobs").exists() and checksum == trained["sha256"](
        tmp_path / "project.db"
    )
    result = run_pipeline(request, check_only=False)
    assert (
        result["status"] == "succeeded"
        and result["predictions_computed"]
        and (result["labels_loaded"] is False)
    )
    assert (
        result["target_unit"] == "synthetic_units"
        and result["evaluation_executed"] is False
        and (result["release_built"] is False)
    )
    assert before == registry.query(
        trained["engine"], "SELECT evaluation_id FROM final_evaluations"
    )
    prediction = pd.read_parquet(result["predictions_path"])
    exported = pd.read_csv(result["export_path"])
    assert exported.id.tolist() == frame.id.iloc[::-1].tolist()
    baseline = prediction.set_index("id").loc[exported.id]
    for original, name in columns.items():
        np.testing.assert_allclose(exported[name], baseline[original], rtol=1e-14, atol=1e-14)
    assert (Path(result["artifact_dir"]) / "pipeline_manifest.json").is_file()
    assert result["attempt_id"] and result["scoring_batch_id"]
    if input_format == "raw":
        assert (Path(result["artifact_dir"]) / "prepared/manifest.json").is_file()
    changed = request.model_copy(update={"expected_release_id": "rel_" + "f" * 24})
    with pytest.raises(ValueError):
        run_pipeline(changed, check_only=True)


@pytest.mark.parametrize("case", ["runtime_mismatch", "target_header"])
def test_native_preflight_failures_write_nothing(trained, tmp_path, case):
    from ml_template.scoring.nn_batch_pipeline import BatchRequest, run_pipeline

    release, root = build(trained, tmp_path)
    frame = pd.DataFrame({"id": [9001], "a": [1.0], "b": [2.0], "k": ["u"]})
    if case == "target_header":
        frame["y"] = 999.0
    else:
        (root / "__init__.py").write_text(
            (root / "__init__.py").read_text() + "\n# mismatched runtime\n"
        )
    source = tmp_path / "input.csv"
    frame.to_csv(source, index=False)
    request = BatchRequest(
        release_dir=release["artifact_dir"],
        expected_release_id=release["release_id"],
        input_path=str(source),
        input_format="canonical",
        runtime_source_root=str(root),
        runtime_python=sys.executable,
        database=str(tmp_path / "project.db"),
        output_root=str(tmp_path / "no_output"),
        timeout_s=180,
    )
    with pytest.raises(RuntimeError):
        run_pipeline(request, check_only=True)
    assert not (tmp_path / "no_output").exists()
