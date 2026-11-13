"""Real synthetic CPU NN evaluate -> sealed release -> registry-free offline predict."""

import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_path = Path(__file__).parents[1] / "models" / "test_nn_lifecycle_01_native.py"
_spec = importlib.util.spec_from_file_location("nn_package_04_training_fixture", _path)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
trained = _module.trained


@pytest.fixture(autouse=True)
def _fixed_synthetic_inference_microbatch(monkeypatch):
    import copy

    from ml_template.training.run_config import RunConfig

    original = RunConfig.model_validate

    def fixed(cls, raw, **kwargs):
        if isinstance(raw, dict) and raw.get("experiment_name") == "nn_package_01":
            raw = copy.deepcopy(raw)
            raw["model"]["params"]["prediction_batch_size"] = 1
        return original(raw, **kwargs)

    monkeypatch.setattr(RunConfig, "model_validate", classmethod(fixed))


@pytest.mark.parametrize("kind", ["final", "cv_mean"])
@pytest.mark.parametrize("variant", ["raw", "intervals", "bias", "affine"])
def test_native_nn_release_offline(trained, tmp_path, monkeypatch, kind, variant):
    from ml_template.evaluation.evaluate_run import evaluate_spec
    from ml_template.evaluation.regression_config import RegressionEvaluationConfig
    from ml_template.models.nn_regressor import NNRegressionAdapter
    from ml_template.preprocessing.nn_preprocessor import NNPreprocessor
    from ml_template.scoring.predictor import ReleasedPredictor
    from ml_template.scoring.regression_release_config import RegressionReleaseConfig
    from ml_template.scoring.release_builder import build_release
    from ml_template.training.regression_calibration import CalibrationConfig, calibrate

    correction = variant if variant in ["bias", "affine"] else "none"
    stage = {"value": "calibrate"}
    old_read = pd.read_parquet
    package_reads = []

    def spy(path, *args, **kwargs):
        path = Path(path)
        if stage["value"] == "package":
            if path == trained["parquet"]:
                columns = kwargs.get("columns")
                assert columns is not None and "y" not in columns and ("y_true" not in columns)
                package_reads.append(tuple(columns))
            if path.name == "holdout_predictions.parquet":
                assert kwargs.get("columns") is not None and "y_true" not in kwargs["columns"]
        elif stage["value"] == "offline":
            if path == trained["parquet"] or path.name == "holdout_predictions.parquet":
                pytest.fail("offline inference opened historical data/evaluation rows")
        return old_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", spy)
    monkeypatch.setattr(NNPreprocessor, "fit", lambda *a, **k: pytest.fail("release fitted Gold"))
    monkeypatch.setattr(
        NNRegressionAdapter, "_train", lambda *a, **k: pytest.fail("release trained NN")
    )
    calibration_id = None
    if variant != "raw":
        c = calibrate(
            trained["engine"],
            CalibrationConfig(
                model_type="nn",
                parent_run_id=trained["rid"],
                predictor_kind=kind,
                correction=correction,
                coverages=[0.5, 0.8],
                target_unit="synthetic_units",
                min_calibration_rows=5,
                min_recalibration_rows=3,
                intended_use="Synthetic only",
            ),
            output_root=tmp_path / "calibrations",
            check_only=False,
            **trained["roots"],
        )
        calibration_id = c["calibration_id"]
    stage["value"] = "evaluate"
    eval_spec = RegressionEvaluationConfig(
        parent_run_id=trained["rid"],
        predictor={"kind": kind},
        calibration_id=calibration_id,
        target_unit="synthetic_units",
        release_variant="recalibrated" if correction != "none" else "raw",
        uncertainty={"enabled": variant != "raw", "coverages": [0.8] if variant != "raw" else []},
    )
    evaluated = evaluate_spec(
        trained["engine"], eval_spec, output_root=tmp_path / "evaluations", **trained["roots"]
    )
    request = RegressionReleaseConfig(
        name="synthetic_nn",
        parent_run_id=trained["rid"],
        evaluation_id=evaluated["evaluation_id"],
        calibration_id=calibration_id,
        intended_use="Synthetic research only",
        acknowledge_research_release=True,
        parity_atol=1e-06,
    )
    stage["value"] = "package"
    old_load = trained["torch"].load
    monkeypatch.setattr(
        trained["torch"], "load", lambda *a, **k: pytest.fail("release check loaded checkpoint")
    )
    output = tmp_path / "releases"
    check = build_release(trained["engine"], request, check_only=True, output_root=output)
    assert (
        check["ready"]
        and (not check["models_loaded"])
        and check["source_features_loaded"]
        and (not check["labels_loaded"])
    )
    assert not check["parity_executed"] and (not output.exists())
    monkeypatch.setattr(trained["torch"], "load", old_load)
    released = build_release(trained["engine"], request, output_root=output)
    assert released["status"] == "succeeded" and (not released["holdout_labels_loaded"])
    assert (
        released["parity"]["labels_loaded"] is False
        and released["parity"]["metrics_recomputed"] is False
    )
    assert released["parity"]["max_abs_diff"] <= 1e-06 and package_reads
    directory = Path(released["artifact_dir"])
    moved = []
    for source in [
        trained["roots"]["silver_root"],
        trained["roots"]["feature_root"],
        trained["roots"]["split_root"],
        trained["root"],
        Path(trained["cfg"].data.dataset_config),
    ]:
        target = source.with_name(source.name + "_offline")
        shutil.move(str(source), str(target))
        moved.append((source, target))
    stage["value"] = "offline"
    try:
        from ml_template.db import evaluation_registry, model_run_registry

        with monkeypatch.context() as isolated:
            isolated.setattr(
                evaluation_registry,
                "connection",
                lambda *a, **k: pytest.fail("offline accessed registry"),
            )
            isolated.setattr(
                model_run_registry,
                "get_run",
                lambda *a, **k: pytest.fail("offline looked up training run"),
            )
            predictor = ReleasedPredictor(directory, expected_release_id=released["release_id"])
            new = pd.DataFrame(
                {
                    "id": [9001, 9002, 9003, 9004],
                    "a": [1.1, 2.2, np.nan, 3.3],
                    "b": [2.0, np.nan, 3.0, 1.0],
                    "k": ["u", "v", None, "u"],
                }
            )
            whole = predictor.predict(new)
            shuffled = predictor.predict(new.iloc[::-1], chunk_size=1).set_index("id").sort_index()
            baseline = whole.set_index("id").sort_index()
            columns = [c for c in baseline if c not in ["release_id"]]
            np.testing.assert_allclose(baseline[columns], shuffled[columns], rtol=0, atol=1e-06)
            assert np.isfinite(baseline[columns].to_numpy()).all()
            assert (
                predictor.info["model_type"] == "nn"
                and predictor.info["target_unit"] == "synthetic_units"
            )
            with pytest.raises(ValueError):
                predictor.predict(new.assign(y=0.0))
            unknown = new.copy()
            unknown.loc[0, "k"] = "unseen"
            with pytest.raises(ValueError):
                predictor.predict(unknown)
    finally:
        for source, target in reversed(moved):
            shutil.move(str(target), str(source))
    (
        directory / "model.ubj" if kind == "final" else directory / "fold_models/fold_0.ubj"
    ).write_bytes(b"tampered")
    monkeypatch.setattr(
        trained["torch"], "load", lambda *a, **k: pytest.fail("tampered release deserialized")
    )
    with pytest.raises(ValueError):
        ReleasedPredictor(directory, expected_release_id=released["release_id"])
