import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ml_template.scoring.input_contract import frame_schema
from ml_template.scoring.predictor import ReleasedPredictor
from ml_template.scoring.release_artifacts import read_json, runtime_info, seal, write_json


@pytest.fixture
def package(tmp_path, monkeypatch):
    class ToyModel:
        def predict_proba(self, frame):
            return (frame.value.to_numpy(dtype=float) + frame.kind.cat.codes.to_numpy()) / 10

    factory = ModuleType("ml_template.models.model_factory")
    factory.load_adapter = lambda cfg, path: ToyModel()
    config = ModuleType("ml_template.training.run_config")
    config.RunConfig = SimpleNamespace(
        model_validate=lambda data: SimpleNamespace(fitted_features=data.get("fitted_features"))
    )
    monkeypatch.setitem(sys.modules, factory.__name__, factory)
    monkeypatch.setitem(sys.modules, config.__name__, config)
    frame = pd.DataFrame(
        {"kind": pd.Categorical(["L", "M", "H"], categories=["L", "M", "H"]), "value": [2, 2, 2]}
    )
    schema = frame_schema(frame)
    write_json(
        tmp_path / "contract.json",
        {
            "schema_version": 1,
            "key_col": "id",
            "key_kind": "int",
            "source_schema": schema,
            "model_schema": schema,
        },
    )
    write_json(
        tmp_path / "feature_recipe.json",
        {"raw_columns": ["kind", "value"], "columns": ["kind", "value"], "blocks": []},
    )
    write_json(tmp_path / "config.json", {"fitted_features": None})
    write_json(tmp_path / "evidence.json", {})
    write_json(
        tmp_path / "release.json",
        {
            "schema_version": 1,
            "name": "test",
            "parent_run_id": "run",
            "calibration_id": None,
            "release_variant": "raw",
            "policy": {"kind": "none"},
        },
    )
    write_json(tmp_path / "runtime.json", runtime_info())
    (tmp_path / "model.ubj").write_bytes(b"toy-model")
    seal(tmp_path)
    frame.insert(0, "id", [30, 10, 20])
    return tmp_path, frame


def change_policy(folder, policy):
    info = read_json(folder / "release.json")
    info["policy"] = policy
    write_json(folder / "release.json", info)
    seal(folder)


def test_category_singleton_and_probability_only(package):
    path, frame = package
    predictor = ReleasedPredictor(path)
    out = predictor.predict(frame)
    assert np.allclose(out.p_raw, [0.2, 0.3, 0.4])
    assert predictor.predict(frame.iloc[[2]]).p_release.iloc[0] == out.p_release.iloc[2]
    assert "policy_flag" not in out


def test_shuffle_chunks_and_no_mutation(package):
    path, frame = package
    before = frame.copy(deep=True)
    predictor = ReleasedPredictor(path)
    first = predictor.predict(frame).set_index("id").sort_index()
    second = predictor.predict(frame.iloc[::-1], chunk_size=1).set_index("id").sort_index()
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(frame, before)


def test_threshold_equality(package):
    path, frame = package
    change_policy(path, {"kind": "threshold", "value": 0.3})
    assert ReleasedPredictor(path).predict(frame).policy_flag.tolist() == [False, True, True]


def test_budget_ties_whole_batch(package):
    path, frame = package
    frame["kind"] = "L"
    change_policy(
        path, {"kind": "alert_rate", "value": 0.34, "batch_definition": "entire scoring batch"}
    )
    predictor = ReleasedPredictor(path)
    out = predictor.predict(frame, chunk_size=1)
    assert out.policy_flag.tolist() == [False, True, True]
    other = predictor.predict(frame.iloc[::-1], chunk_size=2)
    pd.testing.assert_frame_equal(
        out.set_index("id").sort_index(), other.set_index("id").sort_index()
    )


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_chunk_size(package, size):
    path, frame = package
    with pytest.raises(ValueError, match="chunk_size"):
        ReleasedPredictor(path).predict(frame, chunk_size=size)


def test_integrity_precedes_loading(package):
    path, _ = package
    (path / "model.ubj").write_bytes(b"modified")
    with pytest.raises(ValueError, match="artifact"):
        ReleasedPredictor(path)


def test_runtime_mismatch(package):
    path, _ = package
    expected = read_json(path / "runtime.json")
    expected["python"] = "0.0.0"
    write_json(path / "runtime.json", expected)
    seal(path)
    with pytest.raises(ValueError, match="runtime"):
        ReleasedPredictor(path)


def test_no_calibration_fallback(package):
    path, _ = package
    info = read_json(path / "release.json")
    info["release_variant"] = "calibrated"
    write_json(path / "release.json", info)
    seal(path)
    with pytest.raises(ValueError, match="no calibrator"):
        ReleasedPredictor(path)


def test_transformer_required(package):
    path, _ = package
    write_json(path / "config.json", {"fitted_features": [{"step": "test"}]})
    seal(path)
    with pytest.raises(ValueError, match="transformer"):
        ReleasedPredictor(path)


def test_suppress_policy(package):
    path, frame = package
    change_policy(path, {"kind": "threshold", "value": 0.3})
    assert "policy_flag" not in ReleasedPredictor(path).predict(frame, include_policy=False)


def test_calibration_before_threshold(package):
    from ml_template.models.probability_calibrator import ProbabilityCalibrator

    path, frame = package
    calibrator = ProbabilityCalibrator(slope=1.0, intercept=-1.0)
    calibrator.save(path / "calibrator.json")
    info = read_json(path / "release.json")
    info.update(
        calibration_id="cal",
        release_variant="calibrated",
        policy={"kind": "threshold", "value": 0.2},
    )
    write_json(path / "release.json", info)
    seal(path)
    out = ReleasedPredictor(path).predict(frame)
    assert np.allclose(out.p_raw, [0.2, 0.3, 0.4])
    assert np.allclose(out.p_release, calibrator.transform(out.p_raw))
    assert np.array_equal(out.policy_flag, out.p_calibrated >= 0.2)
    assert not np.array_equal(out.policy_flag, out.p_raw >= 0.2)
