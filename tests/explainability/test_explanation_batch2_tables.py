"""Real result/summary business logic, with mocked Parquet I/O and synthetic arrays."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import logit

from ml_template.explainability.explain_batch import read_worker_result, write_outputs
from ml_template.explainability.explanation_inputs import load_saved_batch


@pytest.fixture
def batch2_worker_outputs(batch2_layout, tmp_path, monkeypatch):
    obj = batch2_layout
    saved = load_saved_batch(obj.config)
    directory = tmp_path / "ipc"
    directory.mkdir()
    features = obj.frame.set_index("entity")[["x"]]
    probabilities = saved.predictions.p_raw.to_numpy()
    margins = logit(probabilities)
    np.savez_compressed(
        directory / "contributions.npz",
        contributions=(margins + 1)[:, None],
        base_values=np.full(3, -1.0),
        margins=margins,
        p_raw=probabilities,
    )
    reproduced = saved.predictions.drop(columns=["scoring_batch_id"])
    original = pd.read_parquet

    def read(path, *args, **kwargs):
        if str(path) == str(directory / "features.parquet"):
            return features.copy()
        if str(path) == str(directory / "reproduced_predictions.parquet"):
            return reproduced.copy()
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", read)
    response = {
        "n_rows": 3,
        "n_features": 1,
        "feature_names": ["x"],
        "method": "xgboost_native_tree_shap",
        "feature_dependence": "tree_path_dependent",
        "output_scale": "base_model_log_odds",
    }
    return obj, saved, directory, features, reproduced, response


def test_worker_arrays_validate_against_saved_predictions(batch2_worker_outputs):
    obj, saved, directory, _features, _reproduced, response = batch2_worker_outputs
    result, _ = read_worker_result(obj.config, saved, response, directory)
    assert result.verification["passed"]
    assert result.features.index.tolist() == [30, 10, 20]


def test_bad_array_additivity_rejected(batch2_worker_outputs):
    obj, saved, directory, _features, _reproduced, response = batch2_worker_outputs
    margins = logit(saved.predictions.p_raw.to_numpy())
    np.savez_compressed(
        directory / "contributions.npz",
        contributions=np.ones((3, 1)),
        base_values=np.full(3, -1.0),
        margins=margins,
        p_raw=saved.predictions.p_raw,
    )
    with pytest.raises(ValueError, match="reconstruct"):
        read_worker_result(obj.config, saved, response, directory)


def test_wrong_calibrated_scale_claim_rejected(batch2_worker_outputs):
    obj, saved, directory, _features, _reproduced, response = batch2_worker_outputs
    response["output_scale"] = "calibrated_probability"
    with pytest.raises(ValueError, match="semantics"):
        read_worker_result(obj.config, saved, response, directory)


def test_actual_artifact_writer_table_alignment(batch2_worker_outputs, tmp_path, monkeypatch):
    obj, saved, directory, _features, _reproduced, response = batch2_worker_outputs
    result, reproduced = read_worker_result(obj.config, saved, response, directory)
    output = tmp_path / "output"
    output.mkdir()
    tables = {}

    def capture(frame, path, **kwargs):
        tables[Path(path).name] = frame.copy()
        Path(path).write_bytes(b"mock Parquet transport")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", capture)
    summary = write_outputs(
        output, obj.config, saved, response, result, reproduced, "explain_fixture", 1
    )
    assert summary["persisted_scoring_batch_verified"]
    assert tables["rows.parquet"].index.tolist() == [30, 10, 20]
    assert tables["rows.parquet"].p_raw_saved.tolist() == [0.1, 0.2, 0.3]
    assert tables["contributions.parquet"].columns.tolist() == ["x"]
    assert (output / "report.html").is_file()
