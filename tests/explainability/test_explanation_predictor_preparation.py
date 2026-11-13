from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ml_template.scoring import input_contract as contract_module
from ml_template.scoring import predictor as module
from ml_template.scoring.pre_gold_schema import get_preprocessor_schema


def predictor(monkeypatch):
    obj = object.__new__(module.ReleasedPredictor)
    source = pd.DataFrame({"x": [1.0, 2.0]})
    obj.contract = {
        "schema_version": 1,
        "key_col": "entity",
        "key_kind": "int",
        "source_schema": contract_module.frame_schema(source),
        "model_schema": contract_module.frame_schema(source),
    }

    obj.preprocessor_input_schema = get_preprocessor_schema(obj.contract)
    obj.recipe = {"schema_version": 1, "raw_columns": ["x"], "columns": ["x"], "blocks": []}
    obj.calibrator = None
    obj.info = {"release_variant": "raw"}
    obj.release_id = "rel_test"
    obj.policy = SimpleNamespace(kind="none")
    recorded = []

    def score(X):
        recorded.append(X.copy())
        return np.full(len(X), 0.25)

    obj.model = SimpleNamespace(predict_proba=score)
    return obj, recorded


def test_preparation_matches_scoring_and_preserves_entity_key(monkeypatch):
    obj, recorded = predictor(monkeypatch)
    frame = pd.DataFrame({"entity": [20, 10], "x": [1.0, 2.0]})
    expected = obj.prepare_features(frame)
    obj.predict(frame)
    pd.testing.assert_frame_equal(expected, recorded[0])
    assert expected.index.tolist() == [20, 10]
    assert list(expected.columns) == ["x"]


def test_preparation_rejects_extra_columns(monkeypatch):
    obj, _ = predictor(monkeypatch)
    frame = pd.DataFrame({"entity": [20], "x": [1.0], "target": [1]})
    with pytest.raises(ValueError):
        obj.prepare_features(frame)


def test_preparation_rejects_duplicate_keys(monkeypatch):
    obj, _ = predictor(monkeypatch)
    frame = pd.DataFrame({"entity": [20, 20], "x": [1.0, 2.0]})
    with pytest.raises(ValueError):
        obj.prepare_features(frame)
