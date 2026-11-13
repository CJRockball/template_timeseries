from types import SimpleNamespace

import pandas as pd
import pytest

from ml_template.features import feature_recipe as module


def recipe():
    return {
        "raw_columns": ["v"],
        "columns": ["square"],
        "blocks": [
            {
                "block_name": "test",
                "code_sha256": "same",
                "params": {},
                "derived_from": ["v"],
                "columns": ["square"],
            }
        ],
    }


def block():
    return SimpleNamespace(
        name="test",
        code_sha256=lambda: "same",
        resolve_params=lambda p: p,
        required_columns=lambda p: ["v"],
        run=lambda f, p: pd.DataFrame({"square": f.v**2}),
    )


def test_new_rows_and_keys(monkeypatch):
    monkeypatch.setattr(module, "get_block", lambda name: block())
    frame = pd.DataFrame({"v": [2, 4]}, index=pd.Index([20, 10], name="id"))
    result = module.replay_recipe(frame, recipe())
    assert result.index.equals(frame.index)
    assert result.square.tolist() == [4, 16]


@pytest.mark.parametrize(
    "field,value", [("code_sha256", "changed"), ("derived_from", ["other"]), ("columns", ["other"])]
)
def test_recipe_mismatch(monkeypatch, field, value):
    monkeypatch.setattr(module, "get_block", lambda name: block())
    spec = recipe()
    spec["blocks"][0][field] = value
    with pytest.raises(ValueError):
        module.replay_recipe(pd.DataFrame({"v": [2]}), spec)


def test_raw_recipe():
    frame = pd.DataFrame({"v": [2]})
    pd.testing.assert_frame_equal(
        module.replay_recipe(frame, {"raw_columns": ["v"], "columns": ["v"], "blocks": []}), frame
    )
