import json
from types import SimpleNamespace

import pandas as pd
import pytest

from ml_template.usage_dashboard import app as module


def view():
    index = pd.Index([30, 10], name="entity")
    features = pd.DataFrame({"x": [1.0, 2.0]}, index=index)
    contributions = features * 0.2
    rows = pd.DataFrame(
        {
            "base_value": -1.0,
            "margin": [-0.8, -0.6],
            "p_raw": [0.31, 0.35],
            "p_release": [0.1, 0.2],
            "policy_flag": [False, True],
        },
        index=index,
    )
    importance = pd.DataFrame({"feature": ["x"], "mean_abs_contribution": [0.3]})
    return SimpleNamespace(
        summary={"explanation_id": "fixture"},
        rows=rows,
        features=features,
        contributions=contributions,
        importance=importance,
    )


def test_callback_payloads_are_read_only_data_consumers(monkeypatch):
    data = view()
    monkeypatch.setattr(module, "load_explanation", lambda *a, **k: data)
    result = module.global_payload("unused.db", "fixture")
    assert json.loads(result[0])["explanation_id"] == "fixture"
    assert result[3] == [{"label": "30", "value": 0}, {"label": "10", "value": 1}]
    local = module.local_payload("unused.db", "fixture", 1)
    assert "Entity 10" in local[2]
    assert local[1][0]["feature"] == "x"


def test_separate_dashboard_can_construct(tmp_path, monkeypatch):
    pytest.importorskip("dash")
    monkeypatch.setattr(module, "list_explanations", lambda *a, **k: [])
    app = module.create_app(tmp_path / "unused.db")
    assert app.server.test_client().get("/").status_code == 200
    assert app.server.test_client().get("/_dash-layout").status_code == 200
    assert len(app.callback_map) == 5
    export = app.callback_map["mon-report-download.data"]
    assert export["inputs"] == [{"id": "mon-report-export", "property": "n_clicks"}]
    assert export["state"] == [{"id": "mon-selection", "property": "value"}]
    assert "Released model explanations" in str(app.layout())


def test_no_model_training_or_training_dashboard_imports():
    import ast
    from pathlib import Path

    tree = ast.parse(Path(module.__file__).read_text())
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module]
    assert not any(
        name.startswith(("ml_template.models", "ml_template.training", "ml_template.dashboard"))
        for name in imports
    )
