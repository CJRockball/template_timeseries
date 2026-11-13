import sqlite3

import pytest

from ml_template.dashboard.evaluation_data import list_evaluations, load_evaluation
from ml_template.dashboard.evaluation_figs import evaluation_figures
from ml_template.db import evaluation_registry as reg
from ml_template.evaluation.evaluation_artifacts import manifest, write_json
from ml_template.evaluation.evaluation_config import EvaluationConfig


@pytest.fixture
def saved(tmp_path):
    c = sqlite3.connect(":memory:")
    cfg = EvaluationConfig(parent_run_id="p", expected_silver_dataset_version_id="s")
    reg.start_evaluation(
        c,
        evaluation_id="e",
        identity_hash="hash",
        spec=cfg,
        fingerprint="f",
        artifact_dir=str(tmp_path),
        prior=[],
    )
    data = {
        "summary": {"evaluation_id": "e", "identity_hash": "hash"},
        "spec": cfg.model_dump(mode="json"),
        "provenance": {},
        "warnings": [],
        "metrics": [],
        "differences": [],
        "policy_results": [],
        "reliability": [],
        "subgroups": [],
        "errors": [],
        "curves": [],
        "acceptance": {"criteria": []},
        "development_context": [],
    }
    for name, value in [
        ("report_data.json", data),
        ("summary.json", data["summary"]),
        ("spec.json", data["spec"]),
        ("provenance.json", {}),
    ]:
        write_json(tmp_path / name, value)
    manifest(tmp_path)
    reg.finish_evaluation(c, "e")
    yield c, tmp_path, data
    c.close()


def test_missing_registry_is_read_only():
    c = sqlite3.connect(":memory:")
    assert list_evaluations(c) == []
    assert c.execute("SELECT name FROM sqlite_master").fetchall() == []
    c.close()


def test_loading_report_is_read_only(saved):
    c, _, _ = saved
    changes = c.total_changes
    assert load_evaluation(c, "e")["state"] == "verified"
    assert c.total_changes == changes
    assert list_evaluations(c)[0]["evaluation_id"] == "e"
    assert load_evaluation(c, None)["state"] == "empty"
    assert load_evaluation(c, "missing")["state"] == "missing"


def test_tampered_report_blocked(saved):
    c, directory, _ = saved
    (directory / "report_data.json").write_text("{}")
    assert load_evaluation(c, "e")["state"] == "invalid"


def test_wrong_identity_blocked_even_with_updated_manifest(saved):
    c, directory, data = saved
    data["summary"]["evaluation_id"] = "other"
    write_json(directory / "report_data.json", data)
    manifest(directory)
    assert load_evaluation(c, "e")["state"] == "invalid"


def test_figures_serialize(saved):
    _, _, data = saved
    figures = evaluation_figures(data)
    assert len(figures) == 5
    assert all(f.to_json() for f in figures)


def test_reader_does_not_touch_prediction_artifact(saved, monkeypatch):
    from ml_template.dashboard import evaluation_data

    c, directory, _ = saved
    (directory / "holdout_predictions.parquet").write_text("not accessed by this view")
    manifest(directory)
    original = evaluation_data.sha256

    def guarded(path):
        assert path.name != "holdout_predictions.parquet"
        return original(path)

    monkeypatch.setattr(evaluation_data, "sha256", guarded)
    assert load_evaluation(c, "e")["state"] == "verified"


def test_dash_page_when_installed(saved):
    pytest.importorskip("dash")
    from dash import Dash

    from ml_template.dashboard.pages import evaluation_page

    c, _, _ = saved
    assert evaluation_page.layout(c) is not None
    assert evaluation_page.render_selection(c, "e") is not None
    app = Dash(__name__)
    app.layout = evaluation_page.layout(c)
    evaluation_page.register_callbacks(app, c)
    assert len(app.callback_map) == 3
    required_outputs = {
        ("final-eval-id", "options"),
        ("final-eval-id", "value"),
        ("final-eval-content", "children"),
        ("reg2-eval-controls", "style"),
        ("reg2-eval-predictions", "children"),
    }
    registered_outputs = set()
    for metadata in app.callback_map.values():
        values = metadata["output"]
        values = values if isinstance(values, (list, tuple)) else [values]
        for output in values:
            registered_outputs.add((output.component_id, output.component_property))
    assert required_outputs <= registered_outputs
