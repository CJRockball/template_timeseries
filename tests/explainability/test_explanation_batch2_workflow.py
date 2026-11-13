"""Orchestration unit tests with fake worker/output writers; no actual model or Parquet."""

import sqlite3
from pathlib import Path

import pytest

from ml_template.explainability import explain_batch as workflow
from ml_template.explainability.artifacts import file_entries, verify_explanation, write_json


def install_fakes(monkeypatch):
    def fake_worker(config, binding, directory, *, check_only):
        return {"ready": True, "runtime_sha256": "fixture", "n_rows": 3, "n_features": 1}

    monkeypatch.setattr(workflow, "run_worker", fake_worker)
    monkeypatch.setattr(workflow, "read_worker_result", lambda *a: (None, None))

    def write_outputs(stage, config, saved, response, result, reproduced, identity, elapsed):
        summary = {"explanation_id": identity, "n_rows": 3, "n_features": 1}
        for name in (
            "config.json",
            "provenance.json",
            "feature_metadata.json",
            "verification.json",
        ):
            write_json(stage / name, {})
        write_json(stage / "summary.json", summary)
        for name in (
            "rows.parquet",
            "features.parquet",
            "contributions.parquet",
            "global_summary.parquet",
        ):
            (stage / name).write_bytes(b"mock output: not real Parquet")
        (stage / "report.md").write_text("fixture")
        (stage / "report.html").write_text("fixture")
        return summary

    monkeypatch.setattr(workflow, "write_outputs", write_outputs)


def test_preflight_writes_no_project_state(batch2_layout, monkeypatch):
    install_fakes(monkeypatch)
    obj = batch2_layout
    before = file_entries(obj.score), obj.database.read_bytes()
    response = workflow.preflight(obj.config)
    assert response["registry_written"] is False
    assert response["predictions_computed"] is False
    assert before == (file_entries(obj.score), obj.database.read_bytes())
    assert not Path(obj.config.output_root).exists()


def test_success_publishes_and_registers_new_immutable_id(batch2_layout, monkeypatch):
    install_fakes(monkeypatch)
    obj = batch2_layout
    before = file_entries(obj.release), file_entries(obj.score)
    first = workflow.explain_batch(obj.config)
    second = workflow.explain_batch(obj.config)
    assert first["explanation_id"] != second["explanation_id"]
    verify_explanation(first["artifact_dir"], first["explanation_id"])
    assert before == (file_entries(obj.release), file_entries(obj.score))
    with sqlite3.connect(obj.database) as conn:
        assert conn.execute(
            "SELECT status FROM explanation_runs ORDER BY explanation_id"
        ).fetchall() == [("succeeded",), ("succeeded",)]


def test_worker_failure_records_failed_not_succeeded(batch2_layout, monkeypatch):
    install_fakes(monkeypatch)

    def fail(*args, **kwargs):
        raise ValueError("native explanation failed")

    monkeypatch.setattr(workflow, "run_worker", fail)
    with pytest.raises(ValueError, match="native"):
        workflow.explain_batch(batch2_layout.config)
    with sqlite3.connect(batch2_layout.database) as conn:
        row = conn.execute(
            "SELECT status,artifact_dir,error_summary FROM explanation_runs"
        ).fetchone()
    assert row[0] == "failed" and "native" in row[2]
    assert (Path(row[1]) / "failure.json").exists()
    assert not (Path(row[1]) / "artifacts.json").exists()


def test_source_mutation_during_execution_rejected(batch2_layout, monkeypatch):
    install_fakes(monkeypatch)
    original = workflow.run_worker

    def mutate(config, binding, directory, *, check_only):
        response = original(config, binding, directory, check_only=check_only)
        (batch2_layout.score / "input.csv").write_text("changed")
        return response

    monkeypatch.setattr(workflow, "run_worker", mutate)
    with pytest.raises(ValueError, match="changed"):
        workflow.explain_batch(batch2_layout.config)


def test_unfinished_scoring_record_rejected(batch2_layout):
    obj = batch2_layout
    with sqlite3.connect(obj.database) as conn:
        conn.execute("UPDATE scoring_batches SET status='failed'")
    from ml_template.explainability.explanation_inputs import load_saved_batch

    with pytest.raises(ValueError, match="registry"):
        workflow.check_scoring_record(obj.config, load_saved_batch(obj.config))


@pytest.mark.parametrize("groups", [{"partial": ["x"]}, {"overlap": ["x", "y"], "again": ["x"]}])
def test_invalid_estimator_groups(batch2_layout, groups):
    config = batch2_layout.config.model_copy(update={"groups": groups})
    with pytest.raises(ValueError, match="partition"):
        workflow.validate_groups(config, ["x", "y"])
