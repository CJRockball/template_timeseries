"""Real-project filesystem/source-validation integration; no holdout labels."""

from types import SimpleNamespace

import pytest


def test_changed_selection_file_rejected_before_source_access(tmp_path, monkeypatch):
    pytest.importorskip("ml_template.paths")
    from ml_template.evaluation.policy_selection import validate_policy_selection
    from ml_template.tracking import run_artifacts

    path = tmp_path / "selection.json"
    path.write_text("tampered")
    ref = SimpleNamespace(artifact_path=str(path), artifact_sha256="0" * 64)
    spec = SimpleNamespace(policy=SimpleNamespace(selection=ref))
    monkeypatch.setattr(
        run_artifacts, "verify_artifact", lambda *a: pytest.fail("must reject before source read")
    )
    with pytest.raises(ValueError, match="record hash mismatch"):
        validate_policy_selection(spec, {}, tmp_path)


def test_evaluator_does_not_execute_after_preflight_rejection(monkeypatch):
    pytest.importorskip("sqlalchemy")
    from ml_template.evaluation import evaluate_run
    from ml_template.evaluation.evaluation_config import EvaluationConfig

    def blocked(*args, **kwargs):
        raise ValueError("policy source hash mismatch")

    monkeypatch.setattr(evaluate_run, "prepare_evaluation", blocked)
    monkeypatch.setattr(evaluate_run, "_load_inputs", lambda *a: pytest.fail("holdout read"))
    monkeypatch.setattr(evaluate_run, "_load_model", lambda *a: pytest.fail("model loaded"))
    with pytest.raises(ValueError, match="policy source hash"):
        evaluate_run.evaluate_spec(None, EvaluationConfig(parent_run_id="run_test"))


def test_saved_policy_reference_changes_identity():
    from ml_template.evaluation.evaluation_config import EvaluationConfig

    data = {
        "parent_run_id": "run_test",
        "policy": {"kind": "threshold", "value": 0.2, "source": "manual"},
    }
    first = EvaluationConfig.model_validate(data)
    data["policy"]["value"] = 0.3
    second = EvaluationConfig.model_validate(data)
    assert first.digest() != second.digest()
