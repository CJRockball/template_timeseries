"""IPC contract tests with a simulated subprocess, not native inference tests."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml_template.explainability import regression_bridge as bridge
from ml_template.explainability.artifacts import sha256


@pytest.mark.parametrize("fault", [None, "binding", "worker_hash", "labels", "inference", "exit"])
def test_isolation_and_fail_closed_response(tmp_path, monkeypatch, fault):
    source = tmp_path / "pinned/ml_template"
    source.mkdir(parents=True)
    (source / "__init__.py").touch()
    executable = tmp_path / "venv-python"
    executable.symlink_to(sys.executable)
    output = tmp_path / "ipc"
    output.mkdir()
    binding = {"release_id": "rel_a", "scoring_batch_id": "score_a", "binding_sha256": "hash_a"}
    config = SimpleNamespace(
        runtime_python=str(executable),
        runtime_source_root=str(source),
        release_dir="release",
        scoring_dir="scores",
        chunk_size=2,
        max_rows=10,
        prediction_atol=1e-5,
        prediction_rtol=1e-5,
        timeout_seconds=20,
    )
    monkeypatch.setenv("PYTHONPATH", "must_not_reach_worker")

    def simulated(command, **kwargs):
        assert command[0] == str(executable) and command[1] == "-I"
        assert "PYTHONPATH" not in kwargs["env"]
        worker = Path(command[2])
        assert not worker.is_relative_to(source)
        request = json.loads((output / "request.json").read_text())
        assert request["binding"] == binding
        response = {
            "ready": True,
            **binding,
            "task": "regression",
            "runtime_source_root": str(source),
            "worker_sha256": sha256(worker),
            "labels_loaded": False,
            "model_deserialized": False,
            "predictions_computed": False,
        }
        if fault == "binding":
            response["release_id"] = "wrong"
        elif fault == "worker_hash":
            response["worker_sha256"] = "wrong"
        elif fault == "labels":
            response["labels_loaded"] = True
        elif fault == "inference":
            response["predictions_computed"] = True
        (output / "response.json").write_text(json.dumps(response))
        return SimpleNamespace(returncode=1 if fault == "exit" else 0, stderr="test failure")

    monkeypatch.setattr(bridge.subprocess, "run", simulated)
    if fault:
        with pytest.raises(ValueError):
            bridge.run_worker(config, binding, output, check_only=True)
    else:
        result = bridge.run_worker(config, binding, output, check_only=True)
        assert result["predictions_computed"] is False
