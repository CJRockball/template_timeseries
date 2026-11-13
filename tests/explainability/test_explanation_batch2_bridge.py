import subprocess
from pathlib import Path

import pandas as pd
import pytest

from ml_template.explainability.artifacts import sha256, write_json
from ml_template.explainability.pinned_worker import _aligned_saved, _runtime_difference
from ml_template.explainability.runtime_bridge import run_worker, worker_request


def test_bridge_uses_isolated_interpreter(batch2_layout, monkeypatch, tmp_path):
    obj = batch2_layout
    from ml_template.explainability import runtime_bridge
    from ml_template.explainability.explanation_inputs import load_saved_batch

    saved = load_saved_batch(obj.config)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        directory = Path(command[command.index("--output") + 1])
        response = {
            "ready": True,
            "release_id": saved.binding["release_id"],
            "scoring_batch_id": saved.binding["scoring_batch_id"],
            "binding_sha256": saved.binding["binding_sha256"],
            "worker_sha256": sha256(Path(runtime_bridge.__file__).with_name("pinned_worker.py")),
            "runtime_source_root": obj.config.runtime_source_root,
            "runtime_sha256": "fixture",
            "labels_loaded": False,
            "predictions_computed": False,
            "model_deserialized": False,
        }
        write_json(directory / "response.json", response)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("PYTHONPATH", "/untrusted/current/checkout")
    response = run_worker(obj.config, saved.binding, tmp_path / "ipc", check_only=True)
    assert response["ready"]
    command, kwargs = calls[0]
    assert command[1] == "-I"
    assert "--check" in command
    assert "PYTHONPATH" not in kwargs["env"]
    assert kwargs["cwd"] != Path.cwd()
    assert kwargs.get("shell", False) is False


def test_request_contains_no_labels_or_database(batch2_layout):
    request = worker_request(batch2_layout.config, {"fixture": True})
    assert "registry_path" not in request
    assert "labels" not in request
    assert "output_root" not in request


def test_worker_failures_not_silently_accepted(batch2_layout, monkeypatch, tmp_path):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 2, "", "runtime mismatch"),
    )
    with pytest.raises(ValueError, match="runtime mismatch"):
        run_worker(batch2_layout.config, {}, tmp_path / "ipc", check_only=True)


def test_runtime_difference_reports_missing_changed_extra_files():
    result = _runtime_difference(
        {
            "python": "3.11",
            "packages": {"x": "1"},
            "code_sha256": {"a.py": "old", "gone.py": "old"},
        },
        {
            "python": "3.12",
            "packages": {"x": "2"},
            "code_sha256": {"a.py": "new", "extra.py": "new"},
        },
    )
    assert result["changed_source_count"] == 3
    assert result["package_differences"]["x"] == {"expected": "1", "actual": "2"}


def test_worker_exact_key_alignment():
    frame = pd.DataFrame({"entity": [2, 1]})
    saved = pd.DataFrame({"entity": [1, 2], "p_raw": [0.1, 0.2]})
    assert _aligned_saved(frame, saved, "entity").p_raw.tolist() == [0.2, 0.1]
    with pytest.raises(ValueError):
        _aligned_saved(frame, saved.iloc[:1], "entity")


def test_worker_runtime_guard_precedes_row_read_and_model_load(
    batch2_layout, monkeypatch, tmp_path
):
    import ml_template
    from ml_template.explainability.explanation_inputs import inspect_saved_batch
    from ml_template.explainability.pinned_worker import run

    obj = batch2_layout
    binding = inspect_saved_batch(obj.config)[2]
    monkeypatch.setattr(ml_template, "__file__", str(obj.pinned / "__init__.py"))

    def forbidden(*args, **kwargs):
        raise AssertionError("must fail runtime verification before reading rows")

    monkeypatch.setattr(pd, "read_parquet", forbidden)
    request = worker_request(obj.config, binding)
    with pytest.raises(ValueError, match="pinned runtime mismatch"):
        run(request, tmp_path, check_only=True)


def test_worker_rejects_wrong_source_root(batch2_layout, tmp_path):
    from ml_template.explainability.pinned_worker import run

    request = worker_request(batch2_layout.config, {})
    with pytest.raises(ValueError, match="source root"):
        run(request, tmp_path, check_only=True)


@pytest.mark.parametrize("extra_file", [False, True])
def test_execute_worker_output_inventory_verified(batch2_layout, monkeypatch, tmp_path, extra_file):
    from ml_template.explainability import runtime_bridge
    from ml_template.explainability.artifacts import file_entries
    from ml_template.explainability.explanation_inputs import load_saved_batch

    obj = batch2_layout
    saved = load_saved_batch(obj.config)

    def fake_run(command, **kwargs):
        directory = Path(command[command.index("--output") + 1])
        for name in ("features.parquet", "contributions.npz", "reproduced_predictions.parquet"):
            (directory / name).write_bytes(b"fixture protocol bytes; not real model output")
        response = {
            "ready": True,
            "release_id": saved.binding["release_id"],
            "scoring_batch_id": saved.binding["scoring_batch_id"],
            "binding_sha256": saved.binding["binding_sha256"],
            "worker_sha256": sha256(Path(runtime_bridge.__file__).with_name("pinned_worker.py")),
            "runtime_source_root": obj.config.runtime_source_root,
            "runtime_sha256": "fixture",
            "labels_loaded": False,
            "predictions_computed": True,
            "model_deserialized": True,
            "files": file_entries(directory, exclude=("request.json", "response.json")),
        }
        write_json(directory / "response.json", response)
        if extra_file:
            (directory / "unlisted.txt").write_text("changed")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    if extra_file:
        with pytest.raises(ValueError, match="unlisted"):
            run_worker(obj.config, saved.binding, tmp_path / "ipc", check_only=False)
    else:
        assert run_worker(obj.config, saved.binding, tmp_path / "ipc", check_only=False)["ready"]
