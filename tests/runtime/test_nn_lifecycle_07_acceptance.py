"""Acceptance policy tests are stdlib/pytest only; no model or user data required."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_spec = importlib.util.spec_from_file_location(
    "_nn07_acceptance_under_test", Path(__file__).parents[2] / "scripts/nn_lifecycle_acceptance.py"
)
gate = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = gate
_spec.loader.exec_module(gate)


def phase():
    return {"name": "unit", "tests": ["tests/test_unit.py"], "strict": True}


def evidence():
    return {
        "collected": {"tests/test_unit.py": 1},
        "passed": ["tests/test_unit.py::test_a"],
        "skipped": [],
        "failed": [],
        "xfail": [],
        "deselected": [],
        "collection_errors": [],
        "exitstatus": 0,
    }


def test_strict_all_pass():
    assert gate.assess(evidence(), phase(), 0) == []


@pytest.mark.parametrize(
    "reason",
    [
        "skip",
        "xfail",
        "deselected",
        "failure",
        "collection",
        "empty",
        "missing_file",
        "count_mismatch",
        "duplicate",
        "unexpected_file",
        "exit",
        "process",
        "missing",
        "malformed",
    ],
)
def test_strict_rejects_unearned_success(reason):
    event = evidence()
    code = 0
    if reason == "skip":
        event["skipped"] = [{"reason": "torch absent"}]
    elif reason == "xfail":
        event["xfail"] = [{"reason": "expected failure"}]
    elif reason == "deselected":
        event["deselected"] = ["test_a"]
    elif reason == "failure":
        event["failed"] = [{"reason": "assertion"}]
    elif reason == "collection":
        event["collection_errors"] = [{"reason": "import"}]
    elif reason == "empty":
        event["passed"] = []
    elif reason == "missing_file":
        event["collected"] = {}
    elif reason == "count_mismatch":
        event["collected"]["tests/test_unit.py"] = 2
    elif reason == "duplicate":
        event["passed"] *= 2
    elif reason == "unexpected_file":
        event["collected"]["tests/test_extra.py"] = 1
    elif reason == "exit":
        event["exitstatus"] = 5
    elif reason == "process":
        code = 1
    elif reason == "missing":
        event = None
    else:
        event["passed"] = "wrong type"
    assert gate.assess(event, phase(), code)


def test_compatibility_optional_skips_are_recorded_not_native_acceptance():
    event = evidence()
    event["skipped"] = [{"reason": "optional library"}]
    event["xfail"] = [{"reason": "known issue"}]
    event["deselected"] = ["local_data"]
    assert gate.assess(event, {**phase(), "strict": False}, 0) == []
    assert gate.assess(event, phase(), 0)


@pytest.mark.parametrize(
    "protected", ["src/nested", "data/silver", "data/inference", ".git/objects", ".venv/tmp"]
)
def test_protected_output_refused(tmp_path, protected):
    with pytest.raises(ValueError):
        gate.validate_output(tmp_path, tmp_path / protected)


@pytest.mark.parametrize("marker", gate.SEALED)
def test_sealed_output_refused(tmp_path, marker):
    root = tmp_path / "sealed"
    root.mkdir()
    (root / marker).write_text("{}")
    with pytest.raises(ValueError):
        gate.validate_output(tmp_path, root / "child")


def layout(tmp_path):
    for name in [
        "scripts/nn_lifecycle_acceptance.py",
        "scripts/nn_acceptance_plugin.py",
        "tests/test_unit.py",
    ]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# unit fixture only")
    path = tmp_path / "configs/acceptance/nn_package_07.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema_version": 1, "phases": [phase()]}))
    return tmp_path


def test_preflight_writes_nothing(tmp_path, monkeypatch):
    root = layout(tmp_path)
    monkeypatch.setattr(gate, "REQUIRED", [])
    monkeypatch.setattr(gate, "metadata", lambda root: {"unit_fixture": True})
    output = root / "out"
    before = {
        p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }
    result = gate.run(root, output, check_only=True)
    assert (
        result["status"] == "checked" and result["files_written"] is False and (not output.exists())
    )
    assert before == {
        p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize("invalid", ["missing", "traversal", "duplicate_phase", "empty"])
def test_invalid_plans_rejected(tmp_path, invalid):
    root = layout(tmp_path)
    path = root / "configs/acceptance/nn_package_07.json"
    plan = json.loads(path.read_text())
    if invalid == "missing":
        plan["phases"][0]["tests"] = ["tests/not_present.py"]
    elif invalid == "traversal":
        plan["phases"][0]["tests"] = ["tests/../escape.py"]
    elif invalid == "duplicate_phase":
        plan["phases"] *= 2
    else:
        plan["phases"] = []
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        gate.load_plan(root)


@pytest.mark.parametrize("bad_evidence", [False, True])
def test_runner_persists_pass_or_failure_without_real_models(tmp_path, monkeypatch, bad_evidence):
    root = layout(tmp_path)
    monkeypatch.setattr(gate, "REQUIRED", [])
    monkeypatch.setattr(gate, "metadata", lambda root: {"unit_fixture": True})

    def fake_run(cmd, **kwargs):
        event = evidence()
        if bad_evidence:
            event["skipped"] = [{"reason": "simulated missing torch"}]
        path = Path(kwargs["env"]["ML_TEMPLATE_NN_ACCEPTANCE_EVENTS"])
        path.write_text(json.dumps(event))
        (path.parent / "junit.xml").write_text("<testsuites/>")
        return SimpleNamespace(wait=lambda timeout: 0)

    monkeypatch.setattr(gate.subprocess, "Popen", fake_run)
    result = gate.run(root, root / "out", check_only=False)
    assert result["status"] == ("failed" if bad_evidence else "passed")
    assert result["production_approval"] is False
    directory = Path(result["artifact_dir"])
    assert (directory / "acceptance_manifest.json").is_file()
    assert json.loads((directory / "acceptance.json").read_text())["status"] == result["status"]


def test_pytest_command_does_not_honor_silent_addopts(tmp_path):
    cmd = gate.command(tmp_path, phase(), tmp_path / "events")
    assert "addopts=" in cmd and "nn_acceptance_plugin" in cmd
    assert "-m" not in cmd[3:]
    compat = gate.command(tmp_path, {**phase(), "strict": False}, tmp_path / "events")
    assert compat[-2:] == ["-m", "not local_data"]


@pytest.mark.parametrize("scenario", ["ok", "skip", "xfail", "deselected", "collect_skip"])
def test_real_pytest_plugin_detects_missing_evidence(tmp_path, scenario):
    script_root = Path(__file__).parents[2] / "scripts"
    (tmp_path / "pytest.ini").write_text("[pytest]\nmarkers = exclude: private fixture\n")
    source = "import pytest\ndef test_ok(): assert True\n"
    if scenario == "skip":
        source += '@pytest.mark.skip(reason="missing native dependency")\ndef test_bad(): pass\n'
    elif scenario == "xfail":
        source += 'def test_bad(): pytest.xfail("unsupported")\n'
    elif scenario == "deselected":
        source += "@pytest.mark.exclude\ndef test_other(): pass\n"
    elif scenario == "collect_skip":
        (tmp_path / "test_missing.py").write_text(
            'import pytest\npytest.importorskip("nn07_definitely_absent_dependency")\n'
        )
    (tmp_path / "test_private.py").write_text(source)
    events = tmp_path / "events.json"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-o",
        "addopts=",
        "-p",
        "no:cacheprovider",
        "-p",
        "nn_acceptance_plugin",
        "-q",
        str(tmp_path),
    ]
    if scenario == "deselected":
        command += ["-m", "not exclude"]
    env = {
        **os.environ,
        "PYTHONPATH": str(script_root),
        "ML_TEMPLATE_NN_ACCEPTANCE_EVENTS": str(events),
        "PYTEST_ADDOPTS": "",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert events.is_file(), result.stdout + result.stderr
    value = json.loads(events.read_text())
    issues = gate.assess(
        value, {"name": "private", "tests": ["test_private.py"], "strict": True}, result.returncode
    )
    assert bool(issues) == (scenario != "ok"), (issues, value, result.stdout)
