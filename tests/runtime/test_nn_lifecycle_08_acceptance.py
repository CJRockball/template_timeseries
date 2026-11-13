"""Observable-runner tests; private pytest fixtures, never real model/data acceptance."""

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
_spec = importlib.util.spec_from_file_location(
    "_nn08_live_under_test", ROOT / "scripts/nn_acceptance_live.py"
)
live = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = live
_spec.loader.exec_module(live)


def plan():
    return {"phases": [{"name": "a"}, {"name": "b"}]}


@pytest.mark.parametrize(
    "names,full,selected",
    [
        (None, True, ["a", "b"]),
        (["all"], True, ["a", "b"]),
        (["a"], False, ["a"]),
        (["b", "a"], False, ["a", "b"]),
    ],
)
def test_phase_selection_does_not_invent_full_acceptance(names, full, selected):
    phases, is_full = live.select_phases(plan(), names)
    assert is_full is full and [p["name"] for p in phases] == selected


@pytest.mark.parametrize("names", [["all", "a"], ["a", "a"], ["missing"]])
def test_invalid_phase_selection(names):
    with pytest.raises(ValueError):
        live.select_phases(plan(), names)


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_budgets_rejected_before_io(tmp_path, value):
    with pytest.raises(ValueError):
        live.run(tmp_path, tmp_path / "out", phase_timeout=value)


def test_tail_heartbeat_and_stdout(tmp_path, capsys):
    updates = []
    command = [
        sys.executable,
        "-u",
        "-c",
        'import time; print("started"); time.sleep(.3); print("done")',
    ]
    result = live.supervise(
        command,
        tmp_path,
        dict(os.environ),
        tmp_path,
        phase_timeout=5,
        deadline=time.monotonic() + 5,
        heartbeat=0.05,
        on_progress=lambda *args: updates.append(args),
    )
    output = capsys.readouterr().out
    assert result["returncode"] == 0 and result["reason"] is None and updates
    assert "started" in output and "done" in output and ("elapsed=" in output)
    assert "started" in (tmp_path / "pytest.log").read_text()


@pytest.mark.parametrize("overall", [False, True])
def test_deadlines_kill_waiting_process(tmp_path, overall):
    result = live.supervise(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        tmp_path,
        dict(os.environ),
        tmp_path,
        phase_timeout=5 if overall else 0.2,
        deadline=time.monotonic() + (0.2 if overall else 5),
        heartbeat=0.05,
    )
    assert result["reason"] == ("overall_deadline" if overall else "phase_timeout")
    assert result["returncode"] != 0 and result["duration_s"] < 5


def private_repo(tmp_path, source):
    for name in [
        "nn_lifecycle_acceptance.py",
        "nn_acceptance_plugin.py",
        "nn_acceptance_live.py",
        "nn_acceptance_progress.py",
    ]:
        target = tmp_path / "scripts" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / "scripts" / name).read_bytes())
    test = tmp_path / "tests/test_private.py"
    test.parent.mkdir()
    test.write_text(source)
    config = tmp_path / "configs/acceptance/nn_package_07.json"
    config.parent.mkdir(parents=True)
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "phases": [
                    {"name": "gate_selftest", "tests": ["tests/test_private.py"], "strict": True}
                ],
            }
        )
    )
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    return tmp_path


def test_diagnostic_check_writes_nothing(tmp_path):
    root = private_repo(tmp_path, "def test_ok(): assert True\n")
    result = live.run(root, root / "evidence", names=["gate_selftest"], check_only=True)
    assert (
        result["status"] == "checked"
        and (not result["full_acceptance"])
        and (not (root / "evidence").exists())
    )


@pytest.mark.parametrize("skip", [False, True])
def test_real_diagnostic_run_preserves_evidence_policy(tmp_path, skip):
    source = "import pytest\ndef test_ok(): assert True\n"
    if skip:
        source += '@pytest.mark.skip(reason="not acceptance")\ndef test_skip(): pass\n'
    root = private_repo(tmp_path, source)
    result = live.run(
        root,
        root / "evidence",
        names=["gate_selftest"],
        check_only=False,
        phase_timeout=30,
        total_timeout=45,
        heartbeat=0.1,
    )
    assert result["status"] == ("failed" if skip else "diagnostic_passed")
    assert result["full_acceptance"] is False
    directory = Path(result["artifact_dir"])
    assert (directory / "acceptance_manifest.json").is_file()
    assert result["phases"][0]["evidence"]["passed"]
    if skip:
        assert "strict phase contains skipped" in result["phases"][0]["issues"]


def test_full_run_alone_can_claim_complete_coverage(tmp_path, monkeypatch):
    root = private_repo(tmp_path, "def test_ok(): assert True\n")
    monkeypatch.setattr(live.core, "REQUIRED", ["pytest"])
    data = (root / "configs/acceptance/nn_package_07.json").read_bytes()
    monkeypatch.setattr(
        live,
        "EXPECTED_PLAN_BLOB",
        live.hashlib.sha1(b"blob " + str(len(data)).encode() + b"\x00" + data).hexdigest(),
    )
    result = live.run(
        root, root / "evidence", check_only=False, phase_timeout=30, total_timeout=45, heartbeat=0.1
    )
    assert result["status"] == "passed" and result["full_acceptance"] is True
    assert result["production_approval"] is False


def test_timed_out_diagnostic_is_not_passed(tmp_path):
    root = private_repo(tmp_path, "import time\ndef test_wait(): time.sleep(60)\n")
    result = live.run(
        root,
        root / "evidence",
        names=["gate_selftest"],
        check_only=False,
        phase_timeout=0.25,
        total_timeout=10,
        heartbeat=0.05,
    )
    assert result["status"] == "timed_out" and (not result["full_acceptance"])
    assert result["phases"][0]["reason"] == "phase_timeout"
    assert Path(result["artifact_dir"], "acceptance.json").is_file()


@pytest.mark.skipif(
    os.name != "posix", reason="process-group cancellation guarantee is POSIX/Linux"
)
def test_sigterm_preserves_cancelled_report_and_stops_descendant(tmp_path):
    pidfile = tmp_path / "descendant.pid"
    source = (
        'import subprocess,sys,time\nfrom pathlib import Path\ndef test_wait():\n p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"])\n Path('
        + repr(str(pidfile))
        + ").write_text(str(p.pid))\n time.sleep(60)\n"
    )
    root = private_repo(tmp_path, source)
    output = root / "evidence"
    cmd = [
        sys.executable,
        "-u",
        str(root / "scripts/nn_acceptance_live.py"),
        "--repo",
        str(root),
        "--output-root",
        str(output),
        "--phase",
        "gate_selftest",
        "--execute",
        "--phase-timeout-s",
        "30",
        "--total-timeout-s",
        "40",
        "--heartbeat-s",
        ".1",
    ]
    process = subprocess.Popen(
        cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    try:
        deadline = time.monotonic() + 15
        while not pidfile.exists() and time.monotonic() < deadline and (process.poll() is None):
            time.sleep(0.05)
        assert pidfile.exists(), "private descendant did not start"
        process.send_signal(signal.SIGTERM)
        stdout, _ = process.communicate(timeout=15)
        assert process.returncode == 130, stdout
        reports = list(output.glob("*/acceptance.json"))
        assert len(reports) == 1
        value = json.loads(reports[0].read_text())
        assert value["status"] == "cancelled" and value["full_acceptance"] is False
        assert reports[0].with_name("acceptance_manifest.json").is_file()
        pid = int(pidfile.read_text())
        stat = Path("/proc") / str(pid) / "stat"

        def running():
            try:
                return stat.read_text().split()[2] != "Z"
            except FileNotFoundError:
                return False

        deadline = time.monotonic() + 3
        while running() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not running()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_seal_excludes_temporary_fixture_payloads(tmp_path):
    (tmp_path / "phase/tmp").mkdir(parents=True)
    (tmp_path / "phase/tmp/model.bin").write_bytes(b"private synthetic checkpoint")
    (tmp_path / "phase/events.json").write_text("{}")
    live.seal(tmp_path, {"status": "cancelled", "full_acceptance": False})
    value = json.loads((tmp_path / "acceptance_manifest.json").read_text())
    assert "phase/tmp/model.bin" not in value["files"] and "phase/events.json" in value["files"]


def test_changed_plan_cannot_claim_full_acceptance(tmp_path):
    root = private_repo(tmp_path, "def test_ok(): assert True\n")
    with pytest.raises(ValueError, match="unchanged approved"):
        live.run(root, root / "evidence", check_only=True)
    assert not (root / "evidence").exists()
