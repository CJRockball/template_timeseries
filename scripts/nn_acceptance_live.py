"""Bounded observable wrapper around unchanged Package 07 acceptance policy."""

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import time
import uuid
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

import nn_lifecycle_acceptance as core

EXPECTED_PLAN_BLOB = "b7e1fd4c595fec68cb3a84d9b489adb02b31ec12"


class RunInterrupted(Exception):
    pass


def select_phases(plan, names):
    phases = plan["phases"]
    available = [p["name"] for p in phases]
    if not names or names == ["all"]:
        return (phases, True)
    if "all" in names or len(set(names)) != len(names) or set(names) - set(available):
        raise ValueError("Unknown/duplicate phase; choose from " + ",".join(["all", *available]))
    return ([p for p in phases if p["name"] in names], False)


def progress(path):
    try:
        value = json.loads(Path(path).read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def emit(message):
    print("[nn-acceptance] " + message, flush=True)


def kill_tree(process):
    if os.name == "posix":
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    elif process.poll() is None:
        process.kill()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        raise RuntimeError("Failed to reap cancelled pytest") from None


def supervise(
    command, root, env, directory, *, phase_timeout, deadline, heartbeat=15, on_progress=None
):
    start = time.monotonic()
    next_beat = start + heartbeat
    offset = 0
    last_node = None
    reason = None
    log = directory / "pytest.log"
    current = directory / "progress.json"
    process = None
    emit(
        "START "
        + directory.name
        + " | budget="
        + str(round(min(phase_timeout, max(0, deadline - start)), 1))
        + "s"
    )

    def tail():
        nonlocal offset
        with log.open("r", errors="replace") as source:
            source.seek(offset)
            text = source.read()
            offset = source.tell()
        for line in text.splitlines():
            emit(directory.name + " | " + line)

    try:
        with log.open("w") as stream:
            process = subprocess.Popen(
                command,
                cwd=root,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=os.name == "posix",
            )
            while True:
                tail()
                now = time.monotonic()
                state = progress(current)
                node = state.get("nodeid")
                changed = node != last_node and node is not None
                if changed:
                    emit(directory.name + " | TEST " + str(node))
                    last_node = node
                if now >= next_beat or changed:
                    emit(
                        directory.name
                        + " | elapsed="
                        + str(round(now - start, 1))
                        + "s | "
                        + json.dumps(state, sort_keys=True)
                    )
                    if on_progress:
                        on_progress(state, now - start, process.pid)
                    next_beat = now + heartbeat
                code = process.poll()
                if code is not None:
                    process.wait()
                    break
                if now >= deadline:
                    reason = "overall_deadline"
                    kill_tree(process)
                    code = process.returncode
                    break
                if now - start >= phase_timeout:
                    reason = "phase_timeout"
                    kill_tree(process)
                    code = process.returncode
                    break
                time.sleep(min(0.2, max(0.01, deadline - now)))
            tail()
        return {
            "returncode": code,
            "reason": reason,
            "duration_s": time.monotonic() - start,
            "last_progress": progress(current),
        }
    except BaseException:
        if process is not None:
            kill_tree(process)
        raise


def seal(directory, report):
    core.write_json(directory / "acceptance.json", report)
    entries = {
        p.relative_to(directory).as_posix(): {"sha256": core.sha(p), "bytes": p.stat().st_size}
        for p in sorted(directory.rglob("*"))
        if p.is_file()
        and p.name != "acceptance_manifest.json"
        and ("tmp" not in p.relative_to(directory).parts)
    }
    core.write_json(
        directory / "acceptance_manifest.json",
        {
            "schema_version": 1,
            "scope": "reports_only; excludes pytest basetemp trees",
            "files": entries,
        },
    )


def run(
    root,
    output,
    *,
    names=None,
    check_only=True,
    phase_timeout=900,
    total_timeout=3600,
    heartbeat=15,
):
    if any(not math.isfinite(v) or v <= 0 for v in (phase_timeout, total_timeout, heartbeat)):
        raise ValueError("Budgets/heartbeat must be positive")
    root = Path(root).resolve()
    plan = core.load_plan(root)
    selected, full = select_phases(plan, names)
    if full:
        data = (root / "configs/acceptance/nn_package_07.json").read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\x00" + data).hexdigest()
        if actual != EXPECTED_PLAN_BLOB:
            raise ValueError("Full acceptance requires the unchanged approved Package 07 plan")
    output = core.validate_output(root, output)
    dependencies = (
        core.REQUIRED if full or any(p["name"] != "gate_selftest" for p in selected) else ["pytest"]
    )
    missing = [n for n in dependencies if core.importlib.util.find_spec(n) is None]
    paths = [
        root / "scripts" / n
        for n in [
            "nn_lifecycle_acceptance.py",
            "nn_acceptance_plugin.py",
            "nn_acceptance_live.py",
            "nn_acceptance_progress.py",
        ]
    ]
    if not all(p.is_file() for p in paths):
        raise ValueError("Acceptance scripts missing")
    report = {
        "schema_version": 2,
        "kind": "synthetic_lifecycle_acceptance",
        "baseline": "4371557fe2038893ddaad7b7a262dbca29d58bfb",
        "full_acceptance_requested": full,
        "full_acceptance": False,
        "production_approval": False,
        "fresh_model_quality_evidence": False,
        "selected_phases": [p["name"] for p in selected],
        "unselected_phases": [p["name"] for p in plan["phases"] if p not in selected],
        "phase_timeout_s": phase_timeout,
        "total_timeout_s": total_timeout,
        "heartbeat_s": heartbeat,
        "missing_dependencies": missing,
        "status": "blocked" if missing else "checked",
        "files_written": False,
        "environment": core.metadata(root),
        "phases": [],
        "active_phase": None,
    }
    if check_only:
        return report
    if missing:
        raise ValueError("Required dependencies missing: " + ",".join(missing))
    start = time.monotonic()
    deadline = start + total_timeout
    identity = "nnaccept08_" + datetime.now(UTC).strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:12]
    directory = output / identity
    directory.mkdir(parents=True, exist_ok=False)
    inputs = {p.relative_to(root).as_posix(): core.sha(p) for p in paths}
    inputs["configs/acceptance/nn_package_07.json"] = core.sha(
        root / "configs/acceptance/nn_package_07.json"
    )
    for folder, pattern in [("src/ml_template", "*.py"), ("tests", "test_*.py")]:
        inputs.update(
            {p.relative_to(root).as_posix(): core.sha(p) for p in (root / folder).rglob(pattern)}
        )
    for n in ["uv.lock", "pyproject.toml"]:
        if (root / n).is_file():
            inputs[n] = core.sha(root / n)
    report.update(
        status="running",
        files_written=True,
        acceptance_id=identity,
        artifact_dir=str(directory),
        input_sha256=inputs,
    )
    core.write_json(directory / "acceptance.json", report)
    try:
        for phase in selected:
            if time.monotonic() >= deadline:
                report["status"] = "timed_out"
                report["error"] = "overall_deadline"
                break
            phase_dir = directory / phase["name"]
            phase_dir.mkdir()
            event_path = phase_dir / "events.json"
            report["active_phase"] = {
                "name": phase["name"],
                "started_at": datetime.now(UTC).isoformat(),
            }
            core.write_json(directory / "acceptance.json", report)
            cmd = core.command(root, phase, phase_dir)
            cmd.insert(1, "-u")
            cmd += ["-p", "nn_acceptance_progress"]
            env = {
                **os.environ,
                "PYTEST_ADDOPTS": "",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1",
                "ML_TEMPLATE_NN_ACCEPTANCE_EVENTS": str(event_path),
                "ML_TEMPLATE_NN_PROGRESS": str(phase_dir / "progress.json"),
                "PYTHONPATH": str(root / "scripts") + os.pathsep + os.environ.get("PYTHONPATH", ""),
            }

            def update(state, elapsed, pid):
                report["active_phase"].update(last_progress=state, elapsed_s=elapsed, pid=pid)
                core.write_json(directory / "acceptance.json", report)

            result = supervise(
                cmd,
                root,
                env,
                phase_dir,
                phase_timeout=phase_timeout,
                deadline=deadline,
                heartbeat=heartbeat,
                on_progress=update,
            )
            try:
                events = json.loads(event_path.read_text()) if event_path.is_file() else None
            except ValueError:
                events = None
            issues = core.assess(events, phase, result["returncode"])
            if result["reason"]:
                issues.append(result["reason"])
            if (
                not (phase_dir / "junit.xml").is_file()
                or (phase_dir / "junit.xml").stat().st_size == 0
            ):
                issues.append("JUnit evidence missing")
            item = {
                "name": phase["name"],
                "strict": phase["strict"],
                "command": cmd,
                **result,
                "issues": issues,
                "evidence": events,
                "status": "failed" if issues else "passed",
            }
            evidence = events if isinstance(events, dict) else {}
            emit(
                phase["name"]
                + " | counts="
                + json.dumps(
                    {
                        k: len(evidence.get(k, []))
                        for k in ["passed", "failed", "skipped", "xfail", "deselected"]
                    }
                )
                + " | failed_nodes="
                + str([r.get("nodeid") for r in evidence.get("failed", []) if isinstance(r, dict)])
            )
            report["phases"].append(item)
            core.write_json(directory / "acceptance.json", report)
            emit(
                "END "
                + phase["name"]
                + " | "
                + item["status"]
                + " | "
                + str(round(result["duration_s"], 2))
                + "s | "
                + "; ".join(issues)
            )
            if (phase_dir / "tmp").is_dir():
                core.shutil.rmtree(phase_dir / "tmp")
            if issues:
                report.update(
                    status="timed_out" if result["reason"] else "failed",
                    error=phase["name"] + ": " + "; ".join(issues),
                )
                break
        else:
            if any(
                (not (root / n).is_file() or core.sha(root / n) != v for n, v in inputs.items())
            ):
                raise RuntimeError("Acceptance inputs changed during execution")
            if time.monotonic() >= deadline:
                report.update(status="timed_out", error="overall_deadline")
            else:
                report.update(
                    status="passed" if full else "diagnostic_passed", full_acceptance=full
                )
    except (KeyboardInterrupt, RunInterrupted) as exc:
        report.update(status="cancelled", error=str(exc) or "interrupted")
        emit("CANCELLED; partial evidence retained")
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        emit("FAILED: " + str(exc))
    finally:
        report["last_active_phase"] = report["active_phase"]
        report["active_phase"] = None
        report["duration_s"] = time.monotonic() - start
        seal(directory, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/nn_acceptance"))
    parser.add_argument(
        "--phase", action="append", help="Repeat explicit phase names; omit for full acceptance"
    )
    parser.add_argument("--phase-timeout-s", type=float, default=900)
    parser.add_argument("--total-timeout-s", type=float, default=3600)
    parser.add_argument("--heartbeat-s", type=float, default=15)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if any(
        not math.isfinite(v) or v <= 0
        for v in (args.phase_timeout_s, args.total_timeout_s, args.heartbeat_s)
    ):
        parser.error("Budgets/heartbeat must be positive")
    previous = {}

    def interrupted(signum, frame):
        raise RunInterrupted("signal " + str(signum))

    for signum in [signal.SIGINT, signal.SIGTERM]:
        previous[signum] = signal.signal(signum, interrupted)
    try:
        output = (
            args.output_root
            if args.output_root.is_absolute()
            else args.repo.resolve() / args.output_root
        )
        result = run(
            args.repo,
            output,
            names=args.phase,
            check_only=args.check,
            phase_timeout=args.phase_timeout_s,
            total_timeout=args.total_timeout_s,
            heartbeat=args.heartbeat_s,
        )
        print(json.dumps(result, indent=2), flush=True)
        return (
            130
            if result["status"] == "cancelled"
            else 0
            if result["status"] in {"checked", "passed", "diagnostic_passed"}
            else 1
        )
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
