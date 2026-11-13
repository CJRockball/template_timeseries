"""Synthetic lifecycle acceptance orchestrator. Never a production model release."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

REQUIRED = ["pytest", "numpy", "pandas", "torch", "pyarrow", "sqlalchemy", "dash", "dash_ag_grid"]
SEALED = [
    "manifest.json",
    "artifact_manifest.json",
    "artifacts.json",
    "pipeline_manifest.json",
    "release.json",
    "acceptance_manifest.json",
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temp = Path(path).with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def load_plan(root):
    path = root / "configs/acceptance/nn_package_07.json"
    plan = json.loads(path.read_text())
    if (
        plan.get("schema_version") != 1
        or not isinstance(plan.get("phases"), list)
        or (not plan["phases"])
    ):
        raise ValueError("Invalid acceptance plan")
    seen = set()
    for phase in plan["phases"]:
        if (
            set(phase) != {"name", "tests", "strict"}
            or phase["name"] in seen
            or (not phase["name"].replace("_", "").isalnum())
            or (not isinstance(phase["strict"], bool))
        ):
            raise ValueError("Invalid phase")
        seen.add(phase["name"])
        if (
            not isinstance(phase["tests"], list)
            or not phase["tests"]
            or len(set(phase["tests"])) != len(phase["tests"])
        ):
            raise ValueError("Missing/duplicate tests")
        for name in phase["tests"]:
            path = Path(name)
            if name == "tests" and (not phase["strict"]) and (root / path).is_dir():
                continue
            if (
                path.is_absolute()
                or ".." in path.parts
                or (not name.startswith("tests/"))
                or (not (root / path).is_file())
            ):
                raise ValueError("Invalid/missing acceptance test: " + name)
    return plan


def validate_output(root, output):
    output = Path(output).expanduser().resolve()
    for protected in ["src", "data", ".git", ".venv"]:
        if output.is_relative_to(root / protected):
            raise ValueError("Acceptance output inside protected tree")
    for parent in [output, *output.parents]:
        if any((parent / name).is_file() for name in SEALED):
            raise ValueError("Acceptance output inside sealed artifacts")
    if output.exists() and (not output.is_dir()):
        raise ValueError("Acceptance output is not a directory")
    return output


def assess(events, phase, returncode):
    issues = []
    if returncode != 0:
        issues.append("pytest/process nonzero")
    if not isinstance(events, dict):
        return ["pytest event evidence missing"] + issues
    required = {
        "collected",
        "passed",
        "skipped",
        "failed",
        "xfail",
        "deselected",
        "collection_errors",
        "exitstatus",
    }
    if not required <= events.keys():
        return ["pytest event evidence incomplete"] + issues
    if not isinstance(events["collected"], dict) or any(
        not isinstance(events[k], list)
        for k in ["passed", "skipped", "failed", "xfail", "deselected", "collection_errors"]
    ):
        return ["pytest evidence has wrong types"] + issues
    if events["exitstatus"] != 0:
        issues.append("pytest session nonzero")
    if events["failed"] or events["collection_errors"]:
        issues.append("test or collection failure")
    if not events["passed"]:
        issues.append("no executed passing tests")
    if phase["strict"]:
        for key in ["skipped", "xfail", "deselected"]:
            if events[key]:
                issues.append("strict phase contains " + key)
        if set(events["collected"]) - set(phase["tests"]):
            issues.append("unexpected strict test file")
        if len(set(events["passed"])) != len(events["passed"]):
            issues.append("duplicate pass evidence")
        for path in phase["tests"]:
            actual = sum(
                node.split("::", 1)[0].replace("\\", "/") == path for node in events["passed"]
            )
            if actual != events["collected"].get(path, 0):
                issues.append("collected/pass count mismatch: " + path)
            if events["collected"].get(path, 0) < 1:
                issues.append("required file not collected: " + path)
            if not any(
                node.split("::", 1)[0].replace("\\", "/") == path for node in events["passed"]
            ):
                issues.append("required file has no pass: " + path)
    return issues


def command(root, phase, directory):
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-o",
        "addopts=",
        "-p",
        "no:cacheprovider",
        "-p",
        "nn_acceptance_plugin",
        "-v",
        "-rs",
        "--tb=long",
        "--junitxml=" + str(directory / "junit.xml"),
        "--basetemp=" + str(directory / "tmp"),
        *phase["tests"],
    ]
    if not phase["strict"]:
        cmd.extend(["-m", "not local_data"])
    return cmd


def metadata(root):
    versions = {}
    for distribution in [
        "torch",
        "pytest",
        "numpy",
        "pandas",
        "pyarrow",
        "sqlalchemy",
        "dash",
        "dash-ag-grid",
    ]:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "versions": versions,
        "git_commit": result.stdout.strip() if result.returncode == 0 else None,
    }


def run(root, output, *, check_only=True, timeout=1800):
    root = Path(root).resolve()
    plan = load_plan(root)
    output = validate_output(root, output)
    missing = [name for name in REQUIRED if importlib.util.find_spec(name) is None]
    plugins = (
        root / "scripts/nn_acceptance_plugin.py",
        root / "scripts/nn_lifecycle_acceptance.py",
    )
    if not all(p.is_file() for p in plugins):
        raise ValueError("Acceptance scripts missing")
    report = {
        "schema_version": 1,
        "kind": "synthetic_lifecycle_acceptance",
        "production_approval": False,
        "fresh_model_quality_evidence": False,
        "missing_dependencies": missing,
        "plan": plan,
        "environment": metadata(root),
        "status": "checked" if not missing else "blocked",
        "files_written": False,
        "phases": [],
    }
    if check_only:
        return report
    if missing:
        raise ValueError("Required native/UI dependencies missing: " + ",".join(missing))
    identity = "nnaccept_" + datetime.now(UTC).strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:12]
    directory = output / identity
    directory.mkdir(parents=True, exist_ok=False)
    report.update(
        status="running",
        files_written=True,
        acceptance_id=identity,
        artifact_dir=str(directory),
        source_sha256={p.relative_to(root).as_posix(): sha(p) for p in plugins},
        lock_sha256=sha(root / "uv.lock") if (root / "uv.lock").is_file() else None,
        selected_test_sha256={
            p.relative_to(root).as_posix(): sha(p)
            for p in sorted((root / "tests").rglob("test_*.py"))
        },
        runtime_source_sha256={
            p.relative_to(root).as_posix(): sha(p)
            for p in sorted((root / "src/ml_template").rglob("*.py"))
        },
        plan_sha256=sha(root / "configs/acceptance/nn_package_07.json"),
    )
    write_json(directory / "acceptance.json", report)
    try:
        for phase in plan["phases"]:
            phase_dir = directory / phase["name"]
            phase_dir.mkdir()
            event_path = phase_dir / "events.json"
            cmd = command(root, phase, phase_dir)
            env = {
                **os.environ,
                "PYTEST_ADDOPTS": "",
                "PYTHONDONTWRITEBYTECODE": "1",
                "ML_TEMPLATE_NN_ACCEPTANCE_EVENTS": str(event_path),
                "PYTHONPATH": str(root / "scripts") + os.pathsep + os.environ.get("PYTHONPATH", ""),
            }
            start = datetime.now(UTC)
            error = None
            with (phase_dir / "pytest.log").open("w") as stream:
                process = subprocess.Popen(
                    cmd,
                    cwd=root,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=os.name == "posix",
                )
                try:
                    code = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait()
                    code = -1
                    error = "timeout"
            events = json.loads(event_path.read_text()) if event_path.is_file() else None
            issues = assess(events, phase, code)
            if (
                not (phase_dir / "junit.xml").is_file()
                or (phase_dir / "junit.xml").stat().st_size == 0
            ):
                issues.append("JUnit evidence missing")
            if error:
                issues.append(error)
            item = {
                "name": phase["name"],
                "strict": phase["strict"],
                "command": cmd,
                "duration_s": (datetime.now(UTC) - start).total_seconds(),
                "returncode": code,
                "status": "failed" if issues else "passed",
                "issues": issues,
                "evidence": events,
            }
            if (phase_dir / "tmp").is_dir():
                shutil.rmtree(phase_dir / "tmp")
            report["phases"].append(item)
            write_json(directory / "acceptance.json", report)
            if issues:
                raise RuntimeError(
                    "Acceptance phase failed: " + phase["name"] + "; " + "; ".join(issues)
                )
        if any(
            (sha(root / name) != digest for name, digest in report["selected_test_sha256"].items())
        ):
            raise RuntimeError("Selected tests changed during acceptance")
        if any((sha(root / name) != digest for name, digest in report["source_sha256"].items())):
            raise RuntimeError("Acceptance runner changed during execution")
        if report["lock_sha256"] is not None and sha(root / "uv.lock") != report["lock_sha256"]:
            raise RuntimeError("Lock changed during acceptance")
        if any(
            (sha(root / name) != digest for name, digest in report["runtime_source_sha256"].items())
        ):
            raise RuntimeError("Runtime source changed during acceptance")
        if sha(root / "configs/acceptance/nn_package_07.json") != report["plan_sha256"]:
            raise RuntimeError("Acceptance plan changed during execution")
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=str(exc))
    write_json(directory / "acceptance.json", report)
    entries = {
        p.relative_to(directory).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size}
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p.name != "acceptance_manifest.json"
    }
    write_json(directory / "acceptance_manifest.json", {"schema_version": 1, "files": entries})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/nn_acceptance"))
    parser.add_argument("--timeout-s", type=int, default=1800)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.timeout_s < 1:
        parser.error("timeout must be positive")
    output = (
        args.output_root
        if args.output_root.is_absolute()
        else args.repo.resolve() / args.output_root
    )
    result = run(args.repo, output, check_only=args.check, timeout=args.timeout_s)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] in {"checked", "passed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
