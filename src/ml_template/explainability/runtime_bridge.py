"""Isolated old-runtime execution, not source-hash migration or a verification bypass."""

import os
import subprocess
from pathlib import Path

from ml_template.explainability.artifacts import read_json, sha256, verify_files, write_json


def worker_request(config, binding):
    tolerance_names = ("margin_atol", "margin_rtol", "probability_atol", "probability_rtol")
    return {
        "release_dir": config.release_dir,
        "scoring_dir": config.scoring_dir,
        "runtime_source_root": config.runtime_source_root,
        "binding": binding,
        "chunk_size": config.chunk_size,
        "max_rows": config.max_rows,
        "batch_definition": config.batch_definition,
        "tolerance": {name: getattr(config, name) for name in tolerance_names},
    }


def run_worker(config, binding, directory, *, check_only):
    executable = Path(config.runtime_python)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError("runtime_python must be an executable from the pinned environment")
    source_root = Path(config.runtime_source_root)
    if not (source_root / "__init__.py").is_file():
        raise ValueError("runtime_source_root must identify the pinned ml_template package")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    worker = Path(__file__).with_name("pinned_worker.py").resolve()
    if worker.is_relative_to(source_root.resolve()):
        raise ValueError("use a separate pinned checkout/environment for the inference worker")
    request = worker_request(config, binding)
    write_json(directory / "request.json", request)
    command = [
        str(executable),
        "-I",
        str(worker),
        "--request",
        str(directory / "request.json"),
        "--output",
        str(directory),
    ]
    if check_only:
        command.append("--check")
    environment = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE"):
        environment.pop(name, None)
    completed = subprocess.run(
        command,
        cwd=directory,
        env=environment,
        capture_output=True,
        text=True,
        timeout=config.timeout_seconds,
        check=False,
    )
    path = directory / "response.json"
    response = read_json(path) if path.is_file() else {}
    if completed.returncode != 0 or response.get("ready") is not True:
        message = (
            response.get("error") or completed.stderr[-3000:] or "worker returned no valid response"
        )
        raise ValueError(f"pinned runtime preflight/execution failed: {message}")
    for name in ("release_id", "scoring_batch_id", "binding_sha256"):
        if response.get(name) != binding[name]:
            raise ValueError(f"worker binding mismatch: {name}")
    if response.get("worker_sha256") != sha256(worker):
        raise ValueError("worker code changed during execution")
    if Path(response["runtime_source_root"]).resolve() != source_root.resolve():
        raise ValueError("worker source root mismatch")
    if response.get("labels_loaded") is not False:
        raise ValueError("worker did not affirm label-free execution")
    if check_only:
        if (
            response.get("predictions_computed") is not False
            or response.get("model_deserialized") is not False
        ):
            raise ValueError("check-only worker unexpectedly ran model inference")
    else:
        if response.get("predictions_computed") is not True:
            raise ValueError("worker did not compute predictions")
        expected_files = {"features.parquet", "contributions.npz", "reproduced_predictions.parquet"}
        if set(response.get("files", {})) != expected_files:
            raise ValueError("worker output inventory differs")
        verify_files(directory, response["files"], exclude=("request.json", "response.json"))
    return response
