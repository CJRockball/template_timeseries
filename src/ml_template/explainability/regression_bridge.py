"""Run an external, hashed regression worker in the exact selected release runtime."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from ml_template.explainability.artifacts import read_json, sha256, verify_files, write_json


def run_worker(config, binding, directory, *, check_only):
    executable = Path(config.runtime_python)
    source = Path(config.runtime_source_root).resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError("explicit executable runtime_python required")
    if not (source / "__init__.py").is_file():
        raise ValueError("explicit pinned ml_template source root required")
    directory = Path(directory).resolve()
    request = {
        name: getattr(config, name)
        for name in ("release_dir", "scoring_dir", "runtime_source_root", "chunk_size", "max_rows")
    }
    request.update(
        binding=binding,
        tolerance={
            "prediction_atol": config.prediction_atol,
            "prediction_rtol": config.prediction_rtol,
        },
    )
    write_json(directory / "request.json", request)
    origin = Path(__file__).with_name("regression_worker.py")
    worker_hash = sha256(origin)
    environment = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE"):
        environment.pop(name, None)
    # Copy the standalone script, NOT package code, outside the inference source.
    with tempfile.TemporaryDirectory(prefix="regression-shap-script-") as temp:
        worker = Path(temp) / "worker.py"
        shutil.copyfile(origin, worker)
        if worker.is_relative_to(source) or sha256(worker) != worker_hash:
            raise ValueError("standalone worker placement/hash invalid")
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
    if completed.returncode or response.get("ready") is not True:
        raise ValueError(
            "pinned regression explanation failed: "
            + str(response.get("error") or completed.stderr[-3000:])
        )
    if sha256(origin) != worker_hash or response.get("worker_sha256") != worker_hash:
        raise ValueError("explanation worker implementation changed")
    for name in ("release_id", "scoring_batch_id", "binding_sha256"):
        if response.get(name) != binding[name]:
            raise ValueError("worker identity differs: " + name)
    if (
        Path(response["runtime_source_root"]).resolve() != source
        or response.get("labels_loaded") is not False
        or response.get("task") != "regression"
    ):
        raise ValueError("worker runtime/task/label evidence differs")
    if check_only:
        if (
            response.get("model_deserialized") is not False
            or response.get("predictions_computed") is not False
        ):
            raise ValueError("check unexpectedly executed a model")
    else:
        if (
            response.get("model_deserialized") is not True
            or response.get("predictions_computed") is not True
        ):
            raise ValueError("missing inference evidence")
        expected = {"features.parquet", "contributions.npz", "reproduced_predictions.parquet"}
        if set(response.get("files", {})) != expected:
            raise ValueError("worker output inventory differs")
        verify_files(directory, response["files"], exclude=("request.json", "response.json"))
    return response
