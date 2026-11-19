"""One-command development forecasting workflow; never final-holdout evaluation."""

import argparse
import contextlib
import hashlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
import traceback
import uuid
from datetime import UTC, datetime
from pathlib import Path

STAGES = ("baseline", "materialize", "train", "compare", "report")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def snapshot():
    from ml_template.forecasting import backtest_acceptance, history_contract
    from ml_template.forecasting.temporal_xgb_io import producer

    return {
        "learned": producer(),
        "baseline": history_contract.producer(),
        "report": backtest_acceptance.producer(),
    }


def check_snapshot(expected):
    if snapshot() != expected:
        raise RuntimeError(
            "Producer/runtime changed during workflow. Stop editing source; start a new run."
        )


def sample_matches(path, config, plan, producer):
    from ml_template.forecasting.history import recipe
    from ml_template.forecasting.learned_artifacts import verify
    from ml_template.forecasting.temporal_samples import options

    saved = verify(path, "samples")["identity"]["inputs"]
    wanted = {
        "plan_id": plan["plan_id"],
        "plan_manifest_sha256": plan["manifest_sha256"],
        "silver": plan["identity"]["silver"],
        "recipe": recipe(config["recipe"]),
        "sample_options": options(config["samples"]),
    }
    if any(saved.get(key) != value for key, value in wanted.items()):
        raise ValueError(
            "Explicit samples have different plan/Silver/recipe/options; refusing substitution"
        )
    return saved["producer"] == producer


def baseline_matches(path, config, plan, producer):
    from ml_template.forecasting.history import recipe
    from ml_template.forecasting.history_artifacts import verify_artifact

    saved = verify_artifact(path, "predictions")["identity"]["inputs"]
    wanted = {
        "plan_id": plan["plan_id"],
        "plan_manifest_sha256": plan["manifest_sha256"],
        "silver": plan["identity"]["silver"],
        "recipe": recipe(config["recipe"]),
        "producer": producer,
    }
    return all(saved.get(key) == value for key, value in wanted.items())


def find_baseline(database, config, plan, producer):
    # Read the existing registry; never infer artifacts from timestamps or directory names.
    uri = Path(database).resolve().as_uri() + "?mode=ro"
    with contextlib.closing(sqlite3.connect(uri, uri=True)) as conn:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='temporal_history_artifacts'"
        ).fetchone()
        if not exists:
            return None
        rows = list(
            conn.execute(
                "SELECT path FROM temporal_history_artifacts WHERE kind='predictions' AND plan_id=? ORDER BY artifact_id",
                (plan["plan_id"],),
            )
        )
    matches = []
    for (path,) in rows:
        if not (Path(path) / "manifest.json").is_file():
            continue
        if baseline_matches(path, config, plan, producer):
            matches.append(str(Path(path).resolve()))
    if len(matches) > 1:
        raise ValueError("Multiple compatible baselines; select one explicitly with --baseline")
    return matches[0] if matches else None


def read_config(path):
    import yaml

    path = Path(path).resolve(strict=True)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "plan",
        "silver_dataset",
        "database",
        "output_root",
        "batch_size",
        "recipe",
        "samples",
        "model",
        "seed",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("Expected the existing temporal_xgb YAML schema")
    for key in ["plan", "silver_dataset", "database", "output_root"]:
        value = Path(config[key])
        config[key] = str(
            (path.parent / value).resolve() if not value.is_absolute() else value.resolve()
        )
    if config["model"].get("device", "cpu") not in {"cpu", "cuda"}:
        raise ValueError("Temporal XGBoost device must be cpu or cuda")
    return config


def inspect(config):
    from ml_template.forecasting.temporal_xgb_io import checked_context

    plan, _ = checked_context(config)
    holdout = plan["identity"]["holdout"]["start"]
    windows = plan["identity"]["windows"]
    if not windows or any(
        w["target_date"] >= holdout
        or w["refit_end"] >= holdout
        or w.get("purpose", "development") != "development"
        for w in windows
    ):
        raise ValueError("Only development windows strictly before holdout are permitted")
    if not Path(config["database"]).is_file():
        raise FileNotFoundError("Existing project database required")
    return plan


def action(request):
    from ml_template.forecasting.temporal_xgb import load
    from ml_template.forecasting.temporal_xgb_io import build_samples, fit, score

    check_snapshot(request["producer"])
    if hashlib.sha256(Path(request["config"]).read_bytes()).hexdigest() != request["config_sha256"]:
        raise RuntimeError("Effective configuration changed during workflow")
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != request["runner_sha256"]:
        raise RuntimeError("Runner source changed during workflow")
    config = load(request["config"])
    name = request["stage"]
    paths = request["artifacts"]
    if name == "baseline":
        from ml_template.forecasting.history_io import predict

        result = predict(
            config["plan"],
            config["silver_dataset"],
            config["recipe"],
            Path(request["baseline_root"]),
            config["database"],
            config["batch_size"],
        )
    elif name == "materialize":
        result = build_samples(config)
    elif name == "train":
        result = fit(config, Path(paths["samples"]))
    elif name == "compare":
        result = score(config, Path(paths["models"]), Path(paths["baseline"]))
    elif name == "report":
        from ml_template.forecasting.backtest_acceptance import build

        result = build(
            Path(paths["samples"]),
            Path(paths["models"]),
            Path(paths["scores"]),
            Path(paths["baseline"]),
            Path(request["report_root"]),
            config["database"],
            [Path(p) for p in request["resources"]],
        )
    else:
        raise ValueError("Unknown stage")
    check_snapshot(request["producer"])
    atomic_json(request["result"], result)


def measured_worker(request_path):
    from ml_template.forecasting.backtest_resources import measure

    request = load_json(request_path)
    # Both wrapper and child are in the parent-created process group for cancellation.
    result = measure(
        [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", str(request_path)],
        request["label"],
        Path(request["resource_root"]),
        Path(request["database"]),
        [Path(request["watch"])],
        1.0,
    )
    atomic_json(request["resource_result"], result)
    if result["command_exit_code"] != 0:
        raise RuntimeError("Stage failed; see resource stdout.log: " + result["artifact_path"])


def terminate(process):
    if process.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run_process(command, log, timeout, heartbeat):
    started = time.monotonic()
    with Path(log).open("xb") as handle:
        process = subprocess.Popen(
            command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            while True:
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError(f"Stage exceeded {timeout:g} seconds; log: {log}")
                try:
                    status = process.wait(timeout=min(heartbeat, remaining))
                    break
                except subprocess.TimeoutExpired:
                    print(f"  running: {time.monotonic() - started:.0f}s | log: {log}", flush=True)
            if status:
                raise RuntimeError(f"Stage exited {status}; inspect {log} and resource-result.json")
        finally:
            terminate(process)


def validate_result(result, kind):
    from ml_template.forecasting.acceptance_store import verify as verify_acceptance
    from ml_template.forecasting.history_artifacts import verify_artifact
    from ml_template.forecasting.learned_artifacts import verify

    path = Path(result["artifact_path"]).resolve(strict=True)
    if kind == "baseline":
        manifest = verify_artifact(path, "predictions")
    elif kind == "report":
        manifest = verify_acceptance(path)
        if manifest["identity"]["kind"] != "report":
            raise ValueError("Wrong report kind")
    else:
        manifest = verify(path, kind)
    if manifest["artifact_id"] != result["artifact_id"]:
        raise ValueError("Returned artifact ID/path mismatch")
    return str(path)


def run(args):
    if not sys.platform.startswith("linux"):
        raise RuntimeError("Version 11 supports Linux/WSL (resource measurement/process groups)")
    if args.stage_timeout <= 0 or args.heartbeat < 1:
        raise ValueError("Positive stage timeout and heartbeat >=1 required")
    config = read_config(args.config)
    plan = inspect(config)
    producers = snapshot()
    baseline = (
        str(args.baseline.resolve(strict=True))
        if args.baseline
        else find_baseline(config["database"], config, plan, producers["baseline"])
    )
    if baseline and not baseline_matches(baseline, config, plan, producers["baseline"]):
        raise ValueError("Explicit baseline incompatible with config/current producer")
    samples = str(args.samples.resolve(strict=True)) if args.samples else None
    invalidated = None
    if samples and not sample_matches(samples, config, plan, producers["learned"]):
        invalidated = samples
        samples = None
    needed = [
        name
        for name in STAGES
        if not (name == "baseline" and baseline) and not (name == "materialize" and samples)
    ]
    print("Development workflow only; reserved holdout stays untouched.")
    print("Stages: " + " -> ".join(needed))
    if baseline:
        print("Verified baseline reuse: " + baseline)
    if invalidated:
        print("Sample producer mismatch: rematerialization scheduled (old artifact retained).")
    run_root = args.run_root.resolve()
    output_root = Path(config["output_root"])
    # Do not put bookkeeping in immutable artifacts or watched model output trees.
    protected = [Path(config[k]) for k in ["plan", "silver_dataset", "output_root"]]
    protected += [Path(p) for p in [baseline, samples, invalidated] if p]
    if any(run_root.is_relative_to(p) or p.is_relative_to(run_root) for p in protected):
        raise ValueError("Run root must be disjoint from plan/Silver/output/sample artifacts")
    if args.dry_run:
        print("Dry run complete; no workflow artifacts created.")
        return
    if config["model"].get("device", "cpu") == "cuda":
        from ml_template.forecasting.temporal_xgb_io import probe_cuda

        print("CUDA preflight passed: " + probe_cuda(), flush=True)
    run_root.mkdir(parents=True, exist_ok=True)
    identifier = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    directory = run_root / identifier
    directory.mkdir()
    resource_root = directory / "resources"
    report_root = directory / "reports"
    effective = directory / "effective-config.json"
    atomic_json(effective, config)
    artifacts = {
        key: value for key, value in [("baseline", baseline), ("samples", samples)] if value
    }
    state = {
        "format": "forecast-workflow-11",
        "status": "running",
        "evaluation": "development_only",
        "config_source": str(args.config.resolve()),
        "effective_config": config,
        "producer": producers,
        "invalidated_samples": invalidated,
        "artifacts": artifacts,
        "resources": [],
        "stages": [],
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    state["config_sha256"] = hashlib.sha256(effective.read_bytes()).hexdigest()
    state_path = directory / "workflow.json"
    atomic_json(state_path, state)
    print("Run record: " + str(state_path), flush=True)
    try:
        for stage in needed:
            check_snapshot(producers)
            stage_dir = directory / stage
            stage_dir.mkdir()
            request_path = stage_dir / "request.json"
            request = {
                "stage": stage,
                "config": str(effective),
                "producer": producers,
                "config_sha256": state["config_sha256"],
                "runner_sha256": state["runner_sha256"],
                "artifacts": dict(artifacts),
                "resources": list(state["resources"]),
                "result": str(stage_dir / "result.json"),
                "resource_result": str(stage_dir / "resource-result.json"),
                "resource_root": str(resource_root),
                "database": config["database"],
                "report_root": str(report_root),
                "baseline_root": str(output_root.parent / "workflow_baselines"),
                "watch": str(output_root.parent),
                "label": identifier + "-" + stage,
            }
            atomic_json(request_path, request)
            entry = {"stage": stage, "status": "running", "log": str(stage_dir / "wrapper.log")}
            state["stages"].append(entry)
            atomic_json(state_path, state)
            print(f"[{len(state['stages'])}/{len(needed)}] {stage}", flush=True)
            run_process(
                [
                    sys.executable,
                    "-u",
                    str(Path(__file__).resolve()),
                    "--measured-worker",
                    str(request_path),
                ],
                entry["log"],
                args.stage_timeout,
                args.heartbeat,
            )
            resource = load_json(request["resource_result"])
            if resource["command_exit_code"] != 0:
                raise RuntimeError("Measured child failed")
            state["resources"].append(resource["artifact_path"])
            result = load_json(request["result"])
            key = {
                "baseline": "baseline",
                "materialize": "samples",
                "train": "models",
                "compare": "scores",
                "report": "report",
            }[stage]
            artifacts[key] = validate_result(result, key)
            entry.update(
                status="completed",
                artifact=artifacts[key],
                resource=resource["artifact_path"],
                artifact_reused=bool(result.get("reused")),
            )
            check_snapshot(producers)
            atomic_json(state_path, state)
        state["status"] = "completed"
        report = Path(artifacts["report"])
        summary = (report / "summary.md").read_text(encoding="utf-8")
        (directory / "summary.md").write_text(summary, encoding="utf-8")
        print("\n" + summary)
        print("Report artifact: " + str(report))
        print("Workflow record: " + str(state_path))
    except BaseException:
        state["status"] = "failed"
        state["error"] = traceback.format_exc()
        if state["stages"] and state["stages"][-1]["status"] == "running":
            state["stages"][-1]["status"] = "failed"
        raise
    finally:
        atomic_json(state_path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--samples", type=Path, help="Optional verified temporal Gold/sample artifact"
    )
    parser.add_argument("--baseline", type=Path, help="Optional baseline prediction artifact")
    parser.add_argument("--run-root", type=Path, default=Path("data/forecast_workflows"))
    parser.add_argument("--stage-timeout", type=float, default=7200)
    parser.add_argument("--heartbeat", type=float, default=15)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--measured-worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        action(load_json(args.worker))
    elif args.measured_worker:
        measured_worker(args.measured_worker)
    elif args.config:
        run(args)
    else:
        parser.error("--config is required")


if __name__ == "__main__":
    main()
