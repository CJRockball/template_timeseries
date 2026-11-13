"""Read-only discovery and verified saved-report loading. Never fits or evaluates models."""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from ml_template.evaluation.evaluation_artifacts import sha256


@contextmanager
def _connection(engine):
    owned = not isinstance(engine, sqlite3.Connection)
    c = engine.raw_connection() if owned else engine
    try:
        yield c
    finally:
        if owned:
            c.close()


def _rows(engine, sql, args=()):
    with _connection(engine) as c:
        cur = c.execute(sql, args)
        names = [d[0] for d in cur.description]
        return [dict(zip(names, row, strict=True)) for row in cur.fetchall()]


def list_evaluations(engine):
    exists = _rows(
        engine, "SELECT name FROM sqlite_master WHERE type='table' AND name='final_evaluations'"
    )
    if not exists:
        return []
    return _rows(
        engine,
        "SELECT evaluation_id,parent_run_id,calibration_id,status,started_at,accessed_at,error_summary FROM final_evaluations ORDER BY started_at DESC,evaluation_id DESC",
    )


def load_evaluation(engine, evaluation_id):
    if not evaluation_id:
        return {
            "state": "empty",
            "message": "Select an evaluation. No holdout evaluation is run by this page.",
        }
    if not list_evaluations(engine):
        return {"state": "empty", "message": "No final evaluations registered."}
    records = _rows(
        engine, "SELECT * FROM final_evaluations WHERE evaluation_id=?", (evaluation_id,)
    )
    if not records:
        return {"state": "missing", "message": "Unknown evaluation ID."}
    record = records[0]
    if record["status"] != "succeeded":
        return {
            "state": record["status"],
            "message": record.get("error_summary") or "Evaluation has not completed successfully.",
            "record": record,
        }
    try:
        directory = Path(record["artifact_dir"])
        if not directory.is_absolute():
            from ml_template.paths import ROOT

            directory = ROOT / directory
        directory = directory.resolve()
        manifest = json.loads((directory / "artifact_manifest.json").read_text())
        for name in ["report_data.json", "summary.json", "spec.json", "provenance.json"]:
            item = manifest["files"].get(name)
            if item is None or sha256(directory / name) != item["sha256"]:
                raise ValueError(f"report artifact failed integrity verification: {name}")
        data = json.loads((directory / "report_data.json").read_text())
        summary, spec = data["summary"], data["spec"]
        if (
            summary["evaluation_id"] != evaluation_id
            or summary["identity_hash"] != record["identity_hash"]
        ):
            raise ValueError("evaluation identity mismatch")
        if (
            spec["parent_run_id"] != record["parent_run_id"]
            or spec.get("calibration_id") != record["calibration_id"]
        ):
            raise ValueError("parent/calibration identity mismatch")
        if spec != json.loads(record["spec_json"]):
            raise ValueError("frozen specification differs from registry")
        return {
            "state": "verified",
            "record": record,
            "data": data,
            "message": "Saved report JSON and metadata verified. Predictions/dataset labels were not read; other artifacts were not reverified by this view.",
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {
            "state": "invalid",
            "message": f"Cannot display unverified results: {exc}",
            "record": record,
        }
