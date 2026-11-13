"""Read-only operational inventories and hash-verified saved monitoring assessments."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from ml_template.monitoring.artifacts import read_json, verify

TABLES = {
    "scoring_batches": "started_at",
    "scoring_attempts": "started_at",
    "monitoring_runs": "started_at",
}


@contextmanager
def readonly(database):
    conn = sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def inventory(database, table, *, limit=200):
    if (
        table not in TABLES
        or not isinstance(limit, int)
        or isinstance(limit, bool)
        or not 1 <= limit <= 2000
    ):
        raise ValueError("invalid inventory table or limit")
    with readonly(database) as conn:
        if not conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone():
            return []
        return [
            dict(row)
            for row in conn.execute(
                f"SELECT * FROM {table} ORDER BY {TABLES[table]} DESC,rowid DESC LIMIT ?", (limit,)
            )
        ]


def load_assessment(database, identity):
    with readonly(database) as conn:
        row = conn.execute(
            "SELECT * FROM monitoring_runs WHERE monitoring_id=? AND status='succeeded'",
            (identity,),
        ).fetchone()
    if row is None:
        raise ValueError("successful monitoring execution not found")
    record = dict(row)
    if not record.get("manifest_sha256"):
        raise ValueError("monitoring execution has no registered manifest hash")
    directory = Path(record["artifact_dir"])
    entries = verify(directory, record["manifest_sha256"])
    names = ("summary", "config", "quality", "drift", "performance", "findings")
    if not {name + ".json" for name in names} <= set(entries):
        raise ValueError("monitoring artifact set is incomplete")
    view = {name: read_json(directory / f"{name}.json") for name in names}
    for name in (
        "monitoring_id",
        "scoring_batch_id",
        "release_id",
        "reference_id",
        "label_import_id",
        "assessment",
    ):
        if view["summary"].get(name) != record[name]:
            raise ValueError(f"monitoring summary/registry mismatch: {name}")
    for name in ("scoring_batch_id", "reference_id", "label_import_id"):
        if view["config"].get(name) != record[name]:
            raise ValueError(f"monitoring configuration/registry mismatch: {name}")
    if (
        view["summary"].get("schema_version") not in {1, 2}
        or not isinstance(view["drift"], list)
        or not isinstance(view["findings"], list)
    ):
        raise ValueError("invalid monitoring result schema")
    if view["performance"] is not None and not isinstance(view["performance"], dict):
        raise ValueError("invalid labelled-performance result")
    if view["summary"]["schema_version"] == 2:
        summary = view["summary"]
        if (
            summary.get("task") != "regression"
            or summary.get("prediction_kind") != "continuous"
            or not summary.get("target_unit")
        ):
            raise ValueError("Invalid frozen regression monitoring semantics")
        if view["performance"] is not None and view["performance"].get("task") != "regression":
            raise ValueError("Wrong monitoring performance task")
    elif view["summary"].get("task", "binary_classification") != "binary_classification":
        raise ValueError("Wrong legacy monitoring task")
    from ml_template.monitoring.nn_monitoring_contract import validate_frozen_view

    validate_frozen_view(view)
    return view


def label_context(database, identity):
    if identity is None:
        return (None, None)
    with readonly(database) as conn:
        row = conn.execute(
            "SELECT metadata_json FROM monitoring_objects WHERE object_id=? AND kind='labels'",
            (identity,),
        ).fetchone()
    if row is None:
        raise ValueError("label import metadata missing")
    value = json.loads(row[0])
    return (value.get("outcome_definition"), value.get("sampling"))


def comparable_history(database, selected, *, limit=200):
    context = label_context(database, selected["label_import_id"])
    seen, rows, errors = set(), [], []
    for record in inventory(database, "monitoring_runs", limit=limit):
        if (
            record["status"] != "succeeded"
            or record["release_id"] != selected["release_id"]
            or record["reference_id"] != selected["reference_id"]
        ):
            continue
        try:
            if label_context(database, record["label_import_id"]) != context:
                continue
            batch = record["scoring_batch_id"]
            if batch in seen:
                continue
            seen.add(batch)
            view = load_assessment(database, record["monitoring_id"])
            perf = view["performance"] or {}
            summary = view["summary"]
            if summary.get("task", "binary_classification") != selected.get(
                "task", "binary_classification"
            ):
                continue
            if summary.get("task") == "regression":
                if summary.get("target_unit") != selected.get("target_unit"):
                    continue
                metrics = perf.get("score_metrics", {}).get("y_pred_release", {})
                rows.append(
                    {
                        "monitoring_id": record["monitoring_id"],
                        "scoring_batch_id": batch,
                        "as_of": summary["as_of"],
                        "task": "regression",
                        "target_unit": summary["target_unit"],
                        "rmse": metrics.get("rmse"),
                        "mae": metrics.get("mae"),
                        "bias": metrics.get("bias"),
                        "r2": metrics.get("r2"),
                        "coverage": perf.get("mature_label_coverage"),
                        "n_evaluated": perf.get("n_evaluated"),
                        "support": perf.get("sufficient_support"),
                    }
                )
                continue
            metrics = perf.get("score_metrics", {}).get("p_release", {})
            rows.append(
                {
                    "monitoring_id": record["monitoring_id"],
                    "scoring_batch_id": batch,
                    "as_of": view["summary"]["as_of"],
                    "brier": metrics.get("brier"),
                    "pr_auc": metrics.get("pr_auc"),
                    "coverage": perf.get("mature_label_coverage"),
                    "n_evaluated": perf.get("n_evaluated"),
                    "support": perf.get("sufficient_support"),
                }
            )
        except (ValueError, KeyError, OSError, sqlite3.Error) as exc:
            errors.append(f"{record['monitoring_id']}: {exc}")
    rows.sort(key=lambda row: datetime.fromisoformat(row["as_of"]))
    return rows, errors


def metric_rows(perf):
    if perf and perf.get("task") == "regression":
        from ml_template.usage_dashboard.nn_monitor_report import metric_rows as regression_rows

        return regression_rows(perf)
    if not perf:
        return []
    return [
        {
            "variant": variant,
            "metric": metric,
            "value": value,
            "n_evaluated": perf["n_evaluated"],
            "n_positive": perf["n_positive"],
            "n_negative": perf["n_negative"],
            "sufficient_support": perf["sufficient_support"],
        }
        for variant, metrics in perf.get("score_metrics", {}).items()
        for metric, value in metrics.items()
    ]


def policy_rows(perf):
    policy = (perf or {}).get("policy")
    return (
        []
        if policy is None
        else [{"metric": name, "value": value} for name, value in policy.items()]
    )
