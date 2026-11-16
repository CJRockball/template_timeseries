"""Read-only saved forecasting evidence; no models or raw Silver APIs."""

import csv
import io
import json
import math
import sqlite3
from contextlib import closing
from pathlib import Path

from ml_template.forecasting.acceptance_store import canonical, digest, input_artifact, verify

REQUEST = {
    "request_id",
    "plan_id",
    "purpose",
    "series",
    "origin_date",
    "issued_at",
    "target_date",
    "horizon",
    "observation_id",
}
EXTRA = {"method", "status", "reason", "y_pred", "y_true", "error", "common_supported"}


def finite(value):
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError("Finite number required")
    return float(value)


def connect(database):
    if not database or not Path(database).is_file():
        raise ValueError("Use --forecast-db with your existing project SQLite file")
    conn = sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    return conn


def registry(database, table, kind, identifier=None):
    if table not in {"temporal_acceptance_artifacts", "temporal_xgb_artifacts"}:
        raise ValueError("Unknown registry")
    with closing(connect(database)) as conn:
        if not conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone():
            return []
        query = "SELECT artifact_id,kind,manifest_sha256,path FROM " + table + " WHERE kind=?"
        args = [kind]
        if identifier is not None:
            query += " AND artifact_id=?"
            args.append(identifier)
        return [
            dict(zip(["artifact_id", "kind", "manifest_sha256", "path"], row, strict=True))
            for row in conn.execute(query + " ORDER BY artifact_id LIMIT 201", args)
        ]


def inventory(database):
    try:
        records = registry(database, "temporal_acceptance_artifacts", "report")
        message = [] if records else ["No reports in this DB. Generate an acceptance report first."]
        if len(records) > 200:
            message.append("First 200 report IDs shown; ordering is NOT recency")
        return [
            {"label": row["artifact_id"], "value": row["artifact_id"]} for row in records[:200]
        ], message
    except (ValueError, OSError, sqlite3.Error) as error:
        return [], [str(error)]


def record(database, table, kind, identifier):
    found = registry(database, table, kind, identifier)
    if (
        len(found) != 1
        or found[0]["artifact_id"] != identifier
        or not Path(found[0]["path"]).is_absolute()
    ):
        raise ValueError("Select a registered artifact from this project DB")
    return found[0]


def binding(value, expected):
    if any(value[key] != expected[key] for key in ["artifact_id", "manifest_sha256"]):
        raise ValueError("Artifact binding mismatch")


def snapshot(database, identifier):
    registered = record(database, "temporal_acceptance_artifacts", "report", identifier)
    manifest = verify(registered["path"])
    binding(manifest, registered)
    if manifest["identity"]["kind"] != "report":
        raise ValueError("Not a development report")
    report = json.loads((Path(registered["path"]) / "report.json").read_text())
    if (
        report.get("holdout_scored") is not False
        or report.get("decision") != "descriptive_only_no_promotion"
        or report["bindings"] != manifest["identity"]["inputs"]["bindings"]
    ):
        raise ValueError("Unsafe or misbound report")
    methods = sorted({row["method"] for row in report["metrics"]})
    series = sorted({row["key"] for row in report["metrics"] if row["scope"] == "series"})
    origins = sorted({row["key"] for row in report["metrics"] if row["scope"] == "origin"})
    if (
        not methods
        or len(series) != report["series"]
        or len(origins) != report["origins"]
        or report["requests"] < 1
    ):
        raise ValueError("Invalid report population")
    keys = set()
    for row in report["metrics"]:
        key = tuple(row[field] for field in ["scope", "key", "method", "comparison"])
        if (
            key in keys
            or type(row["n"]) is not int
            or type(row["total_requests"]) is not int
            or not 0 <= row["n"] <= row["total_requests"]
            or row["total_requests"] < 1
        ):
            raise ValueError("Invalid/duplicate metric counts")
        keys.add(key)
        if not math.isclose(
            finite(row["coverage_fraction"]), row["n"] / row["total_requests"], abs_tol=1e-12
        ):
            raise ValueError("Invalid coverage")
        for metric in ["mae", "rmse"]:
            if (row[metric] is None) != (row["n"] == 0) or (
                row[metric] is not None and finite(row[metric]) < 0
            ):
                raise ValueError("Invalid metric")
    expected = {
        (scope, key, method, support)
        for scope, values in [("pooled", [""]), ("origin", origins), ("series", series)]
        for key in values
        for method in methods
        for support in ["own_supported", "common_supported"]
    }
    if keys != expected or set(origins) != {row["origin"] for row in report["rounds"]}:
        raise ValueError("Incomplete metric/origin population")
    binding(verify(registered["path"]), manifest)
    return {
        "manifest": manifest,
        "report": report,
        "methods": methods,
        "series": series,
        "origins": origins,
    }


def model_settings(database, state):
    expected = state["report"]["bindings"]["models"]
    registered = record(database, "temporal_xgb_artifacts", "models", expected["artifact_id"])
    root = Path(registered["path"])
    if root.is_symlink() or (root / "manifest.json").is_symlink():
        raise ValueError("Symlink rejected")
    value = json.loads((root / "manifest.json").read_text())
    binding(value, expected)
    binding(value, registered)
    if (
        value["format"] != "temporal-xgb-v1"
        or value["identity"]["kind"] != "models"
        or value["artifact_id"] != "tsxgb_" + digest(value["identity"])
        or digest({k: v for k, v in value.items() if k != "manifest_sha256"})
        != value["manifest_sha256"]
    ):
        raise ValueError("Model manifest integrity failure")
    return {
        "parameters": value["identity"]["inputs"]["parameters"],
        "seed": value["identity"]["inputs"].get("seed"),
        "scope": "Bound manifest metadata only; weights not opened/executed",
    }


def detail(database, state, max_rows=200000):
    report = state["report"]
    if (
        type(max_rows) is not int
        or max_rows < 1
        or report["requests"] * len(state["methods"]) > max_rows
    ):
        raise ValueError("Detail row budget exceeded; no partial metrics calculated")
    expected = report["bindings"]["scores"]
    registered = record(database, "temporal_xgb_artifacts", "scores", expected["artifact_id"])
    manifest = input_artifact(registered["path"], "scores")
    binding(manifest, expected)
    binding(manifest, registered)
    inputs = manifest["identity"]["inputs"]
    if inputs.get("holdout_scored") is not False or any(
        inputs[k] != report[k] for k in ["plan_id", "plan_manifest_sha256"]
    ):
        raise ValueError("Score plan/holdout differs")
    for label in ["models", "baseline"]:
        if (
            inputs[label + "_id"] != report["bindings"][label]["artifact_id"]
            or inputs[label + "_manifest_sha256"] != report["bindings"][label]["manifest_sha256"]
        ):
            raise ValueError("Score parent differs")
    groups, result = {}, []
    with (Path(registered["path"]) / "scores.jsonl").open() as stream:
        for index, line in enumerate(stream, start=1):
            if index > max_rows:
                raise ValueError("Detail row budget exceeded")
            row = json.loads(line)
            if (
                set(row) != REQUEST | EXTRA
                or row["purpose"] != "development"
                or type(row["horizon"]) is not int
                or row["horizon"] != 1
                or row["plan_id"] != report["plan_id"]
                or row["target_date"] >= report["holdout"]["start"]
                or row["origin_date"] not in state["origins"]
                or row["method"] not in state["methods"]
            ):
                raise ValueError("Score outside development cohort")
            request = {key: row[key] for key in REQUEST}
            if row["request_id"] != "request_" + digest(
                {key: value for key, value in request.items() if key != "request_id"}
            ):
                raise ValueError("Request ID mismatch")
            truth = finite(row["y_true"])
            if row["status"] == "ok":
                if (
                    finite(row["error"]) != finite(finite(row["y_pred"]) - truth)
                    or row["reason"] is not None
                ):
                    raise ValueError("Residual/reason mismatch")
            elif row["status"] == "unsupported":
                if (
                    row["y_pred"] is not None
                    or row["error"] is not None
                    or not isinstance(row["reason"], str)
                    or not row["reason"]
                ):
                    raise ValueError("Invalid unsupported forecast")
            else:
                raise ValueError("Unknown status")
            series = canonical(row["series"])
            if series not in state["series"] or type(row["common_supported"]) is not bool:
                raise ValueError("Invalid series/support")
            group = groups.setdefault(
                row["request_id"], {"request": request, "truth": truth, "methods": {}}
            )
            if (
                group["request"] != request
                or group["truth"] != truth
                or row["method"] in group["methods"]
            ):
                raise ValueError("Duplicate/misaligned scored request")
            group["methods"][row["method"]] = (row["status"] == "ok", row["common_supported"])
            result.append({**row, "series_key": series})
    if len(groups) != report["requests"]:
        raise ValueError("Score population differs")
    for group in groups.values():
        if set(group["methods"]) != set(state["methods"]):
            raise ValueError("Missing explicit method rows")
        common = all(value[0] for value in group["methods"].values())
        if any(value[1] != common for value in group["methods"].values()):
            raise ValueError("Common support mismatch")
    binding(input_artifact(registered["path"], "scores"), expected)
    return result


def summaries(rows, methods, support, scope="pooled", key=""):
    total = len({row["request_id"] for row in rows})
    output = []
    for method in methods:
        errors = [
            row["error"]
            for row in rows
            if row["method"] == method
            and row["status"] == "ok"
            and (support == "own_supported" or row["common_supported"])
        ]
        n = len(errors)
        output.append(
            {
                "scope": scope,
                "key": key,
                "method": method,
                "comparison": support,
                "n": n,
                "total_requests": total,
                "coverage_fraction": n / total if total else 0,
                "mae": finite(math.fsum(abs(e) for e in errors) / n) if n else None,
                "rmse": finite(math.sqrt(math.fsum(e * e for e in errors) / n)) if n else None,
                "bias": finite(math.fsum(errors) / n) if n else None,
            }
        )
    return output


def view(
    state,
    metric="rmse",
    support="common_supported",
    methods=None,
    baseline=None,
    series=(),
    origins=(),
    rows=None,
):
    chosen = state["methods"] if methods is None else methods
    baseline = baseline or state["methods"][0]
    if (
        metric not in {"mae", "rmse"}
        or support not in {"common_supported", "own_supported"}
        or not chosen
        or not set(chosen) <= set(state["methods"])
        or len(set(chosen)) != len(chosen)
        or baseline not in state["methods"]
    ):
        raise ValueError("Choose valid methods/metric/support/baseline")
    if not set(series or ()) <= set(state["series"]) or not set(origins or ()) <= set(
        state["origins"]
    ):
        raise ValueError("Unknown filter")
    if rows is None:
        if series or origins:
            raise ValueError("Enable linked predictions for filtered aggregates")
        metrics = [
            {**row, "bias": None}
            for row in state["report"]["metrics"]
            if row["comparison"] == support
        ]
    else:
        rows = [
            row
            for row in rows
            if (not series or row["series_key"] in series)
            and (not origins or row["origin_date"] in origins)
        ]
        if not rows:
            raise ValueError("No planned requests match these filters")
        metrics = summaries(rows, state["methods"], support)
        for scope, field in [("series", "series_key"), ("origin", "origin_date")]:
            groups = {}
            for row in rows:
                groups.setdefault(row[field], []).append(row)
            for key, group in sorted(groups.items()):
                metrics.extend(summaries(group, state["methods"], support, scope, key))
    pooled = {row["method"]: row for row in metrics if row["scope"] == "pooled"}
    reference = pooled[baseline][metric]
    table = []
    for method in chosen:
        row = dict(pooled[method])
        value = row[metric]
        comparable = support == "common_supported" and reference is not None and value is not None
        row.update(
            {
                "report_id": state["manifest"]["artifact_id"],
                "selected_series": canonical(list(series or [])),
                "selected_origins": canonical(list(origins or [])),
                "baseline": baseline,
                "metric_for_delta": metric,
                "score_minus_baseline": value - reference if comparable else None,
                "improvement_percent": 100 * (1 - value / reference)
                if comparable and reference != 0
                else None,
            }
        )
        table.append(row)
    return {
        "table": table,
        "metrics": [row for row in metrics if row["method"] in chosen],
        "rows": rows,
        "metric": metric,
        "support": support,
        "methods": chosen,
        "note": "Common support is fixed across ALL report methods, even hidden methods. Own-support baseline deltas are disabled.",
    }


def largest_errors(result):
    rows = result["rows"] or []
    supported = [
        row
        for row in rows
        if row["method"] in result["methods"]
        and row["status"] == "ok"
        and (result["support"] == "own_supported" or row["common_supported"])
    ]
    return [
        {**row, "absolute_error": abs(row["error"])}
        for row in sorted(
            supported, key=lambda row: (-abs(row["error"]), row["request_id"], row["method"])
        )[:20]
    ]


def comparison_csv(result):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(result["table"][0]))
    writer.writeheader()
    for row in result["table"]:
        writer.writerow(
            {
                key: "'" + value
                if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@"))
                else value
                for key, value in row.items()
            }
        )
    return stream.getvalue()
