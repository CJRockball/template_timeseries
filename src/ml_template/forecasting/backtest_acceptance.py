"""Read frozen development artifacts; no training, raw Silver reads or promotion."""

import csv
import json
import math
import platform
import sqlite3
import tempfile
from collections import defaultdict
from contextlib import closing
from datetime import date
from pathlib import Path

from .acceptance_store import (
    Publication,
    canonical,
    digest,
    file_hash,
    input_artifact,
    load,
    rows,
    verify,
)

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
FORECAST = {"method", "status", "reason", "y_pred"}


def finite(value):
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError("Finite numeric value required")
    return float(value)


def producer():
    directory = Path(__file__).parent
    return {
        "python": platform.python_version(),
        "source_sha256": {
            name: file_hash(directory / name)
            for name in [
                "acceptance_store.py",
                "backtest_acceptance.py",
                "backtest_resources.py",
                "acceptance_cli.py",
            ]
        },
    }


def reference(value):
    return {key: value[key] for key in ["artifact_id", "manifest_sha256"]}


def aligned(record, request, methods, scored=False):
    extra = {"y_true", "error", "common_supported"} if scored else set()
    if (
        set(record) != REQUEST | FORECAST | extra
        or any(record[key] != request[key] for key in REQUEST)
        or record["method"] not in methods
    ):
        raise ValueError("Misaligned forecast/score row")
    if record["status"] == "ok":
        finite(record["y_pred"])
        if record["reason"] is not None:
            raise ValueError("Supported forecast has reason")
    elif record["status"] == "unsupported":
        if (
            record["y_pred"] is not None
            or not isinstance(record["reason"], str)
            or not record["reason"]
        ):
            raise ValueError("Malformed unsupported forecast")
    else:
        raise ValueError("Invalid forecast status")


def build(samples, models, scores, baseline, output, database, resources=(), max_groups=100000):
    output_path = Path(output).resolve()
    if any(
        output_path.is_relative_to(Path(path).resolve())
        for path in [samples, models, scores, baseline, *resources]
    ):
        raise ValueError("Output must be outside immutable input artifacts")
    if not Path(database).is_file():
        raise FileNotFoundError("Existing project database required")
    if type(max_groups) is not int or max_groups < 1:
        raise ValueError("Positive metric group cap required")
    artifacts = {
        "samples": input_artifact(samples, "samples"),
        "models": input_artifact(models, "models"),
        "scores": input_artifact(scores, "scores"),
        "baseline": input_artifact(baseline, "predictions"),
    }
    context = artifacts["samples"]["identity"]["inputs"]
    trained = artifacts["models"]["identity"]["inputs"]
    evaluated = artifacts["scores"]["identity"]["inputs"]
    base = artifacts["baseline"]["identity"]["inputs"]
    if (
        trained["context"] != context
        or trained["samples_id"] != artifacts["samples"]["artifact_id"]
        or trained["samples_manifest_sha256"] != artifacts["samples"]["manifest_sha256"]
        or trained["producer"] != context["producer"]
    ):
        raise ValueError("Sample/model binding mismatch")
    for label, artifact in [("models", artifacts["models"]), ("baseline", artifacts["baseline"])]:
        if (
            evaluated[label + "_id"] != artifact["artifact_id"]
            or evaluated[label + "_manifest_sha256"] != artifact["manifest_sha256"]
        ):
            raise ValueError("Score input binding mismatch")
    if evaluated["producer"] != context["producer"] or evaluated.get("holdout_scored") is not False:
        raise ValueError("Unsafe score producer/holdout declaration")
    for key in ["plan_id", "plan_manifest_sha256"]:
        if evaluated[key] != context[key]:
            raise ValueError("Score plan binding mismatch")
    for key in ["plan_id", "plan_manifest_sha256", "silver", "recipe"]:
        if base[key] != context[key]:
            raise ValueError("Baseline binding mismatch")
    methods = [*context["recipe"]["methods"], "xgb"]
    if len(set(methods)) != len(methods) or len(methods) < 2:
        raise ValueError("Unique baseline and XGB methods required")
    windows = {window["origin_date"]: window for window in context["windows"]}
    if len(windows) != len(context["windows"]):
        raise ValueError("Duplicate origins")
    holdout = context["holdout"]["start"]
    date.fromisoformat(holdout)
    for window in windows.values():
        if window["target_date"] >= holdout or window["refit_end"] >= holdout:
            raise ValueError("Holdout window prohibited")
    resource_values = []
    for path in resources:
        resource = verify(path)
        if resource["identity"]["kind"] != "resource":
            raise ValueError("Not a resource measurement")
        resource_values.append(
            {**reference(resource), "measurement": load(Path(path) / "resource.json")}
        )
    publication = Publication(output)
    try:
        with (
            tempfile.TemporaryDirectory(prefix="acceptance-index-") as temporary,
            closing(sqlite3.connect(Path(temporary) / "index.sqlite")) as conn,
        ):
            conn.execute("PRAGMA temp_store=FILE")
            conn.execute("PRAGMA cache_size=-32768")
            conn.execute(
                "CREATE TABLE requests(id TEXT PRIMARY KEY,series TEXT,origin TEXT,row TEXT,UNIQUE(series,origin))"
            )
            conn.execute(
                "CREATE TABLE predictions(id TEXT,method TEXT,row TEXT,ok INTEGER,PRIMARY KEY(id,method))"
            )
            conn.execute("CREATE TABLE scored(id TEXT,method TEXT,PRIMARY KEY(id,method))")
            conn.execute("CREATE TABLE truths(id TEXT PRIMARY KEY,y REAL)")
            for row in rows(samples, "requests.jsonl"):
                request = row["request"]
                if (
                    set(request) != REQUEST
                    or request["plan_id"] != context["plan_id"]
                    or request["purpose"] != "development"
                    or type(request["horizon"]) is not int
                    or request["horizon"] != 1
                ):
                    raise ValueError("Invalid development request")
                window = windows.get(request["origin_date"])
                if (
                    not window
                    or request["target_date"] != window["target_date"]
                    or request["issued_at"] != window["issued_at"]
                    or request["target_date"] >= holdout
                ):
                    raise ValueError("Request outside frozen development windows")
                if (
                    not isinstance(request["series"], list)
                    or not request["series"]
                    or any(type(v) not in (str, int) for v in request["series"])
                ):
                    raise ValueError("Invalid series key")
                if request["request_id"] != "request_" + digest(
                    {k: v for k, v in request.items() if k != "request_id"}
                ):
                    raise ValueError("Request ID integrity failure")
                conn.execute(
                    "INSERT INTO requests VALUES (?,?,?,?)",
                    (
                        request["request_id"],
                        canonical(request["series"]),
                        request["origin_date"],
                        canonical(request),
                    ),
                )
            total = conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
            if not total or set(
                dict(conn.execute("SELECT origin,COUNT(*) FROM requests GROUP BY origin"))
            ) != set(windows):
                raise ValueError("Empty or missing development origin")
            for directory, allowed in [(models, ["xgb"]), (baseline, context["recipe"]["methods"])]:
                for row in rows(directory, "predictions.jsonl"):
                    found = conn.execute(
                        "SELECT row FROM requests WHERE id=?", (row.get("request_id"),)
                    ).fetchone()
                    if found is None:
                        raise ValueError("Unexpected prediction request")
                    aligned(row, json.loads(found[0]), allowed)
                    conn.execute(
                        "INSERT INTO predictions VALUES (?,?,?,?)",
                        (row["request_id"], row["method"], canonical(row), row["status"] == "ok"),
                    )
            if conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] != total * len(
                methods
            ):
                raise ValueError("Missing explicit prediction rows")
            sample_report = load(Path(samples) / "report.json")
            if sample_report["requests"] != total:
                raise ValueError("Sample request count mismatch")
            counts = {(row["origin"], row["partition"]): row for row in sample_report["counts"]}
            if len(counts) != len(sample_report["counts"]) or set(counts) != {
                (origin, phase) for origin in windows for phase in ["fit", "stopping", "refit"]
            }:
                raise ValueError("Sample partition counts mismatch")
            for row in counts.values():
                values = [
                    row[key] for key in ["date_candidates", "label_eligible", "feature_complete"]
                ]
                if (
                    any(type(value) is not int for value in values)
                    or not context["sample_options"]["max_rows_per_partition"]
                    >= values[0]
                    >= values[1]
                    >= values[2]
                    >= 0
                ):
                    raise ValueError("Invalid sample eligibility counts")
            model_report = load(Path(models) / "report.json")
            rounds = model_report["models"]
            if (
                model_report.get("holdout_scored") is not False
                or len(rounds) != len(windows)
                or {row["origin"] for row in rounds} != set(windows)
            ):
                raise ValueError("Model report origin/holdout mismatch")
            warnings = []
            for row in rounds:
                origin = row["origin"]
                selected = row["selected_rounds"]
                if (
                    type(selected) is not int
                    or not 1 <= selected <= trained["parameters"]["n_estimators"]
                    or row["feature_order"] != context["sample_options"]["features"]
                ):
                    raise ValueError("Invalid round/feature metadata")
                expected = conn.execute(
                    "SELECT COUNT(*),SUM(p.ok) FROM requests r JOIN predictions p ON r.id=p.id WHERE r.origin=? AND p.method='xgb'",
                    (origin,),
                ).fetchone()
                if (
                    (row["requests"], row["supported"]) != expected
                    or row["refit_rows"] != counts[origin, "refit"]["feature_complete"]
                    or row["model_file"] not in artifacts["models"]["identity"]["files"]
                ):
                    raise ValueError("Model count/file binding mismatch")
                if selected == trained["parameters"]["n_estimators"]:
                    warnings.append("Maximum round count selected at " + origin)
            totals = {("pooled", ""): total}
            for scope, column in [("origin", "origin"), ("series", "series")]:
                query = "SELECT " + column + ",COUNT(*) FROM requests GROUP BY " + column
                for key, n in conn.execute(query):
                    totals[scope, key] = n
                    if len(totals) * len(methods) * 2 > max_groups:
                        raise ValueError("Metric group budget exceeded")
            accumulators = defaultdict(lambda: [0, 0.0, 0.0])
            unsupported = {method: defaultdict(int) for method in methods}
            for row in rows(scores, "scores.jsonl"):
                found = conn.execute(
                    "SELECT r.row,p.row,r.series,r.origin FROM requests r JOIN predictions p ON r.id=p.id WHERE r.id=? AND p.method=?",
                    (row.get("request_id"), row.get("method")),
                ).fetchone()
                if found is None:
                    raise ValueError("Unexpected scored request")
                request, prediction, series, origin = found
                aligned(row, json.loads(request), methods, scored=True)
                if {k: row[k] for k in REQUEST | FORECAST} != json.loads(prediction):
                    raise ValueError("Score/prediction disagreement")
                conn.execute("INSERT INTO scored VALUES (?,?)", (row["request_id"], row["method"]))
                actual = finite(row["y_true"])
                conn.execute(
                    "INSERT OR IGNORE INTO truths VALUES (?,?)", (row["request_id"], actual)
                )
                if (
                    conn.execute(
                        "SELECT y FROM truths WHERE id=?", (row["request_id"],)
                    ).fetchone()[0]
                    != actual
                ):
                    raise ValueError("Inconsistent outcomes across methods")
                common = (
                    conn.execute(
                        "SELECT SUM(ok)=? FROM predictions WHERE id=?",
                        (len(methods), row["request_id"]),
                    ).fetchone()[0]
                    == 1
                )
                if type(row["common_supported"]) is not bool or row["common_supported"] != common:
                    raise ValueError("Common support flag disagreement")
                if row["status"] == "unsupported":
                    if row["error"] is not None:
                        raise ValueError("Unsupported residual must be null")
                    unsupported[row["method"]][row["reason"]] += 1
                    continue
                residual = finite(row["y_pred"] - actual)
                if finite(row["error"]) != residual:
                    raise ValueError("Residual disagreement")
                for scope, key in [("pooled", ""), ("origin", origin), ("series", series)]:
                    for comparison in ["own_supported"] + (["common_supported"] if common else []):
                        accumulator = accumulators[row["method"], comparison, scope, key]
                        accumulator[0] += 1
                        accumulator[1] = finite(math.fsum((accumulator[1], abs(residual))))
                        accumulator[2] = finite(
                            math.fsum((accumulator[2], finite(residual * residual)))
                        )
            if conn.execute("SELECT COUNT(*) FROM scored").fetchone()[0] != total * len(methods):
                raise ValueError("Missing score rows")
            metrics = []
            for (scope, key), denominator in sorted(totals.items()):
                for method in sorted(methods):
                    for comparison in ["own_supported", "common_supported"]:
                        n, absolute, squared = accumulators[method, comparison, scope, key]
                        metrics.append(
                            {
                                "scope": scope,
                                "key": key,
                                "method": method,
                                "comparison": comparison,
                                "n": n,
                                "total_requests": denominator,
                                "coverage_fraction": n / denominator,
                                "mae": absolute / n if n else None,
                                "rmse": math.sqrt(squared / n) if n else None,
                            }
                        )
            cached = load(Path(scores) / "metrics.json")
            existing = cached["metrics"]
            if len(existing) != sum(row["scope"] != "series" for row in metrics) or cached[
                "unsupported"
            ] != {key: dict(value) for key, value in unsupported.items()}:
                raise ValueError("Cached metrics inventory/reason mismatch")
            for row in metrics:
                if row["scope"] == "series":
                    continue
                candidates = [
                    item
                    for item in existing
                    if item["method"] == row["method"]
                    and item["comparison"] == row["comparison"]
                    and item["scope"] == row["scope"]
                    and item["origin_date"] == (row["key"] or None)
                ]
                if len(candidates) != 1:
                    raise ValueError("Cached metric key mismatch")
                for key in ["n", "total_requests", "coverage_fraction", "mae", "rmse"]:
                    left, right = row[key], candidates[0][key]
                    if key in {"n", "total_requests"} and (type(right) is not int or left != right):
                        raise ValueError("Cached metric count mismatch")
                    if (left is None) != (right is None) or (
                        left is not None
                        and not math.isclose(
                            finite(left), finite(right), rel_tol=1e-10, abs_tol=1e-12
                        )
                    ):
                        raise ValueError("Cached metric value mismatch: " + key)
            if any(row["coverage_fraction"] < 1 for row in metrics if row["scope"] == "pooled"):
                warnings.append(
                    "Incomplete own/common support; inspect counts before comparing errors"
                )
            if not resources:
                warnings.append(
                    "No resource measurements supplied; RSS/time/staging peaks are unmeasured"
                )
            bindings = {key: reference(value) for key, value in artifacts.items()}
            artifact_bytes = {
                key: sum(info["bytes"] for info in value["identity"]["files"].values())
                for key, value in artifacts.items()
            }
            if resources:
                warnings.append(
                    "Resource records are user-associated measurements, not independently proven to belong to these model artifacts"
                )
            report = {
                "artifact_payload_bytes": artifact_bytes,
                "bindings": bindings,
                "plan_id": context["plan_id"],
                "plan_manifest_sha256": context["plan_manifest_sha256"],
                "silver": context["silver"],
                "holdout": context["holdout"],
                "holdout_scored": False,
                "requests": total,
                "series": sum(scope == "series" for scope, _ in totals),
                "origins": len(windows),
                "metrics": metrics,
                "rounds": rounds,
                "sample_eligibility": sample_report["counts"],
                "unsupported": {key: dict(value) for key, value in unsupported.items()},
                "warnings": warnings,
                "resources": resource_values,
                "decision": "descriptive_only_no_promotion",
                "truth_source": "frozen development score artifact; raw Silver is not reread",
            }
            (publication.stage / "report.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            with (publication.stage / "metrics.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
                writer.writeheader()
                writer.writerows(metrics)
            lines = [
                "# Development backtest acceptance",
                "",
                "Descriptive audit only. No automatic acceptance, model selection or release.",
                "",
                "Requests: "
                + str(total)
                + " | Series: "
                + str(report["series"])
                + " | Origins: "
                + str(len(windows)),
                "",
                "## Frozen inputs",
            ]
            lines.extend(
                "- " + key + ": " + value["artifact_id"] for key, value in bindings.items()
            )
            lines += [
                "",
                "## Pooled common-supported comparison",
                "",
                "| Method | N | Coverage | MAE | RMSE |",
                "|---|---:|---:|---:|---:|",
            ]
            lines.extend(
                "| "
                + " | ".join(
                    str(row[field]) for field in ["method", "n", "coverage_fraction", "mae", "rmse"]
                )
                + " |"
                for row in metrics
                if row["scope"] == "pooled" and row["comparison"] == "common_supported"
            )
            lines += [
                "",
                "## Warnings",
                *["- " + warning for warning in warnings],
                "",
                "Full per-series/per-origin metrics and exclusions: report.json and metrics.csv.",
                "No raw Silver reads, model fitting, holdout scoring or promotion was performed.",
            ]
            (publication.stage / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        # Recheck all bound files before publication; do not silently report a moving input.
        for key, path in [
            ("samples", samples),
            ("models", models),
            ("scores", scores),
            ("baseline", baseline),
        ]:
            if (
                input_artifact(path, artifacts[key]["identity"]["kind"])["manifest_sha256"]
                != artifacts[key]["manifest_sha256"]
            ):
                raise ValueError("Input changed while reporting")
        for path, value in zip(resources, resource_values, strict=True):
            if reference(verify(path)) != reference(value):
                raise ValueError("Resource input changed while reporting")
        return publication.publish(
            "report",
            {
                "bindings": bindings,
                "producer": producer(),
                "max_groups": max_groups,
                "resources": [reference(value) for value in resource_values],
            },
            database,
        )
    finally:
        publication.close()
