"Shared strict development comparison; baseline artifacts remain read-only."

import json
import math
from collections import defaultdict
from datetime import date
from itertools import chain

from .learned_artifacts import Publication, canonical, rows, verify, write_line
from .temporal_samples import number


def compare(
    plan, requests, model_path, baseline, baseline_rows, outcomes, producer, output_root, database
):
    model = verify(model_path, "models")
    context = model["identity"]["inputs"]["context"]
    if (
        context["producer"] != producer
        or context["plan_id"] != plan["plan_id"]
        or context["plan_manifest_sha256"] != plan["manifest_sha256"]
        or (context["silver"] != plan["identity"]["silver"])
    ):
        raise ValueError("Learned producer/plan/Silver binding mismatch")
    source = baseline["identity"]["inputs"]
    for key in ["plan_id", "plan_manifest_sha256", "silver", "recipe"]:
        if source[key] != context[key]:
            raise ValueError("Baseline binding mismatch: " + key)
    methods = sorted([*context["recipe"]["methods"], "xgb"])
    conn = requests.conn
    publication = Publication(output_root)
    try:
        conn.execute(
            "CREATE TABLE compared_predictions(request_id TEXT,method TEXT,status TEXT,reason TEXT,y REAL,PRIMARY KEY(request_id,method)) WITHOUT ROWID"
        )
        count = 0
        with conn:
            for prediction in chain(rows(model_path, "predictions.jsonl"), baseline_rows):
                expected = conn.execute(
                    "SELECT row FROM requests WHERE request_id=?", (prediction.get("request_id"),)
                ).fetchone()
                if expected is None:
                    raise ValueError("Unexpected prediction request")
                request = json.loads(expected[0])
                if (
                    set(prediction) != set(request) | {"method", "status", "reason", "y_pred"}
                    or any((prediction[k] != v for k, v in request.items()))
                    or prediction["method"] not in methods
                ):
                    raise ValueError("Misaligned prediction payload")
                if prediction["status"] == "ok":
                    value = number(prediction["y_pred"])
                    if prediction["reason"] is not None:
                        raise ValueError("Unexpected supported-row reason")
                elif prediction["status"] == "unsupported":
                    value = None
                    if (
                        prediction["y_pred"] is not None
                        or not isinstance(prediction["reason"], str)
                        or (not prediction["reason"])
                    ):
                        raise ValueError("Malformed unsupported forecast")
                else:
                    raise ValueError("Invalid prediction status")
                conn.execute(
                    "INSERT INTO compared_predictions VALUES (?,?,?,?,?)",
                    (
                        request["request_id"],
                        prediction["method"],
                        prediction["status"],
                        prediction["reason"],
                        value,
                    ),
                )
                count += 1
        if count != requests.count * len(methods):
            raise ValueError("Missing prediction rows; unsupported rows must be explicit")
        conn.execute("CREATE TABLE compared_outcomes(request_id TEXT PRIMARY KEY,y REAL NOT NULL)")
        low, high = conn.execute("SELECT MIN(date),MAX(date) FROM requests").fetchone()
        t = plan["identity"]["silver"]["temporal_schema"]
        expected_columns = {t["observation_id"], *t["series"], t["time"], t["target"]}
        with conn:
            for record in outcomes:
                if set(record) != expected_columns:
                    raise ValueError("Unexpected outcome projection")
                day = record[t["time"]]
                day = day.isoformat() if isinstance(day, date) else day
                if (
                    date.fromisoformat(day).isoformat() != day
                    or day < low
                    or day > high
                    or (day >= plan["identity"]["holdout"]["start"])
                ):
                    raise ValueError("Outcome outside development bounds")
                match = conn.execute(
                    "SELECT request_id,observation_id FROM requests WHERE series=? AND date=?",
                    (canonical([record[c] for c in t["series"]]), day),
                ).fetchone()
                if match:
                    if match[1] != canonical(record[t["observation_id"]]):
                        raise ValueError("Outcome key mismatch")
                    conn.execute(
                        "INSERT INTO compared_outcomes VALUES (?,?)",
                        (match[0], number(record[t["target"]])),
                    )
        if conn.execute("SELECT COUNT(*) FROM compared_outcomes").fetchone()[0] != requests.count:
            raise ValueError("Missing requested outcomes")
        conn.execute(
            "CREATE TEMP TABLE compared_common AS SELECT request_id FROM compared_predictions WHERE status='ok' GROUP BY request_id HAVING COUNT(*)=?",
            (len(methods),),
        )
        totals = {
            None: requests.count,
            **dict(conn.execute("SELECT origin,COUNT(*) FROM requests GROUP BY origin")),
        }
        errors = defaultdict(lambda: [0, 0.0, 0.0])
        reasons = {method: defaultdict(int) for method in methods}
        query = "SELECT r.row,p.method,p.status,p.reason,p.y,o.y,c.request_id FROM requests r JOIN compared_predictions p USING(request_id) JOIN compared_outcomes o USING(request_id) LEFT JOIN compared_common c USING(request_id) ORDER BY r.origin,r.series,p.method"
        with (publication.stage / "scores.jsonl").open("w", encoding="utf-8") as stream:
            for raw, method, status, reason, predicted, actual, common in conn.execute(query):
                request = json.loads(raw)
                error = None
                if status == "ok":
                    error = number(predicted - actual)
                    for scope in [None, request["origin_date"]]:
                        for group in ["own_supported"] + (["common_supported"] if common else []):
                            accumulator = errors[method, group, scope]
                            accumulator[0] += 1
                            accumulator[1] = number(math.fsum((accumulator[1], abs(error))))
                            accumulator[2] = number(
                                math.fsum((accumulator[2], number(error * error)))
                            )
                else:
                    reasons[method][reason] += 1
                write_line(
                    stream,
                    {
                        **request,
                        "method": method,
                        "status": status,
                        "reason": reason,
                        "y_pred": predicted,
                        "y_true": actual,
                        "error": error,
                        "common_supported": common is not None,
                    },
                )
        metrics = []
        for method in methods:
            for group in ["own_supported", "common_supported"]:
                for scope, total in totals.items():
                    n, absolute, squared = errors[method, group, scope]
                    metrics.append(
                        {
                            "method": method,
                            "comparison": group,
                            "origin_date": scope,
                            "scope": "pooled" if scope is None else "origin",
                            "n": n,
                            "total_requests": total,
                            "coverage_fraction": n / total,
                            "mae": absolute / n if n else None,
                            "rmse": math.sqrt(squared / n) if n else None,
                        }
                    )
        (publication.stage / "metrics.json").write_text(
            json.dumps(
                {"metrics": metrics, "unsupported": {k: dict(v) for k, v in reasons.items()}},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        binding = {
            "plan_id": plan["plan_id"],
            "plan_manifest_sha256": plan["manifest_sha256"],
            "models_id": model["artifact_id"],
            "models_manifest_sha256": model["manifest_sha256"],
            "baseline_id": baseline["artifact_id"],
            "baseline_manifest_sha256": baseline["manifest_sha256"],
            "producer": producer,
            "holdout_scored": False,
        }
        return publication.publish("scores", binding, database)
    finally:
        publication.close()
