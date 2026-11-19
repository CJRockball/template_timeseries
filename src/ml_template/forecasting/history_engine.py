"Core generation and SEPARATE scoring; no Silver/Arrow imports in this module."

import json
from collections import Counter

from .history import HistoryIndex, finite, predictions, recipe
from .history_artifacts import Publication, verify_artifact, write_line
from .history_contract import REQUEST_KEYS, RequestIndex, inputs
from .history_metrics import Errors
from .policy import canonical, day


def _predict(
    plan, requests, history_records, configuration, output_root, database, batch_size=5000
):
    configuration = recipe(configuration)
    publication = Publication(output_root)
    index = None
    history = None
    try:
        index = RequestIndex(publication.stage / "requests.sqlite", plan, requests)
        end = max(w["refit_end"] for w in plan["identity"]["windows"])
        history = HistoryIndex(
            publication.stage / "history.sqlite", index.policy, index.temporal, index.holdout, end
        )
        observations = history.add_rows(history_records, batch_size)
        coverage = Counter()
        unsupported = {method: Counter() for method in configuration["methods"]}
        with (
            (publication.stage / "features.jsonl").open("w", encoding="utf-8") as features_file,
            (publication.stage / "predictions.jsonl").open(
                "w", encoding="utf-8"
            ) as predictions_file,
        ):
            for request in index.rows():
                row = history.features(
                    request["series"],
                    request["target_date"],
                    request["origin_date"],
                    request["issued_at"],
                    configuration,
                )
                write_line(features_file, {**request, **row})
                for prediction in predictions(row, configuration):
                    write_line(predictions_file, {**request, **prediction})
                    method = prediction["method"]
                    if prediction["status"] == "ok":
                        coverage[method] += 1
                    else:
                        unsupported[method][prediction["reason"]] += 1
        stats = {
            "requests": index.count,
            "methods": configuration["methods"],
            "history_observations": observations,
            "coverage": {method: coverage[method] for method in configuration["methods"]},
            "unsupported": {
                method: dict(unsupported[method]) for method in configuration["methods"]
            },
            "outcomes_joined": False,
            "holdout_access": False,
        }
        history.close()
        history = None
        index.close()
        index = None
        (publication.stage / "history.sqlite").unlink()
        (publication.stage / "requests.sqlite").unlink()
        return publication.publish("predictions", inputs(plan, configuration), stats, database)
    finally:
        if history is not None:
            history.close()
        if index is not None:
            index.close()
        publication.close()


def _score(plan, requests, prediction_path, outcomes, configuration, output_root, database):
    artifact = verify_artifact(prediction_path, "predictions")
    configuration = recipe(configuration)
    if artifact["identity"]["inputs"] != inputs(plan, configuration):
        raise ValueError("Prediction plan/Silver/recipe/producer binding mismatch")
    publication = Publication(output_root)
    index = None
    try:
        index = RequestIndex(publication.stage / "score.sqlite", plan, requests)
        conn = index.conn
        conn.execute(
            "CREATE TABLE predictions(request_id TEXT,method TEXT,y REAL,status TEXT,reason TEXT,PRIMARY KEY(request_id,method)) WITHOUT ROWID"
        )
        methods = configuration["methods"]
        count = 0
        from pathlib import Path

        with (Path(prediction_path) / "predictions.jsonl").open(encoding="utf-8") as stream, conn:
            for line in stream:
                prediction = json.loads(line)
                if set(prediction) != REQUEST_KEYS | {"method", "y_pred", "status", "reason"}:
                    raise ValueError("Unexpected prediction fields")
                expected = conn.execute(
                    "SELECT row FROM requests WHERE request_id=?", (prediction["request_id"],)
                ).fetchone()
                if expected is None or {k: prediction[k] for k in REQUEST_KEYS} != json.loads(
                    expected[0]
                ):
                    raise ValueError("Unexpected or misaligned prediction")
                method = prediction["method"]
                if method not in methods:
                    raise ValueError("Unexpected baseline method")
                if prediction["status"] == "ok":
                    value = finite(prediction["y_pred"])
                    if prediction["reason"] is not None:
                        raise ValueError("Supported prediction has exclusion reason")
                elif prediction["status"] == "unsupported":
                    value = None
                    if (
                        prediction["y_pred"] is not None
                        or not isinstance(prediction["reason"], str)
                        or (not prediction["reason"])
                    ):
                        raise ValueError("Malformed unsupported prediction")
                else:
                    raise ValueError("Unknown prediction status")
                conn.execute(
                    "INSERT INTO predictions VALUES (?,?,?,?,?)",
                    (
                        prediction["request_id"],
                        method,
                        value,
                        prediction["status"],
                        prediction["reason"],
                    ),
                )
                count += 1
        if count != index.count * len(methods):
            raise ValueError("Missing prediction rows; unsupported requests must be explicit")
        conn.execute("CREATE TABLE outcomes(request_id TEXT PRIMARY KEY,y REAL NOT NULL)")
        t = index.temporal
        columns = {t["observation_id"], *t["series"], t["time"], t["target"]}
        low, high = conn.execute("SELECT MIN(date),MAX(date) FROM requests").fetchone()
        with conn:
            for record in outcomes:
                if set(record) != columns:
                    raise ValueError("Unexpected outcome projection")
                date = day(record[t["time"]]).isoformat()
                if date >= index.holdout or date < low or date > high:
                    raise ValueError("Outcome source returned out-of-bound/holdout date")
                key = canonical([record[c] for c in t["series"]])
                expected = conn.execute(
                    "SELECT request_id,observation_id FROM requests WHERE series=? AND date=?",
                    (key, date),
                ).fetchone()
                if expected is None:
                    continue
                if canonical(record[t["observation_id"]]) != expected[1]:
                    raise ValueError("Outcome observation key mismatch")
                conn.execute(
                    "INSERT INTO outcomes VALUES (?,?)", (expected[0], finite(record[t["target"]]))
                )
        if conn.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0] != index.count:
            raise ValueError("Missing requested outcomes")
        conn.execute(
            "CREATE TEMP TABLE common AS SELECT request_id FROM predictions WHERE status='ok' GROUP BY request_id HAVING COUNT(*)=?",
            (len(methods),),
        )
        origins = [
            row[0] for row in conn.execute("SELECT DISTINCT origin FROM requests ORDER BY origin")
        ]
        errors = {
            (method, comparison, origin): Errors()
            for method in methods
            for comparison in ["own_supported", "common_supported"]
            for origin in [None, *origins]
        }
        reasons = {method: Counter() for method in methods}
        total = {
            None: index.count,
            **dict(conn.execute("SELECT origin,COUNT(*) FROM requests GROUP BY origin")),
        }
        query = "SELECT r.row,p.method,p.y,p.status,p.reason,o.y,c.request_id FROM requests r JOIN predictions p USING(request_id) JOIN outcomes o USING(request_id) LEFT JOIN common c USING(request_id) ORDER BY r.origin,r.series,r.request_id,p.method"
        with (publication.stage / "scores.jsonl").open("w", encoding="utf-8") as stream:
            for raw, method, predicted, status, reason, actual, common in conn.execute(query):
                request = json.loads(raw)
                error = None
                if status == "ok":
                    for comparison in ["own_supported"] + (["common_supported"] if common else []):
                        for origin in [None, request["origin_date"]]:
                            error = errors[method, comparison, origin].add(actual, predicted)
                else:
                    reasons[method][reason] += 1
                write_line(
                    stream,
                    {
                        **request,
                        "method": method,
                        "status": status,
                        "reason": reason,
                        "y_true": actual,
                        "y_pred": predicted,
                        "error": error,
                        "common_supported": common is not None,
                    },
                )
        metrics = []
        for method in methods:
            for comparison in ["own_supported", "common_supported"]:
                for origin in [None, *origins]:
                    result = errors[method, comparison, origin].result()
                    metrics.append(
                        {
                            "method": method,
                            "comparison": comparison,
                            "scope": "pooled" if origin is None else "origin",
                            "origin_date": origin,
                            "total_requests": total[origin],
                            "coverage_fraction": result["n"] / total[origin],
                            **result,
                        }
                    )
        (publication.stage / "metrics.json").write_text(
            json.dumps(
                {"metrics": metrics, "unsupported": {m: dict(reasons[m]) for m in methods}},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        common = conn.execute("SELECT COUNT(*) FROM common").fetchone()[0]
        stats = {
            "requests": index.count,
            "methods": methods,
            "common_supported": common,
            "outcomes_joined": True,
            "outcome_dates": {"min": low, "max": high},
            "holdout_access": False,
        }
        index.close()
        index = None
        (publication.stage / "score.sqlite").unlink()
        binding = {
            **inputs(plan, configuration),
            "predictions_artifact_id": artifact["artifact_id"],
            "predictions_manifest_sha256": artifact["manifest_sha256"],
        }
        return publication.publish("scores", binding, stats, database)
    finally:
        if index is not None:
            index.close()
        publication.close()
