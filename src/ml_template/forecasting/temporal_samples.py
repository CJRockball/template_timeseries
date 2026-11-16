"Materialise plan predicates and replay each historical sample's own origin."

import json
import math
from collections import Counter

from .learned_artifacts import Publication, write_line


def number(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or (not math.isfinite(float(value)))
    ):
        raise ValueError("Finite numeric value required")
    return float(value)


def options(value):
    required = {"features", "max_rows_per_partition"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Sample options require features and max_rows_per_partition")
    features = value["features"]
    if (
        not isinstance(features, list)
        or not features
        or any(not isinstance(x, str) for x in features)
        or (len(set(features)) != len(features))
    ):
        raise ValueError("Explicit ordered unique feature names required")
    cap = value["max_rows_per_partition"]
    if type(cap) is not int or cap < 1:
        raise ValueError("Positive partition cap required")
    return value


def vector(row, features):
    if not set(features) <= set(row["features"]):
        raise ValueError("Unknown selected history feature")
    missing = [f for f in features if row["features"][f] is None]
    return (None, missing) if missing else ([number(row["features"][f]) for f in features], [])


def materialize(plan, requests, history, recipe, configuration, producer, output_root, database):
    """requests is a validated RequestIndex; history is the shared HistoryIndex."""
    configuration = options(configuration)
    features = configuration["features"]
    cap = configuration["max_rows_per_partition"]
    publication = Publication(output_root)
    counts = []
    try:
        conn = history.conn
        conn.execute("CREATE TEMP TABLE IF NOT EXISTS active_series(series TEXT PRIMARY KEY)")
        with (
            (publication.stage / "samples.jsonl").open("w", encoding="utf-8") as samples,
            (publication.stage / "requests.jsonl").open("w", encoding="utf-8") as forecasts,
            (publication.stage / "eligibility.jsonl").open("w", encoding="utf-8") as audit,
        ):
            for window in plan["identity"]["windows"]:
                origin = window["origin_date"]
                if (
                    window["refit_end"] >= plan["identity"]["holdout"]["start"]
                    or window["target_date"] >= plan["identity"]["holdout"]["start"]
                ):
                    raise ValueError("Holdout materialisation prohibited")
                conn.execute("DELETE FROM active_series")
                for (key,) in requests.conn.execute(
                    "SELECT DISTINCT series FROM requests WHERE origin=? ORDER BY series", (origin,)
                ):
                    conn.execute("INSERT INTO active_series VALUES (?)", (key,))
                phases = [
                    ("fit", window["fit_start"], window["fit_end"], window["fit_available_by"]),
                    (
                        "stopping",
                        window["stopping_start"],
                        window["stopping_end"],
                        window["issued_at"],
                    ),
                    ("refit", window["refit_start"], window["refit_end"], window["issued_at"]),
                ]
                for phase, start, end, available in phases:
                    base = " FROM history h JOIN active_series a USING(series) WHERE h.day BETWEEN ? AND ?"
                    n = conn.execute("SELECT COUNT(*)" + base, (start, end)).fetchone()[0]
                    if n > cap:
                        raise ValueError(
                            f"{origin}/{phase}: {n} candidates exceed explicit cap {cap}; no sampling performed"
                        )
                    eligible = Counter()
                    complete = Counter()
                    for key, date, identifier, target in conn.execute(
                        "SELECT h.series,h.day,h.id,h.y"
                        + base
                        + " AND h.available<=? ORDER BY h.series,h.day",
                        (start, end, available),
                    ):
                        eligible[key] += 1
                        values = history.for_sample(json.loads(key), date, recipe)
                        x, missing = vector(values, features)
                        write_line(
                            audit,
                            {
                                "origin": origin,
                                "partition": phase,
                                "series": json.loads(key),
                                "target_date": date,
                                "complete": not missing,
                                "missing_features": missing,
                            },
                        )
                        if missing:
                            continue
                        complete[key] += 1
                        write_line(
                            samples,
                            {
                                "origin": origin,
                                "partition": phase,
                                "series": json.loads(key),
                                "target_date": date,
                                "observation_id": json.loads(identifier),
                                "x": x,
                                "y": number(target),
                            },
                        )
                    for (key,) in conn.execute("SELECT series FROM active_series ORDER BY series"):
                        minimum = (
                            plan["identity"]["policy"]["stopping_periods"]
                            if phase == "stopping"
                            else plan["identity"]["policy"]["min_train_observations"]
                        )
                        if complete[key] < minimum:
                            raise ValueError(
                                f"{origin}/{phase}/{key}: {complete[key]} feature-complete rows, minimum {minimum}; labels eligible {eligible[key]}"
                            )
                    counts.append(
                        {
                            "origin": origin,
                            "partition": phase,
                            "date_candidates": n,
                            "label_eligible": sum(eligible.values()),
                            "feature_complete": sum(complete.values()),
                        }
                    )
                supported = 0
                for emitted, (raw,) in enumerate(
                    requests.conn.execute(
                        "SELECT row FROM requests WHERE origin=? ORDER BY series,request_id",
                        (origin,),
                    ),
                    start=1,
                ):
                    request = json.loads(raw)
                    x, missing = vector(
                        history.for_sample(
                            request["series"],
                            request["target_date"],
                            recipe,
                        ),
                        features,
                    )
                    write_line(
                        forecasts,
                        {
                            "request": request,
                            "x": x,
                            "missing_features": missing,
                        },
                    )
                    supported += x is not None

                    if emitted > cap:
                        raise ValueError("Forecast request count exceeds explicit cap")

                if not supported:
                    raise ValueError("No feature-complete requests at origin " + origin)
        (publication.stage / "report.json").write_text(
            json.dumps(
                {
                    "counts": counts,
                    "requests": requests.count,
                    "missing_policy": "complete_selected_features_only",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        inputs = {
            "plan_id": plan["plan_id"],
            "plan_manifest_sha256": plan["manifest_sha256"],
            "silver": plan["identity"]["silver"],
            "recipe": recipe,
            "sample_options": configuration,
            "windows": plan["identity"]["windows"],
            "holdout": plan["identity"]["holdout"],
            "producer": producer,
        }
        return publication.publish("samples", inputs, database)
    finally:
        publication.close()
