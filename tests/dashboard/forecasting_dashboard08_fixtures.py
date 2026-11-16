"""Synthetic sealed protocols; no trained model evidence."""

import json
import math
import sqlite3

from ml_template.forecasting.acceptance_store import canonical, digest, file_hash


def seal(root, kind, inputs, content, acceptance=False):
    root.mkdir(parents=True, exist_ok=True)
    for name, value in content.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(canonical(row) + "\n" for row in value)
            if name.endswith(".jsonl")
            else json.dumps(value)
        )
    inventory = {
        name: {"bytes": (root / name).stat().st_size, "sha256": file_hash(root / name)}
        for name in content
    }
    format_name = "temporal-acceptance-v1" if acceptance else "temporal-xgb-v1"
    identity = {"format": format_name, "kind": kind, "inputs": inputs, "files": inventory}
    value = {
        "format": format_name,
        "identity": identity,
        "artifact_id": ("acceptance_" if acceptance else "tsxgb_") + digest(identity),
    }
    value["manifest_sha256"] = digest(value)
    (root / "manifest.json").write_text(json.dumps(value))
    return value


def fixture(root, unsupported=False, zero_baseline=False):
    database = root / "project.sqlite"
    methods = ["last_available", "seasonal_naive_7", "xgb"]
    rows = []
    for index, (origin, target) in enumerate(
        [("2020-02-10", "2020-02-11"), ("2020-02-17", "2020-02-18")]
    ):
        for series, actual in [("a", 10.25), ("b", 1000.75)]:
            request = {
                "plan_id": "origin_fixture",
                "purpose": "development",
                "series": [series],
                "origin_date": origin,
                "issued_at": origin + "T23:59:59.999999Z",
                "target_date": target,
                "horizon": 1,
                "observation_id": series + target,
            }
            request["request_id"] = "request_" + digest(request)
            common = not (unsupported and index == 0 and series == "a")
            for method in methods:
                error = {
                    "last_available": 2.0,
                    "seasonal_naive_7": 0.0 if zero_baseline else 4.0,
                    "xgb": float(1 + 2 * index),
                }[method]
                ok = not (unsupported and not common and method == "seasonal_naive_7")
                rows.append(
                    {
                        **request,
                        "method": method,
                        "status": "ok" if ok else "unsupported",
                        "reason": None if ok else "fixture_missing",
                        "y_pred": actual + error if ok else None,
                        "y_true": actual,
                        "error": error if ok else None,
                        "common_supported": common,
                    }
                )
    model_path = root / "models"
    model = seal(
        model_path,
        "models",
        {"parameters": {"n_estimators": 10, "nested_fixture": [3, 4]}, "seed": 42},
        {},
    )
    baseline = {"artifact_id": "history_fixture", "manifest_sha256": "b" * 64}
    inputs = {
        "plan_id": "origin_fixture",
        "plan_manifest_sha256": "p" * 64,
        "holdout_scored": False,
        "models_id": model["artifact_id"],
        "models_manifest_sha256": model["manifest_sha256"],
        "baseline_id": baseline["artifact_id"],
        "baseline_manifest_sha256": baseline["manifest_sha256"],
    }
    score_path = root / "scores"
    score = seal(score_path, "scores", inputs, {"scores.jsonl": rows, "metrics.json": {}})
    refs = {
        "models": model,
        "scores": score,
        "baseline": baseline,
        "samples": {"artifact_id": "samples_fixture", "manifest_sha256": "s" * 64},
    }
    bindings = {
        label: {key: value[key] for key in ["artifact_id", "manifest_sha256"]}
        for label, value in refs.items()
    }
    metrics = []
    for scope, keys in [
        ("pooled", [""]),
        ("origin", ["2020-02-10", "2020-02-17"]),
        ("series", ['["a"]', '["b"]']),
    ]:
        for key in keys:
            selected = [
                r
                for r in rows
                if scope == "pooled"
                or (r["origin_date"] == key if scope == "origin" else canonical(r["series"]) == key)
            ]
            total = len({r["request_id"] for r in selected})
            for method in methods:
                for support in ["own_supported", "common_supported"]:
                    errors = [
                        r["error"]
                        for r in selected
                        if r["method"] == method
                        and r["status"] == "ok"
                        and (support == "own_supported" or r["common_supported"])
                    ]
                    n = len(errors)
                    metrics.append(
                        {
                            "scope": scope,
                            "key": key,
                            "method": method,
                            "comparison": support,
                            "n": n,
                            "total_requests": total,
                            "coverage_fraction": n / total,
                            "mae": sum(abs(e) for e in errors) / n if n else None,
                            "rmse": math.sqrt(sum(e * e for e in errors) / n) if n else None,
                        }
                    )
    report = {
        "bindings": bindings,
        "plan_id": "origin_fixture",
        "plan_manifest_sha256": "p" * 64,
        "silver": {"id": "fixture"},
        "holdout": {"start": "2020-02-22", "end": "2020-03-20"},
        "holdout_scored": False,
        "decision": "descriptive_only_no_promotion",
        "requests": 4,
        "series": 2,
        "origins": 2,
        "metrics": metrics,
        "rounds": [
            {"origin": "2020-02-10", "selected_rounds": 3},
            {"origin": "2020-02-17", "selected_rounds": 4},
        ],
        "unsupported": {},
        "sample_eligibility": [],
        "warnings": ["Synthetic fixture"],
        "resources": [],
    }
    report_path = root / "reports"
    saved = seal(
        report_path,
        "report",
        {"bindings": bindings},
        {"report.json": report, "summary.md": "fixture", "metrics.csv": "fixture"},
        True,
    )
    with sqlite3.connect(database) as conn:
        for table in ["temporal_acceptance_artifacts", "temporal_xgb_artifacts"]:
            conn.execute(
                "CREATE TABLE "
                + table
                + "(artifact_id TEXT PRIMARY KEY,kind TEXT,manifest_sha256 TEXT,path TEXT)"
            )
        conn.execute(
            "INSERT INTO temporal_acceptance_artifacts VALUES (?,?,?,?)",
            (saved["artifact_id"], "report", saved["manifest_sha256"], str(report_path.resolve())),
        )
        conn.executemany(
            "INSERT INTO temporal_xgb_artifacts VALUES (?,?,?,?)",
            [
                (
                    model["artifact_id"],
                    "models",
                    model["manifest_sha256"],
                    str(model_path.resolve()),
                ),
                (
                    score["artifact_id"],
                    "scores",
                    score["manifest_sha256"],
                    str(score_path.resolve()),
                ),
            ],
        )
    return {
        "database": database,
        "identifier": saved["artifact_id"],
        "report_path": report_path,
        "score_path": score_path,
        "model_path": model_path,
    }
