"""Protocol fixtures only; opaque test weights are NOT native XGBoost models."""

import json
import sqlite3
import sys
from pathlib import Path

import pytest

from ml_template.forecasting.acceptance_cli import main
from ml_template.forecasting.acceptance_store import (
    canonical,
    digest,
    file_hash,
    input_artifact,
    verify,
)
from ml_template.forecasting.backtest_acceptance import build
from ml_template.forecasting.backtest_resources import measure


def seal(directory, kind, inputs, content):
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in content.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, bytes):
            path.write_bytes(value)
        elif name.endswith(".jsonl"):
            path.write_text("".join(canonical(row) + "\n" for row in value))
        else:
            path.write_text(json.dumps(value))
    inventory = {
        p.relative_to(directory).as_posix(): {"bytes": p.stat().st_size, "sha256": file_hash(p)}
        for p in directory.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    }
    identity = {"format": "temporal-xgb-v1", "kind": kind, "inputs": inputs, "files": inventory}
    manifest = {"format": "temporal-xgb-v1", "identity": identity}
    prefix = "tsxgb_"
    if kind == "predictions":
        identity.update({"format": "temporal-history-v1", "stats": {"fixture": True}})
        manifest.update(
            {
                "format": "temporal-history-v1",
                "kind": kind,
                "files": inventory,
                "stats": identity["stats"],
            }
        )
        prefix = "history_"
    manifest["artifact_id"] = prefix + digest(identity)
    manifest["manifest_sha256"] = digest(manifest)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return manifest


@pytest.fixture
def bundle(tmp_path):
    origins = ["2020-02-10", "2020-02-17"]
    windows = [
        {
            "origin_date": origin,
            "issued_at": origin + "T23:59:59.999999Z",
            "target_date": target,
            "refit_end": origin,
        }
        for origin, target in zip(origins, ["2020-02-11", "2020-02-18"], strict=True)
    ]
    context = {
        "plan_id": "origin_fixture",
        "plan_manifest_sha256": "f" * 64,
        "silver": {"silver_id": "fixture"},
        "recipe": {"methods": ["last_available", "seasonal_naive_7"]},
        "producer": {"type": "protocol_fixture"},
        "windows": windows,
        "holdout": {"start": "2020-02-22", "end": "2020-03-20"},
        "sample_options": {"features": ["lag_7"], "max_rows_per_partition": 500},
    }
    requests = []
    predictions = []
    scored = []
    actuals = []
    for window in windows:
        for series in ["a", "b"]:
            request = {
                "plan_id": context["plan_id"],
                "purpose": "development",
                "series": [series],
                "origin_date": window["origin_date"],
                "issued_at": window["issued_at"],
                "target_date": window["target_date"],
                "horizon": 1,
                "observation_id": series + window["target_date"],
            }
            request["request_id"] = "request_" + digest(request)
            requests.append(request)
            actual = 10.25 if series == "a" else 1000.75
            actuals.append(actual)
            for method, error in [("xgb", 1.0), ("last_available", 2.0), ("seasonal_naive_7", 3.0)]:
                prediction = {
                    **request,
                    "method": method,
                    "status": "ok",
                    "reason": None,
                    "y_pred": actual + error,
                }
                predictions.append(prediction)
                scored.append(
                    {**prediction, "y_true": actual, "error": error, "common_supported": True}
                )
    paths = {key: tmp_path / key for key in ["samples", "models", "scores", "baseline"]}
    sample = seal(
        paths["samples"],
        "samples",
        context,
        {
            "samples.jsonl": [],
            "requests.jsonl": [
                {"request": request, "x": [1.0], "missing_features": []} for request in requests
            ],
            "eligibility.jsonl": [],
            "report.json": {
                "requests": 4,
                "counts": [
                    {
                        "origin": origin,
                        "partition": phase,
                        "date_candidates": 40,
                        "label_eligible": 38,
                        "feature_complete": 36,
                    }
                    for origin in origins
                    for phase in ["fit", "stopping", "refit"]
                ],
            },
        },
    )
    model_inputs = {
        "context": context,
        "samples_id": sample["artifact_id"],
        "samples_manifest_sha256": sample["manifest_sha256"],
        "producer": context["producer"],
        "parameters": {"n_estimators": 20},
    }
    model = seal(
        paths["models"],
        "models",
        model_inputs,
        {
            "models/0000.ubj": b"OPAQUE TEST FIXTURE",
            "models/0001.ubj": b"OPAQUE TEST FIXTURE",
            "predictions.jsonl": [row for row in predictions if row["method"] == "xgb"],
            "report.json": {
                "holdout_scored": False,
                "models": [
                    {
                        "origin": origin,
                        "selected_rounds": 3,
                        "refit_rows": 36,
                        "requests": 2,
                        "supported": 2,
                        "feature_order": ["lag_7"],
                        "model_file": "models/" + str(index).zfill(4) + ".ubj",
                    }
                    for index, origin in enumerate(origins)
                ],
            },
        },
    )
    baseline = seal(
        paths["baseline"],
        "predictions",
        context,
        {
            "features.jsonl": [],
            "predictions.jsonl": [row for row in predictions if row["method"] != "xgb"],
        },
    )
    metrics = []
    for method, error in [("xgb", 1.0), ("last_available", 2.0), ("seasonal_naive_7", 3.0)]:
        for comparison in ["own_supported", "common_supported"]:
            for origin in [None, *origins]:
                n = 4 if origin is None else 2
                metrics.append(
                    {
                        "method": method,
                        "comparison": comparison,
                        "scope": "pooled" if origin is None else "origin",
                        "origin_date": origin,
                        "n": n,
                        "total_requests": n,
                        "coverage_fraction": 1.0,
                        "mae": error,
                        "rmse": error,
                    }
                )
    score_inputs = {
        "models_id": model["artifact_id"],
        "models_manifest_sha256": model["manifest_sha256"],
        "baseline_id": baseline["artifact_id"],
        "baseline_manifest_sha256": baseline["manifest_sha256"],
        "producer": context["producer"],
        "plan_id": context["plan_id"],
        "plan_manifest_sha256": context["plan_manifest_sha256"],
        "holdout_scored": False,
    }
    seal(
        paths["scores"],
        "scores",
        score_inputs,
        {
            "scores.jsonl": scored,
            "metrics.json": {
                "metrics": metrics,
                "unsupported": {method: {} for method in context["recipe"]["methods"] + ["xgb"]},
            },
        },
    )
    database = tmp_path / "project.sqlite"
    database.touch()
    return paths, database, tmp_path


def run(bundle, **kwargs):
    paths, database, tmp = bundle
    return build(**paths, output=tmp / "reports", database=database, **kwargs)


def reseal_scores(bundle, mutate):
    paths, _, _ = bundle
    manifest = input_artifact(paths["scores"], "scores")
    content = {
        "scores.jsonl": [
            json.loads(line) for line in (paths["scores"] / "scores.jsonl").read_text().splitlines()
        ],
        "metrics.json": json.loads((paths["scores"] / "metrics.json").read_text()),
    }
    mutate(content)
    return seal(paths["scores"], "scores", manifest["identity"]["inputs"], content)


def test_report_reuse_input_preservation(bundle):
    paths, database, _ = bundle
    before = {
        str(path): path.read_bytes()
        for directory in paths.values()
        for path in directory.rglob("*")
        if path.is_file()
    }
    result = run(bundle)
    report = json.loads((Path(result["artifact_path"]) / "report.json").read_text())
    assert (report["requests"], report["origins"], report["series"]) == (4, 2, 2)
    assert len(report["metrics"]) == 30
    assert report["decision"] == "descriptive_only_no_promotion"
    assert run(bundle)["reused"]
    assert all(Path(path).read_bytes() == value for path, value in before.items())
    with sqlite3.connect(database) as conn:
        assert conn.execute("SELECT COUNT(*) FROM temporal_acceptance_artifacts").fetchone()[0] == 1


def test_series_metrics(bundle):
    result = run(bundle)
    metrics = json.loads((Path(result["artifact_path"]) / "report.json").read_text())["metrics"]
    assert {row["key"] for row in metrics if row["scope"] == "series"} == {'["a"]', '["b"]'}
    assert all(row["mae"] == 1 for row in metrics if row["method"] == "xgb")


def test_manifest_tamper(bundle):
    (bundle[0]["models"] / "models/0000.ubj").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        run(bundle)


def test_wrong_score_prediction(bundle):
    reseal_scores(bundle, lambda content: content["scores.jsonl"][0].update(y_pred=500.0))
    with pytest.raises(ValueError, match="disagreement"):
        run(bundle)


def test_wrong_common_flag(bundle):
    reseal_scores(bundle, lambda content: content["scores.jsonl"][0].update(common_supported=False))
    with pytest.raises(ValueError, match="Common"):
        run(bundle)


def test_wrong_residual(bundle):
    reseal_scores(bundle, lambda content: content["scores.jsonl"][0].update(error=-1.0))
    with pytest.raises(ValueError, match="Residual"):
        run(bundle)


def test_inconsistent_truth(bundle):
    reseal_scores(bundle, lambda content: content["scores.jsonl"][1].update(y_true=0.0))
    with pytest.raises(ValueError, match="outcomes"):
        run(bundle)


def test_missing_score(bundle):
    reseal_scores(bundle, lambda content: content["scores.jsonl"].pop())
    with pytest.raises(ValueError, match="Missing score"):
        run(bundle)


def test_duplicate_score(bundle):
    reseal_scores(
        bundle, lambda content: content["scores.jsonl"].append(content["scores.jsonl"][0])
    )
    with pytest.raises(sqlite3.IntegrityError):
        run(bundle)


def test_cached_metrics_disagreement(bundle):
    reseal_scores(bundle, lambda content: content["metrics.json"]["metrics"][0].update(mae=99))
    with pytest.raises(ValueError, match="Cached metric"):
        run(bundle)


def test_holdout_flag(bundle):
    paths = bundle[0]
    manifest = input_artifact(paths["scores"], "scores")
    inputs = manifest["identity"]["inputs"]
    inputs["holdout_scored"] = True
    content = {
        "scores.jsonl": [
            json.loads(line) for line in (paths["scores"] / "scores.jsonl").read_text().splitlines()
        ],
        "metrics.json": json.loads((paths["scores"] / "metrics.json").read_text()),
    }
    seal(paths["scores"], "scores", inputs, content)
    with pytest.raises(ValueError, match="holdout"):
        run(bundle)


def test_group_budget(bundle):
    with pytest.raises(ValueError, match="budget"):
        run(bundle, max_groups=1)


def test_report_verify_tamper(bundle):
    result = run(bundle)
    (Path(result["artifact_path"]) / "metrics.csv").write_text("bad")
    with pytest.raises(ValueError, match="integrity"):
        verify(result["artifact_path"])


def test_cli_report(bundle, capsys):
    paths, database, tmp = bundle
    args = ["report", "--database", str(database), "--output-root", str(tmp / "cli")]
    for key, path in paths.items():
        args += ["--" + key, str(path)]
    main(args)
    assert "acceptance_" in capsys.readouterr().out


def test_cli_help():
    with pytest.raises(SystemExit) as result:
        main(["--help"])
    assert result.value.code == 0


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux measurement wrapper")
def test_resource_measurement(bundle):
    _, database, tmp = bundle
    value = measure(
        [sys.executable, "-c", "print('measurement fixture')"],
        "fixture",
        tmp / "resources",
        database,
        [tmp / "reports"],
    )
    result = json.loads((Path(value["artifact_path"]) / "resource.json").read_text())
    assert result["exit_code"] == 0 and result["elapsed_seconds"] >= 0
    assert result["kernel_reported_child_max_rss_kib"] > 0
    report = run(bundle, resources=[value["artifact_path"]])
    assert verify(report["artifact_path"])["identity"]["kind"] == "report"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux measurement wrapper")
def test_nonzero_command_recorded(bundle):
    _, database, tmp = bundle
    value = measure(
        [sys.executable, "-c", "raise SystemExit(7)"], "failure", tmp / "resources", database
    )
    assert value["command_exit_code"] == 7


def test_extra_cached_metric_rejected(bundle):
    reseal_scores(
        bundle,
        lambda content: content["metrics.json"]["metrics"].append(
            content["metrics.json"]["metrics"][0]
        ),
    )
    with pytest.raises(ValueError, match="inventory"):
        run(bundle)


def test_nonfinite_truth_rejected(bundle):
    # This remains valid JSON; finite arithmetic must reject overflow, too.
    reseal_scores(
        bundle, lambda content: content["scores.jsonl"][0].update(y_true=1e308, error=-1e308)
    )
    with pytest.raises(ValueError):
        run(bundle)


def test_unknown_prediction_method(bundle):
    reseal_scores(bundle, lambda content: content["scores.jsonl"][0].update(method="unknown"))
    with pytest.raises(ValueError, match="Unexpected"):
        run(bundle)


def test_input_symlink_rejected(bundle):
    paths, _, tmp = bundle
    alias = tmp / "alias"
    alias.symlink_to(paths["models"], target_is_directory=True)
    paths["models"] = alias
    with pytest.raises(ValueError, match="Symlink"):
        run(bundle)


def test_python311_syntax():
    import ast
    import token
    import tokenize
    from io import StringIO

    import ml_template.forecasting.acceptance_cli as cli

    directory = Path(cli.__file__).parent
    for name in [
        "acceptance_cli.py",
        "acceptance_store.py",
        "backtest_acceptance.py",
        "backtest_resources.py",
    ]:
        text = (directory / name).read_text()
        ast.parse(text, feature_version=(3, 11))
        # New modules deliberately contain no f-strings: avoid 3.12 quote rewrites.
        if hasattr(token, "FSTRING_START"):
            assert all(
                item.type != token.FSTRING_START
                for item in tokenize.generate_tokens(StringIO(text).readline)
            )


def test_all_common_unsupported_yields_null_metrics(bundle):
    paths, _, _ = bundle
    baseline = input_artifact(paths["baseline"], "predictions")
    forecasts = [
        json.loads(line)
        for line in (paths["baseline"] / "predictions.jsonl").read_text().splitlines()
    ]
    for row in forecasts:
        row.update(status="unsupported", reason="fixture_missing", y_pred=None)
    new_base = seal(
        paths["baseline"],
        "predictions",
        baseline["identity"]["inputs"],
        {"features.jsonl": [], "predictions.jsonl": forecasts},
    )
    score = input_artifact(paths["scores"], "scores")
    content = [
        json.loads(line) for line in (paths["scores"] / "scores.jsonl").read_text().splitlines()
    ]
    for row in content:
        row["common_supported"] = False
        if row["method"] != "xgb":
            row.update(status="unsupported", reason="fixture_missing", y_pred=None, error=None)
    cached = json.loads((paths["scores"] / "metrics.json").read_text())
    for row in cached["metrics"]:
        if row["method"] != "xgb" or row["comparison"] == "common_supported":
            row.update(n=0, coverage_fraction=0.0, mae=None, rmse=None)
    cached["unsupported"] = {
        "xgb": {},
        "last_available": {"fixture_missing": 4},
        "seasonal_naive_7": {"fixture_missing": 4},
    }
    inputs = score["identity"]["inputs"]
    inputs.update(
        baseline_id=new_base["artifact_id"], baseline_manifest_sha256=new_base["manifest_sha256"]
    )
    seal(paths["scores"], "scores", inputs, {"scores.jsonl": content, "metrics.json": cached})
    result = run(bundle)
    report = json.loads((Path(result["artifact_path"]) / "report.json").read_text())
    assert all(
        row["n"] == 0 and row["mae"] is None
        for row in report["metrics"]
        if row["comparison"] == "common_supported"
    )


def test_no_report_inside_immutable_inputs(bundle):
    paths, database, _ = bundle
    with pytest.raises(ValueError, match="outside immutable"):
        build(**paths, output=paths["models"] / "report", database=database)
    assert not (paths["models"] / "report").exists()


def test_resource_cli_nonzero_exit(bundle):
    _, database, tmp = bundle
    if not sys.platform.startswith("linux"):
        pytest.skip("Linux wrapper")
    with pytest.raises(SystemExit) as code:
        main(
            [
                "measure",
                "--label",
                "error",
                "--database",
                str(database),
                "--output-root",
                str(tmp / "resources"),
                "--",
                sys.executable,
                "-c",
                "raise SystemExit(9)",
            ]
        )
    assert code.value.code == 9
