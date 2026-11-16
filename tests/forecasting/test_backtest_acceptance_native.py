"""Mandatory native hardening: real Silver, distinct series and late labels."""

import csv
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import xgboost

from ml_template.forecasting.backtest_acceptance import build as build_report
from ml_template.forecasting.history_io import predict
from ml_template.forecasting.io import build_from_silver
from ml_template.forecasting.temporal_xgb_io import build_samples, fit, score
from ml_template.temporal_stream.silver import build as build_silver
from ml_template.temporal_stream.store import ingest_csv


def pipeline(root, batch=7, file_rows=13):
    assert xgboost.__version__ == "2.1.1"
    root.mkdir(parents=True)
    raw = root / "raw.csv"
    columns = {
        "id": "string",
        "series": "string",
        "date": "date",
        "y": "float",
        "available": "utc_timestamp",
    }
    with raw.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for series, offset, step in [("a", 10.25, 1), ("b", 1000.75, 2)]:
            for index in range(80):
                day = (date(2020, 1, 1) + timedelta(days=index)).isoformat()
                available = (
                    "2020-02-09T00:00:00Z"
                    if series == "a" and day == "2020-01-10"
                    else day + "T00:00:00Z"
                )
                writer.writerow(
                    [series + str(index), series, day, offset + step * index, available]
                )
    temporal = {
        "series": ["series"],
        "time": "date",
        "target": "y",
        "frequency": "D",
        "gaps": "error",
    }
    database = root / "project.sqlite"
    ingest_csv(
        database, "fixture", "v1", raw, {"columns": columns, "key": ["id"], "temporal": temporal}
    )
    schema = {
        "columns": columns,
        "observation_id": "id",
        **temporal,
        "features": [],
        "known_future": [],
        "available_at": "available",
        "target_delay_periods": 0,
        "origin_policy": "end_of_business_day",
    }
    silver = build_silver(
        database,
        {"source": {"source_id": "fixture", "version": "v1"}},
        schema,
        root / "silver",
        batch_size=batch,
        max_rows_per_file=file_rows,
    )
    policy = {
        "development_origins": ["2020-02-10", "2020-02-17"],
        "train_start": "2020-01-01",
        "stopping_periods": 3,
        "min_train_observations": 5,
        "holdout_periods": 28,
        "business_timezone": "UTC",
    }
    plan = build_from_silver(
        silver["dataset_path"], policy, database, root / "plans", batch_size=batch
    )
    recipe = {"lags": [7], "rolling": [], "methods": ["last_available", "seasonal_naive_7"]}
    config = {
        "plan": Path(plan["plan_path"]),
        "silver_dataset": Path(silver["dataset_path"]),
        "database": database,
        "output_root": root / "learned",
        "batch_size": batch,
        "recipe": recipe,
        "samples": {"features": ["last_available"], "max_rows_per_partition": 500},
        "model": {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "n_estimators": 15,
            "early_stopping_rounds": 3,
            "max_depth": 2,
            "device": "cpu",
        },
        "seed": 42,
    }
    sample = build_samples(config)
    values = [
        json.loads(line)
        for line in (Path(sample["artifact_path"]) / "samples.jsonl").read_text().splitlines()
    ]
    assert values and pa.__version__
    return config, sample, values


def test_native_distinct_series_late_label_and_report(tmp_path):
    config, sample, values = pipeline(tmp_path / "native")
    selected = [
        row
        for row in values
        if row["origin"] == "2020-02-10" and row["target_date"] == "2020-01-10"
    ]
    assert selected
    assert {row["partition"] for row in selected if row["series"] == ["a"]} == {"refit"}
    assert any(row["series"] == ["b"] and row["partition"] == "fit" for row in selected)
    expected = {"a": 18.25, "b": 1016.75}
    for row in selected:
        assert row["x"] == [expected[row["series"][0]]]
    replay = [
        row for row in values if row["series"] == ["a"] and row["target_date"] == "2020-01-11"
    ]
    assert replay and all(row["x"] == [18.25] for row in replay)
    baseline = predict(
        config["plan"],
        config["silver_dataset"],
        config["recipe"],
        tmp_path / "baselines",
        config["database"],
        9,
    )
    model = fit(config, sample["artifact_path"])
    scores = score(config, model["artifact_path"], baseline["artifact_path"])
    report = build_report(
        sample["artifact_path"],
        model["artifact_path"],
        scores["artifact_path"],
        baseline["artifact_path"],
        tmp_path / "reports",
        config["database"],
    )
    result = json.loads((Path(report["artifact_path"]) / "report.json").read_text())
    assert result["series"] == 2 and result["requests"] == 4
    assert any(row["scope"] == "series" for row in result["metrics"])


def test_native_batch_and_file_boundary_invariance(tmp_path):
    left, first, left_rows = pipeline(tmp_path / "left", 7, 13)
    right, second, right_rows = pipeline(tmp_path / "right", 11, 23)
    assert left_rows == right_rows
    repeat = build_samples({**left, "batch_size": 19})
    assert repeat["artifact_id"] == first["artifact_id"]
    a = fit(left, first["artifact_path"])
    b = fit(right, second["artifact_path"])

    def predictions(path):
        values = [
            json.loads(line) for line in (Path(path) / "predictions.jsonl").read_text().splitlines()
        ]
        return {(tuple(row["series"]), row["target_date"]): row["y_pred"] for row in values}

    av = predictions(a["artifact_path"])
    bv = predictions(b["artifact_path"])
    assert av.keys() == bv.keys()
    np.testing.assert_allclose(list(av.values()), [bv[key] for key in av], rtol=0, atol=1e-12)


def test_native_explicit_panel_budget_failure(tmp_path):
    config, _, _ = pipeline(tmp_path / "budget")
    limited = {**config, "samples": {**config["samples"], "max_rows_per_partition": 5}}
    with pytest.raises(ValueError, match="cap"):
        build_samples(limited)
