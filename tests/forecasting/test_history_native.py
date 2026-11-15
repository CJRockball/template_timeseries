"Mandatory real Silver/plan/history/scorer gate. No silent Arrow skips."

import csv
import json
from datetime import date, timedelta
from pathlib import Path

import pyarrow as pa
import pytest

from ml_template.forecasting.history_artifacts import file_hash, verify_artifact
from ml_template.forecasting.history_cli import main
from ml_template.forecasting.history_io import predict, score
from ml_template.forecasting.io import build_from_silver
from ml_template.temporal_stream.silver import build
from ml_template.temporal_stream.store import ingest_csv


@pytest.fixture
def native(tmp_path):
    raw = tmp_path / "raw.csv"
    columns = {
        "id": "string",
        "series": "string",
        "date": "date",
        "y": "float",
        "x": "float",
        "available": "utc_timestamp",
    }
    with raw.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(list(columns))
        for series in ["a", "b"]:
            for i in range(140):
                date_string = (date(2020, 1, 1) + timedelta(days=i)).isoformat()
                writer.writerow(
                    [
                        f"{series}-{i}",
                        series,
                        date_string,
                        i + 0.25,
                        9999.0,
                        date_string + "T00:00:00Z",
                    ]
                )
    temporal = {
        "series": ["series"],
        "time": "date",
        "target": "y",
        "frequency": "D",
        "gaps": "error",
    }
    database = tmp_path / "project.db"
    ingest_csv(
        database, "fixture", "v1", raw, {"columns": columns, "key": ["id"], "temporal": temporal}
    )
    schema = {
        "columns": columns,
        "observation_id": "id",
        **temporal,
        "features": ["x"],
        "known_future": [],
        "available_at": "available",
        "target_delay_periods": 0,
        "origin_policy": "end_of_business_day",
    }
    silver = build(
        database,
        {"source": {"source_id": "fixture", "version": "v1"}},
        schema,
        tmp_path / "silver",
        batch_size=9,
        max_rows_per_file=23,
    )
    policy = {
        "development_origins": ["2020-02-10", "2020-02-17"],
        "train_start": "2020-01-01",
        "stopping_periods": 7,
        "min_train_observations": 28,
        "holdout_periods": 28,
        "business_timezone": "UTC",
    }
    plan = build_from_silver(
        silver["dataset_path"], policy, database, tmp_path / "plans", batch_size=11
    )
    return (silver, plan, database, tmp_path / "history")


def test_real_history_score_and_binding(native):
    silver, plan, db, output = native
    result = predict(plan["plan_path"], silver["dataset_path"], None, output, db, batch_size=7)
    before = file_hash(Path(result["artifact_path"]) / "predictions.jsonl")
    report = score(
        plan["plan_path"],
        silver["dataset_path"],
        result["artifact_path"],
        None,
        output,
        db,
        batch_size=13,
    )
    metrics = json.loads((Path(report["artifact_path"]) / "metrics.json").read_text())["metrics"]
    assert (
        next(

                r
                for r in metrics
                if r["method"] == "last_available"
                and r["scope"] == "pooled"
                and (r["comparison"] == "common_supported")

        )["mae"]
        == 1
    )
    assert file_hash(Path(result["artifact_path"]) / "predictions.jsonl") == before
    assert report["stats"]["common_supported"] == 4
    assert pa.__version__


def test_projection_and_date_bounds(native, monkeypatch):
    from ml_template.temporal_stream.silver_io import TemporalSilver

    original = TemporalSilver.scan_batches
    seen = []

    def spy(self, **kwargs):
        seen.append(kwargs)
        return original(self, **kwargs)

    monkeypatch.setattr(TemporalSilver, "scan_batches", spy)
    silver, plan, db, output = native
    first = predict(plan["plan_path"], silver["dataset_path"], None, output, db, batch_size=1)
    second = predict(plan["plan_path"], silver["dataset_path"], None, output, db, batch_size=19)
    assert first["artifact_id"] == second["artifact_id"] and second["reused"]
    assert all(call["end_date"] < plan["identity"]["holdout"]["start"] for call in seen)
    assert all("x" not in call["columns"] for call in seen)
    assert all("y" in call["columns"] and "available" in call["columns"] for call in seen)


def test_full_cli(native, tmp_path, capsys):
    silver, plan, db, output = native
    config = tmp_path / "history.yaml"
    main(
        [
            "init",
            "--plan",
            plan["plan_path"],
            "--silver-dataset",
            silver["dataset_path"],
            "--database",
            str(db),
            "--output-root",
            str(output),
            "--output",
            str(config),
        ]
    )
    main(["predict", "--config", str(config)])
    predictions = next(
        p for p in output.glob("history_*") if verify_artifact(p)["kind"] == "predictions"
    )
    main(["score", "--config", str(config), "--predictions", str(predictions)])
    scores = next(p for p in output.glob("history_*") if verify_artifact(p)["kind"] == "scores")
    main(["verify", "--artifact", str(scores)])
    assert '"outcomes_joined": true' in capsys.readouterr().out
