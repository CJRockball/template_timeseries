"Native Silver bridge gate. Missing Arrow deliberately fails collection."

import csv
from datetime import date, timedelta

import pyarrow as pa
import pytest

from ml_template.forecasting.artifacts import iter_plan_rows
from ml_template.forecasting.io import build_from_silver
from ml_template.forecasting.origins import main
from ml_template.temporal_stream.silver import build
from ml_template.temporal_stream.store import ingest_csv


@pytest.fixture
def real_silver(tmp_path):
    raw = tmp_path / "raw.csv"
    with raw.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "series", "date", "y", "x"])
        for series in ["a", "b"]:
            for i in range(80):
                writer.writerow(
                    [
                        f"{series}-{i}",
                        series,
                        (date(2020, 1, 1) + timedelta(days=i)).isoformat(),
                        123456 + i,
                        i,
                    ]
                )
    columns = {"id": "string", "series": "string", "date": "date", "y": "int", "x": "int"}
    temporal = {
        "series": ["series"],
        "time": "date",
        "target": "y",
        "frequency": "D",
        "gaps": "error",
    }
    db = tmp_path / "project.db"
    ingest_csv(db, "fixture", "v1", raw, {"columns": columns, "key": ["id"], "temporal": temporal})
    schema = {
        "columns": columns,
        "observation_id": "id",
        **temporal,
        "features": ["x"],
        "known_future": [],
        "target_delay_periods": 0,
        "origin_policy": "end_of_business_day",
    }
    silver = build(
        db,
        {"source": {"source_id": "fixture", "version": "v1"}},
        schema,
        tmp_path / "silver",
        batch_size=11,
        max_rows_per_file=37,
    )
    policy = {
        "development_origins": ["2020-02-10", "2020-02-12"],
        "train_start": "2020-01-01",
        "stopping_periods": 7,
        "min_train_observations": 28,
        "holdout_periods": 28,
        "business_timezone": "UTC",
    }
    return (silver, policy, db, tmp_path / "plans")


def test_native_bridge(real_silver):
    silver, policy, db, out = real_silver
    result = build_from_silver(silver["dataset_path"], policy, db, out, batch_size=9)
    requests = list(iter_plan_rows(result["plan_path"]))
    assert len(requests) == 4
    assert result["identity"]["silver"]["manifest_sha256"] == silver["manifest_sha256"]
    assert pa.__version__


def test_never_projects_targets_or_predictors(real_silver, monkeypatch):
    from ml_template.temporal_stream.silver_io import TemporalSilver

    original = TemporalSilver.scan_batches
    seen = []

    def spy(self, **kwargs):
        seen.append(kwargs["columns"])
        return original(self, **kwargs)

    monkeypatch.setattr(TemporalSilver, "scan_batches", spy)
    silver, policy, db, out = real_silver
    build_from_silver(silver["dataset_path"], policy, db, out, batch_size=7)
    assert seen and all(set(cols) == {"id", "series", "date"} for cols in seen)


def test_native_cli(real_silver, tmp_path, capsys):
    silver, _, db, out = real_silver
    main(["bounds", "--silver-dataset", silver["dataset_path"]])
    config = tmp_path / "origins.yaml"
    main(
        [
            "init",
            "--silver-dataset",
            silver["dataset_path"],
            "--database",
            str(db),
            "--output-root",
            str(out),
            "--output",
            str(config),
            "--train-start",
            "2020-01-01",
            "--origin",
            "2020-02-10",
            "--origin",
            "2020-02-12",
            "--business-timezone",
            "UTC",
        ]
    )
    main(["build", "--config", str(config)])
    plans = list(out.glob("origin_*"))
    assert len(plans) == 1
    main(["verify", "--plan", str(plans[0]), "--silver-dataset", silver["dataset_path"]])
    assert '"passed": true' in capsys.readouterr().out
