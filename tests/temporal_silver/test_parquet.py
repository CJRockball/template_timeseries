"Mandatory native Arrow acceptance. Missing pyarrow must fail, not silently skip."

import csv

import pyarrow as pa
import pytest

from ml_template.temporal_stream.silver import build, main
from ml_template.temporal_stream.silver_io import load_temporal_silver
from ml_template.temporal_stream.silver_manifest import verify_inventory
from ml_template.temporal_stream.store import ingest_csv


@pytest.fixture
def spec():
    return {
        "columns": {
            "id": "string",
            "series": "string",
            "date": "date",
            "y": "float",
            "weekday": "string",
        },
        "nullable": [],
        "observation_id": "id",
        "series": ["series"],
        "time": "date",
        "target": "y",
        "features": ["weekday"],
        "known_future": ["weekday"],
        "frequency": "D",
        "gaps": "error",
        "target_delay_periods": 0,
        "origin_policy": "end_of_business_day",
    }


def source(tmp, spec):
    path = tmp / "raw.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "series", "date", "y", "weekday"])
        writer.writerows(
            [
                ["c", "b", "2020-01-01", 3, "Wednesday"],
                ["b", "a", "2020-01-02", 0, "Thursday"],
                ["a", "a", "2020-01-01", 1.25, "Wednesday"],
            ]
        )
    contract = {
        "columns": spec["columns"],
        "key": ["id"],
        "temporal": {
            "series": ["series"],
            "time": "date",
            "frequency": "D",
            "gaps": "error",
            "target": "y",
        },
    }
    database = tmp / "project.db"
    ingest_csv(database, "fixture", "v1", path, contract, batch_size=1)
    return (database, {"source": {"source_id": "fixture", "version": "v1"}})


def test_round_trip_files_and_registry(tmp_path, spec):
    db, inp = source(tmp_path, spec)
    result = build(db, inp, spec, tmp_path / "silver", batch_size=1, max_rows_per_file=2)
    root = result["dataset_path"]
    assert len(result["files"]) == 2
    assert verify_inventory(root)["row_count"] == 3
    dataset = load_temporal_silver(root)
    assert dataset.schema.arrow_schema().field("date").type == pa.date32()
    ids = [row["id"] for batch in dataset.scan_batches(batch_size=1) for row in batch.to_pylist()]
    assert ids == ["a", "b", "c"]
    assert build(db, inp, spec, tmp_path / "silver", batch_size=1, max_rows_per_file=2)["reused"]


def test_layout_independence(tmp_path, spec):
    db, inp = source(tmp_path, spec)
    a = build(db, inp, spec, tmp_path / "silver", batch_size=1, max_rows_per_file=1)
    b = build(db, inp, spec, tmp_path / "silver", batch_size=3, max_rows_per_file=10)
    assert a["logical_data_sha256"] == b["logical_data_sha256"]
    assert a["silver_dataset_version_id"] != b["silver_dataset_version_id"]
    left = [
        row
        for batch in load_temporal_silver(a["dataset_path"]).scan_batches(batch_size=2)
        for row in batch.to_pylist()
    ]
    right = [
        row
        for batch in load_temporal_silver(b["dataset_path"]).scan_batches(batch_size=1)
        for row in batch.to_pylist()
    ]
    assert left == right


@pytest.mark.parametrize("ordered", [True, False])
def test_projection_and_filters(tmp_path, spec, ordered):
    db, inp = source(tmp_path, spec)
    result = build(db, inp, spec, tmp_path / "silver", batch_size=1, max_rows_per_file=1)
    rows = [
        row
        for batch in load_temporal_silver(result["dataset_path"]).scan_batches(
            columns=["id", "y"],
            series_values={"series": ["a"]},
            start_date="2020-01-02",
            end_date="2020-01-02",
            ordered=ordered,
            batch_size=1,
        )
        for row in batch.to_pylist()
    ]
    assert rows == [{"id": "b", "y": 0.0}]


def test_parquet_tamper_after_open(tmp_path, spec):
    from pathlib import Path

    db, inp = source(tmp_path, spec)
    result = build(db, inp, spec, tmp_path / "silver")
    dataset = load_temporal_silver(result["dataset_path"])
    part = Path(result["dataset_path"]) / result["files"][0]["path"]
    with part.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="integrity"):
        list(dataset.scan_batches())


def test_legacy_loader_rejects(tmp_path, spec):
    from ml_template.data.silver_loader import load_silver

    db, inp = source(tmp_path, spec)
    result = build(db, inp, spec, tmp_path / "silver/panel")
    with pytest.raises(ValueError, match="forecasting"):
        load_silver(
            result["silver_dataset_version_id"],
            "panel",
            "id",
            silver_root=tmp_path / "silver",
            task="regression",
        )


def test_verify_cli(tmp_path, spec):
    db, inp = source(tmp_path, spec)
    result = build(db, inp, spec, tmp_path / "silver")
    main(["verify", "--dataset", result["dataset_path"]])


def test_assembly_bridge(tmp_path, spec):
    from ml_template.temporal_stream.enrichment import assemble

    db, inp = source(tmp_path, spec)
    assemble(db, inp["source"], [], tmp_path / "assembly", batch_size=1)
    result = build(
        db, {"assembly": str(tmp_path / "assembly")}, spec, tmp_path / "silver", batch_size=1
    )
    assert result["row_count"] == 3
    assert result["input"]["kind"] == "assembly"


def test_utc_and_nullable_round_trip(tmp_path, spec):
    spec["columns"].update({"available": "utc_timestamp", "optional": "int"})
    spec["available_at"] = "available"
    spec["nullable"] = ["optional"]
    path = tmp_path / "raw.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "series", "date", "y", "weekday", "available", "optional"])
        writer.writerow(["a", "a", "2020-01-01", -1.25, "Wednesday", "2020-01-02T00:00:00Z", ""])
    contract = {"columns": spec["columns"], "key": ["id"], "nullable": ["optional"]}
    db = tmp_path / "project.db"
    ingest_csv(db, "utc", "v1", path, contract)
    result = build(db, {"source": {"source_id": "utc", "version": "v1"}}, spec, tmp_path / "silver")
    dataset = load_temporal_silver(result["dataset_path"])
    assert dataset.schema.arrow_schema().field("available").type == pa.timestamp("us", tz="UTC")
    row = next(dataset.scan_batches()).to_pylist()[0]
    assert row["optional"] is None and row["y"] == -1.25


def test_run_config_rejects_temporal_id():
    from ml_template.training.run_config import RunConfig

    with pytest.raises(ValueError, match="forecasting"):
        RunConfig.model_validate(
            {
                "experiment_name": "temporal_guard",
                "data": {
                    "dataset_name": "panel",
                    "dataset_config": "not_used.yaml",
                    "silver_dataset_version_id": "silver_ts_fixture",
                },
            }
        )
