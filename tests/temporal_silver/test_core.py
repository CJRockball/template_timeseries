"Dependency-light metadata, staging, registry and guard tests. No Arrow mocks."

import json
import sqlite3

import pytest

from ml_template.db.temporal_silver_registry import register_temporal_silver
from ml_template.temporal_stream.silver import stage_rows
from ml_template.temporal_stream.silver_guard import (
    reject_temporal_contract,
    reject_temporal_manifest,
    reject_temporal_path,
)
from ml_template.temporal_stream.silver_manifest import seal, verify_inventory
from ml_template.temporal_stream.silver_schema import (
    FORMAT,
    TemporalSchema,
    file_hash,
    positive,
    sha,
)


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


def records():
    return [
        {"id": "a", "series": "s", "date": "2020-01-01", "y": 1.25, "weekday": "Wednesday"},
        {"id": "b", "series": "s", "date": "2020-01-02", "y": 0.0, "weekday": "Thursday"},
    ]


def inventory(tmp, spec):
    root = tmp / "dataset"
    (root / "data").mkdir(parents=True)
    part = root / "data/part-000000.parquet"
    part.write_bytes(b"inventory-fixture-not-a-real-parquet-file")
    report = root / "validation.json"
    report.write_text(json.dumps({"passed": True, "rows": 2}))
    schema = TemporalSchema.from_dict(spec)
    identity = {
        "temporal_schema": schema.spec,
        "input": {"source": "fixture"},
        "logical_data_sha256": "f" * 64,
    }
    value = {
        "format": FORMAT,
        "dataset_kind": "temporal",
        "eligible_for_legacy_training": False,
        "silver_dataset_version_id": "silver_ts_" + sha(identity),
        "identity": identity,
        "temporal_schema": schema.spec,
        "roles": schema.roles,
        "order_by": schema.order,
        "input": identity["input"],
        "logical_data_sha256": identity["logical_data_sha256"],
        "row_count": 2,
        "validation_passed": True,
        "validation_sha256": file_hash(report),
        "files": [
            {
                "path": "data/part-000000.parquet",
                "rows": 2,
                "bytes": part.stat().st_size,
                "sha256": file_hash(part),
            }
        ],
    }
    value = seal(value)
    (root / "manifest.json").write_text(json.dumps(value))
    return (root, value)


def test_schema_defaults_and_roles(spec):
    schema = TemporalSchema.from_dict(spec)
    assert schema.order == ["series", "date"]
    assert schema.roles["target"] == ["y"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("target_delay_periods", -1),
        ("target_delay_periods", True),
        ("frequency", "H"),
        ("origin_policy", "unknown"),
        ("features", ["y"]),
        ("nullable", ["date"]),
        ("known_future", ["series"]),
        ("series", []),
    ],
)
def test_bad_schema(spec, field, value):
    spec[field] = value
    with pytest.raises(ValueError):
        TemporalSchema.from_dict(spec)


@pytest.mark.parametrize("day", ["2020-02-30", "2020-1-01", "01/01/2020"])
def test_bad_date(spec, day):
    row = records()[0]
    row["date"] = day
    with pytest.raises(ValueError):
        TemporalSchema.from_dict(spec).parse(row)


@pytest.mark.parametrize("number", ["inf", "nan", "-inf"])
def test_nonfinite(spec, number):
    row = records()[0]
    row["y"] = number
    with pytest.raises(ValueError):
        TemporalSchema.from_dict(spec).parse(row)


def test_staging_order_and_bounds(tmp_path, spec):
    schema = TemporalSchema.from_dict(spec)
    count, maximum, digest = stage_rows(tmp_path / "scratch.db", reversed(records()), schema, 1)
    assert count == 2 and maximum == 1 and (len(digest) == 64)
    with sqlite3.connect(tmp_path / "scratch.db") as conn:
        assert conn.execute("SELECT id FROM observations ORDER BY series,date").fetchall() == [
            ("a",),
            ("b",),
        ]


def test_duplicate_across_batches(tmp_path, spec):
    rows = records()
    rows[1]["date"] = rows[0]["date"]
    with pytest.raises(sqlite3.IntegrityError):
        stage_rows(tmp_path / "scratch.db", rows, TemporalSchema.from_dict(spec), 1)


def test_gap(tmp_path, spec):
    rows = records()
    rows[1]["date"] = "2020-01-03"
    with pytest.raises(ValueError, match="gap"):
        stage_rows(tmp_path / "scratch.db", rows, TemporalSchema.from_dict(spec), 1)


def test_valid_inventory(tmp_path, spec):
    root, value = inventory(tmp_path, spec)
    assert verify_inventory(root)["manifest_sha256"] == value["manifest_sha256"]


@pytest.mark.parametrize("change", ["tamper", "extra", "missing", "manifest", "report"])
def test_invalid_inventory(tmp_path, spec, change):
    root, value = inventory(tmp_path, spec)
    part = root / "data/part-000000.parquet"
    if change == "tamper":
        part.write_bytes(b"modified")
    elif change == "extra":
        (root / "data/extra.parquet").write_bytes(b"extra")
    elif change == "missing":
        part.unlink()
    elif change == "manifest":
        value["row_count"] = 9
        (root / "manifest.json").write_text(json.dumps(value))
    else:
        (root / "validation.json").write_text("{}")
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_inventory(root)


def test_path_traversal(tmp_path, spec):
    root, value = inventory(tmp_path, spec)
    value["files"][0]["path"] = "../outside.parquet"
    (root / "manifest.json").write_text(json.dumps(seal(value)))
    with pytest.raises(ValueError, match="Unsafe"):
        verify_inventory(root)


def test_registry_reuse(tmp_path, spec):
    root, value = inventory(tmp_path, spec)
    db = tmp_path / "project.db"
    register_temporal_silver(db, root / "manifest.json")
    register_temporal_silver(db, root / "manifest.json")
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM temporal_silver_datasets").fetchone()[0] == 1


@pytest.mark.parametrize(
    "manifest",
    [
        {"dataset_kind": "temporal"},
        {"eligible_for_legacy_training": False},
        {"temporal_schema": {"x": 1}},
    ],
)
def test_legacy_guard(manifest):
    with pytest.raises(ValueError, match="forecasting"):
        reject_temporal_manifest(manifest)


def test_legacy_unchanged():
    reject_temporal_manifest({"validation_passed": True})
    reject_temporal_contract({"data": {"task": "regression"}})


def test_contract_guard():
    with pytest.raises(ValueError, match="forecasting"):
        reject_temporal_contract({"data": {"temporal": {"time": "date"}}})


def test_path_guard(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"dataset_kind": "temporal"}))
    with pytest.raises(ValueError, match="forecasting"):
        reject_temporal_path(tmp_path)


@pytest.mark.parametrize("value", [0, -1, True])
def test_invalid_batch(value):
    with pytest.raises(ValueError):
        positive(value, "batch")


def test_init_local_config(tmp_path, spec):
    import yaml

    from ml_template.temporal_stream.silver import main

    contract = {
        "columns": spec["columns"],
        "key": ["id"],
        "temporal": {"series": ["series"], "time": "date", "target": "y", "frequency": "D"},
    }
    cfg = tmp_path / "ingest.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "database": "project.db",
                "source_id": "fixture",
                "version": "v1",
                "contract": contract,
            }
        )
    )
    out = tmp_path / "local_silver.yaml"
    command = [
        "init",
        "--ingest-config",
        str(cfg),
        "--output",
        str(out),
        "--target-delay-periods",
        "0",
        "--origin-policy",
        "end_of_business_day",
    ]
    main(command)
    assert yaml.safe_load(out.read_text())["schema"]["features"] == []
    with pytest.raises(FileExistsError):
        main(command)


def test_version_guard():
    from ml_template.temporal_stream.silver_guard import reject_temporal_version

    reject_temporal_version("silver_legacy_123")
    with pytest.raises(ValueError, match="forecasting"):
        reject_temporal_version("silver_ts_123")
