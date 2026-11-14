import json
import sqlite3

import pandas as pd
import pytest

from ml_template.temporal.__main__ import main
from ml_template.temporal.contracts import SourceContract
from ml_template.temporal.enrichment import enrich
from ml_template.temporal.store import ingest_csv, load_source


@pytest.fixture
def spec():
    return {
        "columns": {"id": "string", "series": "string", "date": "date", "y": "int"},
        "key": ["id"],
        "temporal": {
            "series": ["series"],
            "time": "date",
            "frequency": "D",
            "gaps": "error",
            "target": "y",
        },
    }


def write(tmp_path, name, rows):
    p = tmp_path / (name + ".csv")
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def rows():
    return [
        {"id": "a", "series": "x", "date": "2020-01-01", "y": 1},
        {"id": "b", "series": "x", "date": "2020-01-02", "y": 0},
    ]


def test_ingest_reuse_and_conflict(tmp_path, spec):
    p = write(tmp_path, "a", rows())
    db = tmp_path / "p.db"
    first = ingest_csv(db, "sales", "v1", p, spec)
    assert not first["reused"]
    assert ingest_csv(db, "sales", "v1", p, spec)["reused"]
    changed = rows()
    changed[0]["y"] = 9
    write(tmp_path, "a", changed)
    with pytest.raises(ValueError, match="conflict"):
        ingest_csv(db, "sales", "v1", p, spec)
    assert load_source(db, "sales", "v1")[0].y.tolist() == [1, 0]


@pytest.mark.parametrize("bad", ["2020-02-30", "01/01/2020", "2020-1-01", "2020-01-01T00:00:00Z"])
def test_bad_dates(tmp_path, spec, bad):
    r = rows()
    r[0]["date"] = bad
    with pytest.raises(ValueError):
        ingest_csv(tmp_path / "p.db", "x", "1", write(tmp_path, "a", r), spec)


def test_duplicate_observation(tmp_path, spec):
    r = rows()
    r[1]["date"] = r[0]["date"]
    with pytest.raises(ValueError, match="series/date"):
        ingest_csv(tmp_path / "p.db", "x", "1", write(tmp_path, "a", r), spec)


def test_gap(tmp_path, spec):
    r = rows()
    r[1]["date"] = "2020-01-03"
    with pytest.raises(ValueError, match="missing"):
        ingest_csv(tmp_path / "p.db", "x", "1", write(tmp_path, "a", r), spec)
    spec["temporal"]["gaps"] = "allow"
    assert ingest_csv(tmp_path / "p.db", "x", "1", tmp_path / "a.csv", spec)["rows"] == 2


@pytest.mark.parametrize("value", ["1.5", "inf", "9223372036854775808"])
def test_bad_integer(tmp_path, spec, value):
    r = rows()
    r[0]["y"] = value
    with pytest.raises(ValueError):
        ingest_csv(tmp_path / "p.db", "x", "1", write(tmp_path, "a", r), spec)


def test_null_target(tmp_path, spec):
    r = rows()
    r[0]["y"] = None
    with pytest.raises(ValueError, match="Null"):
        ingest_csv(tmp_path / "p.db", "x", "1", write(tmp_path, "a", r), spec)


def test_order_independent_canonical_data(tmp_path, spec):
    db = tmp_path / "p.db"
    a = ingest_csv(db, "x", "1", write(tmp_path, "a", rows()), spec)
    b = ingest_csv(db, "x", "2", write(tmp_path, "b", list(reversed(rows()))), spec)
    assert a["data_sha256"] == b["data_sha256"]


def test_separate_schemas_and_nullable_int(tmp_path, spec):
    db = tmp_path / "p.db"
    ingest_csv(db, "x", "1", write(tmp_path, "a", rows()), spec)
    other = {
        "columns": {"name": "string", "amount": "int"},
        "key": ["name"],
        "nullable": ["amount"],
    }
    p = tmp_path / "b.csv"
    p.write_text("name,amount\na,3\nb,\n")
    ingest_csv(db, "other", "1", p, other)
    assert load_source(db, "other", "1")[0].amount.tolist() == [3, None]


def test_tamper_detected(tmp_path, spec):
    db = tmp_path / "p.db"
    ingest_csv(db, "x", "1", write(tmp_path, "a", rows()), spec)
    with sqlite3.connect(db) as c:
        table = c.execute("SELECT table_name FROM temporal_sources").fetchone()[0]
        c.execute(f'UPDATE "{table}" SET y=99')
    with pytest.raises(ValueError, match="integrity"):
        load_source(db, "x", "1")


def test_failed_insert_rolls_back(tmp_path, spec):
    db = tmp_path / "p.db"
    ingest_csv(db, "x", "1", write(tmp_path, "a", rows()), spec)
    with sqlite3.connect(db) as c:
        c.execute(
            "CREATE TRIGGER deny_new BEFORE INSERT ON temporal_sources BEGIN SELECT RAISE(ABORT,'denied'); END"
        )
        before = set(c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall())
    with pytest.raises(sqlite3.IntegrityError):
        ingest_csv(db, "x", "2", tmp_path / "a.csv", spec)
    with sqlite3.connect(db) as c:
        after = set(c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall())
    assert before == after
    assert len(load_source(db, "x", "1")[0]) == 2


def exact_setup(tmp_path, spec):
    db = tmp_path / "p.db"
    ingest_csv(db, "x", "1", write(tmp_path, "a", rows()), spec)
    cal = {"columns": {"date": "date", "day": "string"}, "key": ["date"]}
    ingest_csv(
        db,
        "cal",
        "1",
        write(tmp_path, "b", [{"date": r["date"], "day": "weekday"} for r in rows()]),
        cal,
    )
    join = {
        "source_id": "cal",
        "version": "1",
        "on": ["date"],
        "columns": {"day": "weekday"},
        "mode": "exact",
        "known_in_advance": True,
    }
    return db, join


def test_exact_join(tmp_path, spec):
    db, j = exact_setup(tmp_path, spec)
    result = enrich(db, {"source_id": "x", "version": "1"}, [j])
    assert result.frame.weekday.tolist() == ["weekday", "weekday"]
    assert not result.manifest["eligible_for_legacy_training"]
    assert enrich(db, {"source_id": "x", "version": "1"}, [j]).manifest == result.manifest


def test_exact_requires_availability_assertion(tmp_path, spec):
    db, j = exact_setup(tmp_path, spec)
    j.pop("known_in_advance")
    with pytest.raises(ValueError, match="known-in-advance"):
        enrich(db, {"source_id": "x", "version": "1"}, [j])


def test_collision(tmp_path, spec):
    db, j = exact_setup(tmp_path, spec)
    j["columns"] = {"day": "y"}
    with pytest.raises(ValueError, match="overwrite"):
        enrich(db, {"source_id": "x", "version": "1"}, [j])


def test_many_to_one_enforced(tmp_path, spec):
    db, j = exact_setup(tmp_path, spec)
    contract = {"columns": {"date": "date", "n": "int", "day": "string"}, "key": ["date", "n"]}
    ingest_csv(
        db,
        "cal",
        "2",
        write(tmp_path, "c", [{"date": "2020-01-01", "n": i, "day": "a"} for i in [1, 2]]),
        contract,
    )
    j["version"] = "2"
    with pytest.raises(ValueError, match="many-to-one"):
        enrich(db, {"source_id": "x", "version": "1"}, [j])


def test_missing_join_fails(tmp_path, spec):
    db, j = exact_setup(tmp_path, spec)
    contract = {"columns": {"date": "date", "day": "string"}, "key": ["date"]}
    ingest_csv(db, "cal", "2", write(tmp_path, "c", [{"date": "2021-01-01", "day": "a"}]), contract)
    j["version"] = "2"
    with pytest.raises(ValueError, match="missingness"):
        enrich(db, {"source_id": "x", "version": "1"}, [j])


def pit_setup(tmp_path, future=999):
    db = tmp_path / "p.db"
    contract = {
        "columns": {"id": "string", "date": "date", "origin": "utc_timestamp"},
        "key": ["id"],
    }
    ingest_csv(
        db,
        "requests",
        "1",
        write(tmp_path, "a", [{"id": "a", "date": "2020-01-03", "origin": "2020-01-02T00:00:00Z"}]),
        contract,
    )
    contract = {
        "columns": {"date": "date", "available": "utc_timestamp", "value": "float"},
        "key": ["date", "available"],
    }
    data = [
        {"date": "2020-01-03", "available": "2020-01-01T00:00:00Z", "value": 5},
        {"date": "2020-01-03", "available": "2020-01-02T01:00:00Z", "value": future},
    ]
    ingest_csv(db, "external", str(future), write(tmp_path, "b", data), contract)
    j = {
        "source_id": "external",
        "version": str(future),
        "on": ["date"],
        "columns": {"value": "prediction"},
        "mode": "point_in_time",
        "available_at": "available",
        "origin": "origin",
    }
    return db, j


def test_point_in_time_future_perturbation(tmp_path):
    db, j = pit_setup(tmp_path, 999)
    first = enrich(db, {"source_id": "requests", "version": "1"}, [j]).frame
    # Rebuild in another directory; future values cannot change earlier-origin features.
    d = tmp_path / "second"
    d.mkdir()
    db, j = pit_setup(d, -100)
    second = enrich(db, {"source_id": "requests", "version": "1"}, [j]).frame
    pd.testing.assert_frame_equal(first, second)
    assert first.prediction.tolist() == [5.0]


@pytest.mark.parametrize("bad", ["2020-01-01", "2020-01-01T00:00:00", "2020-01-01T08:00:00+08:00"])
def test_timestamp_requires_utc_z(bad):
    c = SourceContract.from_dict({"columns": {"t": "utc_timestamp"}, "key": ["t"]})
    with pytest.raises(ValueError):
        c.validate(pd.DataFrame({"t": [bad]}))


def test_cli_round_trip(tmp_path, spec):
    import yaml

    write(tmp_path, "a", rows())
    config = tmp_path / "ingest.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "database": "p.db",
                "source_id": "x",
                "version": "1",
                "input": "a.csv",
                "contract": spec,
            }
        )
    )
    main(["ingest", "--config", str(config)])
    assembly = tmp_path / "assembly.yaml"
    assembly.write_text(
        yaml.safe_dump(
            {"database": "p.db", "base": {"source_id": "x", "version": "1"}, "output": "out"}
        )
    )
    main(["assemble", "--config", str(assembly)])
    assert len(pd.read_csv(tmp_path / "out/data.csv")) == 2
    with pytest.raises(FileExistsError):
        main(["assemble", "--config", str(assembly)])
    assert json.loads((tmp_path / "out/manifest.json").read_text())["rows"] == 2


def test_point_in_time_inclusive_cutoff(tmp_path):
    db, j = pit_setup(tmp_path)
    contract = {
        "columns": {"id": "string", "date": "date", "origin": "utc_timestamp"},
        "key": ["id"],
    }
    p = write(
        tmp_path, "cutoff", [{"id": "a", "date": "2020-01-03", "origin": "2020-01-02T01:00:00Z"}]
    )
    ingest_csv(db, "requests", "2", p, contract)
    assert enrich(db, {"source_id": "requests", "version": "2"}, [j]).frame.prediction.tolist() == [
        999.0
    ]


def test_point_in_time_rejects_no_available_record(tmp_path):
    db, j = pit_setup(tmp_path)
    contract = {
        "columns": {"id": "string", "date": "date", "origin": "utc_timestamp"},
        "key": ["id"],
    }
    p = write(
        tmp_path, "early", [{"id": "a", "date": "2020-01-03", "origin": "2019-01-01T00:00:00Z"}]
    )
    ingest_csv(db, "requests", "2", p, contract)
    with pytest.raises(ValueError, match="missingness"):
        enrich(db, {"source_id": "requests", "version": "2"}, [j])
