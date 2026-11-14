import csv
import json
import sqlite3

import pytest

from ml_template.temporal_stream.__main__ import main
from ml_template.temporal_stream.contracts import SourceContract
from ml_template.temporal_stream.enrichment import assemble
from ml_template.temporal_stream.store import ingest_csv, iter_source, verify_source


@pytest.fixture
def contract():
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


def rows():
    return [
        {"id": "a", "series": "x", "date": "2020-01-01", "y": 1},
        {"id": "b", "series": "x", "date": "2020-01-02", "y": 0},
    ]


def write(tmp, name, data):
    path = tmp / (name + ".csv")
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)
    return path


def ingest(tmp, contract, data=None, version="1", batch=1):
    return ingest_csv(
        tmp / "p.db",
        "sales",
        version,
        write(tmp, "sales", data or rows()),
        contract,
        batch_size=batch,
    )


def test_ingest_batched_reuse_and_verify(tmp_path, contract):
    m = ingest(tmp_path, contract)
    assert m["max_insert_batch_rows"] == 1
    assert verify_source(tmp_path / "p.db", "sales", "1")["rows"] == 2
    assert ingest(tmp_path, contract)["reused"]
    batches = list(iter_source(tmp_path / "p.db", "sales", "1", 1))
    assert len(batches) == 2 and all(len(b) == 1 for b in batches)


def test_conflict(tmp_path, contract):
    ingest(tmp_path, contract)
    r = rows()
    r[0]["y"] = 9
    with pytest.raises(ValueError, match="conflict"):
        ingest(tmp_path, contract, r)
    assert verify_source(tmp_path / "p.db", "sales", "1")["rows"] == 2


@pytest.mark.parametrize("bad", ["2020-02-30", "2020-1-01", "01/01/2020", "2020-01-01T00:00:00Z"])
def test_bad_dates(tmp_path, contract, bad):
    r = rows()
    r[0]["date"] = bad
    with pytest.raises(ValueError):
        ingest(tmp_path, contract, r)


@pytest.mark.parametrize("bad", ["1.5", "inf", "9223372036854775808", ""])
def test_bad_int(tmp_path, contract, bad):
    r = rows()
    r[0]["y"] = bad
    with pytest.raises(ValueError):
        ingest(tmp_path, contract, r)


def test_duplicate_across_batches_rolls_back(tmp_path, contract):
    ingest(tmp_path, contract)
    r = rows()
    r[1]["date"] = r[0]["date"]
    with pytest.raises(sqlite3.IntegrityError):
        ingest(tmp_path, contract, r, version="2")
    with sqlite3.connect(tmp_path / "p.db") as c:
        assert c.execute("SELECT COUNT(*) FROM temporal_sources_v2").fetchone()[0] == 1
        assert (
            c.execute("SELECT COUNT(*) FROM sqlite_master WHERE name LIKE 'ts2_%'").fetchone()[0]
            == 1
        )


def test_gap_across_batch_boundary(tmp_path, contract):
    r = rows()
    r[1]["date"] = "2020-01-03"
    with pytest.raises(ValueError, match="missing"):
        ingest(tmp_path, contract, r)
    contract["temporal"]["gaps"] = "allow"
    assert ingest(tmp_path, contract, r)["rows"] == 2


def test_reordering_hash_stability(tmp_path, contract):
    a = ingest(tmp_path, contract)
    b = ingest(tmp_path, contract, list(reversed(rows())), version="2", batch=9)
    assert a["data_sha256"] == b["data_sha256"]


def test_tamper(tmp_path, contract):
    ingest(tmp_path, contract)
    with sqlite3.connect(tmp_path / "p.db") as c:
        table = c.execute("SELECT table_name FROM temporal_sources_v2").fetchone()[0]
        c.execute(f'UPDATE "{table}" SET y=99')
    with pytest.raises(ValueError, match="integrity"):
        verify_source(tmp_path / "p.db", "sales", "1")


def test_separate_schemas_nullable_integer(tmp_path, contract):
    ingest(tmp_path, contract)
    spec = {"columns": {"name": "string", "n": "int"}, "key": ["name"], "nullable": ["n"]}
    p = write(tmp_path, "other", [{"name": "NA", "n": 3}, {"name": "b", "n": ""}])
    ingest_csv(tmp_path / "p.db", "other", "1", p, spec)
    assert verify_source(tmp_path / "p.db", "other", "1")["rows"] == 2


def calendar(tmp, contract):
    ingest(tmp, contract)
    spec = {"columns": {"date": "date", "value": "string"}, "key": ["date"]}
    p = write(tmp, "cal", [{"date": r["date"], "value": "day"} for r in rows()])
    ingest_csv(tmp / "p.db", "cal", "1", p, spec)
    return {
        "source_id": "cal",
        "version": "1",
        "on": ["date"],
        "columns": {"value": "weekday"},
        "mode": "exact",
        "known_in_advance": True,
    }


def test_exact_and_export_batch(tmp_path, contract):
    j = calendar(tmp_path, contract)
    m = assemble(
        tmp_path / "p.db", {"source_id": "sales", "version": "1"}, [j], tmp_path / "out", 1
    )
    assert m["rows"] == 2 and m["max_export_batch_rows"] == 1
    assert m["reports"][0]["unmatched_rows"] == 0
    assert not m["eligible_for_legacy_training"]
    with (tmp_path / "out/data.csv").open() as f:
        assert [r["weekday"] for r in csv.DictReader(f)] == ["day", "day"]
    with pytest.raises(FileExistsError):
        assemble(
            tmp_path / "p.db", {"source_id": "sales", "version": "1"}, [j], tmp_path / "out", 1
        )


def test_two_sql_joins(tmp_path, contract):
    j = calendar(tmp_path, contract)
    other = {**j, "columns": {"value": "weekday2"}}
    result = assemble(
        tmp_path / "p.db", {"source_id": "sales", "version": "1"}, [j, other], tmp_path / "out", 1
    )
    assert result["rows"] == 2


@pytest.mark.parametrize("kind", ["collision", "unknown", "not_known"])
def test_invalid_join(tmp_path, contract, kind):
    j = calendar(tmp_path, contract)
    if kind == "collision":
        j["columns"] = {"value": "Y"}
    elif kind == "unknown":
        j["nonsense"] = True
    else:
        j.pop("known_in_advance")
    with pytest.raises(ValueError):
        assemble(tmp_path / "p.db", {"source_id": "sales", "version": "1"}, [j], tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_many_to_many_rejected(tmp_path, contract):
    j = calendar(tmp_path, contract)
    spec = {"columns": {"date": "date", "value": "string", "n": "int"}, "key": ["date", "n"]}
    p = write(tmp_path, "dup", [{"date": "2020-01-01", "value": "x", "n": i} for i in [1, 2]])
    ingest_csv(tmp_path / "p.db", "dup", "1", p, spec)
    j["source_id"] = "dup"
    with pytest.raises(ValueError, match="Ambiguous"):
        assemble(tmp_path / "p.db", {"source_id": "sales", "version": "1"}, [j], tmp_path / "out")


def pit(tmp, future=999, origin="2020-01-02T00:00:00Z"):
    spec = {"columns": {"id": "string", "date": "date", "origin": "utc_timestamp"}, "key": ["id"]}
    p = write(tmp, "req", [{"id": "a", "date": "2020-01-03", "origin": origin}])
    ingest_csv(tmp / "p.db", "requests", "1", p, spec)
    spec = {
        "columns": {"date": "date", "available": "utc_timestamp", "value": "float"},
        "key": ["date", "available"],
    }
    p = write(
        tmp,
        "external",
        [
            {"date": "2020-01-03", "available": "2020-01-01T00:00:00Z", "value": 5},
            {"date": "2020-01-03", "available": "2020-01-02T01:00:00Z", "value": future},
        ],
    )
    ingest_csv(tmp / "p.db", "external", "1", p, spec)
    j = {
        "source_id": "external",
        "version": "1",
        "on": ["date"],
        "columns": {"value": "prediction"},
        "mode": "point_in_time",
        "available_at": "available",
        "origin": "origin",
    }
    m = assemble(tmp / "p.db", {"source_id": "requests", "version": "1"}, [j], tmp / "out", 1)
    with (tmp / "out/data.csv").open() as f:
        records = list(csv.DictReader(f))
    return (m, records)


def test_future_perturbation(tmp_path):
    a = tmp_path / "a"
    a.mkdir()
    b = tmp_path / "b"
    b.mkdir()
    assert pit(a, 999)[1] == pit(b, -100)[1]


def test_inclusive_origin(tmp_path):
    assert pit(tmp_path, origin="2020-01-02T01:00:00Z")[1][0]["prediction"] == "999.0"


def test_no_eligible_record(tmp_path):
    with pytest.raises(ValueError, match="missingness"):
        pit(tmp_path, origin="2019-01-01T00:00:00Z")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("bad", ["2020-01-01", "2020-01-01T00:00:00", "2020-01-01T08:00:00+08:00"])
def test_utc_requirement(bad):
    c = SourceContract.from_dict({"columns": {"t": "utc_timestamp"}, "key": ["t"]})
    with pytest.raises(ValueError):
        c.parse({"t": bad})


def test_legacy_tables_untouched(tmp_path, contract):
    with sqlite3.connect(tmp_path / "p.db") as c:
        c.execute("CREATE TABLE raw_records(x TEXT)")
        c.execute("INSERT INTO raw_records VALUES ('old')")
    ingest(tmp_path, contract)
    with sqlite3.connect(tmp_path / "p.db") as c:
        assert c.execute("SELECT x FROM raw_records").fetchone()[0] == "old"


def test_cli(tmp_path, contract):
    import yaml

    write(tmp_path, "sales", rows())
    cfg = tmp_path / "ingest.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "database": "p.db",
                "source_id": "sales",
                "version": "1",
                "input": "sales.csv",
                "contract": contract,
                "batch_size": 1,
            }
        )
    )
    main(["ingest", "--config", str(cfg)])
    cfg.write_text(
        yaml.safe_dump(
            {
                "database": "p.db",
                "base": {"source_id": "sales", "version": "1"},
                "output": "out",
                "batch_size": 1,
            }
        )
    )
    main(["assemble", "--config", str(cfg)])
    assert json.loads((tmp_path / "out/manifest.json").read_text())["rows"] == 2


def test_csv_multiline_bom_and_quotes(tmp_path):
    spec = {"columns": {"id": "string", "value": "string"}, "key": ["id"]}
    p = tmp_path / "special.csv"
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "value"])
        w.writerow(["001", 'line1\nline2,"quoted"'])
    ingest_csv(tmp_path / "p.db", "special", "1", p, spec)
    assert verify_source(tmp_path / "p.db", "special", "1")["rows"] == 1


def test_source_change_during_ingest(tmp_path, contract, monkeypatch):
    from ml_template.temporal_stream import store

    p = write(tmp_path, "mutating", rows())
    original = store.file_hash

    def changed(path):
        result = original(path)
        r = rows()
        r[0]["y"] = 77
        write(tmp_path, "mutating", r)
        return result

    monkeypatch.setattr(store, "file_hash", changed)
    with pytest.raises(ValueError, match="changed during ingestion"):
        ingest_csv(tmp_path / "p.db", "sales", "1", p, contract)


def test_duplicate_header(tmp_path, contract):
    p = tmp_path / "bad.csv"
    p.write_text("id,id,date,y\na,x,2020-01-01,1\n")
    with pytest.raises(ValueError, match="header"):
        ingest_csv(tmp_path / "p.db", "x", "1", p, contract)


def test_temporal_batch_is_positive(tmp_path, contract):
    with pytest.raises(ValueError, match="batch"):
        ingest(tmp_path, contract, batch=0)
