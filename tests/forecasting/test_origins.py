"Real stdlib/SQLite acceptance for planning; no Arrow stand-ins in these tests."

import json
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from ml_template.forecasting.artifacts import iter_plan_rows, verify_plan
from ml_template.forecasting.planner import _build_plan
from ml_template.forecasting.policy import (
    boundary,
    effective_availability,
    utc,
    validate_policy,
    windows,
)


@pytest.fixture
def fixture(tmp_path):
    policy = {
        "development_origins": ["2020-02-10", "2020-02-12"],
        "train_start": "2020-01-01",
        "stopping_periods": 7,
        "min_train_observations": 28,
        "holdout_periods": 28,
        "business_timezone": "UTC",
    }
    rows = [
        {
            "id": f"{series}-{i}",
            "series": series,
            "date": (date(2020, 1, 1) + timedelta(days=i)).isoformat(),
        }
        for series in ["a", "b"]
        for i in range(80)
    ]
    binding = {
        "silver_dataset_version_id": "silver_ts_fixture",
        "manifest_sha256": "a" * 64,
        "logical_data_sha256": "b" * 64,
        "row_count": len(rows),
        "temporal_schema": {
            "observation_id": "id",
            "series": ["series"],
            "time": "date",
            "target": "y",
            "features": [],
            "frequency": "D",
            "target_delay_periods": 0,
            "origin_policy": "end_of_business_day",
        },
    }
    database = tmp_path / "project.db"
    with sqlite3.connect(database):
        pass
    return (rows, binding, policy, database, tmp_path / "plans")


def run(fixture, **kwargs):
    rows, binding, policy, database, output = fixture
    return _build_plan(rows, binding, policy, output, database, **kwargs)


def test_build_reuse_registry(fixture):
    result = run(fixture, batch_size=3)
    assert result["validation"]["requests"] == 4
    assert result["validation"]["max_index_batch_rows"] == 3
    assert result["identity"]["holdout"]["start"] == "2020-02-22"
    assert run(fixture, batch_size=3)["reused"]
    with sqlite3.connect(fixture[3]) as conn:
        assert conn.execute("SELECT COUNT(*) FROM temporal_origin_plans").fetchone()[0] == 1
    requests = list(iter_plan_rows(result["plan_path"]))
    assert len({r["request_id"] for r in requests}) == 4
    assert all(r["purpose"] == "development" and r["target_date"] < "2020-02-22" for r in requests)
    assert result["validation"]["target_values_read"] is False


def test_windows_delay_embargo(fixture):
    temporal = fixture[1]["temporal_schema"]
    temporal["target_delay_periods"] = 2
    w = windows("2020-02-10", validate_policy(fixture[2]), temporal)
    assert w["refit_end"] == "2020-02-08"
    assert w["stopping_start"] == "2020-02-02"
    assert w["fit_end"] == "2020-01-30"
    assert w["fit_available_by"] == "2020-02-01T23:59:59.999999Z"


def test_start_of_day(fixture):
    temporal = fixture[1]["temporal_schema"]
    temporal["origin_policy"] = "start_of_business_day"
    w = windows("2020-02-10", validate_policy(fixture[2]), temporal)
    assert w["target_date"] == "2020-02-10" and w["refit_end"] == "2020-02-09"
    assert w["issued_at"] == "2020-02-10T00:00:00.000000Z"


def test_actual_delayed_stopping_exclusion(fixture):
    rows, binding, _, _, _ = fixture
    binding["temporal_schema"]["available_at"] = "available"
    for row in rows:
        row["available"] = row["date"] + "T00:00:00Z"
        if row["series"] == "b" and row["date"] == "2020-02-07":
            row["available"] = "2020-03-01T00:00:00Z"
    result = run(fixture)
    assert result["validation"]["requests"] == 2
    assert result["validation"]["exclusions"]["incomplete_stopping_window"] == 2


def test_fit_uses_inner_not_outer_availability(fixture):
    rows, binding, _, _, _ = fixture
    binding["temporal_schema"]["available_at"] = "available"
    for row in rows:
        row["available"] = row["date"] + "T00:00:00Z"
        if row["series"] == "b" and row["date"] == "2020-01-15":
            row["available"] = "2020-02-08T00:00:00Z"
    result = run(fixture)
    eligibility = list(iter_plan_rows(result["plan_path"], "eligibility"))
    a = next(r for r in eligibility if r["series"] == ["a"] and r["origin_date"] == "2020-02-10")
    b = next(r for r in eligibility if r["series"] == ["b"] and r["origin_date"] == "2020-02-10")
    assert a["fit_count"] == b["fit_count"] + 1
    assert a["refit_count"] == b["refit_count"]


def test_batch_boundary_parity(fixture):
    a = run(fixture, batch_size=1)
    b = run(fixture, batch_size=17)
    assert a["identity"]["index_sha256"] == b["identity"]["index_sha256"]
    left = list(iter_plan_rows(a["plan_path"], "eligibility"))
    right = list(iter_plan_rows(b["plan_path"], "eligibility"))
    assert left == right
    assert a["plan_id"] != b["plan_id"]


def test_unknown_series(fixture):
    fixture[2]["series_selection"] = [["unknown"]]
    with pytest.raises(ValueError, match="not found"):
        run(fixture)
    assert list(fixture[4].iterdir()) == []


def test_explicit_subset(fixture):
    fixture[2]["series_selection"] = [["b"]]
    result = run(fixture)
    assert result["validation"]["requests"] == 2
    assert all(r["series"] == ["b"] for r in iter_plan_rows(result["plan_path"]))


def test_duplicate_across_batches(fixture):
    fixture[0][-1] = dict(fixture[0][0])
    with pytest.raises(sqlite3.IntegrityError):
        run(fixture, batch_size=1)
    assert list(fixture[4].iterdir()) == []


def test_holdout_overlap_rejected(fixture):
    fixture[2]["development_origins"] = ["2020-02-21"]
    with pytest.raises(ValueError, match="holdout"):
        run(fixture)


def test_missing_target_excluded(fixture):
    rows = fixture[0]
    rows[:] = [r for r in rows if not (r["series"] == "b" and r["date"] == "2020-02-11")]
    fixture[1]["row_count"] = len(rows)
    result = run(fixture)
    assert result["validation"]["exclusions"]["missing_forecast_observation"] == 1


def test_all_ineligible_fails(fixture):
    fixture[2]["min_train_observations"] = 1000
    with pytest.raises(ValueError, match="No eligible"):
        run(fixture)
    assert list(fixture[4].iterdir()) == []


def test_forbidden_target_projection(fixture):
    fixture[0][0]["y"] = 123
    with pytest.raises(ValueError, match="projection"):
        run(fixture)


def test_late_future_metadata_does_not_change_development_eligibility(fixture):
    a = run(fixture)
    fixture[1]["temporal_schema"]["available_at"] = "available"
    for row in fixture[0]:
        row["available"] = (
            row["date"] + "T00:00:00Z" if row["date"] < "2020-02-22" else "2099-01-01T00:00:00Z"
        )
    b = run(fixture)
    assert list(iter_plan_rows(a["plan_path"], "eligibility")) == list(
        iter_plan_rows(b["plan_path"], "eligibility")
    )


@pytest.mark.parametrize(
    "name",
    ["requests.jsonl", "origins.jsonl", "eligibility.jsonl", "validation.json", "manifest.json"],
)
def test_tamper_rejected(fixture, name):
    result = run(fixture)
    path = Path(result["plan_path"]) / name
    if name == "manifest.json":
        manifest = json.loads(path.read_text())
        manifest["validation"]["requests"] += 1
        path.write_text(json.dumps(manifest))
    else:
        path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="integrity"):
        verify_plan(result["plan_path"])


@pytest.mark.parametrize(
    "key,value",
    [
        ("horizon", 2),
        ("horizon", True),
        ("window", "rolling"),
        ("holdout_periods", 0),
        ("stopping_periods", True),
        ("min_train_observations", -1),
        ("extra", 1),
        ("development_origins", []),
        ("development_origins", ["2020-02-10", "2020-02-10"]),
        ("series_selection", [["a"], ["a"]]),
    ],
)
def test_invalid_policy(fixture, key, value):
    fixture[2][key] = value
    with pytest.raises(ValueError):
        run(fixture)


@pytest.mark.parametrize("value", [0, -1, True])
def test_bad_batch(fixture, value):
    with pytest.raises(ValueError):
        run(fixture, batch_size=value)


def test_utc_naive_rejected():
    with pytest.raises(ValueError):
        utc("2020-01-01T00:00:00")


def test_timezone_dst_boundary():
    start = datetime.fromisoformat(
        boundary("2020-11-01", "America/New_York", False).replace("Z", "+00:00")
    )
    end = datetime.fromisoformat(
        boundary("2020-11-01", "America/New_York", True).replace("Z", "+00:00")
    )
    assert end - start + timedelta(microseconds=1) == timedelta(hours=25)


def test_effective_availability_conservative(fixture):
    temporal = fixture[1]["temporal_schema"]
    assert (
        effective_availability("2020-01-01", "2020-01-01T00:00:00Z", fixture[2], temporal)
        == "2020-01-01T23:59:59.999999Z"
    )


def test_registry_failure_rolls_back(fixture, monkeypatch):
    import ml_template.forecasting.planner as planner

    def fail(*args):
        raise RuntimeError("registry failure")

    monkeypatch.setattr(planner, "register_plan", fail)
    with pytest.raises(RuntimeError, match="registry failure"):
        run(fixture)
    assert list(fixture[4].iterdir()) == []


def test_policy_change_new_identity(fixture):
    first = run(fixture)
    fixture[2]["min_train_observations"] = 29
    second = run(fixture)
    assert first["plan_id"] != second["plan_id"]


def test_silver_binding_change_new_identity(fixture):
    first = run(fixture)
    fixture[1]["manifest_sha256"] = "c" * 64
    second = run(fixture)
    assert first["plan_id"] != second["plan_id"]


def test_malformed_row_count(fixture):
    fixture[1]["row_count"] += 1
    with pytest.raises(ValueError, match="count"):
        run(fixture)


def test_cli_help_without_arrow():
    from ml_template.forecasting.origins import main

    with pytest.raises(SystemExit) as exit:
        main(["--help"])
    assert exit.value.code == 0


def test_short_calendar_history(fixture):
    fixture[2]["train_start"] = "2020-02-09"
    with pytest.raises(ValueError, match="calendar history"):
        run(fixture)


def test_init_creates_local_config_and_refuses_overwrite(tmp_path):
    import yaml

    from ml_template.forecasting.origins import main

    silver = tmp_path / "silver"
    silver.mkdir()
    (silver / "manifest.json").write_text("{}")
    db = tmp_path / "project.db"
    db.touch()
    config = tmp_path / "origins.yaml"
    command = [
        "init",
        "--silver-dataset",
        str(silver),
        "--database",
        str(db),
        "--output-root",
        str(tmp_path / "plans"),
        "--output",
        str(config),
        "--train-start",
        "2020-01-01",
        "--origin",
        "2020-02-10",
        "--business-timezone",
        "UTC",
    ]
    main(command)
    value = yaml.safe_load(config.read_text())
    assert value["policy"]["development_origins"] == ["2020-02-10"]
    assert value["policy"]["holdout_periods"] == 28
    with pytest.raises(FileExistsError):
        main(command)


def test_cli_rejects_unknown_config_before_arrow(tmp_path):
    import yaml

    from ml_template.forecasting.origins import main

    cfg = tmp_path / "bad.yaml"
    cfg.write_text(yaml.safe_dump({"unknown": 1}))
    with pytest.raises(ValueError, match="config"):
        main(["build", "--config", str(cfg)])
