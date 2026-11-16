"Core orchestration with explicit history/adapter test doubles, NOT native XGB."

import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from ml_template.forecasting.learned_artifacts import canonical, digest, rows, verify
from ml_template.forecasting.temporal_compare import compare
from ml_template.forecasting.temporal_samples import materialize
from ml_template.forecasting.temporal_train import train


class Requests:
    def __init__(self, path, records):
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE requests(request_id TEXT PRIMARY KEY,series TEXT,date TEXT,origin TEXT,observation_id TEXT,row TEXT)"
        )
        with self.conn:
            for r in records:
                self.conn.execute(
                    "INSERT INTO requests VALUES (?,?,?,?,?,?)",
                    (
                        r["request_id"],
                        canonical(r["series"]),
                        r["target_date"],
                        r["origin_date"],
                        canonical(r["observation_id"]),
                        canonical(r),
                    ),
                )
        self.count = len(records)

    def close(self):
        self.conn.close()


class History:
    def __init__(self, path):
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE history(series TEXT,day TEXT,id TEXT,y REAL,available TEXT,PRIMARY KEY(series,day))"
        )
        self.calls = []
        with self.conn:
            for series in ["a", "b"]:
                for i in range(48):
                    d = (date(2020, 1, 1) + timedelta(days=i)).isoformat()
                    self.conn.execute(
                        "INSERT INTO history VALUES (?,?,?,?,?)",
                        (
                            canonical([series]),
                            d,
                            canonical(f"{series}-{i}"),
                            i + 0.25,
                            d + "T23:59:59.999999Z",
                        ),
                    )

    def for_sample(self, series, target, recipe):
        self.calls.append((series, target))
        target = date.fromisoformat(target)
        issued = (target - timedelta(days=1)).isoformat() + "T23:59:59.999999Z"
        key = canonical(series)
        last = self.conn.execute(
            "SELECT y FROM history WHERE series=? AND day<? AND available<=? ORDER BY day DESC LIMIT 1",
            (key, target.isoformat(), issued),
        ).fetchone()
        lag = self.conn.execute(
            "SELECT y FROM history WHERE series=? AND day=? AND available<=?",
            (key, (target - timedelta(days=7)).isoformat(), issued),
        ).fetchone()
        return {
            "features": {
                "last_available": last[0] if last else None,
                "lag_7": lag[0] if lag else None,
            }
        }

    def close(self):
        self.conn.close()


@pytest.fixture
def fixture(tmp_path):
    t = {"observation_id": "id", "series": ["series"], "time": "date", "target": "y"}
    windows = []
    for origin in ["2020-02-10", "2020-02-17"]:
        d = date.fromisoformat(origin)
        start = (d - timedelta(days=2)).isoformat()
        fit = (d - timedelta(days=3)).isoformat()
        windows.append(
            {
                "origin_date": origin,
                "issued_at": origin + "T23:59:59.999999Z",
                "target_date": (d + timedelta(days=1)).isoformat(),
                "fit_start": "2020-01-01",
                "fit_end": fit,
                "fit_available_by": fit + "T23:59:59.999999Z",
                "stopping_start": start,
                "stopping_end": origin,
                "refit_start": "2020-01-01",
                "refit_end": origin,
            }
        )
    identity = {
        "windows": windows,
        "silver": {
            "temporal_schema": t,
            "silver_dataset_version_id": "silver_ts_fixture",
            "manifest_sha256": "a" * 64,
        },
        "policy": {"min_train_observations": 5, "stopping_periods": 3},
        "holdout": {"start": "2020-02-22", "end": "2020-03-20"},
    }
    plan = {
        "identity": identity,
        "plan_id": "origin_" + digest(identity),
        "manifest_sha256": "b" * 64,
    }
    records = []
    for window in windows:
        for series in ["a", "b"]:
            i = (date.fromisoformat(window["target_date"]) - date(2020, 1, 1)).days
            request = {
                "plan_id": plan["plan_id"],
                "purpose": "development",
                "horizon": 1,
                "series": [series],
                "origin_date": window["origin_date"],
                "issued_at": window["issued_at"],
                "target_date": window["target_date"],
                "observation_id": f"{series}-{i}",
            }
            request["request_id"] = "request_" + digest(request)
            records.append(request)
    request_index = Requests(tmp_path / "requests.db", records)
    history = History(tmp_path / "history.db")
    database = tmp_path / "project.db"
    database.touch()
    config = {"features": ["last_available"], "max_rows_per_partition": 500}
    recipe = {"lags": [7], "rolling": [], "methods": ["last_available", "seasonal_naive_7"]}
    yield (plan, request_index, history, config, recipe, database, tmp_path, records)
    history.close()
    request_index.close()


def samples(fixture):
    p, r, h, c, recipe, db, tmp, _ = fixture
    return materialize(p, r, h, recipe, c, {"kind": "test_double"}, tmp / "artifacts", db)


class Adapter:
    def __init__(self, events):
        self.events = events
        self.early_stopping_history = None

    def fit_with_validation(self, X, y, Xstop, ystop):
        self.events.append(("stop", len(X), len(Xstop)))
        self.value = -123
        return 3

    def refit(self, X, y, rounds):
        self.events.append(("refit", len(X), rounds))
        self.value = float(y.mean())
        self.features = list(X.columns)

    def predict(self, X):
        assert list(X.columns) == self.features
        return np.full(len(X), self.value)

    def save(self, path):
        Path(path).write_text(json.dumps({"value": self.value, "features": self.features}))

    @classmethod
    def load(cls, path):
        state = json.loads(Path(path).read_text())
        model = cls([])
        model.value = state["value"]
        model.features = state["features"]
        return model


def fitted(fixture, sample=None, loader=None):
    sample = sample or samples(fixture)
    events = []
    value = train(
        sample["artifact_path"],
        {"n_estimators": 20},
        42,
        {"kind": "test_double"},
        lambda: Adapter(events),
        loader or Adapter.load,
        fixture[6] / "artifacts",
        fixture[5],
    )
    return (value, events)


def truths(fixture):
    return (
        {
            "id": r["observation_id"],
            "series": r["series"][0],
            "date": r["target_date"],
            "y": (date.fromisoformat(r["target_date"]) - date(2020, 1, 1)).days + 0.25,
        }
        for r in fixture[7]
    )


def comparison(fixture, model, baseline_records=None, outcomes=None):
    p, _, _, _, recipe, db, tmp, requests = fixture
    index = Requests(tmp / ("score-" + str(len(list(tmp.glob("score-*")))) + ".db"), requests)
    context = verify(model["artifact_path"])["identity"]["inputs"]["context"]
    baseline = {
        "artifact_id": "test-baseline",
        "manifest_sha256": "d" * 64,
        "identity": {"inputs": context},
    }
    predictions = (
        baseline_records
        if baseline_records is not None
        else (
            {**request, "method": method, "status": "ok", "reason": None, "y_pred": 0.0}
            for request in requests
            for method in recipe["methods"]
        )
    )
    try:
        return compare(
            p,
            index,
            model["artifact_path"],
            baseline,
            predictions,
            truths(fixture) if outcomes is None else outcomes,
            {"kind": "test_double"},
            tmp / "artifacts",
            db,
        )
    finally:
        index.close()


def test_materialize_and_reuse(fixture):
    result = samples(fixture)
    values = list(rows(result["artifact_path"], "samples.jsonl"))
    assert {row["partition"] for row in values} == {"fit", "stopping", "refit"}
    assert all(row["target_date"] < "2020-02-22" for row in values)
    assert len(list(rows(result["artifact_path"], "requests.jsonl"))) == 4
    assert samples(fixture)["reused"]


def test_label_availability_inner_vs_outer(fixture):
    fixture[2].conn.execute(
        "UPDATE history SET available='2020-02-09T00:00:00.000000Z' WHERE day='2020-01-10'"
    )
    fixture[2].conn.commit()
    result = samples(fixture)
    values = [
        r
        for r in rows(result["artifact_path"], "samples.jsonl")
        if r["origin"] == "2020-02-10" and r["target_date"] == "2020-01-10"
    ]
    assert {r["partition"] for r in values} == {"refit"}


def test_historical_replay_called_for_each_date(fixture):
    samples(fixture)
    assert (["a"], "2020-01-10") in fixture[2].calls
    assert (["a"], "2020-02-11") in fixture[2].calls


def test_complete_feature_exclusions_are_audited(fixture):
    result = samples(fixture)
    report = json.loads((Path(result["artifact_path"]) / "report.json").read_text())
    assert any(r["feature_complete"] < r["label_eligible"] for r in report["counts"])


def test_no_sampling_on_budget_overflow(fixture):
    fixture[3]["max_rows_per_partition"] = 5
    with pytest.raises(ValueError, match="cap"):
        samples(fixture)


def test_stopping_window_feature_gap_fails(fixture):
    fixture[3]["features"] = ["lag_7"]
    fixture[2].conn.execute("DELETE FROM history WHERE day='2020-02-01'")
    fixture[2].conn.commit()
    with pytest.raises(ValueError, match="stopping"):
        samples(fixture)


def test_holdout_window_rejected(fixture):
    fixture[0]["identity"]["windows"][0]["target_date"] = "2020-02-22"
    with pytest.raises(ValueError, match="Holdout"):
        samples(fixture)


def test_fresh_refits_and_rounds(fixture):
    result, events = fitted(fixture)
    assert [event[0] for event in events] == ["stop", "refit", "stop", "refit"]
    assert all(event[2] == 3 for event in events if event[0] == "refit")
    predictions = list(rows(result["artifact_path"], "predictions.jsonl"))
    assert len(predictions) == 4 and all(r["method"] == "xgb" for r in predictions)
    assert all(r["y_pred"] != -123 for r in predictions)


def test_save_load_mismatch_refuses_publication(fixture):

    def bad(path):
        model = Adapter.load(path)
        model.value += 1
        return model

    with pytest.raises(ValueError, match="parity"):
        fitted(fixture, loader=bad)
    assert not any(
        verify(p)["identity"]["kind"] == "models"
        for p in (fixture[6] / "artifacts").glob("tsxgb_*")
    )


def test_comparison_metrics_common_coverage(fixture):
    model, _ = fitted(fixture)
    result = comparison(fixture, model)
    metrics = json.loads((Path(result["artifact_path"]) / "metrics.json").read_text())["metrics"]
    pooled = [
        r for r in metrics if r["comparison"] == "common_supported" and r["scope"] == "pooled"
    ]
    assert len(pooled) == 3 and all(r["n"] == 4 for r in pooled)
    assert {r["method"] for r in pooled} == {"xgb", "last_available", "seasonal_naive_7"}


def test_bad_prediction_before_any_outcome_iteration(fixture):
    model, _ = fitted(fixture)
    bad = [
        {**request, "method": method, "status": "ok", "reason": None, "y_pred": 0.0}
        for request in fixture[7]
        for method in fixture[4]["methods"]
    ]
    bad[0]["observation_id"] = "wrong"

    def forbidden():
        raise AssertionError("truth read too early")
        yield

    with pytest.raises(ValueError, match="Misaligned"):
        comparison(fixture, model, bad, forbidden())


def test_missing_predictions_refused_before_truth(fixture):
    model, _ = fitted(fixture)

    def forbidden():
        raise AssertionError("truth read too early")
        yield

    with pytest.raises(ValueError, match="Missing prediction"):
        comparison(fixture, model, [], forbidden())


def test_holdout_outcome_rejected(fixture):
    model, _ = fitted(fixture)
    outcome = list(truths(fixture)) + [
        {"id": "held", "series": "a", "date": "2020-02-22", "y": 1.25}
    ]
    with pytest.raises(ValueError, match="bounds"):
        comparison(fixture, model, outcomes=iter(outcome))


def test_sample_file_tamper(fixture):
    value = samples(fixture)
    file = Path(value["artifact_path"]) / "samples.jsonl"
    file.write_text(file.read_text() + " ")
    with pytest.raises(ValueError, match="integrity"):
        verify(value["artifact_path"])


def test_unknown_feature_rejected(fixture):
    fixture[3]["features"] = ["y"]
    with pytest.raises(ValueError, match="Unknown"):
        samples(fixture)


@pytest.mark.parametrize("value", [0, -1, True])
def test_bad_cap(fixture, value):
    fixture[3]["max_rows_per_partition"] = value
    with pytest.raises(ValueError):
        samples(fixture)


def test_future_labels_do_not_change_first_origin_samples(fixture):
    first = samples(fixture)
    before = [
        r for r in rows(first["artifact_path"], "samples.jsonl") if r["origin"] == "2020-02-10"
    ]
    fixture[2].conn.execute("UPDATE history SET y=-99999.125 WHERE day>'2020-02-10'")
    fixture[2].conn.commit()
    second = samples(fixture)
    after = [
        r for r in rows(second["artifact_path"], "samples.jsonl") if r["origin"] == "2020-02-10"
    ]
    assert before == after


def test_unsupported_forecast_retained(fixture):
    fixture[3]["features"] = ["lag_7"]
    fixture[2].conn.execute(
        "DELETE FROM history WHERE series=? AND day='2020-02-04'", (canonical(["a"]),)
    )
    fixture[2].conn.commit()
    model, _ = fitted(fixture)
    predictions = list(rows(model["artifact_path"], "predictions.jsonl"))
    assert len(predictions) == 4
    assert sum(p["status"] == "unsupported" for p in predictions) == 1
    result = comparison(fixture, model)
    pooled = [
        r
        for r in json.loads((Path(result["artifact_path"]) / "metrics.json").read_text())["metrics"]
        if r["scope"] == "pooled" and r["comparison"] == "common_supported"
    ]
    assert all(r["n"] == 3 for r in pooled)


def test_model_file_tamper(fixture):
    model, _ = fitted(fixture)
    (Path(model["artifact_path"]) / "models/0000.ubj").write_text("changed")
    with pytest.raises(ValueError, match="integrity"):
        verify(model["artifact_path"])


def test_producer_change_rejected(fixture):
    value = samples(fixture)
    with pytest.raises(ValueError, match="producer"):
        train(
            value["artifact_path"],
            {"n_estimators": 20},
            42,
            {"different": True},
            lambda: Adapter([]),
            Adapter.load,
            fixture[6] / "artifacts",
            fixture[5],
        )


def test_outcome_identity_mismatch(fixture):
    model, _ = fitted(fixture)
    outcome = list(truths(fixture))
    outcome[0]["id"] = "wrong"
    with pytest.raises(ValueError, match="key mismatch"):
        comparison(fixture, model, outcomes=iter(outcome))


def test_missing_outcomes_rejected(fixture):
    model, _ = fitted(fixture)
    with pytest.raises(ValueError, match="Missing requested outcomes"):
        comparison(fixture, model, outcomes=[])


def test_cli_help_without_native_dependencies():
    from ml_template.forecasting.temporal_xgb import main

    with pytest.raises(SystemExit) as result:
        main(["--help"])
    assert result.value.code == 0
