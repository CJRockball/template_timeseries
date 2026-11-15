"Stdlib/SQLite core acceptance. Real Arrow integration lives in a separate gate."

import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pytest

from ml_template.forecasting.history import HistoryIndex, recipe
from ml_template.forecasting.history_artifacts import Publication, verify_artifact
from ml_template.forecasting.history_engine import _predict, _score
from ml_template.forecasting.policy import canonical, digest, validate_policy, windows


@pytest.fixture
def fixture(tmp_path):
    temporal = {
        "columns": {"id": "string", "series": "string", "date": "date", "y": "float"},
        "observation_id": "id",
        "series": ["series"],
        "time": "date",
        "target": "y",
        "frequency": "D",
        "target_delay_periods": 0,
        "origin_policy": "end_of_business_day",
    }
    policy = validate_policy(
        {
            "development_origins": ["2020-02-10", "2020-02-17"],
            "train_start": "2020-01-01",
            "stopping_periods": 7,
            "min_train_observations": 28,
            "holdout_periods": 28,
            "business_timezone": "UTC",
        }
    )
    identity = {
        "silver": {
            "silver_dataset_version_id": "silver_ts_fixture",
            "manifest_sha256": "a" * 64,
            "logical_data_sha256": "b" * 64,
            "row_count": 200,
            "temporal_schema": temporal,
        },
        "policy": policy,
        "holdout": {"start": "2020-03-13", "end": "2020-04-09"},
        "windows": [windows(o, policy, temporal) for o in policy["development_origins"]],
    }
    plan = {
        "plan_id": "origin_" + digest(identity),
        "manifest_sha256": "c" * 64,
        "identity": identity,
        "validation": {"requests": 4},
    }
    requests = []
    for w in identity["windows"]:
        for series in ["a", "b"]:
            i = (date.fromisoformat(w["target_date"]) - date(2020, 1, 1)).days
            request = {
                "plan_id": plan["plan_id"],
                "purpose": "development",
                "series": [series],
                "origin_date": w["origin_date"],
                "issued_at": w["issued_at"],
                "target_date": w["target_date"],
                "horizon": 1,
                "observation_id": f"{series}-{i}",
            }
            request["request_id"] = "request_" + digest(request)
            requests.append(request)
    records = [
        {
            "id": f"{series}-{i}",
            "series": series,
            "date": (date(2020, 1, 1) + timedelta(days=i)).isoformat(),
            "y": float(i) + (100 if series == "b" else 0),
        }
        for series in ["a", "b"]
        for i in range(100)
    ]
    database = tmp_path / "project.db"
    database.touch()
    return (plan, requests, records, database, tmp_path)


def history_rows(fixture):
    end = max(w["refit_end"] for w in fixture[0]["identity"]["windows"])
    return (dict(r) for r in fixture[2] if r["date"] <= end)


def outcome_rows(fixture):
    dates = [r["target_date"] for r in fixture[1]]
    return (dict(r) for r in fixture[2] if min(dates) <= r["date"] <= max(dates))


def predict(fixture, batch=5000):
    plan, requests, _, database, tmp = fixture
    return _predict(
        plan, iter(requests), history_rows(fixture), None, tmp / "predictions", database, batch
    )


def score(fixture, prediction, outcomes=None):
    plan, requests, _, database, tmp = fixture
    return _score(
        plan,
        iter(requests),
        prediction["artifact_path"],
        outcome_rows(fixture) if outcomes is None else outcomes,
        None,
        tmp / "scores",
        database,
    )


def feature_rows(prediction):
    return [
        json.loads(line)
        for line in (Path(prediction["artifact_path"]) / "features.jsonl").read_text().splitlines()
    ]


def direct_index(fixture, tmp_path):
    plan = fixture[0]
    index = HistoryIndex(
        tmp_path / "index.sqlite",
        plan["identity"]["policy"],
        plan["identity"]["silver"]["temporal_schema"],
        plan["identity"]["holdout"]["start"],
        "2020-02-17",
    )
    index.add_rows(history_rows(fixture), batch_size=3)
    return index


def test_end_to_end_and_reuse(fixture):
    prediction = predict(fixture)
    assert prediction["stats"]["requests"] == 4
    assert prediction["stats"]["coverage"] == {"last_available": 4, "seasonal_naive_7": 4}
    assert prediction["stats"]["outcomes_joined"] is False
    assert predict(fixture)["reused"]
    result = score(fixture, prediction)
    metrics = json.loads((Path(result["artifact_path"]) / "metrics.json").read_text())["metrics"]
    pooled = {(r["method"], r["comparison"]): r for r in metrics if r["scope"] == "pooled"}
    assert pooled["last_available", "common_supported"]["mae"] == 1
    assert pooled["last_available", "common_supported"]["rmse"] == 1
    assert pooled["seasonal_naive_7", "common_supported"]["mae"] == 7
    assert result["stats"]["common_supported"] == 4
    assert score(fixture, prediction)["reused"]
    with sqlite3.connect(fixture[3]) as conn:
        assert conn.execute("SELECT COUNT(*) FROM temporal_history_artifacts").fetchone()[0] == 2


def test_batch_parity(fixture):
    first = predict(fixture, 1)
    second = predict(fixture, 11)
    assert first["artifact_id"] == second["artifact_id"] and second["reused"]


def test_future_values_do_not_change_earlier_features(fixture):
    first = feature_rows(predict(fixture))
    for row in fixture[2]:
        if row["date"] > "2020-02-10":
            row["y"] = -98765.125
    second = feature_rows(predict(fixture))
    assert [r for r in first if r["origin_date"] == "2020-02-10"] == [
        r for r in second if r["origin_date"] == "2020-02-10"
    ]


def test_missing_calendar_date_is_not_row_lag(fixture):
    fixture[2][:] = [
        r for r in fixture[2] if not (r["series"] == "a" and r["date"] == "2020-02-04")
    ]
    result = predict(fixture)
    row = next(

            r
            for r in feature_rows(result)
            if r["series"] == ["a"] and r["origin_date"] == "2020-02-10"

    )
    assert row["features"]["lag_7"] is None
    assert row["features"]["mean_7"] is None
    assert row["diagnostics"]["rolling_counts"]["7"] == 6
    scores = score(fixture, result)
    assert scores["stats"]["common_supported"] == 3
    assert result["stats"]["coverage"]["last_available"] == 4
    assert result["stats"]["coverage"]["seasonal_naive_7"] == 3


def test_late_availability(fixture):
    fixture[0]["identity"]["silver"]["temporal_schema"]["available_at"] = "available"
    for row in fixture[2]:
        row["available"] = row["date"] + "T00:00:00Z"
        if row["series"] == "a" and row["date"] == "2020-02-10":
            row["available"] = "2020-02-12T00:00:00Z"
    rows = feature_rows(predict(fixture))
    first = next(r for r in rows if r["series"] == ["a"] and r["origin_date"] == "2020-02-10")
    assert first["features"]["lag_1"] is None
    assert first["unsupported_features"]["lag_1"] == "late_availability"
    assert first["diagnostics"]["last_available_date"] == "2020-02-09"


def test_own_sample_origin_not_outer_origin(fixture, tmp_path):
    index = direct_index(fixture, tmp_path)
    try:
        row = index.for_sample(["a"], "2020-01-20")
        assert row["features"]["last_available"] == 18.0
        with pytest.raises(ValueError, match="own origin"):
            index.features(["a"], "2020-01-20", "2020-02-17", "2020-02-17T23:59:59.999999Z")
    finally:
        index.close()


def test_series_isolation(fixture):
    rows = feature_rows(predict(fixture))
    a = next(r for r in rows if r["series"] == ["a"])
    b = next(r for r in rows if r["series"] == ["b"])
    assert b["features"]["last_available"] - a["features"]["last_available"] == 100


def test_delay_does_not_move_rolling_window(fixture, tmp_path):
    fixture[0]["identity"]["silver"]["temporal_schema"]["target_delay_periods"] = 2
    index = direct_index(fixture, tmp_path)
    try:
        row = index.for_sample(["a"], "2020-02-11")
        assert row["features"]["lag_1"] is None
        assert row["unsupported_features"]["lag_1"] == "unavailable_by_policy"
        assert row["diagnostics"]["rolling_end"] == "2020-02-08"
        assert row["diagnostics"]["last_available_date"] == "2020-02-08"
    finally:
        index.close()


def test_start_of_day(fixture, tmp_path):
    fixture[0]["identity"]["silver"]["temporal_schema"]["origin_policy"] = "start_of_business_day"
    index = direct_index(fixture, tmp_path)
    try:
        row = index.for_sample(["a"], "2020-02-11")
        assert row["diagnostics"]["rolling_end"] == "2020-02-10"
        assert row["features"]["lag_1"] == 40.0
    finally:
        index.close()


def test_float_negative_values_preserved(fixture):
    for row in fixture[2]:
        row["y"] = -1.25
    prediction = predict(fixture)
    assert all(row["features"]["last_available"] == -1.25 for row in feature_rows(prediction))
    result = score(fixture, prediction)
    metrics = json.loads((Path(result["artifact_path"]) / "metrics.json").read_text())["metrics"]
    assert all(row["mae"] == 0 for row in metrics)


def test_no_outcome_iteration_for_bad_predictions(fixture):
    prediction = predict(fixture)
    path = Path(prediction["artifact_path"]) / "predictions.jsonl"
    path.write_text(path.read_text() + " ")

    def forbidden():
        raise AssertionError("outcomes must not be read")
        yield

    with pytest.raises(ValueError, match="integrity"):
        score(fixture, prediction, forbidden())


def test_outcome_key_mismatch(fixture):
    prediction = predict(fixture)
    outcomes = list(outcome_rows(fixture))
    outcomes[0]["id"] = "wrong"
    with pytest.raises(ValueError, match="key mismatch"):
        score(fixture, prediction, iter(outcomes))


def test_missing_outcome(fixture):
    prediction = predict(fixture)
    outcomes = [r for r in outcome_rows(fixture) if r["date"] != "2020-02-11"]
    with pytest.raises(ValueError, match="Missing requested outcomes"):
        score(fixture, prediction, iter(outcomes))


def test_holdout_outcome_rejected(fixture):
    prediction = predict(fixture)
    outcomes = list(outcome_rows(fixture)) + [
        {"id": "held", "series": "a", "date": "2020-03-13", "y": 0.0}
    ]
    with pytest.raises(ValueError, match="holdout"):
        score(fixture, prediction, iter(outcomes))


def test_outcomes_do_not_modify_predictions(fixture):
    prediction = predict(fixture)
    before = verify_artifact(prediction["artifact_path"])
    score(fixture, prediction)
    for row in fixture[2]:
        if row["date"] in {r["target_date"] for r in fixture[1]}:
            row["y"] += 10000
    score(fixture, prediction)
    assert verify_artifact(prediction["artifact_path"]) == before


def test_holdout_request_prohibited(fixture):
    request = fixture[1][0]
    request["target_date"] = "2020-03-13"
    with pytest.raises(ValueError, match="Holdout"):
        predict(fixture)


def test_duplicate_requests_rejected(fixture):
    fixture[1][1] = dict(fixture[1][0])
    with pytest.raises(sqlite3.IntegrityError):
        predict(fixture)


def test_history_does_not_return_holdout(fixture, tmp_path):
    index = direct_index(fixture, tmp_path)
    try:
        with pytest.raises(ValueError, match="Holdout"):
            index.for_sample(["a"], "2020-03-13")
    finally:
        index.close()


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"lags": [0], "rolling": [], "methods": ["last_available"]},
        {"lags": [7, 7], "rolling": [], "methods": ["last_available"]},
        {"lags": [1], "rolling": [], "methods": ["seasonal_naive_7"]},
        {"lags": [7], "rolling": [{"periods": 7, "min_periods": 8}], "methods": ["last_available"]},
    ],
)
def test_bad_recipe(bad):
    with pytest.raises(ValueError):
        recipe(bad)


def test_explicit_partial_rolling_window(fixture, tmp_path):
    fixture[2][:] = [r for r in fixture[2] if r["date"] != "2020-02-04"]
    index = direct_index(fixture, tmp_path)
    try:
        config = {
            "lags": [7],
            "rolling": [{"periods": 7, "min_periods": 6}],
            "methods": ["last_available"],
        }
        row = index.for_sample(["a"], "2020-02-11", config)
        assert row["features"]["mean_7"] is not None
    finally:
        index.close()


def test_all_unsupported_is_explicit(fixture):
    truths = list(outcome_rows(fixture))
    fixture[2][:] = []
    prediction = predict(fixture)
    assert set(prediction["stats"]["coverage"].values()) == {0}
    result = score(fixture, prediction, iter(truths))
    metrics = json.loads((Path(result["artifact_path"]) / "metrics.json").read_text())["metrics"]
    assert all(row["n"] == 0 and row["mae"] is None for row in metrics)


def altered_prediction(fixture, prediction, change):
    source = Path(prediction["artifact_path"])
    publication = Publication(fixture[4] / "altered")
    try:
        (publication.stage / "features.jsonl").write_bytes((source / "features.jsonl").read_bytes())
        rows = [
            json.loads(line) for line in (source / "predictions.jsonl").read_text().splitlines()
        ]
        change(rows)
        (publication.stage / "predictions.jsonl").write_text(
            "".join(canonical(row) + "\n" for row in rows)
        )
        return publication.publish(
            "predictions", prediction["identity"]["inputs"], prediction["stats"], fixture[3]
        )
    finally:
        publication.close()


@pytest.mark.parametrize(
    "change", ["duplicate", "missing", "unexpected", "misaligned", "status", "method"]
)
def test_bad_prediction_payload_before_truth(fixture, change):
    original = predict(fixture)

    def mutate(rows):
        if change == "duplicate":
            rows.append(dict(rows[0]))
        elif change == "missing":
            rows.pop()
        elif change == "unexpected":
            rows[0]["request_id"] = "unexpected"
        elif change == "misaligned":
            rows[0]["observation_id"] = "wrong"
        elif change == "status":
            rows[0]["status"] = "unknown"
        else:
            rows[0]["method"] = "unknown"

    bad = altered_prediction(fixture, original, mutate)

    def forbidden():
        raise AssertionError("outcomes iterated before prediction validation")
        yield

    with pytest.raises((ValueError, sqlite3.IntegrityError)):
        score(fixture, bad, forbidden())


def test_registry_failure_cleans_new_output(fixture, monkeypatch):
    import ml_template.forecasting.history_artifacts as artifacts

    def fail(*args):
        raise RuntimeError("registry failure")

    monkeypatch.setattr(artifacts, "register", fail)
    with pytest.raises(RuntimeError, match="registry failure"):
        predict(fixture)
    assert list((fixture[4] / "predictions").iterdir()) == []


def test_recipe_binding_change_rejected_before_truth(fixture):
    prediction = predict(fixture)
    config = recipe()
    config["rolling"] = []

    def forbidden():
        raise AssertionError("unexpected truth access")
        yield

    with pytest.raises(ValueError, match="binding mismatch"):
        _score(
            fixture[0],
            fixture[1],
            prediction["artifact_path"],
            forbidden(),
            config,
            fixture[4] / "scores",
            fixture[3],
        )


@pytest.mark.parametrize("kind", ["duplicate", "nonfinite"])
def test_bad_outcomes(fixture, kind):
    prediction = predict(fixture)
    truths = list(outcome_rows(fixture))
    if kind == "duplicate":
        truths.append(dict(truths[0]))
    else:
        truths[0]["y"] = float("inf")
    with pytest.raises((ValueError, sqlite3.IntegrityError)):
        score(fixture, prediction, iter(truths))


def test_request_hash_invalid(fixture):
    fixture[1][0]["request_id"] = "request_bad"
    with pytest.raises(ValueError, match="integrity"):
        predict(fixture)


def test_cli_help_without_arrow():
    from ml_template.forecasting.history_cli import main

    with pytest.raises(SystemExit) as result:
        main(["--help"])
    assert result.value.code == 0


@pytest.mark.parametrize("number", [True, float("nan"), float("inf"), "1.25"])
def test_non_numeric_history_rejected(fixture, number):
    fixture[2][0]["y"] = number
    with pytest.raises(ValueError):
        predict(fixture)


def test_history_projection_guard(fixture):
    fixture[2][0]["unexpected"] = 123
    with pytest.raises(ValueError, match="projection"):
        predict(fixture)


def test_cli_init_no_overwrite(tmp_path):
    import yaml

    from ml_template.forecasting.history_cli import main

    plan = tmp_path / "plan"
    silver = tmp_path / "silver"
    plan.mkdir()
    silver.mkdir()
    (plan / "manifest.json").write_text("{}")
    (silver / "manifest.json").write_text("{}")
    database = tmp_path / "project.db"
    database.touch()
    config = tmp_path / "history.yaml"
    command = [
        "init",
        "--plan",
        str(plan),
        "--silver-dataset",
        str(silver),
        "--database",
        str(database),
        "--output-root",
        str(tmp_path / "output"),
        "--output",
        str(config),
    ]
    main(command)
    assert yaml.safe_load(config.read_text())["recipe"]["lags"] == [1, 7, 14, 28]
    with pytest.raises(FileExistsError):
        main(command)


def test_unexpected_config_rejected_without_arrow(tmp_path):
    from ml_template.forecasting.history_cli import main

    config = tmp_path / "bad.yaml"
    config.write_text("wrong: 1")
    with pytest.raises(ValueError, match="config"):
        main(["predict", "--config", str(config)])


def test_late_past_value_mutation_cannot_affect_earlier_forecast(fixture):
    fixture[0]["identity"]["silver"]["temporal_schema"]["available_at"] = "available"
    for row in fixture[2]:
        row["available"] = row["date"] + "T00:00:00Z"
        if row["date"] == "2020-02-04":
            row["available"] = "2020-02-12T00:00:00Z"
    before = feature_rows(predict(fixture))
    for row in fixture[2]:
        if row["date"] == "2020-02-04":
            row["y"] = -999999.875
    after = feature_rows(predict(fixture))
    assert [r for r in before if r["origin_date"] == "2020-02-10"] == [
        r for r in after if r["origin_date"] == "2020-02-10"
    ]


def test_duplicate_history_across_batches(fixture):
    fixture[2].append(dict(fixture[2][0]))
    with pytest.raises(sqlite3.IntegrityError):
        predict(fixture, 1)
    assert list((fixture[4] / "predictions").iterdir()) == []
