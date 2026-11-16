"Mandatory native gate: real Parquet, history replay and XGBoost 2.1.1."

import csv
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest
import xgboost

from ml_template.forecasting.history_io import predict
from ml_template.forecasting.io import build_from_silver
from ml_template.forecasting.learned_artifacts import verify
from ml_template.forecasting.temporal_xgb import main
from ml_template.forecasting.temporal_xgb_io import build_samples, fit, score
from ml_template.temporal_stream.silver import build
from ml_template.temporal_stream.store import ingest_csv


@pytest.fixture
def native(tmp_path):
    assert xgboost.__version__ == "2.1.1"
    raw = tmp_path / "raw.csv"
    columns = {"id": "string", "series": "string", "date": "date", "y": "float"}
    with raw.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(list(columns))
        for series in ["a", "b"]:
            for i in range(80):
                writer.writerow(
                    [
                        f"{series}-{i}",
                        series,
                        (date(2020, 1, 1) + timedelta(days=i)).isoformat(),
                        i + 0.25,
                    ]
                )
    t = {"series": ["series"], "time": "date", "target": "y", "frequency": "D", "gaps": "error"}
    db = tmp_path / "project.db"
    ingest_csv(db, "fixture", "v1", raw, {"columns": columns, "key": ["id"], "temporal": t})
    schema = {
        "columns": columns,
        "observation_id": "id",
        **t,
        "features": [],
        "known_future": [],
        "target_delay_periods": 0,
        "origin_policy": "end_of_business_day",
    }
    silver = build(
        db,
        {"source": {"source_id": "fixture", "version": "v1"}},
        schema,
        tmp_path / "silver",
        batch_size=9,
        max_rows_per_file=23,
    )
    policy = {
        "development_origins": ["2020-02-10", "2020-02-17"],
        "train_start": "2020-01-01",
        "stopping_periods": 3,
        "min_train_observations": 5,
        "holdout_periods": 28,
        "business_timezone": "UTC",
    }
    plan = build_from_silver(silver["dataset_path"], policy, db, tmp_path / "plans", batch_size=11)
    recipe = {"lags": [7], "rolling": [], "methods": ["last_available", "seasonal_naive_7"]}
    baseline = predict(
        plan["plan_path"], silver["dataset_path"], recipe, tmp_path / "baselines", db, 7
    )
    config = {
        "plan": Path(plan["plan_path"]),
        "silver_dataset": Path(silver["dataset_path"]),
        "database": db,
        "output_root": tmp_path / "learned",
        "batch_size": 7,
        "recipe": recipe,
        "samples": {"features": ["last_available", "lag_7"], "max_rows_per_partition": 500},
        "model": {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "n_estimators": 30,
            "early_stopping_rounds": 5,
            "max_depth": 2,
            "device": "cpu",
        },
        "seed": 42,
    }
    return (config, baseline, tmp_path)


def test_native_complete_vertical_slice(native):
    config, baseline, _ = native
    sample = build_samples(config)
    replay = [
        json.loads(line)
        for line in (Path(sample["artifact_path"]) / "samples.jsonl").read_text().splitlines()
    ]
    assert all(row["x"] == [8.25, 2.25] for row in replay if row["target_date"] == "2020-01-10")
    model = fit(config, sample["artifact_path"])
    report = json.loads((Path(model["artifact_path"]) / "report.json").read_text())
    assert len(report["models"]) == 2
    assert all(1 <= r["selected_rounds"] <= 30 for r in report["models"])
    result = score(config, model["artifact_path"], baseline["artifact_path"])
    metrics = json.loads((Path(result["artifact_path"]) / "metrics.json").read_text())["metrics"]
    pooled = [
        r for r in metrics if r["comparison"] == "common_supported" and r["scope"] == "pooled"
    ]
    assert len(pooled) == 3 and all(r["n"] == 4 for r in pooled)
    assert pa.__version__


def test_native_saved_feature_order(native):
    from ml_template.models.xgb_regressor import XGBRegressionAdapter
    from ml_template.training.run_config import XGBRegressionParams

    config, _, _ = native
    sample = build_samples(config)
    model = fit(config, sample["artifact_path"])
    params = XGBRegressionParams.model_validate(config["model"])
    loaded = XGBRegressionAdapter.load(params, 42, Path(model["artifact_path"]) / "models/0000.ubj")
    assert loaded.estimator_feature_names() == ["last_available", "lag_7"]
    with pytest.raises(ValueError, match="feature order"):
        loaded.predict(pd.DataFrame([[1.0, 2.0]], columns=["lag_7", "last_available"]))


def test_native_cli(native, tmp_path, capsys):
    import yaml

    config, baseline, _ = native
    old = tmp_path / "history.yaml"
    old.write_text(
        yaml.safe_dump(
            {
                k: str(v) if isinstance(v, Path) else v
                for k, v in config.items()
                if k not in {"samples", "model", "seed"}
            }
        )
    )
    new = tmp_path / "xgb.yaml"
    main(["init", "--history-config", str(old), "--output", str(new)])
    settings = yaml.safe_load(new.read_text())
    settings["model"] = config["model"]
    settings["samples"] = config["samples"]
    new.write_text(yaml.safe_dump(settings))
    main(["materialize", "--config", str(new)])
    root = Path(settings["output_root"])
    samples = next(p for p in root.glob("tsxgb_*") if verify(p)["identity"]["kind"] == "samples")
    main(["train", "--config", str(new), "--samples", str(samples)])
    models = next(p for p in root.glob("tsxgb_*") if verify(p)["identity"]["kind"] == "models")
    main(
        [
            "compare",
            "--config",
            str(new),
            "--models",
            str(models),
            "--baseline",
            baseline["artifact_path"],
        ]
    )
    comparison = next(p for p in root.glob("tsxgb_*") if verify(p)["identity"]["kind"] == "scores")
    main(["verify", "--artifact", str(comparison)])
    assert '"kind": "scores"' in capsys.readouterr().out
