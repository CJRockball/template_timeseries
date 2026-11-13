"""Synthetic native Tree SHAP and registered-release dashboard replay."""

import importlib.util
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def test_native_contributions_use_saved_tree_range(tmp_path):
    xgb = pytest.importorskip("xgboost", minversion="2.0")
    from ml_template.explainability.regression_worker import native_contributions

    X = pd.DataFrame({"a": np.linspace(-2, 2, 40), "b": np.arange(40) % 3})
    y = X.a * 3 - X.b
    model = xgb.XGBRegressor(
        objective="reg:squarederror", n_estimators=10, max_depth=2, n_jobs=1, tree_method="hist"
    )
    model.fit(X, y)
    path = tmp_path / "model.ubj"
    model.get_booster().save_model(path)
    booster = xgb.Booster()
    booster.load_model(path)
    adapter = SimpleNamespace(_booster=lambda: booster)
    expanded, phi, base = native_contributions(adapter, X)
    assert expanded.equals(X)
    np.testing.assert_allclose(base + phi.sum(axis=1), model.predict(X), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("engineered", [False, True])
@pytest.mark.parametrize("predictor_kind", ["final", "cv_mean"])
def test_regression_explanation_release_replay(tmp_path, monkeypatch, engineered, predictor_kind):
    pytest.importorskip("pyarrow")
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("xgboost", minversion="2.0")
    import ml_template
    from ml_template.scoring import batch as scoring

    original, captured = scoring.score_batch, []

    def capture(engine, *args, **kwargs):
        result = original(engine, *args, **kwargs)
        if not kwargs.get("check_only", False) and result.get("status") == "succeeded":
            captured.append(result)
        return result

    monkeypatch.setattr(scoring, "score_batch", capture)
    source = Path(__file__).parents[1] / "scoring/test_fitted_regression_pipeline_integration.py"
    spec = importlib.util.spec_from_file_location("regression_shap_acceptance", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.test_fitted_regression_end_to_end(tmp_path, monkeypatch, engineered, predictor_kind)
    scored = captured[-1]
    database = tmp_path / "project.db"
    with sqlite3.connect(database) as conn:
        release_dir = conn.execute(
            "SELECT artifact_dir FROM model_releases WHERE release_id=?", (scored["release_id"],)
        ).fetchone()[0]
    from ml_template.explainability.explain_batch import explain_batch
    from ml_template.explainability.explanation_config import ExplanationConfig
    from ml_template.explainability.presentation_data import load_explanation

    config = ExplanationConfig(
        task="regression",
        target_unit="synthetic_units",
        release_dir=release_dir,
        scoring_dir=scored["artifact_dir"],
        expected_release_id=scored["release_id"],
        expected_scoring_batch_id=scored["scoring_batch_id"],
        runtime_python=sys.executable,
        runtime_source_root=str(Path(ml_template.__file__).parent),
        registry_path=str(database),
        output_root=str(tmp_path / "explanations"),
        chunk_size=1,
    ).resolved(tmp_path)
    checked = explain_batch(config, check_only=True)
    assert checked["ready"] and checked["predictions_computed"] is False
    explained = explain_batch(config)
    assert explained["status"] == "succeeded"
    # Presentation must not deserialize a model or call prediction.
    from ml_template.models import model_factory

    def forbidden(*args, **kwargs):
        raise AssertionError("presentation loaded a model")

    monkeypatch.setattr(model_factory, "load_adapter", forbidden)
    view = load_explanation(database, explained["explanation_id"])
    np.testing.assert_allclose(
        view.rows.base_value + view.contributions.sum(axis=1),
        view.rows.y_pred_release,
        atol=1e-5,
        rtol=1e-5,
    )
    assert view.summary["output_scale"] == "prediction_target_units"
    from ml_template.usage_dashboard.app import global_payload, local_payload

    assert "y_pred_release" in {
        col["id"] for col in global_payload(database, explained["explanation_id"])[6]
    }
    assert "y_pred_release" in local_payload(database, explained["explanation_id"], 0)[2]
    # Hash tampering is rejected by the artifact/presentation boundary.
    path = Path(explained["artifact_dir"]) / "contributions.parquet"
    with path.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError):
        load_explanation(database, explained["explanation_id"])
