"""Extend the existing real stateless pipeline acceptance test with fitted state.

Reuses its real training, diagnostics, release, offline replay and export checks.
This does not mock a model or waive any release evidence checks.
"""

import importlib.util
from pathlib import Path

import pytest


@pytest.mark.parametrize("engineered", [False, True])
@pytest.mark.parametrize("predictor_kind", ["final", "cv_mean"])
def test_fitted_regression_end_to_end(tmp_path, monkeypatch, engineered, predictor_kind):
    pytest.importorskip("pyarrow")
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("xgboost", minversion="2.0")
    from ml_template.training.run_config import RunConfig

    original = RunConfig.model_validate

    def with_fitted(cls, raw, **kwargs):
        if isinstance(raw, dict) and raw.get("experiment_name") == "stateless_test":
            raw = dict(raw)
            raw["fitted_features"] = [
                {
                    "step": "numeric_preprocessing",
                    "columns": ["a", "b"],
                    "numeric_strategy": "median",
                    "scaling": "standard",
                }
            ]
        return original(raw, **kwargs)

    monkeypatch.setattr(RunConfig, "model_validate", classmethod(with_fitted))
    source = Path(__file__).with_name("test_stateless_fe_pipeline_integration.py")
    spec = importlib.util.spec_from_file_location("stateless_pipeline_acceptance", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.test_native_regression_stateless_end_to_end(
        tmp_path, monkeypatch, engineered, predictor_kind
    )
