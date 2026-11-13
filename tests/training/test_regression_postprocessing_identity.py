import json

import pytest


def test_default_serialization_preserves_legacy_contracts():
    module = pytest.importorskip("ml_template.training.run_config")
    from ml_template.evaluation.regression_config import RegressionEvaluationConfig

    split = module.SplitConfig()
    assert split.model_dump(mode="json") == dict(
        method="stratified", holdout_fraction=0.2, n_folds=5, seed=42, cv_seed=None
    )
    evaluation = RegressionEvaluationConfig(parent_run_id="run_test")
    data = evaluation.model_dump(mode="json")
    assert "target_unit" not in data and data["uncertainty"] == {"enabled": False}
    assert data["calibration_id"] is None and data["release_variant"] == "raw"
    assert json.loads(evaluation.model_dump_json()) == data


def test_reserved_policy_changes_split_identity():
    module = pytest.importorskip("ml_template.training.run_config")
    from ml_template.training.split import compute_split_id

    old = module.SplitConfig(method="random")
    new = module.SplitConfig(method="random", calibration_fraction=0.15, recalibration_fraction=0.1)
    assert compute_split_id("silver_test", old, "id", "y") != compute_split_id(
        "silver_test", new, "id", "y"
    )
    assert old.model_dump(mode="json").get("calibration_fraction") is None
    with pytest.raises(ValueError):
        module.SplitConfig(calibration_fraction=0.1)
