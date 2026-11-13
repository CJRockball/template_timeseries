import copy
import json
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from ml_template.evaluation.decision_policy import apply_policy, policy_metrics
from ml_template.evaluation.evaluation_config import EvaluationConfig, PolicyConfig
from ml_template.evaluation.policy_selection import (
    DATA_FIELDS,
    SOURCE_FILES,
    save_selection,
    selection_candidate,
    sha256,
    validate_selection_record,
)


@pytest.fixture
def inputs():
    frame = pd.DataFrame(
        {
            "udi": [4, 1, 3, 2],
            "fold": [1, 0, 0, 1],
            "y_true": [1, 0, 1, 0],
            "p_raw": [0.9, 0.1, 0.8, 0.2],
            "p_cal_crossfit": [0.7, 0.01, 0.6, 0.02],
        }
    )
    provenance = {key: "value_" + key for key in DATA_FIELDS}
    provenance.update(parent_run_id="run_test", parent_files_sha256={"model.ubj": "b" * 64})
    return {
        "frame": frame,
        "key_col": "udi",
        "provenance": provenance,
        "source_hashes": {key: "a" * 64 for key in SOURCE_FILES},
    }


def save(inputs, tmp_path, **kwargs):
    return save_selection(
        inputs,
        parent_run_id="run_test",
        calibration_id="cal_test",
        score_variant="calibrated_crossfit",
        objective="f2",
        acknowledge_transfer=True,
        output_root=tmp_path,
        **kwargs,
    )


def test_f1_and_f2_candidates():
    for objective in ["f1", "f2"]:
        candidate = selection_candidate([0, 0, 1, 1], [0.1, 0.2, 0.7, 0.9], objective)
        assert candidate["value"] == 0.7
        assert candidate["diagnostic"]["objective_value"] == 1


def test_highest_cutoff_for_objective_tie():
    candidate = selection_candidate([0, 1], [0.2, 0.8], "f1")
    assert candidate["value"] == 0.8


def test_budget_boundary_is_not_scalar_cutoff():
    candidate = selection_candidate([0, 1, 1, 0], [0.5] * 4, "alert_rate", 0.5)
    assert candidate["kind"] == "alert_rate"
    assert candidate["diagnostic"]["threshold_exact"] is False
    assert candidate["diagnostic"]["flagged"] == 2


@pytest.mark.parametrize(
    "objective,value",
    [
        ("f1", 0.2),
        ("manual", None),
        ("manual", 2),
        ("manual", float("nan")),
        ("alert_rate", 0),
        ("bad", None),
    ],
)
def test_invalid_selection(objective, value):
    with pytest.raises(ValueError):
        selection_candidate([0, 1], [0.1, 0.8], objective, value)


def test_no_positive_optimum():
    with pytest.raises(ValueError, match="no deployable"):
        selection_candidate([0, 0], [0.1, 0.8], "f1")


def test_manual_does_not_search(monkeypatch):
    import ml_template.evaluation.policy_selection as module

    monkeypatch.setattr(
        module, "threshold_profile", lambda *args: pytest.fail("manual policy must not search")
    )
    assert module.selection_candidate([0, 1], [0.1, 0.8], "manual", 0.4)["value"] == 0.4


def test_save_and_roundtrip(inputs, tmp_path):
    import yaml

    result = save(inputs, tmp_path)
    record = json.loads(Path(result["selection_file"]).read_text())
    cfg = EvaluationConfig.model_validate(
        yaml.safe_load(Path(result["evaluation_draft"]).read_text())
    )
    assert cfg.release_variant == "calibrated"
    assert cfg.policy.selection.artifact_sha256 == sha256(result["selection_file"])
    assert cfg.policy.value == 0.6
    assert "selection-biased" in record["assessment"]
    assert not list(tmp_path.rglob("*holdout*"))
    verified = validate_selection_record(cfg, record, inputs["provenance"], inputs["source_hashes"])
    assert verified["selection_id"] == result["selection_id"]


def test_transfer_ack_required_before_write(inputs, tmp_path):
    with pytest.raises(ValueError, match="acknowledge-transfer"):
        save_selection(
            inputs,
            parent_run_id="run_test",
            calibration_id="cal_test",
            score_variant="calibrated_crossfit",
            objective="f1",
            output_root=tmp_path,
        )
    assert not list(tmp_path.iterdir())


def test_p_cal_fit_forbidden(inputs, tmp_path):
    with pytest.raises(ValueError, match="only raw or calibrated_crossfit"):
        save_selection(
            inputs,
            parent_run_id="run_test",
            calibration_id="cal_test",
            score_variant="p_cal_fit",
            objective="f1",
            output_root=tmp_path,
        )


def test_multiple_selections_are_separate(inputs, tmp_path):
    a = save(inputs, tmp_path)
    before = Path(a["selection_file"]).read_bytes()
    b = save(inputs, tmp_path)
    assert a["selection_id"] != b["selection_id"]
    assert Path(a["selection_file"]).read_bytes() == before


@pytest.mark.parametrize(
    "field",
    [
        "parent_run_id",
        "calibration_id",
        "score_variant",
        "objective",
        "value",
        "batch_definition",
        "source_hashes",
        "data_identity",
        "parent_files_sha256",
    ],
)
def test_provenance_mismatch_rejected(inputs, tmp_path, field):
    import yaml

    result = save(inputs, tmp_path)
    record = json.loads(Path(result["selection_file"]).read_text())
    cfg = EvaluationConfig.model_validate(
        yaml.safe_load(Path(result["evaluation_draft"]).read_text())
    )
    changed = copy.deepcopy(record)
    changed[field] = {} if isinstance(changed[field], dict) else "wrong"
    with pytest.raises(ValueError):
        validate_selection_record(cfg, changed, inputs["provenance"], inputs["source_hashes"])


def test_manual_legacy_policy_still_supported():
    cfg = EvaluationConfig(
        parent_run_id="run_test",
        policy=PolicyConfig(kind="threshold", value=0.2, source="predeclared"),
    )
    assert cfg.policy.selection is None
    assert apply_policy([0.1, 0.2], cfg.policy).tolist() == [False, True]


def test_f_scores_and_undefined_case():
    panel = policy_metrics([1, 1, 0, 0], [1, 0, 1, 0])
    assert panel["f1"] == 0.5 and panel["f2"] == 0.5
    assert policy_metrics([0, 0], [0, 0])["f1"] is None
    assert policy_metrics([1, 0], [0, 0])["f2"] == 0


def test_template_identity_is_preserved(inputs, tmp_path):
    template = EvaluationConfig(
        parent_run_id="run_test", calibration_id="cal_test", intended_use="manual inspection"
    )
    result = save(inputs, tmp_path, evaluation_template=template)
    import yaml

    draft = yaml.safe_load(Path(result["evaluation_draft"]).read_text())
    assert draft["intended_use"] == "manual inspection"


def test_wrong_template_writes_nothing(inputs, tmp_path):
    with pytest.raises(ValueError, match="template"):
        save(inputs, tmp_path, evaluation_template=EvaluationConfig(parent_run_id="run_other"))
    assert not list(tmp_path.iterdir())


def test_schema_rejects_score_scale_change(inputs, tmp_path):
    import yaml

    result = save(inputs, tmp_path)
    data = yaml.safe_load(Path(result["evaluation_draft"]).read_text())
    data["release_variant"] = "raw"
    with pytest.raises(ValidationError, match="score scale"):
        EvaluationConfig.model_validate(data)


def test_f_scores_propagate_to_final_metrics_and_bootstrap():
    stats = pytest.importorskip("ml_template.evaluation.evaluation_statistics")
    uncertainty = pytest.importorskip("ml_template.evaluation.evaluation_uncertainty")
    cfg = EvaluationConfig.model_validate(
        {
            "parent_run_id": "run_test",
            "policy": {"kind": "threshold", "value": 0.5, "source": "fixed"},
            "uncertainty": {"enabled": False},
        }
    )
    values = stats.variant_metrics([0, 0, 1, 1], {"raw": [0.1, 0.2, 0.6, 0.9]}, cfg)
    assert values["raw"]["policy_f1"] == 1
    assert values["raw"]["policy_f2"] == 1
    rows, _ = uncertainty.bootstrap_metrics([0, 0, 1, 1], {"raw": [0.1, 0.2, 0.6, 0.9]}, cfg)
    assert {"policy_f1", "policy_f2"}.issubset({r["metric"] for r in rows})
