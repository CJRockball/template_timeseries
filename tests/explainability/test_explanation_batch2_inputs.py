import pytest

from ml_template.explainability.artifacts import write_json
from ml_template.explainability.explanation_inputs import (
    inspect_saved_batch,
    load_saved_batch,
    validate_predictions,
)


def test_saved_batch_aligns_by_key_not_position(batch2_layout):
    obj = batch2_layout
    saved = load_saved_batch(obj.config)
    assert saved.predictions.index.tolist() == [30, 10, 20]
    assert saved.predictions.p_raw.tolist() == [0.1, 0.2, 0.3]
    assert saved.binding["snapshot_name"] == "input.csv"


def test_modified_input_rejected(batch2_layout):
    obj = batch2_layout
    (obj.score / "input.csv").write_text("entity,x\n30,999\n")
    with pytest.raises(ValueError, match="changed"):
        load_saved_batch(obj.config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("release_id", "wrong"),
        ("scoring_batch_id", "wrong"),
        ("labels_loaded", True),
        ("n_rows", 0),
        ("input_sha256", "wrong"),
    ],
)
def test_bad_summary_rejected(batch2_layout, field, value):
    obj = batch2_layout
    obj.summary[field] = value
    obj.reseal()
    with pytest.raises(ValueError):
        inspect_saved_batch(obj.config)


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "missing",
        "foreign",
        "nonfinite",
        "bounds",
        "label",
        "release",
        "batch",
        "key_type",
        "variant",
    ],
)
def test_invalid_predictions(batch2_layout, mutation):
    obj = batch2_layout
    predictions = obj.predictions.copy()
    if mutation == "duplicate":
        predictions.loc[1, "entity"] = predictions.loc[0, "entity"]
    elif mutation == "missing":
        predictions = predictions.iloc[:2]
    elif mutation == "foreign":
        predictions.loc[0, "entity"] = 999
    elif mutation == "nonfinite":
        predictions.loc[0, "p_raw"] = float("nan")
    elif mutation == "bounds":
        predictions.loc[0, "p_raw"] = 1.1
    elif mutation == "label":
        predictions["y_true"] = [0, 0, 1]
    elif mutation == "release":
        predictions.loc[0, "release_id"] = "wrong"
    elif mutation == "batch":
        predictions.loc[0, "scoring_batch_id"] = "wrong"
    elif mutation == "key_type":
        predictions["entity"] = predictions.entity.astype(float)
    elif mutation == "variant":
        predictions.loc[0, "p_release"] = 0.9
    with pytest.raises(ValueError):
        validate_predictions(obj.frame, predictions, obj.contract, obj.info, obj.summary)


def test_target_in_snapshot_rejected(batch2_layout):
    obj = batch2_layout
    frame = obj.frame.copy()
    frame["target"] = 0
    with pytest.raises(ValueError, match="columns"):
        validate_predictions(frame, obj.predictions, obj.contract, obj.info, obj.summary)


def test_unknown_file_rejected(batch2_layout):
    obj = batch2_layout
    (obj.score / "extra.txt").write_text("unexpected")
    with pytest.raises(ValueError):
        load_saved_batch(obj.config)


def test_labels_never_accepted_in_manifest(batch2_layout):
    obj = batch2_layout
    (obj.score / "labels.csv").write_text("no")
    obj.reseal()
    with pytest.raises(ValueError, match="exactly"):
        inspect_saved_batch(obj.config)


def test_alert_policy_acknowledgement(batch2_layout):
    obj = batch2_layout
    obj.info["policy"] = {"kind": "alert_rate", "value": 0.1, "batch_definition": "entire batch"}
    write_json(obj.release / "release.json", obj.info)
    from ml_template.scoring.release_artifacts import seal

    manifest = seal(obj.release)
    obj.summary["release_id"] = manifest["release_id"]
    obj.summary["content_sha256"] = manifest["content_sha256"]
    obj.summary["policy"] = obj.info["policy"]
    obj.reseal()
    config = obj.config.model_copy(update={"expected_release_id": manifest["release_id"]})
    with pytest.raises(ValueError, match="acknowledgement"):
        inspect_saved_batch(config)
