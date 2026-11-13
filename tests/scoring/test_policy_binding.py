from copy import deepcopy
from types import SimpleNamespace

import pytest

from ml_template.evaluation.policy_selection import (
    DATA_FIELDS,
    SOURCE_FILES,
    validate_selection_record,
)


def binding():
    provenance = {key: f"value_{key}" for key in DATA_FIELDS}
    provenance["parent_files_sha256"] = {"model.ubj": "modelhash"}
    hashes = {key: f"hash_{key}" for key in SOURCE_FILES}
    record = {
        "schema_version": 1,
        "parent_run_id": "run",
        "calibration_id": "cal",
        "score_variant": "calibrated_crossfit",
        "objective": "f2",
        "transfer_acknowledged": True,
        "batch_definition": "entire scoring batch",
        "kind": "threshold",
        "value": 0.2,
        "data_identity": {key: provenance[key] for key in DATA_FIELDS},
        "parent_files_sha256": provenance["parent_files_sha256"],
        "source_hashes": hashes,
    }
    ref = SimpleNamespace(
        **{
            key: record[key]
            for key in [
                "parent_run_id",
                "calibration_id",
                "score_variant",
                "objective",
                "transfer_acknowledged",
            ]
        }
    )
    policy = SimpleNamespace(
        selection=ref, kind="threshold", value=0.2, batch_definition="entire scoring batch"
    )
    return (
        SimpleNamespace(
            policy=policy, parent_run_id="run", calibration_id="cal", release_variant="calibrated"
        ),
        record,
        provenance,
        hashes,
    )


def test_binding_accepted():
    spec, record, provenance, hashes = binding()
    assert validate_selection_record(spec, record, provenance, hashes) == record


@pytest.mark.parametrize(
    "field,value",
    [
        ("parent_run_id", "other"),
        ("calibration_id", "other"),
        ("score_variant", "raw"),
        ("value", 0.3),
        ("batch_definition", "different"),
        ("transfer_acknowledged", False),
    ],
)
def test_binding_mismatch(field, value):
    spec, record, provenance, hashes = binding()
    record = deepcopy(record)
    record[field] = value
    with pytest.raises(ValueError):
        validate_selection_record(spec, record, provenance, hashes)


def test_source_changed():
    spec, record, provenance, hashes = binding()
    with pytest.raises(ValueError, match="changed"):
        validate_selection_record(
            spec, record, provenance, {**hashes, "calibrator.json": "changed"}
        )
