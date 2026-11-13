import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from ml_template.data.silver_loader import load_silver  # noqa: E402
from ml_template.tracking.run_artifacts import sha256_file  # noqa: E402

SID = "silver_test_abc"
ROLES = {"id": ["udi", "product_id"], "feature": ["type", "temp"], "target": ["machine_failure"]}


def _frame(n=40):
    return pd.DataFrame(
        {
            "dataset_version_id": "ds_v1",
            "source_row_number": np.arange(1, n + 1),
            "udi": np.arange(1, n + 1),
            "product_id": [f"P{i}" for i in range(n)],
            "type": pd.Categorical(["L", "M"] * (n // 2)),
            "temp": np.linspace(300, 310, n),
            "machine_failure": [0, 1] * (n // 2),
            "twf": 0,  # an extra column that no role uses
        }
    )


def _build(tmp_path, *, roles=None, df=None, passed=True):
    out = tmp_path / "ds" / SID
    out.mkdir(parents=True)
    df = _frame() if df is None else df
    df.to_parquet(out / "data.parquet", index=False)
    manifest = {
        "silver_dataset_version_id": SID,
        "columns": {c: str(t) for c, t in df.dtypes.items()},
        "roles": ROLES if roles is None else roles,
        "parquet_sha256": sha256_file(out / "data.parquet"),
        "validation_passed": passed,
    }
    (out / "manifest.json").write_text(json.dumps(manifest))
    return out


def test_loads_key_features_and_target_only(tmp_path):
    _build(tmp_path)
    silver = load_silver(SID, "ds", "udi", tmp_path)
    assert silver.features == ["type", "temp"] and silver.target == "machine_failure"
    frame = silver.model_frame()
    assert list(frame.columns) == ["type", "temp", "machine_failure"]
    assert frame.index.name == "udi" and frame.index.is_monotonic_increasing
    assert set(silver.df.columns) == {"udi", "type", "temp", "machine_failure"}


def test_dataset_without_outcome_role_loads(tmp_path):
    _build(tmp_path, roles=ROLES)  # ROLES has no outcome at all
    assert load_silver(SID, "ds", "udi", tmp_path).target == "machine_failure"


def test_extra_roles_and_plural_names_are_accepted(tmp_path):
    roles = {
        "ids": ["udi"],
        "features": ["type", "temp"],
        "targets": ["machine_failure"],
        "outcome": ["twf"],
        "weights": ["temp"],
    }
    _build(tmp_path, roles=roles)
    silver = load_silver(SID, "ds", "udi", tmp_path)
    assert silver.features == ["type", "temp"]
    assert set(silver.roles) == {"id", "feature", "target"}


def test_hash_mismatch_detected(tmp_path):
    out = _build(tmp_path)
    df = pd.read_parquet(out / "data.parquet")
    df.loc[0, "temp"] = 999.0
    df.to_parquet(out / "data.parquet", index=False)
    with pytest.raises(ValueError, match="hash"):
        load_silver(SID, "ds", "udi", tmp_path)
    load_silver(SID, "ds", "udi", tmp_path, verify_hash=False)


def test_missing_and_unvalidated_silver(tmp_path):
    with pytest.raises(FileNotFoundError, match="build Silver"):
        load_silver(SID, "ds", "udi", tmp_path)
    _build(tmp_path, passed=False)
    with pytest.raises(ValueError, match="validation"):
        load_silver(SID, "ds", "udi", tmp_path)


def test_feature_role_overlap_rejected(tmp_path):
    roles = {**ROLES, "feature": ["type", "temp", "machine_failure"]}
    _build(tmp_path, roles=roles)
    with pytest.raises(ValueError, match="overlaps"):
        load_silver(SID, "ds", "udi", tmp_path)


def test_single_target_enforced(tmp_path):
    _build(tmp_path, roles={**ROLES, "target": ["machine_failure", "twf"]})
    with pytest.raises(ValueError, match="single-target"):
        load_silver(SID, "ds", "udi", tmp_path)


def test_required_roles_and_key_role(tmp_path):
    _build(tmp_path, roles={"id": ["udi"], "target": ["machine_failure"]})
    with pytest.raises(ValueError, match="feature"):
        load_silver(SID, "ds", "udi", tmp_path)


def test_key_must_have_id_role(tmp_path):
    _build(tmp_path)
    with pytest.raises(ValueError, match="id role"):
        load_silver(SID, "ds", "temp", tmp_path)


def test_role_column_missing_from_manifest(tmp_path):
    _build(tmp_path, roles={**ROLES, "feature": ["type", "ghost"]})
    with pytest.raises(ValueError, match="ghost"):
        load_silver(SID, "ds", "udi", tmp_path)


def test_non_binary_target_rejected(tmp_path):
    df = _frame()
    df.loc[0, "machine_failure"] = 2
    _build(tmp_path, df=df)
    with pytest.raises(ValueError, match="binary"):
        load_silver(SID, "ds", "udi", tmp_path)
