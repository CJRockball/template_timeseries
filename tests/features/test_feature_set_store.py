import logging

import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from ml_template.features import feature_set_store as store  # noqa: E402
from ml_template.features.block_registry import (  # noqa: E402
    discover_blocks,
    register_block,
    unregister_block,
)

discover_blocks()
SID = "silver_test_abc"
RAW_COLS = [
    "type",
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]


def _raw():
    return pd.DataFrame(
        {
            "type": ["L", "M", "H", "L", "M"],
            "air_temperature_k": [300.0, 300.0, 299.0, 300.0, 301.0],
            "process_temperature_k": [310.0, 308.0, 308.0, 310.0, 311.0],
            "rotational_speed_rpm": [1500.0, 1300.0, 2800.0, 1400.0, 1600.0],
            "torque_nm": [40.0, 50.0, 60.0, 10.0, 30.0],
            "tool_wear_min": [100.0, 220.0, 250.0, 10.0, 50.0],
        },
        index=pd.Index([1, 2, 3, 4, 5], name="udi"),
    )


def _thermal(tmp_path, raw=None):
    return store.build_block("thermal_power", _raw() if raw is None else raw, SID, root=tmp_path)


def _set(tmp_path, block_ids, **kw):
    return store.create_feature_set(SID, block_ids, raw_columns=RAW_COLS, root=tmp_path, **kw)


def test_build_block_writes_new_columns_only_with_lineage(tmp_path):
    m = _thermal(tmp_path)
    assert m["columns"] == ["temp_diff_k", "power_w"]
    assert m["silver_dataset_version_id"] == SID and m["key_col"] == "udi"
    assert m["derived_from"] == [
        "air_temperature_k",
        "process_temperature_k",
        "rotational_speed_rpm",
        "torque_nm",
    ]
    assert len(m["code_sha256"]) == 64 and len(m["parquet_sha256"]) == 64
    _, frame = store.load_block(m["block_id"], tmp_path)
    assert list(frame.columns) == ["temp_diff_k", "power_w"]
    assert frame.index.name == "udi" and frame.index.tolist() == [1, 2, 3, 4, 5]


def test_build_block_is_idempotent_and_param_sensitive(tmp_path):
    first = _thermal(tmp_path)
    again = _thermal(tmp_path)
    assert again["block_id"] == first["block_id"]
    assert again["created_at"] == first["created_at"]
    a = store.build_block("strain", _raw(), SID, root=tmp_path)
    b = store.build_block(
        "strain", _raw(), SID, {"osf_limit_minnm": {"L": 1.0, "M": 2.0, "H": 3.0}}, tmp_path
    )
    assert a["block_id"] != b["block_id"]


def test_tampered_block_file_is_detected(tmp_path):
    m = _thermal(tmp_path)
    path = tmp_path / m["block_id"] / "data.parquet"
    df = pd.read_parquet(path)
    df.loc[0, "power_w"] = 1.0
    df.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="hash"):
        store.load_block(m["block_id"], tmp_path)
    with pytest.raises(ValueError, match="hash"):
        _thermal(tmp_path)


def test_sanity_mismatch_logs_warning_and_is_recorded(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="ml_template.features.feature_set_store"):
        m = _thermal(tmp_path)
    assert "hdf_rule_rows" in caplog.text
    assert m["sanity"]["hdf_rule_rows"] == {"observed": 1.0, "expected": 115.0, "ok": False}


def test_create_feature_set_columns_and_identity(tmp_path):
    t = _thermal(tmp_path)
    fs = _set(tmp_path, [t["block_id"]])
    assert fs["columns"] == [*RAW_COLS, "temp_diff_k", "power_w"]
    assert fs["column_sources"]["power_w"] == "thermal_power"
    assert fs["column_sources"]["torque_nm"] == "silver"
    assert fs["parent_feature_set_id"] is None and fs["n_features"] == 8
    assert _set(tmp_path, [t["block_id"]])["feature_set_id"] == fs["feature_set_id"]


def test_selection_is_a_new_set_not_new_data(tmp_path):
    t = _thermal(tmp_path)
    full = _set(tmp_path, [t["block_id"]])
    sel = _set(
        tmp_path,
        [t["block_id"]],
        columns=["power_w", "torque_nm"],
        parent=full["feature_set_id"],
    )
    assert sel["feature_set_id"] != full["feature_set_id"]
    assert sel["columns"] == ["power_w", "torque_nm"] and sel["selection"] == sel["columns"]
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([t["block_id"], "_sets"])
    with pytest.raises(ValueError, match="invalid column selection"):
        _set(tmp_path, [t["block_id"]], columns=["nope"])


def test_duplicate_column_across_blocks_rejected(tmp_path):
    register_block("t_dup_power", requires=("torque_nm",), description="clash")(
        lambda df, p: pd.DataFrame({"power_w": df["torque_nm"]}, index=df.index)
    )
    try:
        t = _thermal(tmp_path)
        d = store.build_block("t_dup_power", _raw(), SID, root=tmp_path)
        with pytest.raises(ValueError, match="duplicate column"):
            _set(tmp_path, [t["block_id"], d["block_id"]])
    finally:
        unregister_block("t_dup_power")


def test_block_from_other_silver_rejected(tmp_path):
    t = store.build_block("thermal_power", _raw(), "silver_other", root=tmp_path)
    with pytest.raises(ValueError, match="another Silver"):
        _set(tmp_path, [t["block_id"]])


def test_extend_feature_set_sets_parent_and_appends(tmp_path):
    t = _thermal(tmp_path)
    s = store.build_block("strain", _raw(), SID, root=tmp_path)
    fs1 = _set(tmp_path, [t["block_id"]])
    fs2 = store.extend_feature_set(fs1["feature_set_id"], [s["block_id"]], root=tmp_path)
    assert fs2["parent_feature_set_id"] == fs1["feature_set_id"]
    assert fs2["blocks"] == [t["block_id"], s["block_id"]]
    assert fs2["columns"][-2:] == ["wear_torque_minnm", "osf_margin_minnm"]
    assert store.extend_feature_set(fs2["feature_set_id"], [s["block_id"]], root=tmp_path) == fs2


def test_extend_keeps_parent_selection(tmp_path):
    t = _thermal(tmp_path)
    s = store.build_block("strain", _raw(), SID, root=tmp_path)
    fs1 = _set(tmp_path, [t["block_id"]], columns=["power_w"])
    fs2 = store.extend_feature_set(fs1["feature_set_id"], [s["block_id"]], root=tmp_path)
    assert fs2["columns"] == ["power_w", "wear_torque_minnm", "osf_margin_minnm"]


def test_resolve_joins_blocks_and_follows_row_order(tmp_path):
    raw = _raw()
    t = _thermal(tmp_path)
    fs = _set(tmp_path, [t["block_id"]])
    out = store.resolve_feature_set(fs["feature_set_id"], raw, SID, tmp_path)
    assert list(out.columns) == fs["columns"] and out.index.equals(raw.index)
    assert out.loc[2, "temp_diff_k"] == pytest.approx(8.0)
    rev = raw.iloc[::-1]
    out_rev = store.resolve_feature_set(fs["feature_set_id"], rev, SID, tmp_path)
    assert out_rev.index.equals(rev.index)
    assert out_rev.loc[2, "temp_diff_k"] == pytest.approx(8.0)


def test_resolve_guards(tmp_path):
    raw = _raw()
    t = _thermal(tmp_path)
    fid = _set(tmp_path, [t["block_id"]])["feature_set_id"]
    with pytest.raises(ValueError, match="belongs to"):
        store.resolve_feature_set(fid, raw, "silver_other", tmp_path)
    with pytest.raises(ValueError, match="columns changed"):
        store.resolve_feature_set(fid, raw.drop(columns=["type"]), SID, tmp_path)
    other_keys = raw.copy()
    other_keys.index = pd.Index([1, 2, 3, 4, 99], name="udi")
    with pytest.raises(ValueError, match="keys"):
        store.resolve_feature_set(fid, other_keys, SID, tmp_path)
    with pytest.raises(FileNotFoundError):
        store.resolve_feature_set("fs_missing", raw, SID, tmp_path)


def test_database_index_rows_are_idempotent(tmp_path):
    sa = pytest.importorskip("sqlalchemy")
    from ml_template.db.tracking_tables import feature_blocks, feature_sets

    t = _thermal(tmp_path)
    fs = _set(tmp_path, [t["block_id"]])
    engine = sa.create_engine("sqlite://")
    for _ in range(2):
        store.register_block_record(engine, t)
        store.register_feature_set_record(engine, fs, tmp_path)
    with engine.connect() as conn:
        assert conn.execute(sa.select(sa.func.count()).select_from(feature_blocks)).scalar() == 1
        row = conn.execute(sa.select(feature_sets)).mappings().one()
    assert row["feature_set_id"] == fs["feature_set_id"] and row["n_features"] == 8
