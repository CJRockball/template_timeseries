"""Metadata-only plans and shared stateless preparation; no user data."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ml_template.features import prepared_recipe as prepared
from ml_template.features.block_registry import get_block, register_block, unregister_block


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def world(tmp_path):
    block_name = "test_stateless_replay_fixture"
    unregister_block(block_name)

    @register_block(
        block_name, requires=["a", "b", "kind"], description="Synthetic test only", params={}
    )
    def toy(frame, params):
        if (frame.b == 0).any():
            raise ValueError("zero denominator")
        return pd.DataFrame(
            {
                "ab_ratio": frame.a / frame.b,
                "kind_pair": frame.kind.astype("string") + "|" + frame.kind.astype("string"),
            },
            index=frame.index,
        )

    raw = pd.DataFrame(
        {"a": [1.0, 2.0, 4.0, 8.0], "b": [2, 2, 2, 4], "kind": ["u", "v", "u", "v"]},
        index=pd.Index([1, 2, 3, 4], name="id"),
    )
    full = pd.concat([raw, toy(raw, {})], axis=1)
    for col in ["kind", "kind_pair"]:
        full[col] = full[col].astype("category")
    info = {
        "features": list(full),
        "dtypes": {c: str(t) for c, t in full.dtypes.items()},
        "categorical_schema": {
            c: {"categories": full[c].cat.categories.tolist(), "ordered": False}
            for c in ["kind", "kind_pair"]
        },
    }
    block = get_block(block_name)
    bid = block.block_id("silver_test", {})
    folder = tmp_path / bid
    folder.mkdir()
    (folder / "data.parquet").write_bytes(b"never parse cached features during preflight")
    meta = {
        "block_id": bid,
        "block_name": block_name,
        "silver_dataset_version_id": "silver_test",
        "key_col": "id",
        "code_sha256": block.code_sha256(),
        "params": block.resolve_params({}),
        "derived_from": ["a", "b", "kind"],
        "columns": ["ab_ratio", "kind_pair"],
        "dtypes": {"ab_ratio": "float64", "kind_pair": "string"},
        "n_rows": 4,
        "parquet_sha256": prepared.sha256(folder / "data.parquet"),
    }
    dump(folder / "manifest.json", meta)
    identity = {"silver": "silver_test", "blocks": [bid], "columns": list(full), "parent": None}
    fsid = "fs_" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    fs = {
        "feature_set_id": fsid,
        "silver_dataset_version_id": "silver_test",
        "parent_feature_set_id": None,
        "blocks": [bid],
        "raw_columns": list(raw),
        "columns": list(full),
    }
    dump(tmp_path / "_sets" / (fsid + ".json"), fs)
    cfg = SimpleNamespace(
        fitted_features=None,
        data=SimpleNamespace(feature_set_id=fsid, silver_dataset_version_id="silver_test"),
    )
    yield {
        "raw": raw,
        "model": full,
        "info": info,
        "cfg": cfg,
        "root": tmp_path,
        "block": meta,
        "folder": folder,
        "fs": fs,
    }
    unregister_block(block_name)


def plan(w):
    return prepared.verify_plan(w["cfg"], list(w["raw"]), w["info"], "id", 4, w["root"])


def test_preflight_reads_no_parquet_values(world, monkeypatch):
    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("decoded feature/label values")),
    )
    result = plan(world)
    assert result["recipe"]["columns"] == world["info"]["features"]
    assert len(result["input_hashes"]) == 3


def test_replay_new_keys_chunk_and_shuffle_without_live_cache(world):
    import shutil

    recipe = plan(world)
    raw = world["raw"].copy()
    raw.index = pd.Index([1001, 1002, 1003, 1004], name="id")
    whole = prepared.prepare_frame(raw, recipe["recipe"], recipe["model_schema"])
    chunks = pd.concat(
        [
            prepared.prepare_frame(raw.iloc[i : i + 1], recipe["recipe"], recipe["model_schema"])
            for i in range(len(raw))
        ]
    )
    pd.testing.assert_frame_equal(whole, chunks)
    pd.testing.assert_frame_equal(
        whole.sort_index(),
        prepared.prepare_frame(
            raw.iloc[::-1], recipe["recipe"], recipe["model_schema"]
        ).sort_index(),
    )
    shutil.rmtree(world["root"] / "_sets")
    shutil.rmtree(world["folder"])
    pd.testing.assert_frame_equal(
        whole, prepared.prepare_frame(raw, recipe["recipe"], recipe["model_schema"])
    )
    assert str(whole.b.dtype) == "Int64"


def test_schema_selection_reorder_supported(world):
    info = copy.deepcopy(world["info"])
    selected = ["kind_pair", "ab_ratio"]
    info["features"] = selected
    info["dtypes"] = {k: info["dtypes"][k] for k in selected}
    info["categorical_schema"] = {"kind_pair": info["categorical_schema"]["kind_pair"]}
    fs = copy.deepcopy(world["fs"])
    fs["columns"] = selected
    identity = {
        "silver": "silver_test",
        "blocks": fs["blocks"],
        "columns": selected,
        "parent": None,
    }
    fsid = "fs_" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    fs["feature_set_id"] = fsid
    dump(world["root"] / "_sets" / (fsid + ".json"), fs)
    cfg = SimpleNamespace(
        fitted_features=None,
        data=SimpleNamespace(feature_set_id=fsid, silver_dataset_version_id="silver_test"),
    )
    value = prepared.verify_plan(cfg, list(world["raw"]), info, "id", 4, world["root"])
    assert (
        list(prepared.prepare_frame(world["raw"], value["recipe"], value["model_schema"]))
        == selected
    )


@pytest.mark.parametrize("bad", ["missing", "extra", "order", "category", "zero"])
def test_invalid_inference_inputs_fail_closed(world, bad):
    value = plan(world)
    raw = world["raw"].copy()
    if bad == "missing":
        raw = raw.drop(columns="b")
    elif bad == "extra":
        raw = raw.assign(y_true=123.0)
    elif bad == "order":
        raw = raw[list(raw)[::-1]]
    elif bad == "category":
        raw.loc[1, "kind"] = "new"
    else:
        raw.loc[1, "b"] = 0
    with pytest.raises(ValueError):
        prepared.prepare_frame(raw, value["recipe"], value["model_schema"])


def test_cached_feature_tampering_rejected(world):
    (world["folder"] / "data.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Cached"):
        plan(world)


def test_feature_set_identity_and_code_tampering_rejected(world):
    path = world["folder"] / "manifest.json"
    meta = copy.deepcopy(world["block"])
    meta["code_sha256"] = "0" * 64
    dump(path, meta)
    with pytest.raises(ValueError, match="implementation"):
        plan(world)


def test_partition_projects_raw_columns_and_selected_labels_only(world, monkeypatch):
    value = plan(world)
    silver = world["root"] / "silver.parquet"
    silver.write_bytes(b"source snapshot")
    p = SimpleNamespace(
        run_cfg=world["cfg"],
        feature_info=world["info"],
        key_col="id",
        target_col="y",
        parquet_path=silver,
        assignments=pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "partition": ["dev", "dev", "holdout", "holdout"],
                "fold": [0, 1, -1, -1],
            }
        ),
        roots={"feature_root": world["root"]},
        provenance={
            "feature_preparation": value,
            "input_hashes": {"silver_parquet": prepared.sha256(silver)},
        },
    )
    frame = world["raw"].reset_index().assign(y=[10.0, 20.0, np.inf, np.inf])
    reads = []

    def read(path, *, engine, columns, filters):
        assert Path(path) == silver and columns == ["id", "a", "b", "kind", "y"]
        assert filters == [("id", "in", [1, 2])]
        reads.append(columns)
        return frame[frame.id.isin(filters[0][2])][columns].copy()

    monkeypatch.setattr(pd, "read_parquet", read)
    X, y = prepared.load_partition(p, [1, 2])
    assert list(X) == world["info"]["features"] and y.tolist() == [10.0, 20.0] and len(reads) == 1


def test_packaged_recipe_binding_and_legacy_raw(world):
    value = plan(world)
    source_schema = [{"name": name} for name in world["raw"]]
    contract = {"source_schema": source_schema}
    schema = [{**item, "observed_missing_rate": 0.0} for item in value["model_schema"]]
    provenance = {"feature_preparation": value, "input_hashes": dict(value["input_hashes"])}
    assert (
        prepared.validate_packaged(
            value["recipe"], contract, schema, world["cfg"], provenance, world["info"]
        )["feature_set_id"]
        == world["cfg"].data.feature_set_id
    )
    changed = copy.deepcopy(value["recipe"])
    changed["columns"] = changed["columns"][::-1]
    with pytest.raises(ValueError):
        prepared.validate_packaged(
            changed, contract, schema, world["cfg"], provenance, world["info"]
        )
    cfg = SimpleNamespace(fitted_features=None, data=SimpleNamespace(feature_set_id=None))
    legacy = {"schema_version": 1, "kind": "identity", "raw_columns": ["a"], "columns": ["a"]}
    assert (
        prepared.validate_packaged(
            legacy, {"source_schema": [{"name": "a"}]}, [{"name": "a"}], cfg, {}
        )["blocks"]
        == []
    )


def test_fitted_and_incomplete_vocab_remain_rejected(world):
    world["cfg"].fitted_features = [{"type": "anything"}]
    with pytest.raises(ValueError):
        plan(world)
    info = copy.deepcopy(world["info"])
    info["categorical_schema"].pop("kind")
    with pytest.raises(ValueError):
        prepared.model_schema(info)


@pytest.mark.parametrize(
    "name",
    [
        "models/a.py",
        "features/a.py",
        "preprocessing/a.py",
        "training/run_config.py",
        "evaluation/regression_inputs.py",
    ],
)
def test_prediction_code_guard_is_not_weakened(name):
    with pytest.raises(ValueError, match="implementation changed"):
        prepared.verify_prediction_code({name: "old"}, {name: "new"})
    with pytest.raises(ValueError):
        prepared.verify_prediction_code({name: "old"}, {})


def test_nonprediction_code_change_is_still_permitted_by_existing_policy():
    prepared.verify_prediction_code(
        {"dashboard/a.py": "old", "models/a.py": "same"},
        {"dashboard/a.py": "new", "models/a.py": "same"},
    )
