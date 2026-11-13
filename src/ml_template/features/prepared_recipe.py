"""Shared, task-neutral stateless preparation from frozen raw-feature recipes.

Metadata verification never deserialises feature/target Parquet values. Prediction
replays the recipe on NEW rows; it never joins cached historical feature values.
Learned/fitted transforms intentionally remain outside this layer.
"""

import hashlib
import json
import re
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe(root, name):
    if not isinstance(name, str) or not name:
        raise ValueError("Artifact name must be a relative string")
    root = Path(root).resolve()
    part = PurePosixPath(name)
    if part.is_absolute() or ".." in part.parts or "\\" in name or part.as_posix() != name:
        raise ValueError("Unsafe feature artifact path")
    path = root.joinpath(*part.parts)
    if (
        not path.resolve().is_relative_to(root)
        or path.is_symlink()
        or any(p.is_symlink() for p in path.parents if p != root and p.is_relative_to(root))
    ):
        raise ValueError("Unsafe feature artifact symlink")
    return path


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate feature metadata key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("Nonfinite feature JSON constant: " + value)

    return json.loads(
        Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=invalid
    )


def names(values, label):
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(v, str) or not v for v in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError("Invalid unique column names: " + label)
    return list(values)


def normalize_recipe(recipe):
    """Accept the old raw identity format and native schema-v1 stateless recipes."""
    if not isinstance(recipe, dict) or recipe.get("schema_version") != 1:
        raise ValueError("Unsupported feature recipe version")
    if recipe.get("kind") == "identity":
        if (
            set(recipe) != {"schema_version", "kind", "raw_columns", "columns"}
            or recipe["raw_columns"] != recipe["columns"]
        ):
            raise ValueError("Invalid legacy identity recipe")
        recipe = {
            "schema_version": 1,
            "raw_columns": recipe["raw_columns"],
            "columns": recipe["columns"],
            "blocks": [],
        }
    if set(recipe) - {"schema_version", "raw_columns", "columns", "blocks", "feature_set_id"}:
        raise ValueError("Unexpected feature recipe fields")
    raw, selected = names(recipe["raw_columns"], "source"), names(recipe["columns"], "model")
    if not isinstance(recipe["blocks"], list):
        raise ValueError("Feature blocks must be ordered")
    available = set(raw)
    seen = set()
    for block in recipe["blocks"]:
        if set(block) != {
            "block_id",
            "block_name",
            "code_sha256",
            "params",
            "derived_from",
            "columns",
        }:
            raise ValueError("Invalid frozen block metadata")
        if not isinstance(block["block_id"], str) or block["block_id"] in seen:
            raise ValueError("Duplicate/invalid feature block ID")
        seen.add(block["block_id"])
        columns = names(block["columns"], "block")
        if available & set(columns) or not set(block["derived_from"]) <= set(raw):
            raise ValueError("Duplicate outputs or non-source dependencies")
        if re.fullmatch(r"[0-9a-f]{64}", str(block["code_sha256"])) is None:
            raise ValueError("Invalid block implementation hash")
        available.update(columns)
    if not set(selected) <= available:
        raise ValueError("Unknown selected feature columns")
    return recipe


def model_schema(info):
    """Match V3 pre-Gold nullable schemas using SAVED estimator input metadata."""
    columns = names(info["features"], "saved model")
    if set(info["dtypes"]) != set(columns):
        raise ValueError("Saved feature dtype coverage differs")
    categories = info.get("categorical_schema", {})
    if not set(categories) <= set(columns):
        raise ValueError("Saved category schema has unknown columns")
    schema = []
    for column in columns:
        declared = info["dtypes"][column]
        if declared == "category":
            item = categories.get(column)
            if not isinstance(item, dict) or type(item.get("ordered")) is not bool:
                raise ValueError("Categorical model input lacks its saved vocabulary")
            levels = item["categories"]
            if (
                not isinstance(levels, list)
                or any(
                    v is None
                    or not isinstance(v, (str, int, float, bool))
                    or isinstance(v, float)
                    and not np.isfinite(v)
                    for v in levels
                )
                or pd.Index(levels).has_duplicates
            ):
                raise ValueError("Invalid saved categorical vocabulary")
            value = {"kind": "category", "categories": levels, "ordered": item["ordered"]}
        else:
            if column in categories:
                raise ValueError("Numeric dtype/category schema conflict")
            dtype = pd.api.types.pandas_dtype(declared)
            if pd.api.types.is_bool_dtype(dtype):
                value = {"kind": "bool", "dtype": "boolean"}
            elif pd.api.types.is_integer_dtype(dtype):
                native = np.dtype(getattr(dtype, "numpy_dtype", dtype))
                prefix = "UInt" if native.kind == "u" else "Int"
                value = {"kind": "integer", "dtype": prefix + str(native.itemsize * 8)}
            elif pd.api.types.is_float_dtype(dtype):
                value = {"kind": "float", "dtype": str(dtype)}
            else:
                raise ValueError(
                    "Unsupported saved estimator dtype; text needs a frozen category schema"
                )
        schema.append({"name": column, **value, "nullable": True})
    return schema


def prepare_frame(raw, recipe, schema):
    """The same raw -> recipe -> frozen coercion path for evaluation and inference."""
    from ml_template.features.feature_recipe import replay_recipe
    from ml_template.scoring.input_contract import coerce_frame

    recipe = normalize_recipe(recipe)
    if (
        not isinstance(raw, pd.DataFrame)
        or raw.empty
        or not raw.index.is_unique
        or raw.index.hasnans
    ):
        raise ValueError("Preparation requires a nonempty uniquely keyed raw feature frame")
    if list(raw.columns) != recipe["raw_columns"]:
        raise ValueError("Raw feature order differs from the frozen recipe")
    if [item["name"] for item in schema] != recipe["columns"]:
        raise ValueError("Model schema order differs from frozen feature selection")
    result = replay_recipe(raw.copy(deep=True), recipe)
    if not result.index.equals(raw.index):
        raise ValueError("Recipe changed entity-key alignment")
    return coerce_frame(result, schema)


def verify_plan(cfg, raw_columns, info, key_col, n_rows, feature_root):
    """Bind fs identity, manifests, cached byte hashes and current registered code."""
    from ml_template.features.block_registry import get_block

    raw_columns = names(list(raw_columns), "Silver source")
    schema = model_schema(info)
    selected = list(info["features"])
    fsid = cfg.data.feature_set_id
    from ml_template.preprocessing.numeric_preprocessing import validate_regression_specs

    validate_regression_specs(cfg.fitted_features)
    if (info.get("fitted_features") or []) != (cfg.fitted_features or []):
        raise ValueError("Saved fitted feature specification differs from config")
    hashes = {}
    if fsid is None:
        recipe = {
            "schema_version": 1,
            "raw_columns": raw_columns,
            "columns": raw_columns,
            "blocks": [],
        }
        if selected != raw_columns:
            raise ValueError("Raw model inputs differ from Silver source order")
    else:
        if not isinstance(fsid, str) or re.fullmatch(r"fs_[0-9a-f]{12}", fsid) is None:
            raise ValueError("Invalid feature_set_id")
        root = Path(feature_root)
        path = safe(root, "_sets/" + fsid + ".json")
        before = sha256(path)
        fs = read_json(path)
        identity = {
            "silver": fs["silver_dataset_version_id"],
            "blocks": fs["blocks"],
            "columns": fs["columns"],
            "parent": fs["parent_feature_set_id"],
        }
        actual_id = (
            "fs_" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
        )
        if (
            actual_id != fsid
            or fs["feature_set_id"] != fsid
            or fs["silver_dataset_version_id"] != cfg.data.silver_dataset_version_id
        ):
            raise ValueError("Feature-set identity/Silver version differs")
        if (
            set(names(fs["raw_columns"], "feature-set source")) != set(raw_columns)
            or fs["columns"] != selected
        ):
            raise ValueError("Feature-set source or saved model column order differs")
        if not isinstance(fs["blocks"], list) or len(set(fs["blocks"])) != len(fs["blocks"]):
            raise ValueError("Duplicate/invalid feature-set blocks")
        items = []
        for block_id in fs["blocks"]:
            manifest_path = safe(root, block_id + "/manifest.json")
            manifest_hash = sha256(manifest_path)
            block = read_json(manifest_path)
            data_path = safe(root, block_id + "/data.parquet")
            if (
                block["block_id"] != block_id
                or block["silver_dataset_version_id"] != cfg.data.silver_dataset_version_id
                or block["key_col"] != key_col
                or block["n_rows"] != n_rows
            ):
                raise ValueError("Feature block dataset/key/row inventory differs")
            registered = get_block(block["block_name"])
            if (
                registered.block_id(cfg.data.silver_dataset_version_id, block["params"]) != block_id
                or registered.code_sha256() != block["code_sha256"]
                or registered.resolve_params(block["params"]) != block["params"]
                or list(registered.required_columns(block["params"])) != block["derived_from"]
            ):
                raise ValueError("Feature implementation/parameters/dependencies changed")
            cached_hash = sha256(data_path)
            if cached_hash != block["parquet_sha256"] or sha256(manifest_path) != manifest_hash:
                raise ValueError("Cached feature bytes/metadata changed")
            item = {
                name: block[name]
                for name in [
                    "block_id",
                    "block_name",
                    "code_sha256",
                    "params",
                    "derived_from",
                    "columns",
                ]
            }
            if set(block["dtypes"]) != set(item["columns"]):
                raise ValueError("Feature block dtype metadata differs")
            items.append(item)
            hashes["feature_block/" + block_id + "/manifest"] = manifest_hash
            hashes["feature_block/" + block_id + "/parquet"] = cached_hash
        if sha256(path) != before:
            raise ValueError("Feature-set metadata changed during verification")
        hashes["feature_set/" + fsid] = before
        recipe = {
            "schema_version": 1,
            "raw_columns": raw_columns,
            "columns": selected,
            "feature_set_id": fsid,
            "blocks": items,
        }
    recipe = normalize_recipe(recipe)
    return {
        "schema_version": 1,
        "feature_set_id": fsid,
        "recipe": recipe,
        "model_schema": schema,
        "input_hashes": hashes,
    }


def verify_prediction_code(expected_hashes, current_hashes):
    """Same critical-code policy as the pinned regression release builder."""
    for name, expected in expected_hashes.items():
        critical = name.startswith(("models/", "features/", "preprocessing/")) or name in {
            "training/run_config.py",
            "evaluation/regression_inputs.py",
        }
        if critical and current_hashes.get(name) != expected:
            raise ValueError("Evaluated prediction implementation changed: " + name)


def verify_prepared(p):
    frozen = p.provenance["feature_preparation"]
    current = verify_plan(
        p.run_cfg,
        frozen["recipe"]["raw_columns"],
        p.feature_info,
        p.key_col,
        len(p.assignments),
        p.roots["feature_root"],
    )
    if current != frozen:
        raise ValueError("Feature preparation changed after preflight")


def load_partition(p, keys):
    """Read source columns and targets for only the explicitly approved partition."""
    verify_prepared(p)
    if sha256(p.parquet_path) != p.provenance["input_hashes"]["silver_parquet"]:
        raise ValueError("Silver bytes changed after preflight")
    keys = list(keys)
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("Partition keys must be unique and nonempty")
    plan = p.provenance["feature_preparation"]
    columns = [p.key_col, *plan["recipe"]["raw_columns"], p.target_col]
    if len(set(columns)) != len(columns):
        raise ValueError("Source/key/target role overlap")
    frame = pd.read_parquet(
        p.parquet_path, engine="pyarrow", columns=columns, filters=[(p.key_col, "in", keys)]
    )
    if (
        frame[p.key_col].isna().any()
        or frame[p.key_col].duplicated().any()
        or set(frame[p.key_col]) != set(keys)
    ):
        raise ValueError("Partition key coverage differs")
    frame = frame.set_index(p.key_col).loc[keys]
    target = frame[p.target_col]
    if not pd.api.types.is_numeric_dtype(target) or pd.api.types.is_bool_dtype(target):
        raise ValueError("Regression target must be numeric")
    y = target.astype(float)
    if not np.isfinite(y.to_numpy()).all():
        raise ValueError("Partition target must be finite")
    X = prepare_frame(frame[plan["recipe"]["raw_columns"]], plan["recipe"], plan["model_schema"])
    if sha256(p.parquet_path) != p.provenance["input_hashes"]["silver_parquet"]:
        raise ValueError("Silver changed while loading partition")
    verify_prepared(p)
    return X, y


def validate_packaged(recipe, contract, schema, cfg, provenance, info=None):
    """Pure package binding. Never look up a live historical feature set at scoring."""
    canonical = normalize_recipe(recipe)
    if canonical["raw_columns"] != [s["name"] for s in contract["source_schema"]] or canonical[
        "columns"
    ] != [s["name"] for s in schema]:
        raise ValueError("Packaged source/model schema differs from recipe")
    from ml_template.preprocessing.numeric_preprocessing import validate_regression_specs

    validate_regression_specs(cfg.fitted_features)
    if cfg.fitted_features and (info is None or info.get("fitted_features") != cfg.fitted_features):
        raise ValueError("Packaged fitted specification differs from run config")
    if canonical.get("feature_set_id") != cfg.data.feature_set_id:
        raise ValueError("Recipe feature-set ID differs from run configuration")
    plan = provenance.get("feature_preparation")
    if plan is None:
        if (
            cfg.data.feature_set_id is not None
            or canonical["blocks"]
            or canonical["raw_columns"] != canonical["columns"]
            or recipe.get("kind") != "identity"
        ):
            raise ValueError("Legacy releases may only use the original raw identity recipe")
        return canonical
    if (
        plan.get("schema_version") != 1
        or plan["feature_set_id"] != cfg.data.feature_set_id
        or normalize_recipe(plan["recipe"]) != canonical
    ):
        raise ValueError("Packaged recipe differs from evaluated feature preparation")
    if info is None or model_schema(info) != plan["model_schema"]:
        raise ValueError("Missing/different evaluated model input metadata")
    semantic = [{k: v for k, v in item.items() if k != "observed_missing_rate"} for item in schema]
    if semantic != plan["model_schema"]:
        raise ValueError("Packaged preprocessor schema differs from evaluated model schema")
    for name, expected in plan["input_hashes"].items():
        if provenance["input_hashes"].get(name) != expected:
            raise ValueError("Feature artifact/evaluation hash binding differs")
    return canonical
