"""Read-only archived regression data. No model or source-label loading."""

import copy
import hashlib
import json
import os
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

ERRORS = (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError, sqlite3.Error)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Invalid JSON constant: {value}")

    return json.loads(
        Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=invalid
    )


def safe(root, name):
    if not isinstance(name, str) or not name:
        raise ValueError("Artifact path must be a relative string")
    root = Path(root).resolve()
    part = PurePosixPath(name)
    if part.is_absolute() or ".." in part.parts or "\\" in name or (part.as_posix() != name):
        raise ValueError("Unsafe artifact path")
    path = root.joinpath(*part.parts)
    if (
        not path.resolve().is_relative_to(root)
        or path.is_symlink()
        or any(p.is_symlink() for p in path.parents if p != root and p.is_relative_to(root))
    ):
        raise ValueError("Unsafe artifact symlink")
    return path


def manifest(root):
    value = read_json(safe(root, "artifact_manifest.json"))
    entries = value["files"]
    if not isinstance(entries, dict):
        raise ValueError("Invalid artifact manifest")
    for name, item in entries.items():
        safe(root, name)
        if (
            not isinstance(item, dict)
            or re.fullmatch("[0-9a-f]{64}", str(item.get("sha256", ""))) is None
        ):
            raise ValueError("Invalid artifact hash")
    return entries


def verify(root, entries, names):
    for name in names:
        if name not in entries:
            raise ValueError(f"Unlisted artifact: {name}")
        path = safe(root, name)
        if not path.is_file() or sha256(path) != entries[name]["sha256"]:
            raise ValueError(f"Missing or changed artifact: {name}")


@contextmanager
def connection(engine):
    owned = not isinstance(engine, sqlite3.Connection)
    conn = engine.raw_connection() if owned else engine
    try:
        yield conn
    finally:
        if owned:
            conn.close()


def rows(engine, sql, params=()):
    with connection(engine) as conn:
        cursor = conn.execute(sql, params)
        names = [item[0] for item in cursor.description]
        return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


def header(engine, run_id):
    if not rows(engine, "SELECT name FROM sqlite_master WHERE type='table' AND name='model_runs'"):
        raise ValueError("No model run registry")
    found = rows(engine, "SELECT * FROM model_runs WHERE run_id=?", (run_id,))
    if len(found) != 1:
        raise ValueError("Unknown run ID")
    return found[0]


def run_directory(h):
    from ml_template.paths import ARTIFACTS_RUNS, ROOT

    stored = h.get("artifact_dir")
    valid = stored is not None and (not pd.isna(stored)) and str(stored)
    path = Path(stored) if valid else ARTIFACTS_RUNS / str(h.get("run_id", "unknown"))
    return (path if path.is_absolute() else ROOT / path).resolve()


def task_hint(h):
    """Routing hint only; the regression loader separately verifies archived config."""
    regression_metric = h.get("primary_metric") in {"rmse", "mae", "r2"}
    path = run_directory(h) / "config.json"
    if path.is_file():
        try:
            task = read_json(path).get("task", "binary_classification")
            return "regression" if regression_metric and task != "regression" else task
        except ERRORS:
            return "regression" if regression_metric else "unknown"
    return "regression" if regression_metric else "binary_classification"


def is_regression_run(engine, run_id):
    if not run_id:
        return False
    try:
        return task_hint(header(engine, run_id)) == "regression"
    except ERRORS:
        return False


def guard_binary_labels(h):
    if task_hint(h) != "binary_classification":
        raise ValueError(
            "Binary raw-label loading blocked for regression/unknown task; use saved regression artifacts"
        )


def snapshot_hash(cfg):
    """Pure implementation of the recorded schema-v2 hash; no training imports."""
    if cfg.get("schema_version") != 2:
        raise ValueError("Unsupported regression snapshot version")
    value = copy.deepcopy(cfg)
    value.pop("analysis", None)
    if value.get("task") == "binary_classification":
        value.pop("task", None)
    if value.get("fitted_features") is None:
        value.pop("fitted_features", None)
    if value["split"].get("cv_seed") is None:
        value["split"].pop("cv_seed", None)
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]


def context(engine, run_id):
    h = header(engine, run_id)
    if h["status"] != "succeeded":
        raise ValueError("Select a successful run")
    root = run_directory(h)
    entries = manifest(root)
    verify(root, entries, ["config.json", "features.json"])
    cfg = read_json(safe(root, "config.json"))
    if cfg.get("task") != "regression" or snapshot_hash(cfg) != h["config_hash"]:
        raise ValueError("Run task/config/registry binding differs")
    info = read_json(safe(root, "features.json"))
    if (
        cfg["data"]["silver_dataset_version_id"] != h["silver_dataset_version_id"]
        or info["silver_dataset_version_id"] != h["silver_dataset_version_id"]
        or info["split_id"] != h["split_id"]
    ):
        raise ValueError("Dataset/split binding differs")
    if (
        cfg["data"].get("feature_set_id") != h.get("feature_set_id")
        or cfg["seed"] != h["seed"]
        or cfg["model"]["type"] != h["model_type"]
    ):
        raise ValueError("Run model/seed/features differ")
    count = cfg["split"]["n_folds"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise ValueError("Invalid fold count")
    if not isinstance(info["features"], list) or len(info["features"]) != len(
        set(info["features"])
    ):
        raise ValueError("Invalid feature inventory")
    return {"header": h, "root": root, "entries": entries, "config": cfg, "features": info}


def optional_table(ctx, name):
    path = safe(ctx["root"], name)
    if name not in ctx["entries"] and (not path.exists()):
        return None
    verify(ctx["root"], ctx["entries"], [name])
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def analysis_root():
    from ml_template.paths import ARTIFACTS, ROOT

    path = Path(
        os.environ.get("ML_TEMPLATE_ANALYSIS_ROOT", str(ARTIFACTS / "analyses"))
    ).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def external_metadata(ctx, directory):
    entries = manifest(directory)
    verify(
        directory,
        entries,
        ["analysis.json", "provenance.json", "report_data.json", "residual_summary.json"],
    )
    state = read_json(safe(directory, "analysis.json"))
    provenance = read_json(safe(directory, "provenance.json"))
    rid = ctx["header"]["run_id"]
    if (
        state.get("status") != "succeeded"
        or state.get("analysis_id") != directory.name
        or state.get("parent_run_id") != rid
        or (provenance.get("parent_run_id") != rid)
    ):
        raise ValueError("Analysis state/parent/ID differs")
    bound = provenance["input_provenance"]["input_hashes"]
    for alias, name in [("config", "config.json"), ("features", "features.json")]:
        if bound.get(alias) != ctx["entries"][name]["sha256"]:
            raise ValueError("Analysis belongs to a different run snapshot")
    if provenance.get("analysis_kind") is not None:
        from ml_template.evaluation.residual_artifacts import verify_external

        verify_external(ctx, directory, entries, provenance, state)
        return entries
    descriptor = provenance["fold_descriptor"]
    count = ctx["config"]["split"]["n_folds"]
    members = descriptor.get("members", [])
    if (
        descriptor.get("schema_version") != 2
        or descriptor.get("task") != "regression"
        or descriptor.get("prediction_kind") != "continuous"
        or (descriptor.get("kind") != "cv_mean")
        or (descriptor.get("fitted_features") is not False)
        or (descriptor.get("weighting") != "equal")
        or ([m.get("fold_id") for m in members] != list(range(count)))
    ):
        raise ValueError("Invalid fold inventory descriptor")
    expected_files = {"cv_folds.csv"}
    for k, member in enumerate(members):
        name = f"fold_models/fold_{k}.ubj"
        if (
            member.get("model_path") != name
            or member.get("transformer_path") is not None
            or (not np.isclose(member.get("weight", np.nan), 1 / count, rtol=0, atol=1e-15))
        ):
            raise ValueError("Invalid fold member")
        expected_files.add(name)
    if set(descriptor["files_sha256"]) != expected_files:
        raise ValueError("Incomplete fold hash coverage")
    for name, expected in descriptor["files_sha256"].items():
        if ctx["entries"].get(name, {}).get("sha256") != expected:
            raise ValueError("Analysis/model hashes differ")
        verify(ctx["root"], ctx["entries"], [name])
    return entries


def discover(ctx):
    options, mapping, warnings = ([], {}, [])
    root = analysis_root()
    if root.is_dir():
        for directory in sorted(root.iterdir()):
            if (
                directory.is_symlink()
                or not directory.is_dir()
                or (not re.fullmatch("analysis_[A-Za-z0-9_]+", directory.name))
            ):
                continue
            try:
                state = read_json(safe(directory, "analysis.json"))
                if (
                    state.get("parent_run_id") != ctx["header"]["run_id"]
                    or state.get("status") != "succeeded"
                ):
                    continue
                entries = external_metadata(ctx, directory)
                key = "analysis:" + directory.name
                options.append({"label": directory.name + " (standalone)", "value": key})
                mapping[key] = (directory, entries, "standalone")
            except ERRORS as exc:
                warnings.append(f"{directory.name}: {exc}")
    prefix = "regression_diagnostics/"
    required = [
        "report_data.json",
        "residual_summary.json",
        "oof_residuals.parquet",
        "error_sample.csv",
        "permutation_importance.csv",
    ]
    if prefix + "report_data.json" in ctx["entries"]:
        try:
            verify(ctx["root"], ctx["entries"], [prefix + name for name in required])
            if "analysis_errors.json" in ctx["entries"]:
                verify(ctx["root"], ctx["entries"], ["analysis_errors.json"])
                if "regression_diagnostics" in read_json(safe(ctx["root"], "analysis_errors.json")):
                    raise ValueError("Training recorded a diagnostic failure")
            key = "run:" + ctx["header"]["run_id"] + ":diagnostics"
            options.append({"label": "Diagnostics sealed with this run", "value": key})
            nested = {
                name[len(prefix) :]: item
                for name, item in ctx["entries"].items()
                if name.startswith(prefix)
            }
            mapping[key] = (ctx["root"] / "regression_diagnostics", nested, "training_run")
        except ERRORS as exc:
            warnings.append(str(exc))
    return (options, mapping, warnings)


def finite(table, columns):
    for name in columns:
        if not np.isfinite(pd.to_numeric(table[name], errors="raise").to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite saved values: {name}")


def validate_curves(table, role, folds):
    if table is None:
        return
    if table.empty or not {"fold", "round", "dataset", "metric", "value"} <= set(table):
        raise ValueError("Incomplete curve table")
    finite(table, ["fold", "round", "value"])
    if (
        set(table.fold) != set(folds)
        or (table.fold % 1 != 0).any()
        or (table["round"] < 1).any()
        or (table["round"] % 1 != 0).any()
    ):
        raise ValueError("Invalid curve fold/round IDs")
    labels = {"inner_fit", "inner_stop"} if role == "inner" else {"train", "val"}
    if (
        set(table.dataset) != labels
        or table.metric.nunique() != 1
        or (not set(table.metric) <= {"rmse", "mae"})
    ):
        raise ValueError("Curve task/role differs")
    for _, part in table.groupby("fold"):
        if part.groupby("dataset").size().nunique() != 1:
            raise ValueError("Curve lengths differ")
        for _, group in part.groupby("dataset"):
            if sorted(group["round"].tolist()) != list(range(1, len(group) + 1)):
                raise ValueError("Curve rounds must be unique and consecutive")
        if role == "inner":
            if not {"selected_round", "direction", "curve_role"} <= set(part):
                raise ValueError("Missing inner stopping provenance")
            finite(part, ["selected_round"])
            selected = float(part.selected_round.iloc[0])
            if (
                part.selected_round.nunique() != 1
                or set(part.direction) != {"min"}
                or set(part.curve_role) != {"inner_early_stopping"}
                or (not selected.is_integer())
                or (selected < 1)
                or (selected > part["round"].max())
            ):
                raise ValueError("Invalid selected-round provenance")


def load_analysis(ctx, identifier, mapping):
    if identifier not in mapping:
        raise ValueError("Select an analysis explicitly for this run")
    directory, entries, origin = mapping[identifier]
    verify(directory, entries, list(entries))
    report = read_json(safe(directory, "report_data.json"))
    summary = read_json(safe(directory, "residual_summary.json"))
    if report["summary"] != summary or summary.get("holdout_labels_loaded") is not False:
        raise ValueError("Analysis summary provenance differs")
    original = optional_table(ctx, "oof_predictions.parquet")
    if original is None:
        raise ValueError("Original OOF artifact unavailable")
    keys = [c for c in original if c not in {"fold", "y_pred_oof"}]
    if len(keys) != 1 or original.columns.duplicated().any():
        raise ValueError("Ambiguous OOF key schema")
    key = keys[0]
    residuals = pd.read_parquet(safe(directory, "oof_residuals.parquet"))
    required = {
        key,
        "fold",
        "y_true",
        "y_pred_oof",
        "residual",
        "absolute_error",
        "squared_error",
        "error_kind",
    }
    if not required <= set(residuals) or residuals.empty:
        raise ValueError("Incomplete residual table")
    for frame in [original, residuals]:
        if frame[key].isna().any() or frame[key].duplicated().any():
            raise ValueError("Duplicate/null OOF or residual keys")
    if set(original[key]) != set(residuals[key]):
        raise ValueError("Residual/OOF key coverage differs")
    r, old = (residuals.set_index(key).sort_index(), original.set_index(key).sort_index())
    finite(r, ["fold", "y_true", "y_pred_oof", "residual", "absolute_error", "squared_error"])
    if set(r.fold) != set(range(ctx["config"]["split"]["n_folds"])) or not np.array_equal(
        r.fold, old.fold
    ):
        raise ValueError("Residual/OOF fold identity differs")
    if not np.allclose(r.y_pred_oof, old.y_pred_oof, rtol=0, atol=1e-07):
        raise ValueError("Analysis differs from original OOF predictions")
    expected = r.y_true - r.y_pred_oof
    if (
        not np.allclose(r.residual, expected)
        or not np.allclose(r.absolute_error, np.abs(expected))
        or (not np.allclose(r.squared_error, expected**2))
    ):
        raise ValueError("Residual arithmetic differs")
    direction = np.where(
        expected > 0, "underprediction", np.where(expected < 0, "overprediction", "exact")
    )
    if not np.array_equal(r.error_kind, direction) or summary["n_development"] != len(r):
        raise ValueError("Residual labels/support differ")
    importance = pd.read_csv(safe(directory, "permutation_importance.csv"))
    cols = {"feature", "fold", "mean", "repeat_std", "between_fold_std", "metric", "target_unit"}
    if not cols <= set(importance):
        raise ValueError("Invalid importance schema")
    enabled = summary["permutation"]["enabled"]
    if not isinstance(enabled, bool) or enabled != (not importance.empty):
        raise ValueError("Importance enablement differs")
    if not importance.empty:
        finite(importance, ["fold", "mean"])
        coverage = {(f, k) for f in ctx["features"]["features"] for k in [-1, *set(old.fold)]}
        if (
            importance.duplicated(["feature", "fold"]).any()
            or set(zip(importance.feature, importance.fold, strict=True)) != coverage
        ):
            raise ValueError("Incomplete importance feature/fold inventory")
        if (
            set(importance.metric) != {summary["permutation"]["metric"]}
            or not set(importance.metric) <= {"rmse", "mae"}
            or set(importance.target_unit) != {summary["target_unit"]}
        ):
            raise ValueError("Importance metric/units differ")
        for name in ["repeat_std", "between_fold_std"]:
            values = pd.to_numeric(importance[name], errors="raise").dropna().to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("Invalid importance variability")
        for _, group in importance.groupby("feature"):
            folds, pooled = (group[group.fold >= 0], group[group.fold == -1].iloc[0])
            if not np.isclose(pooled["mean"], folds["mean"].mean()) or not np.isclose(
                pooled.between_fold_std, folds["mean"].std(ddof=1)
            ):
                raise ValueError("Importance aggregation differs")
            if not pd.isna(pooled.repeat_std) or folds.between_fold_std.notna().any():
                raise ValueError("Repeat and fold variability are mixed")
    sample = pd.read_csv(safe(directory, "error_sample.csv"), dtype={key: "string"})
    reference = residuals.copy()
    reference[key] = reference[key].astype("string")
    reference = reference.set_index(key)
    if (
        key not in sample
        or sample[key].isna().any()
        or sample[key].duplicated().any()
        or (not set(sample[key]) <= set(reference.index))
    ):
        raise ValueError("Error sample key coverage differs")
    for name in ["y_true", "y_pred_oof", "residual", "absolute_error"]:
        if name not in sample or not np.allclose(
            pd.to_numeric(sample[name], errors="raise"), reference.loc[sample[key], name]
        ):
            raise ValueError("Error sample values differ")
    return {
        "id": identifier,
        "origin": origin,
        "summary": summary,
        "residuals": residuals,
        "importance": importance,
        "errors": sample,
    }


def load_bundle(engine, run_id, selected=None):
    if not run_id:
        return {"state": "empty", "message": "Select a run"}
    try:
        ctx = context(engine, run_id)
        options, mapping, warnings = discover(ctx)
        inner = optional_table(ctx, "early_stopping_curves.parquet")
        outer = optional_table(ctx, "learning_curves.parquet")
        folds = list(range(ctx["config"]["split"]["n_folds"]))
        validate_curves(inner, "inner", folds)
        validate_curves(outer, "outer", folds)
        for name, role, table in [
            ("early_stopping_curves_metadata.json", "inner_early_stopping", inner),
            ("learning_curves_metadata.json", "outer_diagnostic_refit", outer),
        ]:
            if name in ctx["entries"]:
                verify(ctx["root"], ctx["entries"], [name])
                meta = read_json(safe(ctx["root"], name))
                if meta.get("curve_role") != role or meta.get("used_holdout") is not False:
                    raise ValueError("Curve metadata role differs")
            elif table is not None:
                warnings.append(
                    f"{name} missing: displaying the verified legacy table under its known artifact role"
                )
        inventory = optional_table(ctx, "cv_folds.csv")
        if inventory is not None and inner is not None:
            if (
                not {"fold", "best_round"} <= set(inventory)
                or inventory.fold.duplicated().any()
                or set(inventory.fold) != set(folds)
            ):
                raise ValueError("Invalid CV round inventory")
            if not np.array_equal(
                inventory.set_index("fold").best_round.sort_index(),
                inner.groupby("fold").selected_round.first().sort_index(),
            ):
                raise ValueError("Inner selected rounds differ from CV inventory")
        exists = rows(
            engine, "SELECT name FROM sqlite_master WHERE type='table' AND name='run_metrics'"
        )
        metrics = (
            pd.DataFrame(
                rows(
                    engine,
                    "SELECT split,fold,metric,value FROM run_metrics WHERE run_id=? AND split='cv'",
                    (run_id,),
                )
            )
            if exists
            else pd.DataFrame(columns=["split", "fold", "metric", "value"])
        )
        analysis = load_analysis(ctx, selected, mapping) if selected else None
        return {
            "state": "verified",
            "context": ctx,
            "options": options,
            "warnings": warnings,
            "inner": inner,
            "outer": outer,
            "metrics": metrics,
            "analysis": analysis,
        }
    except ERRORS as exc:
        return {
            "state": "invalid",
            "message": f"Cannot display unverified regression artifacts: {exc}",
        }
