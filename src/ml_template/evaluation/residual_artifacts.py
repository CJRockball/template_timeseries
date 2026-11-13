"""Pure saved-OOF residual summaries and read-only analysis verification. No model API."""

import numpy as np
import pandas as pd

KIND = "oof_residuals_v1"


def check_keys(values):
    values = pd.Index(values)
    if values.empty or values.hasnans or not values.is_unique:
        raise ValueError("Keys must be nonempty, unique and nonmissing")
    kinds = set()
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("Boolean keys are forbidden")
        if isinstance(value, (int, np.integer)):
            kinds.add("int")
        elif isinstance(value, str):
            kinds.add("str")
        else:
            raise ValueError("Keys must be integers or strings")
    if len(kinds) != 1:
        raise ValueError("Mixed key types are forbidden")


def statistics(frame):
    if frame.columns.duplicated().any() or not {"y_true", "y_pred_oof"} <= set(frame):
        raise ValueError("Ambiguous residual summary schema")
    for name in ("y_true", "y_pred_oof"):
        dtype = frame[name].dtype
        if (
            not pd.api.types.is_numeric_dtype(dtype)
            or pd.api.types.is_bool_dtype(dtype)
            or pd.api.types.is_complex_dtype(dtype)
        ):
            raise ValueError("Summary vectors must be real numeric, not booleans/objects")
    y = frame.y_true.to_numpy(dtype=float)
    p = frame.y_pred_oof.to_numpy(dtype=float)
    if not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Residual summaries need nonempty finite real vectors")
    with np.errstate(over="ignore", invalid="ignore"):
        r = y - p
        sse = float(np.sum(r**2))
        centered = float(np.sum((y - np.mean(y)) ** 2))
        values = {
            "n_rows": len(r),
            "rmse": float(np.sqrt(np.mean(r**2))),
            "mae": float(np.mean(np.abs(r))),
            "mean_residual": float(np.mean(r)),
            "residual_std_population": float(np.std(r)),
            "r2": 1.0 - sse / centered if centered > 0 else None,
        }
        quantiles = {
            str(q): float(np.quantile(r, q)) for q in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
        }
    if not np.isfinite(sse) or not np.isfinite(centered):
        raise ValueError("Residual arithmetic overflow")
    if any(not np.isfinite(v) for v in values.values() if v is not None):
        raise ValueError("Nonfinite residual summary")
    values["residual_quantiles"] = quantiles
    return values


def augment_summary(result):
    result.summary.update(
        analysis_kind=KIND,
        pooled=statistics(result.residuals),
        fold_summaries=[
            {"fold": int(fold), **statistics(part)}
            for fold, part in result.residuals.groupby("fold", sort=True)
        ],
    )
    return result


def verify_external(ctx, directory, entries, provenance, state):
    # Imports only read-only dashboard helpers; never the analysis executor.
    from ml_template.dashboard import regression_data as base

    if (
        provenance.get("schema_version") != 1
        or provenance.get("analysis_kind") != KIND
        or state.get("analysis_kind") != KIND
        or provenance.get("prediction_source") != "original_saved_oof"
    ):
        raise ValueError("Unsupported residual-only analysis semantics")
    for name in (
        "models_loaded",
        "predictions_computed",
        "holdout_labels_loaded",
        "registry_rows_written",
    ):
        if provenance.get(name) is not False:
            raise ValueError("Residual-only provenance forbids: " + name)
    source = provenance["input_provenance"]["input_hashes"]
    if source.get("oof") != ctx["entries"].get("oof_predictions.parquet", {}).get("sha256"):
        raise ValueError("Residual analysis original OOF binding differs")
    base.verify(ctx["root"], ctx["entries"], ["oof_predictions.parquet"])
    base.verify(directory, entries, list(entries))
    summary = base.read_json(base.safe(directory, "residual_summary.json"))
    if (
        summary.get("analysis_kind") != KIND
        or summary.get("holdout_labels_loaded") is not False
        or summary.get("permutation", {}).get("enabled") is not False
    ):
        raise ValueError("Residual summary scope differs")
    unit = summary.get("target_unit")
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError("Residual analysis requires an explicit target unit")
    original = pd.read_parquet(base.safe(ctx["root"], "oof_predictions.parquet"))
    keys = [c for c in original if c not in {"fold", "y_pred_oof"}]
    if len(keys) != 1 or original.columns.duplicated().any():
        raise ValueError("Invalid original OOF schema")
    key = keys[0]
    residuals = pd.read_parquet(base.safe(directory, "oof_residuals.parquet"))
    check_keys(original[key])
    check_keys(residuals[key])
    old, new = original.set_index(key).sort_index(), residuals.set_index(key).sort_index()
    if not old.index.equals(new.index) or not np.array_equal(old.fold, new.fold):
        raise ValueError("Residual original-key/fold binding differs")
    if not np.array_equal(old.y_pred_oof, new.y_pred_oof):
        raise ValueError("Residual predictions differ from original OOF")
    if summary.get("pooled") != statistics(new):
        raise ValueError("Pooled residual summary differs")
    folds = [{"fold": int(k), **statistics(part)} for k, part in new.groupby("fold", sort=True)]
    if summary.get("fold_summaries") != folds:
        raise ValueError("Per-fold residual summary differs")
