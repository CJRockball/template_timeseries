"""Pure OOF residuals and fold-local permutation error increases."""

import html
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Diagnostics:
    residuals: pd.DataFrame
    error_sample: pd.DataFrame
    summary: dict
    importance: pd.DataFrame
    repeats: pd.DataFrame
    samples: dict
    registry_rows: list


def finite_vector(values, n, name):
    arr = np.asarray(values)
    if arr.shape != (n,) or arr.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a numeric vector, not booleans/objects/matrices")
    arr = arr.astype(float)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    return arr


def error_metric(y, prediction, metric):
    if metric not in {"rmse", "mae"}:
        raise ValueError("Permutation metric must be rmse or mae")
    y = finite_vector(y, len(y), "target")
    p = finite_vector(prediction, len(y), "prediction")
    if not len(y):
        raise ValueError("Empty metric input")
    residual = y - p
    with np.errstate(over="ignore", invalid="ignore"):
        value = (
            float(np.sqrt(np.mean(residual**2)))
            if metric == "rmse"
            else float(np.mean(np.abs(residual)))
        )
    if not np.isfinite(value):
        raise ValueError("Regression error metric overflowed")
    return value


def predict(adapter, X):
    return finite_vector(adapter.predict(X.copy(deep=True)), len(X), "prediction")


def regression_permutation_importance(
    adapter, X, y, *, metric="rmse", n_repeats=10, seed=42, max_rows=None, parity_atol=1e-7
):
    if not X.index.is_unique or X.index.hasnans or not X.index.equals(y.index):
        raise ValueError("Feature/target indices must be unique, non-null and aligned")
    if X.empty or X.columns.duplicated().any():
        raise ValueError("Nonempty uniquely named features required")
    if isinstance(n_repeats, bool) or not isinstance(n_repeats, int) or n_repeats < 1:
        raise ValueError("n_repeats must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if max_rows is not None and (
        isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1
    ):
        raise ValueError("max_rows must be a positive integer or None")
    rng = np.random.default_rng(seed)
    positions = np.arange(len(X))
    if max_rows is not None and len(X) > max_rows:
        positions = np.sort(rng.choice(len(X), max_rows, replace=False))
    sample = X.iloc[positions].copy(deep=True)
    truth = finite_vector(y.iloc[positions].to_numpy(), len(sample), "target")
    baseline_prediction = predict(adapter, sample)
    baseline = error_metric(truth, baseline_prediction, metric)
    rows = []
    for column in sample:
        for repeat in range(n_repeats):
            work = sample.copy(deep=True)
            shuffled = sample[column].iloc[rng.permutation(len(sample))].copy()
            shuffled.index = sample.index
            work[column] = shuffled
            perturbed = error_metric(truth, predict(adapter, work), metric)
            rows.append(
                {
                    "feature": column,
                    "repeat": repeat,
                    "baseline": baseline,
                    "permuted": perturbed,
                    "importance": perturbed - baseline,
                    "metric": metric,
                    "n_rows": len(sample),
                }
            )
    again = predict(adapter, sample)
    if not np.allclose(again, baseline_prediction, rtol=0, atol=parity_atol):
        raise ValueError("Predictions changed during diagnostic computation")
    repeats = pd.DataFrame(rows)
    table = repeats.groupby("feature", sort=False).importance.agg(["mean", "std"]).reset_index()
    table = table.rename(columns={"std": "repeat_std"})
    table["metric"] = metric
    table["n_rows"] = len(sample)
    table["n_repeats"] = n_repeats
    table["baseline"] = baseline
    return table, repeats, sample.index.tolist()


def align_oof(oof, X, y, assignments, key_col):
    if (
        not X.index.is_unique
        or X.index.hasnans
        or not y.index.is_unique
        or not X.index.equals(y.index)
    ):
        raise ValueError("Feature/target indices must be aligned unique non-null keys")
    if assignments[key_col].isna().any() or assignments[key_col].duplicated().any():
        raise ValueError("Split keys must be unique and non-null")
    allowed_partitions = ["dev", "holdout", "calibration", "recalibration"]
    if not assignments.partition.isin(allowed_partitions).all():
        raise ValueError("Unknown split partition")
    non_development = ~assignments.partition.eq("dev")
    if not assignments.loc[non_development, "fold"].eq(-1).all():
        raise ValueError("Non-development rows must use fold -1")
    dev = assignments.loc[assignments.partition.eq("dev"), [key_col, "fold"]]
    if dev.empty or dev.fold.isna().any() or (dev.fold < 0).any() or (dev.fold % 1 != 0).any():
        raise ValueError("Invalid development folds")
    if len(dev.fold.unique()) < 2:
        raise ValueError("At least two development folds required")
    if not set(dev[key_col]) <= set(X.index):
        raise ValueError("Development rows missing from inputs")
    required = [key_col, "fold", "y_pred_oof"]
    if not set(required) <= set(oof) or oof.columns.duplicated().any():
        raise ValueError("Regression OOF schema is missing/ambiguous")
    if (
        oof[key_col].isna().any()
        or oof[key_col].duplicated().any()
        or set(oof[key_col]) != set(dev[key_col])
    ):
        raise ValueError("OOF must cover exactly development keys; holdout forbidden")
    merged = oof[required].merge(dev, on=key_col, validate="one_to_one", suffixes=("", "_stored"))
    if not merged.fold.eq(merged.fold_stored).all():
        raise ValueError("OOF/stored fold identities differ")
    merged = (
        merged.drop(columns="fold_stored")
        .sort_values(key_col, kind="stable")
        .reset_index(drop=True)
    )
    truth = finite_vector(y.loc[merged[key_col]].to_numpy(), len(merged), "development target")
    prediction = finite_vector(merged.y_pred_oof.to_numpy(), len(merged), "OOF prediction")
    merged["y_true"] = truth
    merged["residual"] = truth - prediction
    merged["absolute_error"] = np.abs(merged.residual)
    with np.errstate(over="ignore"):
        merged["squared_error"] = merged.residual**2
    if not np.isfinite(merged.squared_error).all():
        raise ValueError("Squared residuals overflowed")
    merged["error_kind"] = np.where(
        merged.residual > 0,
        "underprediction",
        np.where(merged.residual < 0, "overprediction", "exact"),
    )
    return merged


def compute_diagnostics(
    oof,
    X,
    y,
    assignments,
    key_col,
    *,
    model_loader=None,
    permutation=True,
    metric="rmse",
    n_repeats=10,
    seed=42,
    max_rows=None,
    error_sample_size=25,
    target_unit="target_units",
    parity_atol=1e-7,
):
    if metric not in {"rmse", "mae"}:
        raise ValueError("Permutation metric must be rmse or mae")
    if (
        isinstance(error_sample_size, bool)
        or not isinstance(error_sample_size, int)
        or error_sample_size < 0
    ):
        raise ValueError("error_sample_size must be a nonnegative integer")
    if not np.isfinite(parity_atol) or parity_atol <= 0:
        raise ValueError("parity_atol must be positive and finite")
    residuals = align_oof(oof, X, y, assignments, key_col)
    clash = set(X.columns) & set(residuals.columns)
    if clash:
        raise ValueError(f"Feature/report column names clash: {sorted(clash)}")
    sample = (
        residuals.sort_values("absolute_error", ascending=False, kind="stable")
        .head(error_sample_size)
        .copy()
    )
    feature_rows = X.loc[sample[key_col]].reset_index(drop=True)
    sample = pd.concat([sample.reset_index(drop=True), feature_rows], axis=1)
    point = {
        "rmse": error_metric(residuals.y_true, residuals.y_pred_oof, "rmse"),
        "mae": error_metric(residuals.y_true, residuals.y_pred_oof, "mae"),
    }
    residual = residuals.residual.to_numpy()
    summary = {
        "n_development": len(residuals),
        "metric_values": point,
        "residual_convention": "y_true - y_pred_oof; positive means underprediction",
        "residual_mean": float(np.mean(residual)),
        "residual_std_population": float(np.std(residual)),
        "target_unit": target_unit,
        "holdout_labels_loaded": False,
        "residual_quantiles": {
            str(q): float(np.quantile(residual, q)) for q in [0.05, 0.25, 0.5, 0.75, 0.95]
        },
    }
    tables, repeat_tables, samples, registry_rows = [], [], {}, []
    if permutation:
        if model_loader is None:
            raise ValueError("Permutation diagnostics require saved fold models")
        for fold in sorted(int(k) for k in residuals.fold.unique()):
            part = residuals.loc[residuals.fold.eq(fold)].copy()
            keys = part[key_col].to_numpy()
            adapter = model_loader(fold)
            try:
                baseline = predict(adapter, X.loc[keys])
                if not np.allclose(baseline, part.y_pred_oof.to_numpy(), rtol=0, atol=parity_atol):
                    raise ValueError(f"Saved fold {fold} predictions differ from original OOF")
                table, repeats, chosen = regression_permutation_importance(
                    adapter,
                    X.loc[keys],
                    y.loc[keys],
                    metric=metric,
                    n_repeats=n_repeats,
                    seed=seed + fold,
                    max_rows=max_rows,
                    parity_atol=parity_atol,
                )
                if not np.allclose(
                    predict(adapter, X.loc[keys]), baseline, rtol=0, atol=parity_atol
                ):
                    raise ValueError(f"Saved fold {fold} predictions changed after permutation")
            finally:
                del adapter
            table = table.assign(fold=fold, target_unit=target_unit)
            repeats = repeats.assign(fold=fold, target_unit=target_unit)
            tables.append(table)
            repeat_tables.append(repeats)
            samples[str(fold)] = [v.item() if isinstance(v, np.generic) else v for v in chosen]
            for row in table.itertuples():
                registry_rows.append(
                    {
                        "split": "cv",
                        "fold": fold,
                        "method": f"permutation_{metric}_increase",
                        "feature": row.feature,
                        "value": float(row.mean),
                        "std": float(row.repeat_std),
                    }
                )
        per_fold = pd.concat(tables, ignore_index=True)
        pooled = []
        for feature, group in per_fold.groupby("feature", sort=False):
            pooled.append(
                {
                    "feature": feature,
                    "fold": -1,
                    "mean": float(group["mean"].mean()),
                    "repeat_std": np.nan,
                    "between_fold_std": float(group["mean"].std(ddof=1)),
                    "metric": metric,
                    "n_rows": int(group.n_rows.sum()),
                    "n_repeats": n_repeats,
                    "baseline": np.nan,
                    "target_unit": target_unit,
                }
            )
            registry_rows.append(
                {
                    "split": "cv",
                    "fold": -1,
                    "method": f"permutation_{metric}_increase",
                    "feature": feature,
                    "value": pooled[-1]["mean"],
                    "std": pooled[-1]["between_fold_std"],
                }
            )
        per_fold["between_fold_std"] = np.nan
        importance = pd.concat([per_fold, pd.DataFrame(pooled)], ignore_index=True)
        repeats = pd.concat(repeat_tables, ignore_index=True)
    else:
        importance = pd.DataFrame(
            columns=[
                "feature",
                "fold",
                "mean",
                "repeat_std",
                "between_fold_std",
                "metric",
                "n_rows",
                "n_repeats",
                "baseline",
                "target_unit",
            ]
        )
        repeats = pd.DataFrame(
            columns=[
                "feature",
                "repeat",
                "baseline",
                "permuted",
                "importance",
                "metric",
                "n_rows",
                "fold",
                "target_unit",
            ]
        )
    summary["permutation"] = {
        "enabled": permutation,
        "metric": metric,
        "direction": "error_increase",
        "n_repeats": n_repeats,
        "max_rows_per_fold": max_rows,
        "seed": seed,
        "pooling": "unweighted mean of fold mean importances; not pooled OOF RMSE change",
        "repeat_std": "sample std across shuffles within one fold",
        "between_fold_std": "sample std across fold means; not a confidence interval",
        "full_fold_prediction_parity_before_and_after": permutation,
    }
    return Diagnostics(residuals, sample, summary, importance, repeats, samples, registry_rows)


def write_diagnostics(result, directory, *, plots=True):
    from ml_template.evaluation.evaluation_artifacts import manifest, write_json

    directory = Path(directory)
    result.residuals.to_parquet(directory / "oof_residuals.parquet", index=False)
    result.error_sample.to_csv(directory / "error_sample.csv", index=False)
    result.importance.to_csv(directory / "permutation_importance.csv", index=False)
    result.repeats.to_parquet(directory / "permutation_repeats.parquet", index=False)
    write_json(directory / "sample_keys.json", result.samples)
    write_json(directory / "residual_summary.json", result.summary)
    warnings = [
        "Exploratory development diagnostics, not fresh validation or deployment evidence.",
        "Correlated features can mask each other's marginal permutation importance.",
        "Negative importance is retained; shuffling can occasionally reduce error.",
    ]
    report = {
        "summary": result.summary,
        "warnings": warnings,
        "importance": result.importance.to_dict(orient="records"),
    }
    write_json(directory / "report_data.json", report)
    if plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        r = result.residuals
        axes[0].scatter(r.y_true, r.y_pred_oof, s=9, alpha=0.6)
        low, high = (
            float(min(r.y_true.min(), r.y_pred_oof.min())),
            float(max(r.y_true.max(), r.y_pred_oof.max())),
        )
        axes[0].plot([low, high], [low, high], color="grey", ls="--")
        axes[0].set(xlabel="observed", ylabel="OOF predicted")
        axes[1].scatter(r.y_pred_oof, r.residual, s=9, alpha=0.6)
        axes[1].axhline(0, color="grey", ls="--")
        axes[1].set(xlabel="OOF predicted", ylabel="residual")
        axes[2].hist(r.residual, bins=min(30, max(5, int(np.sqrt(len(r))))))
        axes[2].set(xlabel="residual", ylabel="rows")
        fig.suptitle(f"Development OOF residuals — {result.summary['target_unit']}")
        fig.tight_layout()
        fig.savefig(directory / "residual_diagnostics.png", dpi=150)
        plt.close(fig)
    image = '<img src="residual_diagnostics.png" alt="OOF residual plots">' if plots else ""
    safe = html.escape(json.dumps(result.summary, indent=2))
    (directory / "report.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Regression diagnostics</title><h1>Development OOF diagnostics</h1>'
        + image
        + "<pre>"
        + safe
        + "</pre>"
        + result.importance.to_html(index=False, escape=True),
        encoding="utf-8",
    )
    manifest(directory)
