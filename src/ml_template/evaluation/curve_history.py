"""Recorded inner stopping history, distinct from outer diagnostic refits."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

INNER_COLUMNS = [
    "fold",
    "round",
    "dataset",
    "metric",
    "value",
    "selected_round",
    "direction",
    "curve_role",
]


def history_table(history, metric, selected_round, *, inner):
    """Convert two evaluation trajectories; never fit or select a round here."""
    labels = ("inner_fit", "inner_stop") if inner else ("train", "val")
    parts = []
    lengths = []
    for key, label in zip(("validation_0", "validation_1"), labels, strict=True):
        values = np.asarray(history[key][metric], dtype=float)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
            raise ValueError("Expected a nonempty finite metric history")
        lengths.append(len(values))
        parts.append(
            pd.DataFrame(
                {
                    "round": np.arange(1, len(values) + 1),
                    "dataset": label,
                    "metric": metric,
                    "value": values,
                }
            )
        )
    if lengths[0] != lengths[1]:
        raise ValueError("Training and validation history lengths differ")
    table = pd.concat(parts, ignore_index=True)
    if inner:
        if metric not in {"rmse", "mae"}:
            raise ValueError("Regression stopping history supports rmse/mae")
        if not 1 <= int(selected_round) <= lengths[0]:
            raise ValueError("Selected round is outside recorded history")
        table = table.assign(
            selected_round=int(selected_round), direction="min", curve_role="inner_early_stopping"
        )
    return table


def capture_inner_history(adapter, best_round, fold):
    history = getattr(adapter, "early_stopping_history", None)
    if history is None:
        return None
    if not isinstance(history, pd.DataFrame) or history.empty:
        raise ValueError("early_stopping_history must be a nonempty DataFrame")
    table = history.copy(deep=True).assign(fold=int(fold))
    missing = set(INNER_COLUMNS) - set(table.columns)
    if missing:
        raise ValueError(f"Missing stopping-history columns: {sorted(missing)}")
    if not table.selected_round.eq(int(best_round)).all():
        raise ValueError("Recorded selected round disagrees with CV best_round")
    if set(table.dataset) != {"inner_fit", "inner_stop"}:
        raise ValueError("Expected inner_fit and inner_stop history")
    if set(table.curve_role) != {"inner_early_stopping"} or set(table.direction) != {"min"}:
        raise ValueError("Invalid stopping history provenance")
    if table.metric.nunique() != 1 or table.metric.iloc[0] not in {"rmse", "mae"}:
        raise ValueError("Invalid stopping history metric")
    if not np.isfinite(table.value.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite stopping history")
    for _, part in table.groupby("dataset"):
        if part.duplicated("round").any() or sorted(part["round"]) != list(range(1, len(part) + 1)):
            raise ValueError("History rounds must be consecutive and unique")
        if int(best_round) not in set(part["round"]):
            raise ValueError("Selected round missing from history")
    return table[INNER_COLUMNS]


def save_inner_history(run_dir, cv):
    table = getattr(cv, "early_stopping_curves", None)
    if table is None or table.empty:
        return
    run_dir = Path(run_dir)
    table.to_parquet(run_dir / "early_stopping_curves.parquet", index=False)
    metadata = {
        "schema_version": 1,
        "curve_role": "inner_early_stopping",
        "selection_source": "original inner stopping fits; not reconstructed",
        "datasets": {"inner_fit": "inner training slice", "inner_stop": "inner validation slice"},
        "direction": "min",
        "round_indexing": "one_based",
        "used_holdout": False,
    }
    (run_dir / "early_stopping_curves_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def _plot_inner_axes(table):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if table.empty:
        raise ValueError("No recorded inner stopping curves")
    folds = sorted(int(k) for k in table.fold.unique())
    fig, axes = plt.subplots(
        1, len(folds), figsize=(3.6 * len(folds), 3.5), sharey=True, squeeze=False
    )
    for ax, fold in zip(axes[0], folds, strict=True):
        part = table.loc[table.fold == fold]
        for dataset, group in part.groupby("dataset"):
            group = group.sort_values("round")
            ax.plot(group["round"], group.value, label=dataset)
        ax.axvline(
            int(part.selected_round.iloc[0]),
            ls="--",
            color="grey",
            lw=1,
            label="selected on inner_stop",
        )
        ax.set(title=f"fold {fold}", xlabel="round")
    axes[0][0].set_ylabel(str(table.metric.iloc[0]))
    axes[0][-1].legend(fontsize=8)
    return fig, axes


def plot_early_stopping_curves(table, path, title=""):
    import matplotlib.pyplot as plt

    fig, _ = _plot_inner_axes(table)
    try:
        fig.suptitle("Recorded inner early stopping" + (f" — {title}" if title else ""))
        fig.tight_layout()
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)


def run_regression_curve_diagnostics(cfg, run_dir, inputs, cv, make_adapter, logger):
    """Optional analyses never alter CV/refit outputs; errors remain visible."""
    if not cfg.analysis.learning_curves:
        return
    from ml_template.evaluation.learning_curves import learning_curves_for_run, plot_learning_curves

    run_dir = Path(run_dir)
    errors = {}
    actions = []
    inner = getattr(cv, "early_stopping_curves", None)
    if inner is not None and not inner.empty:
        actions.append(
            (
                "inner_stopping_plot",
                lambda: plot_early_stopping_curves(inner, run_dir / "early_stopping_curves.png"),
            )
        )

    def outer():
        table = learning_curves_for_run(
            make_adapter,
            inputs.X,
            inputs.y,
            inputs.assignments,
            inputs.contract.entity_id,
            cv.folds,
            metric=cfg.analysis.learning_curve_metric,
            max_rounds=cfg.analysis.learning_curve_rounds,
        )
        table.to_parquet(run_dir / "learning_curves.parquet", index=False)
        metadata = {
            "schema_version": 1,
            "curve_role": "outer_diagnostic_refit",
            "datasets": {"train": "outer_train", "val": "outer_validation"},
            "best_round": "inner-selected reference only; not selected on this curve",
            "early_stopping": False,
            "used_holdout": False,
            "metric": cfg.analysis.learning_curve_metric,
        }
        (run_dir / "learning_curves_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        plot_learning_curves(table, run_dir / "learning_curves.png")

    actions.append(("outer_diagnostic_curves", outer))
    for name, action in actions:
        try:
            action()
        except Exception as exc:
            logger.exception("Regression curve diagnostic failed: %s", name)
            errors[name] = f"{type(exc).__name__}: {exc}"
    if errors:
        path = run_dir / "analysis_errors.json"
        previous = json.loads(path.read_text()) if path.exists() else {}
        path.write_text(json.dumps({**previous, **errors}, indent=2), encoding="utf-8")


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(
        description="Plot recorded inner history without reconstructing old fits"
    )
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(argv)
    table = pd.read_parquet(args.run_dir / "early_stopping_curves.parquet")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "early_stopping_curves.png"
    if output.exists():
        raise FileExistsError(output)
    plot_early_stopping_curves(table, output, args.run_dir.name)
    return output


if __name__ == "__main__":
    main()
