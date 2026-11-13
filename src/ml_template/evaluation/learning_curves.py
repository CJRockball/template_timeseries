"""Per-round training and validation metric curves, saved as a table.

For every CV fold a separate diagnostic model is fitted for a fixed number of rounds on that
fold's training rows. The metric is recorded each round on those rows and on the fold's
validation rows. There is no early stopping, and nothing here feeds back into the model, the
round count or the CV score. The table is the source of truth; plots are drawn from it.

Table columns: fold, round (1-based), dataset (train or val), metric, value, best_round.

    python -m ml_template.evaluation.learning_curves --run-id <run_id>
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

TABLE_COLUMNS = ["fold", "round", "dataset", "metric", "value", "best_round"]
ROUND_FACTOR = 2.0  # default length of the diagnostic fit, relative to the fold's best round
MIN_EXTRA_ROUNDS = 10


def diagnostic_rounds(best_round: int, max_rounds: int | None, cap: int) -> int:
    """Rounds for one diagnostic fit: a fixed value if given, else twice the best round."""
    if max_rounds is not None:
        n = int(max_rounds)
    else:
        n = max(round(best_round * ROUND_FACTOR), best_round + MIN_EXTRA_ROUNDS)
    return max(1, min(n, cap))


def learning_curves_for_run(
    make_adapter: Callable[[], Any],
    X: pd.DataFrame,
    y: pd.Series,
    assignments: pd.DataFrame,
    key_col: str,
    folds: pd.DataFrame,
    *,
    metric: str,
    max_rounds: int | None = None,
) -> pd.DataFrame:
    """One diagnostic fit per fold on development rows only; the holdout is never used."""
    dev = assignments[assignments["partition"] == "dev"]
    parts = []
    for row in folds.itertuples():
        k, best = int(row.fold), int(row.best_round)
        val_keys = dev.loc[dev["fold"] == k, key_col].to_numpy()
        train_keys = dev.loc[dev["fold"] != k, key_col].to_numpy()
        adapter = make_adapter()
        n_rounds = diagnostic_rounds(best, max_rounds, adapter.max_rounds)
        curve = adapter.learning_curve(
            X.loc[train_keys], y.loc[train_keys], X.loc[val_keys], y.loc[val_keys], n_rounds, metric
        )
        curve = curve.assign(fold=k, best_round=best)
        parts.append(curve[TABLE_COLUMNS])
    return pd.concat(parts, ignore_index=True)


def _outer_curve_axes(table, show_inner_reference=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if table.empty:
        raise ValueError("No outer diagnostic curves")
    folds = sorted(int(k) for k in table["fold"].unique())
    metric = str(table["metric"].iloc[0])
    fig, axes = plt.subplots(
        1, len(folds), figsize=(3.6 * len(folds), 3.5), sharey=True, squeeze=False
    )
    for ax, k in zip(axes[0], folds, strict=True):
        part = table[table["fold"] == k]
        for dataset, group in part.groupby("dataset"):
            group = group.sort_values("round")
            label = {"train": "outer_train", "val": "outer_validation"}.get(dataset, dataset)
            ax.plot(group["round"], group["value"], label=label)
        if show_inner_reference:
            ax.axvline(
                int(part["best_round"].iloc[0]),
                ls="--",
                color="grey",
                lw=1,
                label="inner-selected round — reference only",
            )
        ax.set(title=f"fold {k}", xlabel="round")
    axes[0][0].set_ylabel(metric)
    axes[0][-1].legend(fontsize=8)
    return fig, axes


def plot_learning_curves(
    table: pd.DataFrame, path: Path, title: str = "", *, show_inner_reference: bool = False
) -> None:
    """Outer diagnostic refit: never use this plot to select the stopping round."""
    import matplotlib.pyplot as plt

    fig, _ = _outer_curve_axes(table, show_inner_reference)
    try:
        heading = "Outer-fold diagnostic fit — no early stopping"
        fig.suptitle(heading + (f" — {title}" if title else ""))
        fig.tight_layout()
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)


def main(argv: list[str] | None = None) -> Path:
    from ml_template.paths import ARTIFACTS_RUNS

    p = argparse.ArgumentParser(description="Plot the learning curves saved for a run.")
    p.add_argument("--run-id", required=True)
    args = p.parse_args(argv)
    run_dir = Path(ARTIFACTS_RUNS) / args.run_id
    source = run_dir / "learning_curves.parquet"
    if not source.exists():
        raise FileNotFoundError(f"{source} not found; run with analysis.learning_curves: true")
    out = run_dir / "learning_curves.png"
    plot_learning_curves(pd.read_parquet(source), out, title=args.run_id)
    print(out)
    return out


if __name__ == "__main__":
    main()
