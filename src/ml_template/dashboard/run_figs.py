"""Plotly figures for the single-run page. Pure functions: DataFrame in, Figure out."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import average_precision_score, precision_recall_curve

COLORS = {"train": "#1f77b4", "val": "#d62728"}
TEMPLATE = "plotly_white"


def empty_fig(message: str) -> go.Figure:
    """Placeholder for sections whose artifact was not written for this run."""
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font={"size": 14})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(template=TEMPLATE, height=240)
    return fig


def learning_curve_fig(table: pd.DataFrame | None, overlay: bool = False) -> go.Figure:
    """Train and val metric per round. One panel per fold, or all folds overlaid."""
    if table is None or table.empty:
        return empty_fig("No learning curves for this run (set analysis.learning_curves: true)")
    folds = sorted(int(k) for k in table["fold"].unique())
    metric = str(table["metric"].iloc[0])
    n_cols = 1 if overlay else len(folds)
    fig = make_subplots(
        rows=1,
        cols=n_cols,
        shared_yaxes=True,
        subplot_titles=None if overlay else [f"fold {k}" for k in folds],
    )
    for i, k in enumerate(folds):
        col = 1 if overlay else i + 1
        part = table[table["fold"] == k]
        for dataset, group in part.groupby("dataset"):
            fig.add_scatter(
                x=group["round"],
                y=group["value"],
                mode="lines",
                name=str(dataset),
                legendgroup=str(dataset),
                showlegend=(i == 0),
                line={"color": COLORS.get(str(dataset)), "width": 1.5 if overlay else 2},
                opacity=0.6 if overlay else 1.0,
                hovertemplate=f"fold {k} round %{{x}}: %{{y:.4f}}<extra>{dataset}</extra>",
                row=1,
                col=col,
            )
        fig.add_vline(
            x=int(part["best_round"].iloc[0]), line_dash="dash", line_color="grey", row=1, col=col
        )
    fig.update_xaxes(title_text="round")
    fig.update_yaxes(title_text=metric, col=1)
    fig.update_layout(template=TEMPLATE, height=340, margin={"t": 40})
    return fig


def gap_fig(table: pd.DataFrame | None) -> go.Figure:
    """Val minus train metric per round, one line per fold. Meaning depends on the metric."""
    if table is None or table.empty:
        return empty_fig("No learning curves for this run")
    wide = table.pivot_table(
        index=["fold", "round"], columns="dataset", values="value"
    ).reset_index()
    wide["gap"] = wide["val"] - wide["train"]
    fig = go.Figure()
    for k, group in wide.groupby("fold"):
        fig.add_scatter(x=group["round"], y=group["gap"], mode="lines", name=f"fold {int(k)}")
    fig.add_hline(y=0, line_color="grey", line_width=1)
    metric = str(table["metric"].iloc[0])
    fig.update_layout(
        template=TEMPLATE, height=300, xaxis_title="round", yaxis_title=f"val - train ({metric})"
    )
    return fig


def fold_metric_fig(metrics: pd.DataFrame | None, metric: str, split: str = "cv") -> go.Figure:
    """Per-fold bars for one metric, with the pooled value (fold -1) as a reference line."""
    if metrics is None or metrics.empty:
        return empty_fig("No metrics for this run")
    part = metrics[(metrics["split"] == split) & (metrics["metric"] == metric)]
    folds = part[part["fold"] >= 0].sort_values("fold")
    if folds.empty:
        return empty_fig(f"No per-fold values for {metric}")
    fig = go.Figure(go.Bar(x=[f"fold {int(k)}" for k in folds["fold"]], y=folds["value"]))
    mean = folds["value"].mean()
    fig.add_hline(y=mean, line_dash="dash", annotation_text=f"mean {mean:.4f}")
    pooled = part[part["fold"] == -1]["value"]
    if not pooled.empty and pd.notna(pooled.iloc[0]):
        fig.add_hline(
            y=float(pooled.iloc[0]),
            line_color="grey",
            annotation_text=f"pooled {float(pooled.iloc[0]):.4f}",
            annotation_position="bottom right",
        )
    fig.update_layout(template=TEMPLATE, height=300, yaxis_title=metric, showlegend=False)
    return fig


def importance_fig(
    importance: pd.DataFrame | None, method: str | None = None, top_n: int = 15
) -> go.Figure:
    """Horizontal bars with error bars. Columns: method, feature, value, std."""
    if importance is None or importance.empty:
        return empty_fig("No feature importance for this run")
    df = importance
    if method is not None:
        df = df[df["method"] == method]
    df = df.sort_values("value", ascending=False).head(top_n).iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=df["value"],
            y=df["feature"],
            orientation="h",
            error_x={"type": "data", "array": df["std"].fillna(0.0)},
        )
    )
    fig.update_layout(
        template=TEMPLATE,
        height=max(260, 24 * len(df) + 90),
        xaxis_title="importance (mean over folds)",
    )
    return fig


def pr_curve_fig(y_true: np.ndarray, p: np.ndarray) -> go.Figure:
    """Precision-recall curve with the base-rate line and average precision."""
    y = np.asarray(y_true)
    if y.size == 0 or y.sum() == 0:
        return empty_fig("No positive rows to draw a PR curve")
    precision, recall, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)
    fig = go.Figure()
    fig.add_scatter(x=recall, y=precision, mode="lines", name=f"AP {ap:.3f}")
    base = float(y.mean())
    fig.add_hline(
        y=base, line_dash="dash", line_color="grey", annotation_text=f"base rate {base:.3f}"
    )
    fig.update_layout(
        template=TEMPLATE,
        height=340,
        xaxis_title="recall",
        yaxis_title="precision",
        yaxis_range=[0, 1.02],
    )
    return fig


def reliability_fig(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> go.Figure:
    """Mean predicted against observed rate in equal-count bins; bubble size is bin count."""
    y, p = np.asarray(y_true, dtype=float), np.asarray(p, dtype=float)
    if y.size < n_bins:
        return empty_fig("Too few rows for a reliability curve")
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)
    mean_p = np.array([p[b].mean() for b in bins])
    obs = np.array([y[b].mean() for b in bins])
    size = np.array([len(b) for b in bins])
    top = float(max(mean_p.max(), obs.max(), 1e-6)) * 1.05
    fig = go.Figure()
    fig.add_scatter(
        x=[0, top], y=[0, top], mode="lines", line={"dash": "dash", "color": "grey"}, name="ideal"
    )
    fig.add_scatter(
        x=mean_p,
        y=obs,
        mode="lines+markers",
        name="model",
        marker={"size": 6 + 14 * size / size.max()},
    )
    fig.update_layout(
        template=TEMPLATE,
        height=340,
        xaxis_title="mean predicted probability",
        yaxis_title="observed positive rate",
    )
    return fig


def score_hist_fig(y_true: np.ndarray, p: np.ndarray, threshold: float | None = None) -> go.Figure:
    """Score distribution by class on a log count axis, with an optional cutoff line."""
    y, p = np.asarray(y_true), np.asarray(p)
    fig = go.Figure()
    for label, name in ((0, "negative"), (1, "positive")):
        fig.add_histogram(x=p[y == label], name=name, opacity=0.65, nbinsx=40)
    if threshold is not None:
        fig.add_vline(x=threshold, line_dash="dash", line_color="black")
    fig.update_layout(
        template=TEMPLATE,
        height=300,
        barmode="overlay",
        xaxis_title="predicted probability",
        yaxis_type="log",
        yaxis_title="count",
    )
    return fig


def cost_fig(runs: pd.DataFrame | None, run_id: str) -> go.Figure:
    """Duration against primary score for all runs; the selected run is highlighted."""
    if runs is None or runs.empty:
        return empty_fig("No runs to compare cost")
    df = runs.dropna(subset=["duration_s", "primary_value"])
    others = df[df["run_id"] != run_id]
    chosen = df[df["run_id"] == run_id]
    fig = go.Figure()
    fig.add_scatter(
        x=others["duration_s"],
        y=others["primary_value"],
        mode="markers",
        name="other runs",
        text=others["run_id"],
        marker={"color": "lightgrey"},
    )
    fig.add_scatter(
        x=chosen["duration_s"],
        y=chosen["primary_value"],
        mode="markers",
        name="selected",
        text=chosen["run_id"],
        marker={"size": 12, "color": COLORS["val"]},
    )
    metric = str(df["primary_metric"].iloc[0]) if "primary_metric" in df and len(df) else "score"
    fig.update_layout(template=TEMPLATE, height=320, xaxis_title="duration (s)", yaxis_title=metric)
    return fig


def threshold_summary(y_true: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    """Confusion counts and rates when everything with p >= threshold is flagged."""
    y = np.asarray(y_true).astype(int)
    flag = np.asarray(p) >= threshold
    tp = int((flag & (y == 1)).sum())
    fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum())
    tn = int((~flag & (y == 0)).sum())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "alert_rate": float(flag.mean()) if flag.size else float("nan"),
    }
