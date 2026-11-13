"""Pure Plotly figures for multi-run development-data comparisons."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
from sklearn.metrics import average_precision_score, precision_recall_curve


def empty(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False)
    fig.update_layout(template="plotly_white", height=330)
    return fig


def pr_overlay(tables: dict[str, pd.DataFrame]) -> go.Figure:
    if not tables:
        return empty("Aligned OOF predictions unavailable; see warnings")
    fig = go.Figure()
    first = next(iter(tables.values()))
    if first["y_true"].nunique() != 2:
        return empty("PR comparison needs both target classes")
    for rid, frame in tables.items():
        precision, recall, _ = precision_recall_curve(frame["y_true"], frame["p_oof"])
        ap = average_precision_score(frame["y_true"], frame["p_oof"])
        fig.add_scatter(x=recall, y=precision, mode="lines", name=f"{rid} | AP {ap:.4f}")
    fig.add_hline(y=float(first["y_true"].mean()), line_dash="dash", annotation_text="base rate")
    fig.update_layout(
        template="plotly_white",
        height=400,
        xaxis_title="Recall",
        yaxis_title="Precision",
        yaxis_range=[0, 1.02],
    )
    return fig


def reliability_overlay(tables: dict[str, pd.DataFrame], bins: int = 10) -> go.Figure:
    if not tables:
        return empty("Aligned OOF predictions unavailable")
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="ideal", line={"dash": "dash"})
    for rid, frame in tables.items():
        q = pd.qcut(frame["p_oof"], q=min(bins, len(frame)), duplicates="drop")
        # Constant predictions form one bin; tied probabilities are never split arbitrarily.
        group = pd.Series(0, index=frame.index) if q.isna().all() else q
        summary = (
            frame.groupby(group, observed=True)
            .agg(predicted=("p_oof", "mean"), observed=("y_true", "mean"), n=("y_true", "size"))
            .sort_values("predicted")
        )
        fig.add_scatter(
            x=summary["predicted"],
            y=summary["observed"],
            mode="lines+markers",
            name=rid,
            customdata=summary[["n"]].to_numpy(),
            hovertemplate="predicted %{x:.4f}<br>observed %{y:.4f}"
            "<br>rows %{customdata[0]}<extra>%{fullData.name}</extra>",
        )
    fig.update_layout(
        template="plotly_white",
        height=400,
        xaxis_title="Mean stored probability",
        yaxis_title="Observed positive rate",
    )
    return fig


def metric_deltas(metrics: pd.DataFrame, baseline: str, metric: str, higher: bool) -> go.Figure:
    part = metrics[
        (metrics["split"] == "cv") & (metrics["fold"] == -1) & (metrics["metric"] == metric)
    ].set_index("run_id")["value"]
    if baseline not in part or pd.isna(part[baseline]):
        return empty("Selected metric is not available for the baseline")
    deltas = part.drop(index=baseline) - part[baseline]
    colours = [
        "#16803c" if (value >= 0 if higher else value <= 0) else "#bc3434" for value in deltas
    ]
    fig = go.Figure(go.Bar(x=deltas.index, y=deltas, marker_color=colours))
    fig.add_hline(y=0)
    fig.update_layout(
        template="plotly_white",
        height=330,
        yaxis_title=f"{metric}: run minus baseline (raw difference)",
    )
    return fig


def fold_overlay(metrics: pd.DataFrame, metric: str) -> go.Figure:
    part = metrics[
        (metrics["split"] == "cv") & (metrics["fold"] >= 0) & (metrics["metric"] == metric)
    ]
    if part.empty:
        return empty("No per-fold values recorded for this metric")
    fig = go.Figure()
    for rid, group in part.groupby("run_id", sort=False):
        group = group.sort_values("fold")
        fig.add_scatter(x=group["fold"], y=group["value"], mode="lines+markers", name=rid)
    fig.update_layout(
        template="plotly_white", height=330, xaxis_title="Stored CV fold", yaxis_title=metric
    )
    return fig


def importance_heatmap(
    table: pd.DataFrame, run_ids: list[str], method: str | None, top_n: int = 20
) -> go.Figure:
    part = table[table["method"] == method]
    if part.empty:
        return empty("No CV importance for the selected method")
    values = part.pivot(index="feature", columns="run_id", values="value").reindex(columns=run_ids)
    top = values.abs().max(axis=1).nlargest(top_n).index
    values = values.loc[top]
    fig = go.Figure(
        go.Heatmap(
            x=values.columns,
            y=values.index,
            z=values.to_numpy(),
            colorscale="RdBu_r",
            zmid=0,
            hoverongaps=False,
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=max(350, 24 * len(values) + 120),
        yaxis={"autorange": "reversed"},
        title="Mean CV importance; blank = unrecorded",
    )
    return fig


def learning_overlay(
    curves: dict[str, pd.DataFrame | None], metric: str | None, fold: int | None
) -> go.Figure:
    fig = go.Figure()
    for i, (rid, frame) in enumerate(curves.items()):
        if frame is None or frame.empty or metric is None or fold is None:
            continue
        part = frame[(frame["metric"] == metric) & (frame["fold"] == fold)]
        for dataset, group in part.groupby("dataset"):
            group = group.sort_values("round")
            fig.add_scatter(
                x=group["round"],
                y=group["value"],
                mode="lines",
                name=f"{rid} | {dataset}",
                legendgroup=rid,
                line={
                    "color": qualitative.Plotly[i % len(qualitative.Plotly)],
                    "dash": "dash" if dataset == "train" else "solid",
                },
            )
    if not fig.data:
        return empty("No diagnostic learning curves for this fold and metric")
    fig.update_layout(
        template="plotly_white",
        height=370,
        xaxis_title="Diagnostic boosting round",
        yaxis_title=metric,
        title=f"Fold {fold}: dashed train, solid validation",
    )
    return fig


def cost_plot(headers: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for row in headers.itertuples():
        fig.add_scatter(
            x=[row.duration_s],
            y=[row.primary_value],
            mode="markers",
            name=row.run_id,
            marker={"size": 12},
        )
    metric = headers["primary_metric"].iloc[0] if len(headers) else "CV score"
    fig.update_layout(
        template="plotly_white",
        height=330,
        xaxis_title="Whole-run duration (s)",
        yaxis_title=metric,
    )
    return fig
