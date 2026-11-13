"""Shared Plotly threshold views; data-only functions, no Dash dependency."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def _base(fig, title, ylabel):
    fig.update_layout(
        template="plotly_white",
        height=370,
        title=title,
        xaxis_title="Named development score cutoff (p >= cutoff)",
        yaxis_title=ylabel,
    )
    return fig


def objective_overlay(profiles: dict[str, pd.DataFrame], objective="f1", current_threshold=None):
    if objective not in {"f1", "f2"}:
        raise ValueError("Objective must be f1 or f2")
    fig = go.Figure()
    for run_id, profile in profiles.items():
        valid = profile[(profile["threshold"] <= 1) & profile[objective].notna()]
        if valid.empty:
            continue
        fig.add_scatter(
            x=valid["threshold"],
            y=valid[objective],
            mode="lines",
            name=run_id,
            line_shape="vh",
            legendgroup=run_id,
        )
        best = valid[objective].max()
        choice = valid[valid[objective] >= best - 1e-12].iloc[-1]
        fig.add_scatter(
            x=[choice["threshold"]],
            y=[choice[objective]],
            mode="markers",
            marker={"symbol": "diamond", "size": 11},
            name=f"{run_id} max {objective.upper()}",
            legendgroup=run_id,
            showlegend=False,
        )
    if current_threshold is not None:
        fig.add_vline(x=current_threshold, line_dash="dash", annotation_text="current cutoff")
    if not fig.data:
        fig.add_annotation(text="No defined objective: positive labels required", showarrow=False)
    return _base(
        fig,
        f"{objective.upper()} versus cutoff — diamond marks estimated optimum",
        objective.upper(),
    )


def operating_curves(profile: pd.DataFrame, current_threshold=None):
    fig = go.Figure()
    valid = profile[profile["threshold"] <= 1]
    for metric in ("precision", "recall", "alert_rate"):
        fig.add_scatter(
            x=valid["threshold"], y=valid[metric], mode="lines", name=metric, line_shape="vh"
        )
    if current_threshold is not None:
        fig.add_vline(x=current_threshold, line_dash="dash", annotation_text="current cutoff")
    return _base(fig, "Precision, recall and workload", "Fraction")


def fold_optima(table: pd.DataFrame):
    fig = go.Figure()
    for objective, part in table.groupby("objective", sort=False):
        part = part.dropna(subset=["threshold"])
        fig.add_scatter(
            x=part["fold"],
            y=part["threshold"],
            mode="markers",
            name=objective.upper(),
            customdata=part[["objective_value"]].to_numpy(),
            hovertemplate="fold %{x}<br>cutoff %{y:.6f}<br>selected score %{customdata[0]:.4f}<extra></extra>",
        )
    fig.update_layout(
        template="plotly_white",
        height=300,
        xaxis_title="Original CV fold",
        yaxis_title="Separately selected cutoff",
        title="Fold sensitivity — not independent validation or confidence intervals",
    )
    return fig


# calibrated-policy update v1
