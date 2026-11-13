"""Pure Plotly parameter figures for sweep diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ml_template.dashboard import sweep_data as sd


def placeholder(message):
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False)
    fig.update_layout(template="plotly_white", height=360)
    return fig


def parameter_plot(summary: pd.DataFrame, axes: list[str], higher: bool = True):
    if not axes or summary.empty:
        return placeholder("No successful scored seeds for these parameters and context")
    if len(axes) == 1:
        axis = axes[0]
        numeric = pd.to_numeric(summary[axis], errors="coerce")
        order = (
            numeric.sort_values().index
            if numeric.notna().all()
            else summary[axis].astype(str).sort_values().index
        )
        part = summary.loc[order]
        fig = go.Figure(
            go.Scatter(
                x=pd.to_numeric(part[axis]).to_numpy() if numeric.notna().all() else part[axis],
                y=part["mean"],
                mode="markers",
                error_y={"type": "data", "array": part["std"].fillna(0)},
                customdata=part[["n_seeds", "std"]].to_numpy(),
                hovertemplate="%{x}<br>mean %{y:.5f}<br>seeds %{customdata[0]}"
                "<br>sample std %{customdata[1]:.5f}<extra></extra>",
            )
        )
        fig.update_layout(
            xaxis_title=axis.removeprefix(sd.PARAM_PREFIX), yaxis_title="Mean CV primary score"
        )
    else:
        x, y = axes
        means = summary.pivot(index=y, columns=x, values="mean")
        counts = summary.pivot(index=y, columns=x, values="n_seeds").reindex_like(means)
        stds = summary.pivot(index=y, columns=x, values="std").reindex_like(means)
        custom = np.stack([counts.to_numpy(), stds.to_numpy()], axis=-1)
        fig = go.Figure(
            go.Heatmap(
                x=means.columns.astype(str),
                y=means.index.astype(str),
                z=means.to_numpy(),
                customdata=custom,
                colorscale="Viridis" if higher else "Viridis_r",
                hoverongaps=False,
                texttemplate="%{z:.4f}",
                hovertemplate="x %{x}<br>y %{y}<br>mean %{z:.5f}<br>seeds %{customdata[0]}"
                "<br>sample std %{customdata[1]:.5f}<extra></extra>",
                colorbar={"title": "Mean CV score"},
            )
        )
        fig.update_layout(
            xaxis_title=x.removeprefix(sd.PARAM_PREFIX), yaxis_title=y.removeprefix(sd.PARAM_PREFIX)
        )
    fig.update_layout(template="plotly_white", height=420)
    return fig
