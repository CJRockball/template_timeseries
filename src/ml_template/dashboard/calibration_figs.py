"""Pure Plotly calibration views; no database, filesystem or model fitting."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = {"raw": "#d95f02", "calibrated_crossfit": "#1b9e77"}
LABELS = {"raw": "Raw OOF", "calibrated_crossfit": "Calibrated cross-fit diagnostic"}


def empty_fig(message):
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(template="plotly_white", height=260)
    return fig


def reliability_fig(table):
    if table is None or table.empty:
        return empty_fig("Verified reliability table unavailable")
    fig = go.Figure()
    fig.add_scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Ideal", line={"color": "grey", "dash": "dash"}
    )
    for variant in COLORS:
        part = table.loc[table.variant.eq(variant)].sort_values("mean_probability")
        if part.empty:
            continue
        fig.add_scatter(
            x=part.mean_probability,
            y=part.event_rate,
            mode="lines+markers",
            name=LABELS[variant],
            marker={"color": COLORS[variant], "size": 8},
            line={"color": COLORS[variant]},
            error_y={
                "type": "data",
                "symmetric": False,
                "array": (part.event_rate_high - part.event_rate).clip(lower=0),
                "arrayminus": (part.event_rate - part.event_rate_low).clip(lower=0),
            },
            customdata=part[["n_rows", "n_positives"]].to_numpy(),
            hovertemplate="Mean p=%{x:.4f}<br>Event rate=%{y:.4f}<br>Rows=%{customdata[0]}<br>Positives=%{customdata[1]}<extra>%{fullData.name}</extra>",
        )
    fig.update_layout(
        template="plotly_white",
        title="Development reliability with pointwise intervals",
        xaxis_title="Mean predicted probability",
        yaxis_title="Observed event rate",
        xaxis_range=[0, 1],
        yaxis_range=[0, 1],
        height=440,
    )
    return fig


def fold_metric_fig(metrics, metric="brier"):
    if metrics is None or metrics.empty:
        return empty_fig("No verified per-fold metrics")
    part = metrics.loc[metrics.metric.eq(metric) & metrics.fold.ge(0)]
    if part.empty:
        return empty_fig(f"No per-fold {metric} values")
    fig = go.Figure()
    for variant in COLORS:
        group = part.loc[part.variant.eq(variant)].sort_values("fold")
        if not group.empty:
            fig.add_scatter(
                x=group.fold,
                y=group.value,
                mode="lines+markers",
                name=LABELS[variant],
                line={"color": COLORS[variant]},
            )
    fig.update_layout(
        template="plotly_white",
        title=f"Per-fold {metric} (descriptive, not confidence intervals)",
        xaxis_title="Stored development fold",
        yaxis_title=metric,
        height=340,
    )
    fig.update_xaxes(dtick=1)
    return fig


def loss_fig(comparison):
    if comparison is None or comparison.empty:
        return empty_fig("No verified pooled probability losses")
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Brier score", "Log loss"])
    for col, metric in [(1, "brier"), (2, "log_loss")]:
        row = comparison.loc[comparison.metric.eq(metric)]
        if row.empty:
            continue
        for variant in COLORS:
            fig.add_bar(
                x=[LABELS[variant]],
                y=[row[variant].iloc[0]],
                name=LABELS[variant],
                marker_color=COLORS[variant],
                showlegend=col == 1,
                legendgroup=variant,
                row=1,
                col=col,
            )
    fig.update_layout(
        template="plotly_white", height=360, title="Pooled development losses — lower is better"
    )
    return fig


def distribution_fig(table):
    if table is None or table.empty:
        return empty_fig("Verified development predictions unavailable")
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Negative rows", "Positive rows"])
    for col, label in [(1, 0), (2, 1)]:
        for variant in COLORS:
            part = table.loc[table["class"].eq(label) & table.variant.eq(variant)]
            fig.add_scatter(
                x=part.bin_mid,
                y=part["count"],
                mode="lines",
                name=LABELS[variant],
                line={"color": COLORS[variant], "shape": "hv"},
                legendgroup=variant,
                showlegend=col == 1,
                row=1,
                col=col,
            )
    fig.update_xaxes(title_text="Predicted probability", range=[0, 1])
    fig.update_yaxes(title_text="Rows per fixed probability bin")
    fig.update_layout(
        template="plotly_white", height=340, title="Score distributions by observed class"
    )
    return fig


def mapping_fig(calibrator):
    if not calibrator:
        return empty_fig("Verified final calibrator unavailable")
    try:
        method = calibrator["method"]
        x = np.linspace(0, 1, 501)
        if method == "none":
            y = x
        elif method == "sigmoid":
            eps = float(calibrator["epsilon"])
            slope, intercept = float(calibrator["slope"]), float(calibrator["intercept"])
            if not np.isfinite([eps, slope, intercept]).all() or not 0 < eps < 0.01 or slope <= 0:
                raise ValueError("invalid mapping parameters")
            p = np.clip(x, eps, 1 - eps)
            z = slope * (np.log(p) - np.log1p(-p)) + intercept
            y = 1 / (1 + np.exp(-np.clip(z, -700, 700)))
        else:
            raise ValueError("unsupported mapping")
    except (KeyError, TypeError, ValueError):
        return empty_fig("Unsupported or invalid final calibrator parameters")
    fig = go.Figure()
    fig.add_scatter(x=x, y=x, name="Identity", mode="lines", line={"dash": "dash", "color": "grey"})
    fig.add_scatter(
        x=x,
        y=y,
        name="Final saved mapping",
        mode="lines",
        line={"color": COLORS["calibrated_crossfit"]},
    )
    fig.update_layout(
        template="plotly_white",
        height=350,
        title="Final fitted mapping — not assessment evidence",
        xaxis_title="Raw base-model probability",
        yaxis_title="Mapped probability",
        xaxis_range=[0, 1],
        yaxis_range=[0, 1],
    )
    return fig
