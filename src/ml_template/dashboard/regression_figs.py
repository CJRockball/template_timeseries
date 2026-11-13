"""Plot verified saved regression tables; never fit or evaluate."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def empty(message):
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    return fig


def metrics_table(metrics, unit):
    records = []
    for name in ["rmse", "mae", "r2"]:
        part = metrics[metrics.metric == name] if "metric" in metrics else pd.DataFrame()
        if part.empty:
            continue
        pooled = part[part.fold == -1]
        values = pd.to_numeric(part.loc[part.fold >= 0, "value"], errors="coerce").dropna()
        records.append(
            {
                "metric": name,
                "direction": "max" if name == "r2" else "min",
                "unit": "dimensionless" if name == "r2" else unit,
                "pooled_OOF": pooled.value.iloc[0] if len(pooled) else None,
                "fold_mean": values.mean() if len(values) else None,
                "fold_std_descriptive": values.std(ddof=1) if len(values) > 1 else None,
            }
        )
    return pd.DataFrame(records)


def fold_metrics(metrics, unit):
    if metrics.empty:
        return empty("No saved CV metrics")
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Errors: minimise", "R²: maximise"])
    for name in ["rmse", "mae", "r2"]:
        part = metrics[(metrics.metric == name) & (metrics.fold >= 0)].sort_values("fold")
        if not part.empty:
            fig.add_trace(
                go.Scatter(x=part.fold, y=part.value, mode="lines+markers", name=name),
                row=1,
                col=2 if name == "r2" else 1,
            )
    fig.update_xaxes(title_text="validation fold")
    fig.update_yaxes(title_text=unit, row=1, col=1)
    fig.update_yaxes(title_text="dimensionless", row=1, col=2)
    return fig


def curves(table, role, unit):
    if table is None:
        return empty(
            "Inner stopping history not recorded"
            if role == "inner"
            else "Outer diagnostic curves not recorded"
        )
    folds = sorted(int(k) for k in table.fold.unique())
    count = min(3, len(folds))
    row_count = (len(folds) + count - 1) // count
    fig = make_subplots(rows=row_count, cols=count, subplot_titles=[f"fold {k}" for k in folds])
    for i, fold in enumerate(folds):
        r, c = (i // count + 1, i % count + 1)
        part = table[table.fold == fold]
        for name, group in part.groupby("dataset"):
            group = group.sort_values("round")
            label = {"train": "outer_train", "val": "outer_validation"}.get(name, name)
            fig.add_trace(
                go.Scatter(
                    x=group["round"],
                    y=group.value,
                    mode="lines",
                    name=label,
                    legendgroup=label,
                    showlegend=i == 0,
                ),
                row=r,
                col=c,
            )
        if role == "inner":
            fig.add_vline(
                x=int(part.selected_round.iloc[0]),
                line_dash="dash",
                line_color="grey",
                row=r,
                col=c,
            )
    title = (
        "Recorded inner stopping — selected on inner_stop"
        if role == "inner"
        else "Outer diagnostic refits — no early stopping / selection marker"
    )
    fig.update_layout(title=title, height=300 * row_count)
    fig.update_xaxes(title_text="round")
    fig.update_yaxes(title_text=f"{table.metric.iloc[0]} ({unit})")
    return fig


def residual_figures(table, unit, max_points=5000):
    if table is None:
        return [empty("Select a saved diagnostic analysis") for _ in range(3)]
    part = (
        table
        if len(table) <= max_points
        else table.iloc[
            np.sort(np.random.default_rng(0).choice(len(table), max_points, replace=False))
        ]
    )
    suffix = f" ({len(part)} displayed of {len(table)} OOF rows)"
    fig = go.Figure(
        go.Scattergl(
            x=part.y_true, y=part.y_pred_oof, mode="markers", marker={"size": 5, "opacity": 0.6}
        )
    )
    lo = float(min(part.y_true.min(), part.y_pred_oof.min()))
    hi = float(max(part.y_true.max(), part.y_pred_oof.max()))
    fig.add_trace(
        go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line={"dash": "dash"}, name="identity")
    )
    fig.update_layout(
        title="Observed vs OOF predicted" + suffix,
        xaxis_title=f"observed ({unit})",
        yaxis_title=f"predicted ({unit})",
    )
    residual = go.Figure(
        go.Scattergl(
            x=part.y_pred_oof, y=part.residual, mode="markers", marker={"size": 5, "opacity": 0.6}
        )
    )
    residual.add_hline(y=0, line_dash="dash")
    residual.update_layout(
        title="OOF residuals" + suffix,
        xaxis_title=f"predicted ({unit})",
        yaxis_title=f"y_true − prediction ({unit})",
    )
    histogram = go.Figure(go.Histogram(x=part.residual, nbinsx=30))
    histogram.update_layout(
        title="OOF residual distribution" + suffix,
        xaxis_title=f"residual ({unit})",
        yaxis_title="displayed rows",
    )
    return [fig, residual, histogram]


def importance(table, fold=-1):
    if table is None or table.empty:
        return empty("Permutation importance not recorded/enabled")
    part = table[table.fold == int(fold)].copy()
    if part.empty:
        return empty("No importance for this fold")
    part = part.iloc[np.argsort(np.abs(part["mean"].to_numpy()))[-50:]].sort_values("mean")
    std = "between_fold_std" if int(fold) == -1 else "repeat_std"
    title = (
        "Across-fold SD: descriptive, not CI"
        if int(fold) == -1
        else "Across-repeat SD: descriptive, not CI"
    )
    fig = go.Figure(
        go.Bar(
            x=part["mean"],
            y=part.feature,
            orientation="h",
            error_x={
                "type": "data",
                "array": [None if pd.isna(v) else float(v) for v in part[std]],
            },
        )
    )
    fig.add_vline(x=0, line_color="grey")
    fig.update_layout(
        title=title,
        xaxis_title=f"{part.metric.iloc[0]} increase ({part.target_unit.iloc[0]})",
        height=max(300, len(part) * 23),
    )
    return fig
