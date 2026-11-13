"""Pure shared Plotly figures; all contributions are base-model log-odds."""

from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ml_template.explainability.presentation_data import selected_entity


def output_unit(view):
    return (
        escape(view.summary["target_unit"])
        if getattr(view, "summary", {}).get("task") == "regression"
        else "log-odds"
    )


def placeholder(message):
    figure = go.Figure()
    figure.add_annotation(
        text=escape(str(message)), showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper"
    )
    figure.update_layout(template="plotly_white", height=350)
    return figure


def sample_positions(view, max_points=2000):
    if isinstance(max_points, bool) or not isinstance(max_points, int) or max_points < 1:
        raise ValueError("positive max_points required")
    order = np.argsort(view.rows.index.to_numpy(), kind="stable")
    selected = np.linspace(0, len(order) - 1, min(max_points, len(order)), dtype=int)
    return order[selected]


def importance_figure(view, top_n=20):
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise ValueError("positive top_n required")
    table = view.importance.head(top_n).iloc[::-1]
    figure = go.Figure(
        go.Bar(
            x=table.mean_abs_contribution,
            y=[escape(str(v)) for v in table.feature],
            orientation="h",
        )
    )
    figure.update_layout(
        template="plotly_white",
        title=f"Global importance — all {len(view.rows)} scoring rows",
        xaxis_title=f"Mean absolute contribution ({output_unit(view)})",
        height=max(400, 30 * len(table)),
    )
    return figure


def distribution_figure(view, *, top_n=10, max_points=2000):
    if getattr(view, "summary", {}).get("task") == "regression":
        from ml_template.explainability.regression_figures import (
            distribution_figure as regression_distribution,
        )

        return regression_distribution(view, top_n=top_n, max_points=max_points)
    positions = sample_positions(view, max_points)
    figure = go.Figure()
    for name in view.importance.head(top_n).feature:
        figure.add_trace(
            go.Box(
                x=view.contributions[name].iloc[positions],
                name=escape(str(name)),
                boxpoints=False,
                orientation="h",
                showlegend=False,
            )
        )
    figure.add_vline(x=0, line_dash="dot", line_color="gray")
    figure.update_layout(
        template="plotly_white",
        title=f"Contribution distributions — {len(positions)} of {len(view.rows)} rows",
        xaxis_title="Signed contribution (log-odds)",
        height=max(400, 40 * min(top_n, len(view.importance))),
    )
    return figure


def dependence_figure(view, feature, *, max_points=2000):
    if feature in getattr(view, "summary", {}).get("varying_feature_values", []):
        return placeholder(
            "Dependence unavailable: this estimator input differs across fold transforms"
        )
    if feature not in view.features:
        raise ValueError("feature not found in saved estimator inputs")
    positions = sample_positions(view, max_points)
    values = view.features[feature].iloc[positions]
    numeric = pd.api.types.is_numeric_dtype(values.dtype) and not pd.api.types.is_bool_dtype(
        values.dtype
    )
    x = (
        values.to_numpy(dtype=float, na_value=np.nan)
        if numeric
        else [escape(str(v)) for v in values]
    )
    keys = [escape(str(k)) for k in view.rows.index[positions]]
    figure = go.Figure(
        go.Scattergl(
            x=x,
            y=view.contributions[feature].iloc[positions],
            mode="markers",
            text=keys,
            marker={
                "color": view.rows[
                    "y_pred_release"
                    if getattr(view, "summary", {}).get("task") == "regression"
                    else "p_release"
                ].iloc[positions],
                "colorscale": "Viridis",
                "showscale": True,
                "colorbar": {
                    "title": "Release prediction"
                    if getattr(view, "summary", {}).get("task") == "regression"
                    else "Release<br>probability"
                },
                "size": 6,
                "opacity": 0.7,
            },
            hovertemplate="Entity: %{text}<br>Value: %{x}<br>Contribution: %{y:.4f}<extra></extra>",
        )
    )
    figure.add_hline(y=0, line_dash="dot", line_color="gray")
    figure.update_layout(
        template="plotly_white",
        title=f"{escape(str(feature))} — {len(positions)} sampled rows",
        xaxis_title=escape(str(feature)),
        yaxis_title=f"Contribution ({output_unit(view)})",
        height=450,
    )
    return figure


def waterfall_figure(view, position, *, max_features=12):
    if isinstance(max_features, bool) or not isinstance(max_features, int) or max_features < 1:
        raise ValueError("positive max_features required")
    key, row, table = selected_entity(view, position)
    selected = table.head(max_features)
    names = [escape(str(name)) for name in selected.feature]
    values = selected.contribution.astype(float).tolist()
    if len(table) > max_features:
        names.append("Other features (signed sum)")
        values.append(float(table.contribution.iloc[max_features:].sum()))
    figure = go.Figure(
        go.Waterfall(
            orientation="h",
            measure=["absolute", *(["relative"] * len(values)), "total"],
            y=[
                "Baseline",
                *names,
                "Prediction"
                if getattr(view, "summary", {}).get("task") == "regression"
                else "Prediction margin",
            ],
            x=[float(row.base_value), *values, 0],
            connector={"line": {"color": "gray"}},
            increasing={"marker": {"color": "#c44240"}},
            decreasing={"marker": {"color": "#3280b9"}},
        )
    )
    figure.update_layout(
        template="plotly_white",
        title=f"Entity {escape(str(key))}: "
        + (
            "released predictor explanation"
            if getattr(view, "summary", {}).get("task") == "regression"
            else "base-model explanation"
        ),
        xaxis_title=f"Prediction ({output_unit(view)})"
        if getattr(view, "summary", {}).get("task") == "regression"
        else "Base-model log-odds — not probability percentage points",
        height=max(450, 33 * (len(values) + 2)),
        yaxis={"autorange": "reversed"},
    )
    return figure
