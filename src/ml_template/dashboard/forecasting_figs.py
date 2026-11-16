"""Plotly charts for verified frozen forecasts, not model execution."""

import plotly.graph_objects as go

COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]


def empty(message):
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    fig.update_layout(template="plotly_white", height=350)
    return fig


def figures(result, all_methods, plot_series=None, max_points=5000):
    metric = result["metric"]
    colors = {method: COLORS[index % len(COLORS)] for index, method in enumerate(all_methods)}
    pooled, origins = go.Figure(), go.Figure()
    for row in result["table"]:
        pooled.add_trace(
            go.Bar(
                x=[row["method"]],
                y=[row[metric]],
                name=row["method"],
                marker_color=colors[row["method"]],
                text=["N=" + str(row["n"]) + "; coverage=" + str(row["coverage_fraction"])],
            )
        )
    pooled.update_layout(
        title="Pooled " + metric.upper() + " — " + result["support"],
        template="plotly_white",
        yaxis_title="Saved target scale",
    )
    for method in result["methods"]:
        values = sorted(
            (r for r in result["metrics"] if r["scope"] == "origin" and r["method"] == method),
            key=lambda r: r["key"],
        )
        origins.add_trace(
            go.Scatter(
                x=[r["key"] for r in values],
                y=[r[metric] for r in values],
                name=method,
                mode="lines+markers",
                connectgaps=False,
                line={"color": colors[method]},
            )
        )
    origins.update_layout(title=metric.upper() + " by evaluated origin", template="plotly_white")
    keys = sorted({r["key"] for r in result["metrics"] if r["scope"] == "series"})
    matrix = {
        (r["key"], r["method"]): r[metric] for r in result["metrics"] if r["scope"] == "series"
    }
    heat = go.Figure(
        go.Heatmap(
            x=result["methods"],
            y=keys[:100],
            z=[[matrix.get((key, method)) for method in result["methods"]] for key in keys[:100]],
            colorscale="Blues",
        )
    )
    heat.update_layout(
        title="Per-series raw "
        + metric.upper()
        + " (first "
        + str(min(100, len(keys)))
        + " of "
        + str(len(keys))
        + ")",
        template="plotly_white",
    )
    actual, errors = (
        empty("Enable linked predictions and choose a plot series"),
        empty("Enable linked predictions for residuals"),
    )
    notes = []
    if result["rows"] is not None:
        rows = result["rows"]
        supported = [
            r
            for r in rows
            if r["status"] == "ok"
            and r["method"] in result["methods"]
            and (result["support"] == "own_supported" or r["common_supported"])
        ]
        errors = go.Figure()
        for method in result["methods"]:
            values = sorted(
                (r for r in supported if r["method"] == method),
                key=lambda r: (r["origin_date"], r["request_id"]),
            )
            if len(values) > max_points:
                notes.append(
                    "Residual plot limited to first "
                    + str(max_points)
                    + " points for "
                    + method
                    + "; metrics use every filtered row"
                )
            errors.add_trace(
                go.Histogram(
                    x=[r["error"] for r in values[:max_points]],
                    name=method,
                    opacity=0.5,
                    marker_color=colors[method],
                )
            )
        errors.update_layout(
            title="Signed error: prediction minus actual",
            barmode="overlay",
            template="plotly_white",
        )
        truth = {
            r["request_id"]: r
            for r in rows
            if r["series_key"] == plot_series
            and (result["support"] == "own_supported" or r["common_supported"])
        }
        ordered = sorted(truth.values(), key=lambda r: r["target_date"])
        if ordered:
            if len(ordered) > max_points:
                notes.append(
                    "Actual/prediction chart limited to first evaluated dates; metric population is unchanged"
                )
            ordered = ordered[:max_points]
            ids = {r["request_id"] for r in ordered}
            actual = go.Figure(
                go.Scatter(
                    x=[r["target_date"] for r in ordered],
                    y=[r["y_true"] for r in ordered],
                    mode="markers",
                    name="Actual at evaluated target dates",
                    marker={"symbol": "diamond", "color": "#222", "size": 9},
                )
            )
            for method in result["methods"]:
                values = sorted(
                    (r for r in supported if r["method"] == method and r["request_id"] in ids),
                    key=lambda r: r["target_date"],
                )
                actual.add_trace(
                    go.Scatter(
                        x=[r["target_date"] for r in values],
                        y=[r["y_pred"] for r in values],
                        mode="markers",
                        name=method,
                        marker={"color": colors[method]},
                    )
                )
            actual.update_layout(
                title="Forecast-date observations — " + str(plot_series), template="plotly_white"
            )
    return {
        "pooled": pooled,
        "origins": origins,
        "series": heat,
        "actual": actual,
        "residuals": errors,
        "notes": notes,
    }
