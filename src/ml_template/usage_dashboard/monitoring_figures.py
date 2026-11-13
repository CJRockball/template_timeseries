"""Pure Plotly figures for saved results; no inference or policy changes."""

import plotly.graph_objects as go


def empty(message):
    figure = go.Figure()
    figure.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    return figure


def drift_figure(rows):
    if not rows:
        return empty("No distribution measurements")
    fig = go.Figure(go.Bar(x=[r["name"] for r in rows], y=[r["js_divergence"] for r in rows]))
    fig.update_layout(
        title="Descriptive distribution change, not a significance test",
        yaxis_title="Jensen-Shannon divergence (bits)",
        yaxis_range=[0, 1],
    )
    return fig


def reliability_figure(perf):
    if perf and perf.get("task") == "regression":
        return empty(
            "Probability reliability is not applicable to regression; see saved interval coverage metrics"
        )
    if not perf or not perf.get("n_evaluated"):
        return empty("No mature labelled rows selected")
    fig = go.Figure(
        go.Scatter(
            x=[0, 1], y=[0, 1], name="Perfect calibration", mode="lines", line={"dash": "dash"}
        )
    )
    for variant, rows in perf.get("reliability", {}).items():
        rows = [r for r in rows if r["n_rows"]]
        fig.add_trace(
            go.Scatter(
                x=[r["mean_probability"] for r in rows],
                y=[r["observed_fraction"] for r in rows],
                text=[f"n={r['n_rows']}" for r in rows],
                mode="lines+markers",
                name=variant,
            )
        )
    fig.update_layout(
        title="Reliability on mature labelled rows",
        xaxis_title="Mean saved probability",
        yaxis_title="Observed outcome fraction",
    )
    return fig


def history_figure(rows):
    if rows and rows[0].get("task") == "regression":
        from ml_template.usage_dashboard.nn_monitor_report import regression_history

        return regression_history(rows)
    if not rows:
        return empty("No comparable assessment history")
    fig = go.Figure()
    for name in ("brier", "pr_auc", "coverage"):
        fig.add_trace(
            go.Scatter(
                x=[r["as_of"] for r in rows],
                y=[r[name] for r in rows],
                mode="markers",
                name=name,
                text=[
                    f"{r['scoring_batch_id']} | evaluated={r['n_evaluated']} | support={r['support']}"
                    for r in rows
                ],
            )
        )
    fig.update_layout(
        title="Latest assessment per batch: same release/reference/outcome context",
        xaxis_title="Assessment as-of, not business event time",
        yaxis_title="Metric or coverage",
    )
    return fig
