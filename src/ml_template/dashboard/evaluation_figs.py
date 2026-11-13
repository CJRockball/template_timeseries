"""Pure Plotly figures for saved evaluations; no database, Dash callbacks or fitting."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def evaluation_figures(data):
    figures = []
    loss = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Brier score (lower is better)", "Log loss (lower is better)"],
    )
    for col, metric in enumerate(["brier", "log_loss"], 1):
        rows = [r for r in data.get("metrics", []) if r["metric"] == metric]
        loss.add_trace(
            go.Bar(
                x=[r["variant"] for r in rows],
                y=[r["value"] for r in rows],
                name=metric,
                showlegend=False,
            ),
            row=1,
            col=col,
        )
    loss.update_layout(title="Holdout probability quality", template="plotly_white")
    figures.append(loss)
    fig = go.Figure()
    for v in sorted({r["variant"] for r in data.get("reliability", [])}):
        rows = [r for r in data["reliability"] if r["variant"] == v]
        fig.add_trace(
            go.Scatter(
                x=[r["mean_probability"] for r in rows],
                y=[r["event_rate"] for r in rows],
                name=v,
                mode="lines+markers",
                customdata=[[r["n_rows"], r["n_positives"]] for r in rows],
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": [r["event_rate_high"] - r["event_rate"] for r in rows],
                    "arrayminus": [r["event_rate"] - r["event_rate_low"] for r in rows],
                },
                hovertemplate="mean p=%{x:.4f}<br>event rate=%{y:.4f}<br>rows=%{customdata[0]}<br>positives=%{customdata[1]}<extra>%{fullData.name}</extra>",
            )
        )
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="ideal", mode="lines", line={"dash": "dash"}))
    fig.update_layout(
        title="Reliability (pointwise 95% Wilson intervals)",
        xaxis_title="Mean predicted probability",
        yaxis_title="Observed event rate",
        template="plotly_white",
    )
    figures.append(fig)
    for kind, title, x, y in [
        ("precision_recall", "Precision–recall", "Recall", "Precision"),
        ("roc", "ROC", "False-positive rate", "True-positive rate"),
        ("capture", "Cumulative capture", "Flagged fraction", "Recall"),
    ]:
        fig = go.Figure()
        for v in sorted({r["variant"] for r in data.get("curves", []) if r["kind"] == kind}):
            rows = [r for r in data["curves"] if r["kind"] == kind and r["variant"] == v]
            fig.add_trace(
                go.Scatter(x=[r["x"] for r in rows], y=[r["y"] for r in rows], name=v, mode="lines")
            )
        fig.update_layout(title=title, xaxis_title=x, yaxis_title=y, template="plotly_white")
        figures.append(fig)
    return figures
