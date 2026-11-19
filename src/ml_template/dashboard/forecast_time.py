"""Read-only lead diagnostics over the forecasting page's verified score rows."""

from collections import defaultdict
from datetime import date
from math import sqrt


def lead_metrics(result):
    buckets = defaultdict(list)
    for row in result["rows"] or ():
        if row["method"] not in result["methods"]:
            continue
        lead = (
            date.fromisoformat(str(row["target_date"]))
            - date.fromisoformat(str(row["origin_date"]))
        ).days
        if lead < 1:
            raise ValueError("Nonpositive forecast lead")
        values = buckets[row["method"], lead]
        if row["status"] == "ok" and (
            result["support"] == "own_supported" or row["common_supported"]
        ):
            values.append(float(row["error"]))
    return [
        dict(
            method=m,
            lead=h,
            n=len(v),
            mae=sum(map(abs, v)) / len(v) if v else None,
            rmse=sqrt(sum(e**2 for e in v) / len(v)) if v else None,
            bias=sum(v) / len(v) if v else None,
        )
        for (m, h), v in sorted(buckets.items())
    ]


def panel(result):
    from dash import dcc, html
    from plotly import graph_objects as go

    if result["rows"] is None:
        return html.P("Enable linked DEVELOPMENT scores to see time analysis.")
    records = lead_metrics(result)
    if not records:
        return html.P("No linked rows match the current selection.")
    fig = go.Figure()
    metric = result["metric"]
    for method in result["methods"]:
        subset = [r for r in records if r["method"] == method]
        fig.add_scatter(
            x=[r["lead"] for r in subset],
            y=[r[metric] for r in subset],
            text=[f"Matched N={r['n']} · bias={r['bias']}" for r in subset],
            name=method,
            mode="lines+markers",
            connectgaps=False,
        )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Days from each forecast issuance",
        yaxis_title=metric.upper(),
    )
    note = (
        "Only lead 1 is present. Rolling origin dates are not longer leads."
        if {r["lead"] for r in records} == {1}
        else "Uses the current series, origin, method and support selections."
    )
    return html.Div(
        [html.H3("Time analysis"), html.P(note), dcc.Graph(id="forecast-chart-lead", figure=fig)]
    )
