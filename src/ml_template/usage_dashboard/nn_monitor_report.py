"""Pure saved monitoring report/figure helpers. No model or source-label reads."""

import csv
import hashlib
import html
import io
import json
import zipfile


def metric_rows(perf):
    if not perf:
        return []
    out = []
    for variant, metrics in perf.get("score_metrics", {}).items():
        for metric, value in metrics.items():
            if metric == "residual_quantiles":
                out.extend(
                    (
                        {
                            "variant": variant,
                            "metric": "residual_q" + str(q),
                            "value": v,
                            "n_evaluated": perf["n_evaluated"],
                            "sufficient_support": perf["sufficient_support"],
                        }
                        for q, v in value.items()
                    )
                )
            else:
                out.append(
                    {
                        "variant": variant,
                        "metric": metric,
                        "value": value,
                        "n_evaluated": perf["n_evaluated"],
                        "sufficient_support": perf["sufficient_support"],
                    }
                )
    for level, metrics in perf.get("interval_metrics", {}).items():
        out.extend(
            (
                {
                    "variant": "prediction_interval",
                    "level": level,
                    "metric": metric,
                    "value": value,
                    "n_evaluated": metrics["n_evaluated"],
                    "sufficient_support": perf["sufficient_support"],
                }
                for metric, value in metrics.items()
                if metric != "n_evaluated"
            )
        )
    return out


def regression_history(rows):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for name in ["rmse", "mae", "bias"]:
        fig.add_trace(
            go.Scatter(
                x=[r["as_of"] for r in rows],
                y=[r.get(name) for r in rows],
                mode="markers",
                name=name,
                text=[str(r["scoring_batch_id"]) + " | support=" + str(r["support"]) for r in rows],
            ),
            secondary_y=False,
        )
    fig.add_trace(
        go.Scatter(
            x=[r["as_of"] for r in rows],
            y=[r.get("coverage") for r in rows],
            mode="markers",
            name="mature label coverage",
        ),
        secondary_y=True,
    )
    fig.update_yaxes(
        title_text="Errors / signed bias (" + str(rows[0]["target_unit"]) + ")", secondary_y=False
    )
    fig.update_yaxes(title_text="Mature label coverage", range=[0, 1], secondary_y=True)
    fig.update_layout(
        title="Saved regression history; support in hover, not independent replications",
        xaxis_title="Assessment as-of",
    )
    return fig


def bundle_bytes(view, identity):
    from ml_template.monitoring.nn_monitoring_contract import validate_frozen_view

    validate_frozen_view(view)
    document = {
        k: view[k] for k in ["summary", "config", "quality", "drift", "performance", "findings"]
    }
    if document["summary"]["monitoring_id"] != identity:
        raise ValueError("Report identity differs")
    members = {
        name + ".json": json.dumps(value, indent=2, allow_nan=False).encode()
        for name, value in document.items()
    }
    body = html.escape(json.dumps(document, indent=2, allow_nan=False))
    members["report.html"] = (
        '<!doctype html><meta charset="utf-8"><title>Saved monitoring report</title><h1>Saved monitoring evidence</h1><p>Verified saved artifacts only; no new scoring, fitting or evaluation. No warning is not proof of health.</p><pre>'
        + body
        + "</pre>"
    ).encode()
    perf = view["performance"]
    rows = (
        metric_rows(perf)
        if (perf or {}).get("task") == "regression"
        else [
            {
                "variant": variant,
                "metric": metric,
                "value": value,
                "n_evaluated": perf["n_evaluated"],
                "sufficient_support": perf["sufficient_support"],
            }
            for variant, metrics in (perf or {}).get("score_metrics", {}).items()
            for metric, value in metrics.items()
        ]
    )
    stream = io.StringIO()
    columns = ["variant", "level", "metric", "value", "n_evaluated", "sufficient_support"]
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        safe = {
            k: "'" + v if isinstance(v, str) and v[:1] in {"=", "+", "-", "@"} else v
            for k, v in row.items()
        }
        writer.writerow(safe)
    members["metrics.csv"] = stream.getvalue().encode()
    members["SHA256SUMS.json"] = json.dumps(
        {k: hashlib.sha256(v).hexdigest() for k, v in members.items()}, indent=2
    ).encode()
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    return result.getvalue()
