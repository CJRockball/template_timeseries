"""Offline HTML/Markdown reporting and regeneration from verified saved data."""

import argparse
import html
from pathlib import Path

from ml_template.evaluation.evaluation_artifacts import manifest, verify, write_json


def table(rows):
    if not rows:
        return "<p>Not requested / not available.</p>"
    columns = list(dict.fromkeys(k for r in rows for k in r))
    head = "".join(f"<th>{html.escape(str(k))}</th>" for k in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(r.get(k)))}</td>" for k in columns) + "</tr>"
        for r in rows
    )
    return f"<div class='scroll'><table><tr>{head}</tr>{body}</table></div>"


def render_report(data, output_dir):
    from plotly.offline import get_plotlyjs

    from ml_template.dashboard.evaluation_figs import evaluation_figures

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sections = [
        ("Summary", [data["summary"]]),
        ("Frozen specification", [{"field": k, "value": v} for k, v in data["spec"].items()]),
        ("Warnings", [{"warning": w} for w in data["warnings"]]),
        ("Acceptance (no automatic promotion)", data["acceptance"]["criteria"]),
        ("Metrics; pr_auc means average precision", data["metrics"]),
        ("Calibrated minus raw", data["differences"]),
        ("Frozen policy", data["policy_results"]),
        (
            "Development policy-selection provenance (exploratory)",
            [data["policy_selection"]] if data.get("policy_selection") else [],
        ),
        ("Reliability support", data["reliability"]),
        ("Subgroups: global policy, exploratory", data["subgroups"]),
        ("Policy errors", data["errors"]),
        ("Development context, not independent final-test evidence", data["development_context"]),
        ("Provenance", [{"field": k, "value": v} for k, v in data["provenance"].items()]),
    ]
    body = "".join(f"<h2>{html.escape(title)}</h2>{table(rows)}" for title, rows in sections)
    plots = "".join(
        f.to_html(full_html=False, include_plotlyjs=False) for f in evaluation_figures(data)
    )
    style = "body{font:15px system-ui;margin:2rem}td,th{border:1px solid #ddd;padding:7px;max-width:650px;overflow-wrap:anywhere;text-align:left;vertical-align:top}table{border-collapse:collapse}.scroll{overflow:auto;max-height:650px}"
    page = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Final evaluation</title><style>"
        + style
        + "</style><script>"
        + get_plotlyjs()
        + "</script></head><body><h1>Final model evaluation</h1><p>Completion is not deployment approval. No fitting or selection occurs here.</p>"
        + body
        + plots
        + "</body></html>"
    )
    (out / "report.html").write_text(page, encoding="utf-8")
    lines = ["# Final model evaluation", "", "## Summary", ""]
    lines += [f"- {html.escape(str(k))}: {html.escape(str(v))}" for k, v in data["summary"].items()]
    lines += ["", "## Metrics", "", "|Variant|Metric|Value|Interval|", "|---|---|---:|---|"]
    lines += [
        f"|{r['variant']}|{r['metric']}|{r['value']}|[{r.get('ci_low')}, {r.get('ci_high')}]|"
        for r in data["metrics"]
    ]
    lines += ["", "## Frozen policy", ""]
    lines += ["- " + html.escape(str(row)) for row in data["policy_results"]]
    if data.get("policy_selection"):
        lines += [
            "",
            "## Development policy-selection provenance (exploratory)",
            "",
            html.escape(str(data["policy_selection"])),
        ]
    lines += ["", "## Warnings", ""] + [
        "- " + html.escape(str(w)).replace("\n", " ") for w in data["warnings"]
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def regenerate_report(source, output):
    if Path(output).exists():
        raise FileExistsError("use a new output directory")
    data = verify(source)
    render_report(data, output)
    write_json(
        Path(output) / "regeneration.json",
        {"source_evaluation_id": data["summary"]["evaluation_id"], "holdout_access": False},
    )
    manifest(output)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation-dir", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    regenerate_report(args.evaluation_dir, args.output)


if __name__ == "__main__":
    main()

# calibrated-policy update v1
