"""Export rich standalone HTML without modifying the source explanation package."""

import argparse
import json
import uuid
from datetime import UTC, datetime
from html import escape
from pathlib import Path

import numpy as np
import plotly
from plotly.offline import get_plotlyjs

from ml_template.explainability.artifacts import (
    file_entries,
    fingerprint,
    sha256,
    verify_explanation,
    write_json,
)
from ml_template.explainability.figures import (
    dependence_figure,
    distribution_figure,
    importance_figure,
    waterfall_figure,
)
from ml_template.explainability.presentation_data import load_explanation, selected_entity


def script_json(text):
    return (
        text.replace("&", r"\u0026")
        .replace("<", r"\u003c")
        .replace(">", r"\u003e")
        .replace("\u2028", r"\u2028")
        .replace("\u2029", r"\u2029")
    )


def figure_html(figure, identifier):
    payload = script_json(figure.to_json())
    return (
        f'<div id="{identifier}"></div><script type="application/json" id="{identifier}-data">{payload}</script>'
        f'<script>{{const f=JSON.parse(document.getElementById("{identifier}-data").textContent);'
        f'Plotly.newPlot("{identifier}",f.data,f.layout,{{responsive:true}});}}</script>'
    )


def entity_positions(view, entities=None, max_local=3):
    if isinstance(max_local, bool) or not isinstance(max_local, int) or max_local < 0:
        raise ValueError("nonnegative max_local required")
    if entities is None:
        return np.argsort(-view.rows.p_release.to_numpy(), kind="stable")[:max_local].tolist()
    if not isinstance(entities, list):
        raise TypeError("entities must be a JSON list")
    if len(entities) > max_local:
        raise ValueError("entity list exceeds max_local")
    if any(isinstance(k, bool) or not isinstance(k, (int, str)) for k in entities):
        raise ValueError("entity keys must be JSON integers or strings")
    if len(set(entities)) != len(entities):
        raise ValueError("duplicate entity selection")
    positions = []
    for key in entities:
        try:
            position = view.rows.index.get_loc(key)
        except KeyError as exc:
            raise ValueError(f"entity not present: {key}") from exc
        positions.append(int(position))
    return positions


def render_html(view, *, entities=None, max_local=3, max_points=2000):
    positions = entity_positions(view, entities, max_local)
    summary = escape(json.dumps(view.summary, indent=2))
    sections = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<title>Released model explanations</title><style>body{font-family:system-ui;margin:2rem;max-width:1400px}table{border-collapse:collapse}td,th{padding:.4rem;border:1px solid #ccc}pre{white-space:pre-wrap}.notice{background:#f4f4f4;padding:1rem}</style>",
        "<script>" + get_plotlyjs() + "</script></head><body><h1>Released model explanations</h1>",
        "<p class='notice'>Contributions explain base-model log-odds, not calibrated probability or policy. They are not causal effects or evidence of predictive accuracy. Global figures describe the complete original scoring batch. Plot samples are display-only; no policy is recalculated.</p>",
        f"<details><summary>Identity and provenance summary</summary><pre>{summary}</pre></details>",
        figure_html(importance_figure(view), "importance"),
        figure_html(distribution_figure(view, max_points=max_points), "distribution"),
    ]
    if len(view.importance):
        sections.append(
            figure_html(
                dependence_figure(view, view.importance.feature.iloc[0], max_points=max_points),
                "dependence",
            )
        )
    sections += [
        "<h2>Individual examples</h2>",
        "<p>"
        + (
            "Explicitly selected entities."
            if entities is not None
            else "Highest release-probability examples; these are not a representative sample."
        )
        + "</p>",
    ]
    for number, position in enumerate(positions):
        key, row, table = selected_entity(view, position)
        scores = {
            name: row[name]
            for name in ("p_raw", "p_calibrated", "p_release", "policy_flag")
            if name in row.index
        }
        score_text = ", ".join(f"{name}={value}" for name, value in scores.items())
        sections += [
            f"<h3>Entity {escape(str(key))}</h3><p>{escape(score_text)}</p>",
            figure_html(waterfall_figure(view, position), f"waterfall-{number}"),
            table.to_html(index=False, escape=True),
        ]
    sections.append("</body></html>")
    return "\n".join(sections), positions


def export_report(
    database,
    explanation_id,
    output_root,
    *,
    entities=None,
    max_local=3,
    max_points=2000,
    max_rows=100000,
):
    view = load_explanation(database, explanation_id, max_rows=max_rows)
    root = Path(output_root).resolve()
    source = Path(view.record["artifact_dir"]).resolve()
    if root == source or root.is_relative_to(source):
        raise ValueError("report exports must not be written inside immutable source explanations")
    text, positions = render_html(
        view, entities=entities, max_local=max_local, max_points=max_points
    )
    if verify_explanation(source, explanation_id) != view.manifest:
        raise ValueError("source explanation changed during rendering")
    report_id = f"report_{datetime.now(UTC):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:12]}"
    destination = root / report_id
    destination.mkdir(parents=True, exist_ok=False)
    try:
        (destination / "report.html").write_text(text)
        renderer_files = ("presentation_data.py", "figures.py", "export_report.py")
        write_json(
            destination / "report.json",
            {
                "schema_version": 1,
                "report_id": report_id,
                "explanation_id": explanation_id,
                "source_content_sha256": view.manifest["content_sha256"],
                "release_id": view.summary["release_id"],
                "scoring_batch_id": view.summary["scoring_batch_id"],
                "selected_entities": [
                    view.rows.index[p].item()
                    if isinstance(view.rows.index[p], np.generic)
                    else view.rows.index[p]
                    for p in positions
                ],
                "selection": "explicit" if entities is not None else "highest_release_probability",
                "max_points": max_points,
                "plotly_version": plotly.__version__,
                "offline_self_contained": True,
                "renderer_sha256": {
                    name: sha256(Path(__file__).with_name(name)) for name in renderer_files
                },
                "created_at": datetime.now(UTC).isoformat(),
            },
        )
        payload = {
            "schema_version": 1,
            "kind": "explanation_report",
            "report_id": report_id,
            "files": file_entries(destination),
        }
        write_json(
            destination / "artifacts.json", {**payload, "content_sha256": fingerprint(payload)}
        )
    except Exception as exc:
        write_json(destination / "failure.json", {"complete": False, "error": str(exc)})
        raise
    return {
        "report_id": report_id,
        "artifact_dir": str(destination),
        "explanation_id": explanation_id,
        "source_explanation_modified": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--explanation-id", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/explanation_reports"))
    parser.add_argument("--entities-json")
    parser.add_argument("--max-local", type=int, default=3)
    parser.add_argument("--max-points", type=int, default=2000)
    parser.add_argument("--max-rows", type=int, default=100000)
    args = parser.parse_args(argv)
    entities = json.loads(args.entities_json) if args.entities_json else None
    print(
        json.dumps(
            export_report(
                args.database,
                args.explanation_id,
                args.output_root,
                entities=entities,
                max_local=args.max_local,
                max_points=args.max_points,
                max_rows=args.max_rows,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
