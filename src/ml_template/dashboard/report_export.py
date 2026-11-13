"""Memory-only reports from browser-rendered snapshots; never load models or datasets."""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import zipfile
from datetime import UTC, datetime

TABS = {
    "assets": "Assets",
    "sweeps": "Sweeps",
    "experiments": "Experiments",
    "run": "Single run",
    "compare": "Compare",
    "calibration": "Calibration",
    "evaluation": "Final evaluation",
}
MAX_BYTES = 12 * 1024 * 1024
MAX_CELLS = 100000


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _text(value, limit=250000):
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError("Invalid or oversized text")
    return value


def validate(snapshot):
    if not isinstance(snapshot, dict):
        raise ValueError("No browser snapshot")
    if len(canonical(snapshot).encode()) > MAX_BYTES:
        raise ValueError("Snapshot exceeds 12 MiB; reduce the selected view")
    if type(snapshot.get("schema_version")) is not int or snapshot["schema_version"] != 1:
        raise ValueError("Unsupported snapshot version")
    if snapshot.get("active_tab") not in TABS:
        raise ValueError("Unknown tab")
    stamp = datetime.fromisoformat(_text(snapshot.get("captured_at"), 64).replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("Timestamp needs a timezone")
    if type(snapshot.get("include_final_evaluation")) is not bool:
        raise ValueError("Missing consent state")
    if snapshot["active_tab"] == "evaluation" and not snapshot["include_final_evaluation"]:
        raise ValueError("Explicitly allow visible Final evaluation content first")
    controls = snapshot.get("controls", [])
    if not isinstance(controls, list) or len(controls) > 250:
        raise ValueError("Invalid control state")
    for item in controls:
        if not isinstance(item, dict):
            raise ValueError("Invalid control")
        for key in ("id", "kind", "value"):
            _text(item.get(key, ""), 20000)
    blocks = snapshot.get("blocks")
    if not isinstance(blocks, list) or not 1 <= len(blocks) <= 800:
        raise ValueError("No rendered content or too many blocks")
    cells = 0
    for block in blocks:
        if not isinstance(block, dict):
            raise ValueError("Invalid block")
        kind = block.get("kind")
        if kind in {"paragraph", "pre", "heading"}:
            _text(block.get("text"))
            if kind == "heading" and (
                type(block.get("level")) is not int or not 1 <= block["level"] <= 6
            ):
                raise ValueError("Invalid heading")
        elif kind == "table":
            rows = block.get("rows")
            if not isinstance(rows, list) or not 1 <= len(rows) <= 10000:
                raise ValueError("Invalid table")
            _text(block.get("caption", ""), 1000)
            for row in rows:
                if not isinstance(row, list) or not row:
                    raise ValueError("Invalid table row")
                cells += len(row)
                for cell in row:
                    _text(cell)
            if cells > MAX_CELLS:
                raise ValueError("Displayed-cell limit exceeded")
        elif kind == "figure":
            fig = block.get("figure")
            if (
                not isinstance(fig, dict)
                or set(fig) - {"data", "layout"}
                or not isinstance(fig.get("data"), list)
                or not isinstance(fig.get("layout"), dict)
            ):
                raise ValueError("Invalid figure")
            remote = {
                "scattergeo",
                "choropleth",
                "scattermapbox",
                "choroplethmapbox",
                "densitymapbox",
                "scattermap",
                "choroplethmap",
                "densitymap",
            }
            for trace in fig["data"]:
                if not isinstance(trace, dict) or trace.get("type") in remote:
                    raise ValueError("Offline reports do not support geographic/map traces")
                if "source" in trace:
                    _image(trace["source"])
            template = fig["layout"].get("template", {})
            if not isinstance(template, dict):
                raise ValueError("Invalid figure template")
            for layout in (fig["layout"], template.get("layout", {})):
                if not isinstance(layout, dict) or not isinstance(layout.get("images", []), list):
                    raise ValueError("Invalid layout images")
                for image in layout.get("images", []):
                    _image(image.get("source") if isinstance(image, dict) else None)
        else:
            raise ValueError("Unsupported report block")
    return snapshot


def _image(source):
    if not isinstance(source, str) or not source.startswith(
        ("data:image/png;base64,", "data:image/jpeg;base64,", "data:image/webp;base64,")
    ):
        raise ValueError("External images are not bundled; use embedded PNG/JPEG/WebP")


def metadata(snapshot):
    return {
        "schema_version": 1,
        "exported_at": datetime.now(UTC).isoformat(),
        "captured_at": snapshot["captured_at"],
        "active_tab": snapshot["active_tab"],
        "controls": snapshot.get("controls", []),
        "include_final_evaluation": snapshot["include_final_evaluation"],
        "snapshot_sha256": hashlib.sha256(canonical(snapshot).encode()).hexdigest(),
        "capture_kind": "rendered_current_tab",
        "underlying_data_reloaded": False,
        "artifact_integrity_reverified": False,
        "limitations": [
            "Rendered UI snapshot, not a model artifact, verified provenance or deployment approval.",
            "Tables retain displayed precision, pagination and row/column limits, not full source data.",
            "Control values are displayed labels, not a machine-restorable selection state.",
            "Hidden tabs, collapsed content and unrendered rows are excluded.",
            "AG Grid column panes are separate; pinned-top/bottom rows are not captured.",
            "Displayed figures include chart data, even legend-hidden traces; review before sharing.",
            "Charts retain captured data/layout, not Dash callbacks, animation or hover state.",
            "Maps/external images and external network requests are unsupported; LaTeX assets are not bundled.",
            "Source artifact IDs/hashes are not independently recovered or reverified.",
            "CSV formula-like text is apostrophe-prefixed; snapshot.json retains original displayed text.",
        ],
    }


def script_json(value):
    return (
        canonical(value)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace(chr(0x2028), "\\u2028")
        .replace(chr(0x2029), "\\u2029")
    )


def _render(snapshot, info):
    escape = html.escape
    parts, figures = [], []
    for block in snapshot["blocks"]:
        kind = block["kind"]
        if kind == "heading":
            tag = "h" + str(block["level"])
            parts.append("<" + tag + ">" + escape(block["text"]) + "</" + tag + ">")
        elif kind in {"paragraph", "pre"}:
            tag = "pre" if kind == "pre" else "p"
            parts.append("<" + tag + ">" + escape(block["text"]) + "</" + tag + ">")
        elif kind == "table":
            rows = "".join(
                "<tr>" + "".join("<td>" + escape(cell) + "</td>" for cell in row) + "</tr>"
                for row in block["rows"]
            )
            parts.append(
                "<h3>"
                + escape(block.get("caption", "Displayed table"))
                + "</h3><div class='scroll'><table>"
                + rows
                + "</table></div>"
            )
        else:
            identifier = "report-figure-" + str(len(figures) + 1)
            figures.append({"id": identifier, "figure": block["figure"]})
            parts.append("<div class='figure' id='" + identifier + "'></div>")
    charts = ""
    if figures:
        from plotly.offline import get_plotlyjs

        charts = (
            "<script>"
            + get_plotlyjs()
            + "</script><script>const reportFigures="
            + script_json(figures)
            + ";reportFigures.forEach(function(x){"
            "Plotly.newPlot(x.id,x.figure.data,x.figure.layout,{responsive:true,displaylogo:false})"
            ".catch(function(e){document.getElementById(x.id).textContent='Chart unavailable: '+String(e);});});</script>"
        )
    controls = "".join(
        "<tr><td>"
        + escape(c.get("id", ""))
        + "</td><td>"
        + escape(c.get("value", ""))
        + "</td></tr>"
        for c in info["controls"]
    )
    title = "Dashboard report — " + TABS[snapshot["active_tab"]]
    csp = "default-src 'none'; script-src 'unsafe-inline' 'unsafe-eval'; style-src 'unsafe-inline'; img-src data: blob:; font-src data:; connect-src 'none'; worker-src blob:; object-src 'none'; base-uri 'none'"
    css = "body{font-family:system-ui,sans-serif;margin:28px;line-height:1.45}table{border-collapse:collapse}td{border:1px solid #ccc;padding:6px;vertical-align:top;white-space:pre-wrap}.scroll{overflow-x:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f4f4;padding:12px}.figure{min-height:400px;width:100%}.notice{background:#edf5ff;padding:12px;border-left:4px solid #2673b8}"
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        '<meta http-equiv="Content-Security-Policy" content="' + escape(csp, quote=True) + '">'
        "<title>"
        + escape(title)
        + "</title><style>"
        + css
        + "</style></head><body><h1>"
        + escape(title)
        + "</h1><p>Captured: "
        + escape(info["captured_at"])
        + "</p>"
        "<div class='notice'>Read-only rendered snapshot, not new evaluation or full-precision source data.</div>"
        "<h2>Selected controls — displayed labels</h2><table>"
        + controls
        + "</table><hr>"
        + "".join(parts)
        + "<hr><h2>Scope and limitations</h2><ul>"
        + "".join("<li>" + escape(t) + "</li>" for t in info["limitations"])
        + "</ul><details><summary>Export metadata</summary><pre>"
        + escape(json.dumps(info, ensure_ascii=False, indent=2))
        + "</pre></details>"
        + charts
        + "</body></html>"
    )


def build_html(snapshot):
    snapshot = validate(snapshot)
    return _render(snapshot, metadata(snapshot))


def _csv_cell(value):
    return (
        "'" + value
        if value.lstrip().startswith(("=", "+", "-", "@")) or value.startswith(("\t", "\r", "\n"))
        else value
    )


def build_bundle(snapshot):
    snapshot = validate(snapshot)
    info = metadata(snapshot)
    members = {
        "report.html": _render(snapshot, info).encode(),
        "metadata.json": json.dumps(info, ensure_ascii=False, indent=2).encode(),
        "snapshot.json": json.dumps(
            snapshot, ensure_ascii=False, indent=2, allow_nan=False
        ).encode(),
    }
    tables, figures = 0, 0
    for block in snapshot["blocks"]:
        if block["kind"] == "table":
            tables += 1
            stream = io.StringIO(newline="")
            csv.writer(stream).writerows([[_csv_cell(c) for c in row] for row in block["rows"]])
            members[f"tables/table_{tables:03d}.csv"] = ("\ufeff" + stream.getvalue()).encode()
        elif block["kind"] == "figure":
            figures += 1
            members[f"figures/figure_{figures:03d}.json"] = canonical(block["figure"]).encode()
    members["README.txt"] = (
        b"Extract this bundle and open report.html in a browser. Plotly.js is embedded when charts exist.\n"
        b"Tables are rendered/rounded/limited text, not full source data. CSV formula-like cells have an apostrophe prefix.\n"
        b"snapshot.json preserves the displayed text. Metadata is not independently verified artifact provenance.\n"
        b"No models, database or source dataset are included. Treat chart/table data as potentially sensitive.\n"
    )
    members["SHA256SUMS.json"] = json.dumps(
        {n: hashlib.sha256(v).hexdigest() for n, v in members.items()}, indent=2
    ).encode()
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    return stream.getvalue()
