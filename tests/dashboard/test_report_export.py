"""Actual renderer tests, independent of models, registry and source datasets."""

import copy
import csv
import hashlib
import io
import json
import zipfile

import pytest

from ml_template.dashboard import report_export as report


def snapshot():
    return {
        "schema_version": 1,
        "format": "html",
        "active_tab": "experiments",
        "captured_at": "2026-11-11T14:39:00Z",
        "include_final_evaluation": False,
        "controls": [
            {"id": "cohort", "kind": "dropdown_display_labels", "value": "concrete | MPa"}
        ],
        "blocks": [
            {"kind": "heading", "level": 3, "text": "Recorded model parameters"},
            {"kind": "table", "caption": "NN parameters", "rows": [["hidden_dims"], ["[64, 32]"]]},
            {"kind": "paragraph", "text": "Showing 1 of 200 rows; development evidence only"},
        ],
    }


def test_html_preserves_displayed_content():
    result = report.build_html(snapshot())
    assert "[64, 32]" in result and "concrete | MPa" in result
    assert "Showing 1 of 200 rows" in result and "displayed precision" in result
    assert "<script>" not in result and "connect-src" in result


def test_text_is_escaped():
    value = snapshot()
    value["blocks"][0]["text"] = "<script>window.injected=1</script>"
    result = report.build_html(value)
    assert "<script>window.injected" not in result
    assert "&lt;script&gt;" in result


def test_zip_checksums_and_snapshot():
    value = snapshot()
    with zipfile.ZipFile(io.BytesIO(report.build_bundle(value))) as archive:
        assert archive.testzip() is None
        assert json.loads(archive.read("snapshot.json")) == value
        info = json.loads(archive.read("metadata.json"))
        assert (
            info["underlying_data_reloaded"] is False
            and info["artifact_integrity_reverified"] is False
        )
        for name, digest in json.loads(archive.read("SHA256SUMS.json")).items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
        assert "tables/table_001.csv" in archive.namelist()


@pytest.mark.parametrize(
    "value", ["=cmd", "+1", "-2.5", "@name", "  =cmd", "\ttext", "\rtext", "\ntext"]
)
def test_spreadsheet_formula_escaping(value):
    snap = snapshot()
    snap["blocks"][1]["rows"] = [[value]]
    with zipfile.ZipFile(io.BytesIO(report.build_bundle(snap))) as archive:
        rows = list(
            csv.reader(io.StringIO(archive.read("tables/table_001.csv").decode("utf-8-sig")))
        )
        assert rows == [["'" + value]]
        assert json.loads(archive.read("snapshot.json"))["blocks"][1]["rows"] == [[value]]


@pytest.mark.parametrize("tab", list(report.TABS))
def test_tab_labels(tab):
    snap = snapshot()
    snap["active_tab"] = tab
    snap["include_final_evaluation"] = tab == "evaluation"
    assert report.TABS[tab] in report.build_html(snap)


@pytest.mark.parametrize(
    "override",
    [
        {"schema_version": True},
        {"schema_version": 2},
        {"active_tab": "../bad"},
        {"captured_at": "2026-11-11"},
        {"captured_at": "bad"},
        {"include_final_evaluation": "yes"},
        {"blocks": []},
        {"controls": "bad"},
        {"active_tab": "evaluation"},
        {"blocks": [{"kind": "html", "text": "bad"}]},
        {"blocks": [{"kind": "heading", "level": 9, "text": "bad"}]},
        {"blocks": [{"kind": "table", "rows": [[None]]}]},
    ],
)
def test_invalid_snapshots_rejected(override):
    snap = snapshot()
    snap.update(override)
    with pytest.raises((ValueError, TypeError)):
        report.build_html(snap)


def test_size_limits(monkeypatch):
    monkeypatch.setattr(report, "MAX_BYTES", 20)
    with pytest.raises(ValueError):
        report.build_html(snapshot())


def test_cell_limits(monkeypatch):
    monkeypatch.setattr(report, "MAX_CELLS", 1)
    with pytest.raises(ValueError):
        report.build_bundle(snapshot())


def test_plotly_embedded_once_and_chart_json_escaped():
    pytest.importorskip("plotly")
    snap = snapshot()
    attack = "</script><script>window.reportInjected=true</script>"
    fig = {
        "data": [{"type": "scatter", "x": [1, 2], "y": [3, 4], "text": [attack, "safe"]}],
        "layout": {},
    }
    snap["blocks"] += [
        {"kind": "figure", "figure": fig},
        {"kind": "figure", "figure": copy.deepcopy(fig)},
    ]
    html = report.build_html(snap)
    assert attack not in html and "\\u003c/script\\u003e" in html
    assert html.count("plotly.js v") == 1 and "<script src=" not in html
    with zipfile.ZipFile(io.BytesIO(report.build_bundle(snap))) as archive:
        assert "figures/figure_002.json" in archive.namelist()


@pytest.mark.parametrize(
    "fig",
    [
        {"data": [{"type": "scattergeo"}], "layout": {}},
        {"data": [{"type": "image", "source": "https://example.invalid/image.png"}], "layout": {}},
        {"data": [], "layout": {"images": [{"source": "file:///secret"}]}},
        {"data": [{"x": [float("nan")]}], "layout": {}},
    ],
)
def test_external_or_nonfinite_figures_rejected(fig):
    snap = snapshot()
    snap["blocks"].append({"kind": "figure", "figure": fig})
    with pytest.raises(ValueError):
        report.build_html(snap)


def test_safe_bundle_paths_and_no_side_effects(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    snap = snapshot()
    snap["blocks"][1]["caption"] = "../../model.ubj"
    original = copy.deepcopy(snap)
    with zipfile.ZipFile(io.BytesIO(report.build_bundle(snap))) as archive:
        assert not any(".." in name for name in archive.namelist())
    assert snap == original and not list(tmp_path.iterdir())
