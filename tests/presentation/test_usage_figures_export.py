import json

import numpy as np
import pytest

from ml_template.explainability.export_report import (
    entity_positions,
    export_report,
    render_html,
    script_json,
)
from ml_template.explainability.figures import (
    dependence_figure,
    distribution_figure,
    importance_figure,
    sample_positions,
    waterfall_figure,
)


def test_all_figures_construct_and_serialize(presentation_fixture):
    view = presentation_fixture.view()
    for figure in (
        importance_figure(view),
        distribution_figure(view),
        dependence_figure(view, "x"),
        dependence_figure(view, "kind"),
        waterfall_figure(view, 0, max_features=1),
    ):
        assert json.loads(figure.to_json())["data"]


def test_waterfall_preserves_signed_total(presentation_fixture):
    view = presentation_fixture.view()
    figure = waterfall_figure(view, 0, max_features=1)
    assert np.isclose(sum(figure.data[0].x[:-1]), view.rows.margin.iloc[0])
    assert "Other features (signed sum)" in figure.data[0].y


def test_sampling_is_stable_and_display_only(presentation_fixture):
    view = presentation_fixture.view()
    assert sample_positions(view, 2).tolist() == sample_positions(view, 2).tolist()
    assert len(sample_positions(view, 2)) == 2
    assert len(view.rows) == 3
    with pytest.raises(ValueError):
        sample_positions(view, 0)


def test_script_data_escaping():
    encoded = script_json('{"name":"</script><script>bad()</script>&"}')
    assert "</script>" not in encoded and "<" not in encoded
    assert json.loads(encoded)["name"] == "</script><script>bad()</script>&"


def test_offline_report_is_separate_and_source_unchanged(presentation_fixture, tmp_path):
    obj = presentation_fixture
    before = {p.name: p.read_bytes() for p in obj.directory.iterdir()}
    result = export_report(
        obj.database, "explain_fixture", tmp_path / "reports", entities=[10], max_local=1
    )
    from pathlib import Path

    report = Path(result["artifact_dir"])
    html = (report / "report.html").read_text()
    assert "plotly.js" in html.lower()
    assert "<script src=" not in html
    assert "Entity 10" in html
    assert "not calibrated probability" in html
    assert (report / "artifacts.json").is_file()
    assert before == {p.name: p.read_bytes() for p in obj.directory.iterdir()}


def test_export_inside_source_forbidden(presentation_fixture):
    obj = presentation_fixture
    with pytest.raises(ValueError, match="immutable"):
        export_report(obj.database, "explain_fixture", obj.directory / "reports")


@pytest.mark.parametrize("entities", [[True], ["missing"], [10, 10], [10, 20, 30]])
def test_bad_entity_selection(presentation_fixture, entities):
    with pytest.raises(ValueError):
        entity_positions(presentation_fixture.view(), entities=entities, max_local=2)


def test_render_does_not_mutate_data(presentation_fixture):
    view = presentation_fixture.view()
    original = view.contributions.copy()
    text, positions = render_html(view, max_local=1)
    assert positions == [int(np.argmax(view.rows.p_release))]
    assert "not a representative sample" in text
    assert view.contributions.equals(original)


def test_non_list_entity_selection(presentation_fixture):
    with pytest.raises(TypeError, match="JSON list"):
        entity_positions(
            presentation_fixture.view(),
            entities="10",
        )
