import pandas as pd
import pytest


def test_real_parquet_roundtrip(presentation_fixture, monkeypatch):
    pytest.importorskip("pyarrow")
    from pandas.io.parquet import read_parquet

    from ml_template.explainability.export_report import export_report

    obj = presentation_fixture
    for name, frame in obj.tables.items():
        frame.to_parquet(obj.directory / name, index=True)
    obj.reseal()
    monkeypatch.setattr(pd, "read_parquet", read_parquet)
    view = obj.view()
    assert isinstance(view.features.kind.dtype, pd.CategoricalDtype)
    assert view.rows.index.tolist() == [30, 10, 20]
    result = export_report(obj.database, "explain_fixture", obj.directory.parent / "real_reports")
    assert result["source_explanation_modified"] is False
