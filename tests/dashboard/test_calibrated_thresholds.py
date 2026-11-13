import pandas as pd
import pytest

cd = pytest.importorskip("ml_template.dashboard.calibration_data")


def bundle():
    return {
        "header": {"parent_run_id": "run_test", "calibration_id": "cal_test"},
        "provenance": {"key_col": "udi"},
        "predictions": pd.DataFrame(
            {
                "udi": [3, 1, 2],
                "fold": [0, 0, 1],
                "y_true": [1, 0, 0],
                "p_raw": [0.8, 0.2, 0.4],
                "p_cal_crossfit": [0.6, 0.01, 0.1],
                "p_cal_fit": [0.9, 0, 0.02],
            }
        ),
    }


def test_calibrated_adapter_sorts_and_excludes_fit_scores():
    frames, metadata = cd.threshold_inputs(bundle())
    frame = frames["Calibrated cross-fit"]
    assert frame.udi.tolist() == [1, 2, 3]
    assert frame.p_oof.tolist() == [0.01, 0.1, 0.6]
    assert "p_cal_fit" not in frame
    assert metadata["Calibrated cross-fit"]["calibration_id"] == "cal_test"
    assert metadata["Calibrated cross-fit"]["score_variant"] == "calibrated_crossfit"


def test_raw_adapter_uses_raw_scale():
    frames, metadata = cd.threshold_inputs(bundle(), "raw")
    assert frames["Raw OOF"].p_oof.tolist() == [0.2, 0.4, 0.8]
    assert metadata["Raw OOF"]["score_variant"] == "raw"


def test_empty_and_forbidden_variants():
    b = bundle()
    b["predictions"] = pd.DataFrame()
    assert cd.threshold_inputs(b) == ({}, {})
    with pytest.raises(ValueError):
        cd.threshold_inputs(bundle(), "p_cal_fit")


def test_shared_panel_preserves_metadata():
    pytest.importorskip("dash")
    pytest.importorskip("dash_ag_grid")
    import json

    from plotly.utils import PlotlyJSONEncoder

    from ml_template.dashboard.threshold_panels import render_threshold_panel

    frames, metadata = cd.threshold_inputs(bundle())
    panel = render_threshold_panel(frames, 0.1, 0.5, metadata=metadata)
    encoded = json.dumps(panel, cls=PlotlyJSONEncoder)
    assert "calibrated_crossfit" in encoded and "cal_test" in encoded


def test_calibration_page_has_threshold_controls():
    pytest.importorskip("dash")
    pytest.importorskip("dash_ag_grid")
    import inspect

    from ml_template.dashboard.pages.calibration_page import build_view

    signature = inspect.signature(build_view)
    assert {"score_variant", "threshold", "budget"}.issubset(signature.parameters)
