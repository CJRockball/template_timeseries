from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ml_template.evaluation.regression_postprocessing import add_monitoring, evaluate
from ml_template.models.regression_postprocessing import apply, fit_quantiles, interval_columns


def parameters():
    return {
        "schema_version": 1,
        "correction": {"method": "bias", "slope": 1.0, "intercept": 2.0},
        "n_calibration": 99,
        "quantiles": fit_quantiles(np.zeros(99), np.arange(99.0), [0.9]),
    }


def test_evaluation_keeps_raw_predictions_and_reports_intervals():
    state = parameters()
    p = SimpleNamespace(
        provenance={"postprocessing": {"state": {"parameters": state}}},
        spec=SimpleNamespace(
            release_variant="recalibrated",
            uncertainty=SimpleNamespace(enabled=True, coverages=[0.9]),
        ),
    )
    raw = np.array([1.0, 2.0])
    point, columns, metrics = evaluate(p, raw, np.array([4.0, 5.0]))
    np.testing.assert_allclose(point, [3.0, 4.0])
    np.testing.assert_array_equal(columns["y_pred_raw"], raw)
    assert {row["metric"] for row in metrics} == {
        "empirical_coverage_c900000",
        "mean_width_c900000",
    }


@pytest.mark.parametrize(
    "support,coverage,sampling,expected",
    [
        (True, 1.0, "all_rows", True),
        (False, 1.0, "all_rows", False),
        (True, 0.5, "all_rows", False),
        (True, 1.0, "selected", False),
        (True, 1.0, "unknown", False),
    ],
)
def test_interval_warnings_obey_existing_support_and_selection_gates(
    support, coverage, sampling, expected
):
    state = parameters()
    _, bounds = apply(np.zeros(2), state, [0.9])
    predictions = pd.DataFrame({"id": [1, 2], **bounds})
    batch = SimpleNamespace(
        summary={
            "postprocessing": True,
            "target_unit": "units",
            "interval_columns": interval_columns(state, [0.9]),
        },
        key="id",
        predictions=predictions,
    )
    labels = {
        "frame": pd.DataFrame(
            {
                "id": [1, 2],
                "y_true": [10000.0, 10000.0],
                "eligible_at": pd.Timestamp("2026-01-01", tz="UTC"),
                "observed_at": pd.Timestamp("2026-01-02", tz="UTC"),
            }
        ),
        "metadata": {"sampling": sampling},
    }
    config = SimpleNamespace(
        target_unit="units",
        as_of=pd.Timestamp("2026-01-03", tz="UTC"),
        min_label_coverage=0.9,
        interval_coverage_slack=0.05,
    )
    perf = {"sufficient_support": support, "mature_label_coverage": coverage}
    findings = []
    add_monitoring(config, batch, labels, perf, findings)
    assert bool(findings) == expected
    assert perf["interval_metrics"]["c900000"]["empirical_coverage"] == 0.0


def test_no_mature_labels_and_no_label_request():
    state = parameters()
    _, bounds = apply(np.zeros(1), state, [0.9])
    batch = SimpleNamespace(
        summary={
            "postprocessing": True,
            "target_unit": "units",
            "interval_columns": interval_columns(state, [0.9]),
        },
        key="id",
        predictions=pd.DataFrame({"id": [1], **bounds}),
    )
    config = SimpleNamespace(
        target_unit="units",
        as_of=pd.Timestamp("2026-01-01", tz="UTC"),
        min_label_coverage=0.9,
        interval_coverage_slack=0.05,
    )
    labels = {
        "frame": pd.DataFrame(
            {
                "id": [1],
                "y_true": [np.nan],
                "eligible_at": pd.Timestamp("2026-02-01", tz="UTC"),
                "observed_at": [pd.NaT],
            }
        ),
        "metadata": {"sampling": "all_rows"},
    }
    perf = {"sufficient_support": False, "mature_label_coverage": None}
    add_monitoring(config, batch, labels, perf, [])
    assert perf["interval_metrics"]["c900000"]["empirical_coverage"] is None
    add_monitoring(config, batch, None, None, [])


def test_units_cannot_be_relabelled():
    batch = SimpleNamespace(summary={"postprocessing": True, "target_unit": "kg"})
    with pytest.raises(ValueError):
        add_monitoring(SimpleNamespace(target_unit="lb"), batch, None, None, [])
