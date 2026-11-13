"""Regression protection for Patch 4's diagnostic and parity boundaries."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ml_template.evaluation import regression_postprocessing as postprocessing
from ml_template.evaluation.regression_diagnostics import align_oof


@pytest.fixture
def world():
    keys = pd.Index(range(1, 8), name="id")
    X = pd.DataFrame({"x": np.arange(7, dtype=float)}, index=keys)
    y = pd.Series([0.0, 2.0, 4.0, 6.0, np.nan, np.nan, np.nan], index=keys)
    assignments = pd.DataFrame(
        {
            "id": keys,
            "partition": ["dev", "dev", "dev", "dev", "holdout", "calibration", "recalibration"],
            "fold": [0, 0, 1, 1, -1, -1, -1],
        }
    )
    oof = pd.DataFrame(
        {"id": [1, 2, 3, 4], "fold": [0, 0, 1, 1], "y_pred_oof": [0.0, 2.0, 4.0, 6.0]}
    )
    return X, y, assignments, oof


def test_reserved_rows_are_not_diagnostic_targets(world):
    X, y, assignments, oof = world
    result = align_oof(oof, X, y, assignments, "id")
    assert result.id.tolist() == [1, 2, 3, 4]
    assert result.y_true.notna().all()
    assert result.residual.eq(0).all()
    assert y.loc[[5, 6, 7]].isna().all()


def test_development_only_inputs_remain_supported(world):
    X, y, assignments, oof = world
    result = align_oof(oof.iloc[::-1], X.loc[[1, 2, 3, 4]], y.loc[[1, 2, 3, 4]], assignments, "id")
    assert result.id.tolist() == [1, 2, 3, 4]


def test_legacy_partition_vocabulary_remains_supported(world):
    X, y, assignments, oof = world
    assignments = assignments.loc[assignments.id <= 5]
    result = align_oof(oof, X, y, assignments, "id")
    assert len(result) == 4


@pytest.mark.parametrize("role", ["holdout", "calibration", "recalibration"])
def test_oof_cannot_include_any_reserved_population(world, role):
    X, y, assignments, oof = world
    key = int(assignments.loc[assignments.partition.eq(role), "id"].iloc[0])
    bad = pd.concat(
        [oof, pd.DataFrame({"id": [key], "fold": [0], "y_pred_oof": [1.0]})], ignore_index=True
    )
    with pytest.raises(ValueError, match="exactly development keys"):
        align_oof(bad, X, y, assignments, "id")


@pytest.mark.parametrize("role", ["holdout", "calibration", "recalibration"])
@pytest.mark.parametrize("bad_fold", [0, np.nan])
def test_non_development_fold_must_be_minus_one(world, role, bad_fold):
    X, y, assignments, oof = world
    assignments = assignments.copy()
    assignments.loc[assignments.partition.eq(role), "fold"] = bad_fold
    with pytest.raises(ValueError, match="fold -1"):
        align_oof(oof, X, y, assignments, "id")


def test_unknown_role_still_rejected(world):
    X, y, assignments, oof = world
    assignments = assignments.copy()
    assignments.loc[assignments.partition.eq("calibration"), "partition"] = "calibraton"
    with pytest.raises(ValueError, match="Unknown split partition"):
        align_oof(oof, X, y, assignments, "id")


def test_stored_fold_binding_still_required(world):
    X, y, assignments, oof = world
    oof = oof.copy()
    oof.loc[0, "fold"] = 1
    with pytest.raises(ValueError, match="fold identities differ"):
        align_oof(oof, X, y, assignments, "id")


def test_minimal_raw_parity_context_needs_no_provenance():
    material = {"prepared": SimpleNamespace(key_col="id", hold_keys=[1, 2])}
    assert postprocessing.reference_columns(material) == []


def test_explicit_raw_provenance_needs_no_spec():
    assert postprocessing.reference_columns({"prepared": SimpleNamespace(provenance={})}) == []


def test_postprocessed_reference_retains_selected_bound_columns(monkeypatch):
    parameters = {"opaque": "state"}
    calls = []

    def columns(state, chosen):
        calls.append((state, chosen))
        return {"c900000": {"lower": "pi_lower_c900000", "upper": "pi_upper_c900000"}}

    monkeypatch.setattr(postprocessing, "interval_columns", columns)
    p = SimpleNamespace(
        provenance={"postprocessing": {"state": {"parameters": parameters}}},
        spec=SimpleNamespace(uncertainty=SimpleNamespace(enabled=True, coverages=[0.9])),
    )
    assert postprocessing.reference_columns({"prepared": p}) == [
        "y_pred_raw",
        "pi_lower_c900000",
        "pi_upper_c900000",
    ]
    assert calls == [(parameters, [0.9])]


def test_declared_but_malformed_postprocessing_does_not_fall_back_to_raw():
    p = SimpleNamespace(provenance={"postprocessing": {}})
    with pytest.raises(KeyError):
        postprocessing.reference_columns({"prepared": p})
