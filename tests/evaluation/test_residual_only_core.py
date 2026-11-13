"""Pure residual summaries, config and partition integrity tests."""

import copy
import hashlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ml_template.evaluation.residual_analysis_run import (
    ResidualConfig,
    split_identity,
    validate_inventory,
)
from ml_template.evaluation.residual_artifacts import augment_summary, check_keys, statistics


def inventory():
    a = pd.DataFrame(
        {
            "id": range(8),
            "partition": ["dev"] * 4 + ["holdout"] * 2 + ["calibration", "recalibration"],
            "fold": [0, 1, 0, 1, -1, -1, -1, -1],
        }
    )
    o = pd.DataFrame({"id": [0, 1, 2, 3], "fold": [0, 1, 0, 1], "y_pred_oof": [0.2, 1.2, 2.2, 3.2]})
    m = {
        "n_rows": 8,
        "n_holdout": 2,
        "n_calibration": 1,
        "n_recalibration": 1,
        "fold_sizes": {"0": 2, "1": 2},
        "assignments_sha256": hashlib.sha256(a.to_csv(index=False).encode()).hexdigest(),
    }
    return a, o, m


def test_known_signed_residual_statistics():
    s = statistics(pd.DataFrame({"y_true": [1.0, 3.0, 5.0], "y_pred_oof": [0.0, 5.0, 4.0]}))
    assert s["mean_residual"] == 0
    assert s["rmse"] == pytest.approx(np.sqrt(2))
    assert s["mae"] == pytest.approx(4 / 3)
    assert s["r2"] == pytest.approx(0.25)
    assert s["residual_quantiles"]["0.5"] == 1


def test_constant_target_r2_is_unavailable_not_infinite():
    s = statistics(pd.DataFrame({"y_true": [2.0, 2.0], "y_pred_oof": [2.0, 3.0]}))
    assert s["r2"] is None


@pytest.mark.parametrize(
    "values", [[], [True, False], ["1", "2"], [1.0, np.nan], [1.0, np.inf], [1j, 2j]]
)
def test_invalid_summary_vectors(values):
    with pytest.raises(ValueError):
        statistics(pd.DataFrame({"y_true": values, "y_pred_oof": values}))


def test_per_fold_summaries_are_separate():
    frame = pd.DataFrame(
        {"fold": [0, 0, 1, 1], "y_true": [1.0, 2.0, 4.0, 5.0], "y_pred_oof": [0.0, 1.0, 6.0, 7.0]}
    )
    result = SimpleNamespace(residuals=frame, summary={})
    augment_summary(result)
    assert result.summary["pooled"]["mean_residual"] == -0.5
    assert [r["mean_residual"] for r in result.summary["fold_summaries"]] == [1.0, -2.0]


@pytest.mark.parametrize("keys", [[], [1, 1], [1, None], [True, False], [1.0, 2.0], ["1", 2]])
def test_bad_keys_rejected(keys):
    with pytest.raises(ValueError):
        check_keys(keys)


@pytest.mark.parametrize("keys", [[1, 2], ["a", "b"]])
def test_supported_key_types(keys):
    check_keys(keys)


def test_shuffled_oof_is_joined_by_key():
    a, o, m = inventory()
    assert validate_inventory(a, o.iloc[::-1], "id", 2, m) == [0, 1, 2, 3]


@pytest.mark.parametrize(
    "kind", ["missing", "duplicate", "holdout", "wrongfold", "nonfinite", "boolean", "extra"]
)
def test_bad_oof_rejected(kind):
    a, o, m = inventory()
    if kind == "missing":
        o = o.iloc[:-1]
    elif kind == "duplicate":
        o.loc[1, "id"] = 0
    elif kind == "holdout":
        o.loc[0, "id"] = 4
    elif kind == "wrongfold":
        o.loc[0, "fold"] = 1
    elif kind == "nonfinite":
        o.loc[0, "y_pred_oof"] = np.inf
    elif kind == "boolean":
        o["y_pred_oof"] = True
    else:
        o["p_oof"] = 0.5
    with pytest.raises(ValueError):
        validate_inventory(a, o, "id", 2, m)


@pytest.mark.parametrize(
    "field,value",
    [
        ("n_rows", 9),
        ("n_holdout", 3),
        ("n_calibration", 0),
        ("fold_sizes", {"0": 1, "1": 3}),
        ("assignments_sha256", "bad"),
    ],
)
def test_split_manifest_mismatches_rejected(field, value):
    a, o, m = inventory()
    m[field] = value
    with pytest.raises(ValueError):
        validate_inventory(a, o, "id", 2, m)


def test_reserved_fold_ids_rejected():
    a, o, m = inventory()
    a.loc[7, "fold"] = 0
    with pytest.raises(ValueError):
        validate_inventory(a, o, "id", 2, m)


def test_config_optional_and_separate_from_training():
    cfg = ResidualConfig(run_id="run_test", target_unit="MPa")
    assert cfg.enabled is False and cfg.feature_columns == []


@pytest.mark.parametrize(
    "options",
    [
        {"enabled": "true"},
        {"error_sample_size": True},
        {"target_unit": "target_units"},
        {"target_unit": " "},
        {"feature_columns": ["x", "x"]},
        {"permutation": True},
    ],
)
def test_bad_config_rejected(options):
    raw = {"run_id": "run_test", "target_unit": "MPa", **options}
    with pytest.raises(ValueError):
        ResidualConfig.model_validate(raw)


def test_split_identity_matches_original_serialization():
    policy = {"method": "random", "n_folds": 2, "seed": 42, "holdout_fraction": 0.2}
    assert split_identity("silver_test", policy, "id", "y").startswith("split_")
    assert split_identity("silver_test", policy, "id", "y") != split_identity(
        "silver_test", {**policy, "seed": 43}, "id", "y"
    )


def test_statistics_do_not_modify_input():
    f = pd.DataFrame({"y_true": [-1.5, 0.25, 3.0], "y_pred_oof": [-1.0, 0.5, 2.0]})
    before = copy.deepcopy(f)
    statistics(f)
    pd.testing.assert_frame_equal(f, before)
