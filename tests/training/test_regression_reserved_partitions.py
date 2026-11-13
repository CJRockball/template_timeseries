from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ml_template.training.regression_partitions import reserve, validate_assignments


def cfg(**kwargs):
    return SimpleNamespace(
        **{
            **dict(
                seed=42,
                n_folds=3,
                method="random",
                calibration_fraction=0.15,
                recalibration_fraction=0.1,
            ),
            **kwargs,
        }
    )


def test_roles_disjoint_and_holdout_unchanged():
    dev, parts = reserve(100, np.arange(80), cfg())
    assert set(np.flatnonzero(parts == "holdout")) == set(range(80, 100))
    assert len(dev) == 55
    assert np.sum(parts == "calibration") == 15 and np.sum(parts == "recalibration") == 10
    assert not set(dev) & set(np.flatnonzero(parts != "dev"))
    np.testing.assert_array_equal(parts, reserve(100, np.arange(80)[::-1], cfg())[1])


def test_unset_roles_preserve_membership():
    dev, parts = reserve(10, [1, 3, 5, 7], cfg(calibration_fraction=0, recalibration_fraction=0))
    assert dev.tolist() == [1, 3, 5, 7] and set(parts) == {"dev", "holdout"}


@pytest.mark.parametrize(
    "policy", [cfg(method="stratified"), cfg(calibration_fraction=0.7, recalibration_fraction=0.1)]
)
def test_invalid_reservations(policy):
    with pytest.raises(ValueError):
        reserve(100, np.arange(80), policy)


def test_manifest_counts():
    _, parts = reserve(100, np.arange(80), cfg())
    assignments = pd.DataFrame({"partition": parts})
    manifest = {"n_calibration": 15, "n_recalibration": 10}
    validate_assignments(cfg(), assignments, manifest)
    with pytest.raises(ValueError):
        validate_assignments(cfg(), assignments, {**manifest, "n_calibration": 14})
