import hashlib
import json

import numpy as np
import pandas as pd

from ml_template.training.run_config import SplitConfig
from ml_template.training.split import (
    compute_split_id,
    holdout_keys,
    make_split,
    split_policy,
)


def _data():
    return pd.DataFrame(
        {
            "udi": np.arange(1, 1001),
            "machine_failure": np.tile([0, 0, 0, 0, 1], 200),
        }
    )


def test_unset_cv_seed_preserves_legacy_split_id():
    cfg = SplitConfig()

    legacy_payload = json.dumps(
        {
            "silver": "silver_test",
            "policy": {
                "method": "stratified",
                "holdout_fraction": 0.2,
                "n_folds": 5,
                "seed": 42,
            },
            "key": "udi",
            "target": "machine_failure",
        },
        sort_keys=True,
    )
    expected = "split_" + hashlib.sha256(legacy_payload.encode()).hexdigest()[:12]

    assert compute_split_id("silver_test", cfg, "udi", "machine_failure") == expected
    assert "cv_seed" not in split_policy(cfg)


def test_changing_cv_seed_preserves_holdout_but_changes_folds():
    df = _data()
    original_cfg = SplitConfig(seed=42)
    changed_cfg = SplitConfig(seed=42, cv_seed=43)

    original = make_split(df, original_cfg, "udi", "machine_failure")
    changed = make_split(df, changed_cfg, "udi", "machine_failure")

    assert set(holdout_keys(original, "udi")) == set(holdout_keys(changed, "udi"))
    pd.testing.assert_series_equal(
        original["partition"],
        changed["partition"],
    )

    dev = original["partition"].eq("dev")
    assert (original.loc[dev, "fold"] != changed.loc[dev, "fold"]).any()

    assert compute_split_id(
        "silver_test", original_cfg, "udi", "machine_failure"
    ) != compute_split_id("silver_test", changed_cfg, "udi", "machine_failure")


def test_new_cv_seed_is_reproducible():
    df = _data()
    cfg = SplitConfig(seed=42, cv_seed=43)

    first = make_split(df, cfg, "udi", "machine_failure")
    second = make_split(df, cfg, "udi", "machine_failure")

    pd.testing.assert_frame_equal(first, second)


def test_explicit_original_seed_preserves_assignments():
    df = _data()

    legacy = make_split(df, SplitConfig(seed=42), "udi", "machine_failure")
    explicit = make_split(df, SplitConfig(seed=42, cv_seed=42), "udi", "machine_failure")

    pd.testing.assert_frame_equal(legacy, explicit)
