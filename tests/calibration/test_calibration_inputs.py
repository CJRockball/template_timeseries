import pandas as pd
import pytest

from ml_template.training.calibration_inputs import align_development


def fixture():
    split = pd.DataFrame(
        {"id": [1, 2, 3, 4, 5], "partition": ["dev"] * 4 + ["holdout"], "fold": [0, 1, 0, 1, -1]}
    )
    oof = pd.DataFrame({"id": [4, 2, 1, 3], "fold": [1, 1, 0, 0], "p_oof": [0.8, 0.7, 0.1, 0.2]})
    target = pd.DataFrame({"id": [3, 1, 4, 2], "target": [0, 0, 1, 1]})
    return oof, split, target


def test_join_is_keyed_not_positional():
    oof, split, target = fixture()
    frame = align_development(oof, split, target, "id", "target")
    assert frame.id.tolist() == [1, 2, 3, 4]
    assert frame.y_true.tolist() == [0, 1, 0, 1]
    assert frame.p_raw.tolist() == [0.1, 0.7, 0.2, 0.8]


@pytest.mark.parametrize(
    "mutation", ["holdout", "missing", "duplicate", "wrong_fold", "bad_target"]
)
def test_bad_alignment_rejected(mutation):
    oof, split, target = fixture()
    if mutation == "holdout":
        oof.loc[0, "id"] = 5
    elif mutation == "missing":
        target = target.iloc[:-1]
    elif mutation == "duplicate":
        oof.loc[0, "id"] = oof.loc[1, "id"]
    elif mutation == "wrong_fold":
        oof.loc[0, "fold"] = 0
    else:
        target.loc[0, "target"] = 2
    with pytest.raises(ValueError):
        align_development(oof, split, target, "id", "target")
