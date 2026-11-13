import pytest

from ml_template.data.contract import load_dataset_contract

TOP = """
data:
  entity_id: ID
  target: Y
roles:
  id: [ID]
  feature: [A, B]
  target: [Y]
silver_eda_contract:
  canonical_names: {ID: id, A: a, B: b, Y: y}
  canonical_dtypes: {id: int64, a: float64, b: category, y: int8}
"""

NESTED = """
data:
  entity_id: ID
  target: Y
bronze_source_contract:
  rows: 3
  roles:
    id: [ID]
    feature: [A, B]
    target: [Y]
silver_eda_contract:
  canonical_names: {ID: id, A: a, B: b, Y: y}
  canonical_dtypes: {id: int64, a: float64, b: category, y: int8}
"""


def _write(tmp_path, text=TOP):
    path = tmp_path / "c.yaml"
    path.write_text(text)
    return path


@pytest.mark.parametrize("text", [TOP, NESTED])
def test_contract_in_canonical_names(tmp_path, text):
    c = load_dataset_contract(_write(tmp_path, text))
    assert (c.entity_id, c.target) == ("id", "y")
    assert c.features == ("a", "b")
    assert c.dtypes == {"a": "float64", "b": "category"}
    assert set(c.excluded_from_features) == {"id", "y"}


def test_other_roles_are_ignored(tmp_path):
    text = TOP.replace("  target: [Y]\n", "  target: [Y]\n  outcome: [F1, F2]\n")
    c = load_dataset_contract(_write(tmp_path, text))  # F1, F2 have no canonical names
    assert c.features == ("a", "b")


def test_missing_roles_message(tmp_path):
    with pytest.raises(ValueError, match="roles"):
        load_dataset_contract(_write(tmp_path, NESTED.replace("  roles:", "  rolez:")))


def test_missing_canonical_name_rejected(tmp_path):
    with pytest.raises(ValueError, match="canonical name"):
        load_dataset_contract(_write(tmp_path, TOP.replace("B: b, ", "")))


def test_overlap_rejected(tmp_path):
    with pytest.raises(ValueError, match="both"):
        load_dataset_contract(
            _write(tmp_path, TOP.replace("feature: [A, B]", "feature: [A, B, Y]"))
        )


def test_single_target_only(tmp_path):
    text = TOP.replace("target: [Y]", "target: [Y, A]")
    with pytest.raises(ValueError, match="single-target"):
        load_dataset_contract(_write(tmp_path, text))


def test_target_mismatch_rejected(tmp_path):
    with pytest.raises(ValueError, match="disagrees"):
        load_dataset_contract(_write(tmp_path, TOP.replace("target: Y\nroles", "target: A\nroles")))
