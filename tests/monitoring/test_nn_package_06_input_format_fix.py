"""Regression: Package 05 records the format only inside request."""

import pytest

from ml_template.monitoring.nn_monitoring_contract import _pipeline_input_format


@pytest.mark.parametrize(
    "state,expected",
    [
        ({"request": {"input_format": "canonical"}}, "canonical"),
        ({"request": {"input_format": "raw"}}, "raw"),
        ({}, None),
        ({"input_format": "canonical"}, None),
        ({"request": None}, None),
        ({"request": []}, None),
        ({"request": {}}, None),
        ({"request": {"input_format": "invalid"}}, None),
        ({"request": {"input_format": False}}, None),
    ],
)
def test_package_05_nested_request_format(state, expected):
    if expected is None:
        with pytest.raises(ValueError, match="Package 05"):
            _pipeline_input_format(state)
    else:
        assert "input_format" not in state
        assert _pipeline_input_format(state) == expected
