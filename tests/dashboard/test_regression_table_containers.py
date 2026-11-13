import numpy as np
import pandas as pd
import pytest


def test_regression_table_accepts_container_parameters():
    pytest.importorskip("dash")

    from ml_template.dashboard.pages.regression_run_page import table

    frame = pd.DataFrame(
        {
            "parameter": [
                "hidden_dims",
                "array_value",
                "preprocessing",
                "empty_list",
                "missing",
            ],
            "value": [
                [64, 32],
                np.array([64, 32]),
                {"numeric": ["cement", "age"]},
                [],
                pd.NA,
            ],
        }
    )

    rendered = table(frame)
    html_table = rendered.children[0]
    body = html_table.children[1]
    cells = [row.children[1].children for row in body.children]

    assert cells == [
        "[64, 32]",
        "[64 32]",
        "{'numeric': ['cement', 'age']}",
        "[]",
        "—",
    ]
