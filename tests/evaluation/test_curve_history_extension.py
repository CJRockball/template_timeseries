"""History provenance, curve configuration and diagnostic noninterference."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ml_template.evaluation.curve_history import (
    _plot_inner_axes,
    capture_inner_history,
    history_table,
    run_regression_curve_diagnostics,
    save_inner_history,
)
from ml_template.evaluation.learning_curves import _outer_curve_axes


def recorded():
    return history_table(
        {"validation_0": {"rmse": [4.0, 3.0, 2.0]}, "validation_1": {"rmse": [4.5, 2.5, 3.0]}},
        "rmse",
        2,
        inner=True,
    )


def test_original_history_provenance():
    table = capture_inner_history(SimpleNamespace(early_stopping_history=recorded()), 2, 4)
    assert set(table.dataset) == {"inner_fit", "inner_stop"}
    assert set(table.fold) == {4} and set(table.selected_round) == {2}
    assert set(table.direction) == {"min"}
    assert table.loc[table.dataset == "inner_stop", "value"].tolist() == [4.5, 2.5, 3.0]
    assert capture_inner_history(SimpleNamespace(), 2, 0) is None


def test_invalid_history_rejected():
    with pytest.raises(ValueError, match="disagrees"):
        capture_inner_history(SimpleNamespace(early_stopping_history=recorded()), 1, 0)
    with pytest.raises(ValueError):
        history_table(
            {"validation_0": {"rmse": [1.0]}, "validation_1": {"rmse": [np.nan]}},
            "rmse",
            1,
            inner=True,
        )
    with pytest.raises(ValueError):
        history_table(
            {"validation_0": {"rmse": [1.0]}, "validation_1": {"rmse": [1.0, 2.0]}},
            "rmse",
            1,
            inner=True,
        )


def test_plot_markers_have_correct_provenance():
    import matplotlib.pyplot as plt

    table = capture_inner_history(SimpleNamespace(early_stopping_history=recorded()), 2, 0)
    fig, axes = _plot_inner_axes(table)
    assert len(axes[0][0].lines) == 3
    assert axes[0][0].lines[-1].get_label() == "selected on inner_stop"
    plt.close(fig)
    outer = table.rename(columns={"selected_round": "best_round"}).replace(
        {"dataset": {"inner_fit": "train", "inner_stop": "val"}}
    )
    fig, axes = _outer_curve_axes(outer)
    assert len(axes[0][0].lines) == 2
    assert {line.get_label() for line in axes[0][0].lines} == {"outer_train", "outer_validation"}
    plt.close(fig)
    fig, axes = _outer_curve_axes(outer, show_inner_reference=True)
    assert "reference only" in axes[0][0].lines[-1].get_label()
    plt.close(fig)


def test_history_artifact_writer(tmp_path):
    pytest.importorskip("pyarrow")
    table = capture_inner_history(SimpleNamespace(early_stopping_history=recorded()), 2, 0)
    save_inner_history(tmp_path, SimpleNamespace(early_stopping_curves=table))
    pd.testing.assert_frame_equal(
        table, pd.read_parquet(tmp_path / "early_stopping_curves.parquet")
    )
    metadata = json.loads((tmp_path / "early_stopping_curves_metadata.json").read_text())
    assert metadata["used_holdout"] is False
    assert "original" in metadata["selection_source"]


def config_raw():
    return {
        "experiment_name": "curve_test",
        "task": "regression",
        "data": {
            "dataset_name": "test",
            "dataset_config": "unused.yaml",
            "silver_dataset_version_id": "silver_test",
        },
        "split": {"method": "random", "n_folds": 3},
        "model": {
            "type": "xgb",
            "params": {
                "objective": "reg:squarederror",
                "eval_metric": "mae",
                "n_estimators": 20,
                "early_stopping_rounds": 3,
            },
        },
    }


def test_metric_defaults_and_hash():
    from ml_template.training.run_config import RunConfig, config_hash

    base = config_raw()
    cfg = RunConfig.model_validate(base)
    assert cfg.analysis.learning_curve_metric == "mae"
    enabled = RunConfig.model_validate({**base, "analysis": {"learning_curves": True}})
    assert enabled.analysis.learning_curve_metric == "mae"
    assert config_hash(enabled) == config_hash(cfg)
    assert RunConfig.model_validate_json(enabled.model_dump_json()) == enabled
    with pytest.raises(ValueError, match="incompatible"):
        RunConfig.model_validate(
            {**base, "analysis": {"learning_curves": True, "learning_curve_metric": "logloss"}}
        )
    # Old disabled-analysis snapshots remain readable, even with the inert old default.
    RunConfig.model_validate(
        {**base, "analysis": {"learning_curves": False, "learning_curve_metric": "logloss"}}
    )


def test_cv_capture_and_holdout_exclusion():
    from ml_template.training.cross_validation import cross_validate
    from ml_template.training.run_config import SplitConfig
    from ml_template.training.split import make_split

    df = pd.DataFrame(
        {"id": np.arange(90), "x": np.linspace(0, 1, 90), "y": np.linspace(-3.5, 50.5, 90)}
    )
    assign = make_split(df, SplitConfig(method="random", n_folds=3), "id", "y")
    hold = set(assign.loc[assign.partition == "holdout", "id"])
    X, y = df.set_index("id")[["x"]], df.set_index("id").y

    class Fake:
        max_rounds = 20
        early_stopping_history = None

        def fit_with_validation(self, Xt, yt, Xv, yv):
            assert not set(Xt.index) & set(Xv.index)
            assert not (set(Xt.index) | set(Xv.index)) & hold
            self.early_stopping_history = recorded()
            return 2

        def refit(self, X, y, n_rounds):
            assert not set(X.index) & hold
            self.mean = float(y.mean())

        def predict(self, X):
            return np.full(len(X), self.mean)

    result = cross_validate(
        Fake,
        X,
        y,
        assign,
        "id",
        validation_fraction=0.2,
        seed=42,
        metric_names=["rmse", "mae", "r2"],
        alert_rates=[],
        task="regression",
    )
    assert set(result.early_stopping_curves.fold) == {0, 1, 2}
    assert len(result.early_stopping_curves) == 18
    assert set(result.oof.id) == set(X.index) - hold
    assert result.n_rounds_final == 2


def test_optional_diagnostics_do_not_mutate_cv(tmp_path):
    pytest.importorskip("pyarrow")
    X = pd.DataFrame({"x": np.arange(12.0)}, index=np.arange(12))
    y = pd.Series(np.linspace(-1.5, 8.5, 12), index=X.index)
    assign = pd.DataFrame(
        {
            "id": np.arange(12),
            "partition": ["dev"] * 10 + ["holdout"] * 2,
            "fold": [0] * 5 + [1] * 5 + [-1] * 2,
        }
    )
    calls = []

    class Fake:
        max_rounds = 20

        def learning_curve(self, Xt, yt, Xv, yv, n_rounds, metric):
            assert not (set(Xt.index) | set(Xv.index)) & {10, 11}
            calls.append((set(Xt.index), set(Xv.index)))
            return pd.DataFrame(
                {
                    "round": [1, 2, 1, 2],
                    "dataset": ["train", "train", "val", "val"],
                    "metric": metric,
                    "value": [3.0, 2.0, 4.0, 3.0],
                }
            )

    cv = SimpleNamespace(
        folds=pd.DataFrame({"fold": [0, 1], "best_round": [2, 2]}),
        early_stopping_curves=None,
        n_rounds_final=2,
        oof=pd.DataFrame({"y_pred_oof": [1.25, 2.5]}),
    )
    original = cv.oof.copy(deep=True)
    inputs = SimpleNamespace(X=X, y=y, assignments=assign, contract=SimpleNamespace(entity_id="id"))
    cfg = SimpleNamespace(
        analysis=SimpleNamespace(
            learning_curves=False, learning_curve_metric="rmse", learning_curve_rounds=2
        )
    )
    import logging

    run_regression_curve_diagnostics(cfg, tmp_path, inputs, cv, Fake, logging.getLogger(__name__))
    assert not calls and not list(tmp_path.iterdir())
    cfg.analysis.learning_curves = True
    run_regression_curve_diagnostics(cfg, tmp_path, inputs, cv, Fake, logging.getLogger(__name__))
    assert len(calls) == 2
    pd.testing.assert_frame_equal(cv.oof, original)
    assert cv.n_rounds_final == 2
    assert not (tmp_path / "analysis_errors.json").exists()
    metadata = json.loads((tmp_path / "learning_curves_metadata.json").read_text())
    assert (
        metadata["curve_role"] == "outer_diagnostic_refit" and metadata["early_stopping"] is False
    )


def test_real_xgb_history_and_prediction_parity():
    pytest.importorskip("xgboost", minversion="2.0")
    from ml_template.models.xgb_regressor import XGBRegressionAdapter
    from ml_template.training.run_config import XGBRegressionParams

    rng = np.random.default_rng(71)
    X = pd.DataFrame({"a": rng.normal(size=90), "b": rng.normal(size=90)})
    y = pd.Series(5 * X.a - 2 * X.b + 0.25)
    params = XGBRegressionParams(
        n_estimators=20, early_stopping_rounds=3, subsample=1.0, colsample_bytree=1.0
    )
    first, second = XGBRegressionAdapter(params, 42), XGBRegressionAdapter(params, 42)
    best = first.fit_with_validation(X.iloc[:60], y.iloc[:60], X.iloc[60:], y.iloc[60:])
    assert set(first.early_stopping_history.dataset) == {"inner_fit", "inner_stop"}
    # Reference model matches the old validation-only monitoring setup.
    reference = second._build(second.max_rounds, params.early_stopping_rounds)
    reference.fit(X.iloc[:60], y.iloc[:60], eval_set=[(X.iloc[60:], y.iloc[60:])], verbose=False)
    assert best == int(reference.best_iteration) + 1
    np.testing.assert_allclose(first.predict(X), reference.predict(X), rtol=0, atol=1e-6)
    before = first.predict(X).copy()
    first.learning_curve(X.iloc[:60], y.iloc[:60], X.iloc[60:], y.iloc[60:], 5, "rmse")
    np.testing.assert_array_equal(first.predict(X), before)
