"""Pure regression diagnostics; no holdout data, DB or fitting required."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ml_template.evaluation.regression_analysis_run import ensure_analysis_destination
from ml_template.evaluation.regression_diagnostics import (
    align_oof,
    compute_diagnostics,
    regression_permutation_importance,
)


@pytest.fixture
def world():
    index = pd.Index(np.arange(12), name="id")
    X = pd.DataFrame(
        {
            "signal": np.linspace(-2.0, 3.0, 12),
            "noise": np.zeros(12),
            "cat": pd.Categorical(["a", "b"] * 6),
        },
        index=index,
    )
    y = 3 * X.signal + 0.25
    # Holdout values are deliberately unusable: diagnostics must never inspect them.
    y.iloc[-2:] = np.nan
    X.loc[10:, "signal"] = np.inf
    assignments = pd.DataFrame(
        {
            "id": index,
            "partition": ["dev"] * 10 + ["holdout"] * 2,
            "fold": [0] * 5 + [1] * 5 + [-1] * 2,
        }
    )
    oof = pd.DataFrame(
        {
            "id": np.arange(10),
            "fold": [0] * 5 + [1] * 5,
            "y_pred_oof": 3 * X.loc[:9, "signal"].to_numpy() + 0.25,
        }
    )
    calls = []

    class Adapter:
        def predict(self, frame):
            assert not set(frame.index) & {10, 11}
            assert isinstance(frame.cat.dtype, pd.CategoricalDtype)
            calls.append(frame.index.tolist())
            result = 3 * frame.signal.to_numpy() + 0.25
            frame.iloc[0, 0] = 999999
            return result

        def fit(self, *args):
            raise AssertionError("Diagnostics must not fit")

        refit = fit

    return X, y, assignments, oof, Adapter, calls


def test_fold_validation_only_importance_and_no_mutation(world):
    X, y, a, oof, Adapter, calls = world
    original = X.copy(deep=True)
    result = compute_diagnostics(
        oof,
        X,
        y,
        a,
        "id",
        model_loader=lambda fold: Adapter(),
        n_repeats=5,
        max_rows=4,
        error_sample_size=3,
        target_unit="MPa",
    )
    pd.testing.assert_frame_equal(X, original)
    assert set(result.residuals.id) == set(range(10))
    assert result.summary["metric_values"]["rmse"] == 0
    pooled = result.importance[result.importance.fold == -1].set_index("feature")
    assert pooled.loc["signal", "mean"] > 0
    assert pooled.loc["noise", "mean"] == 0
    assert len(result.repeats) == 2 * 3 * 5
    assert all(len(keys) == 4 for keys in result.samples.values())
    assert all(not set(keys) & {10, 11} for keys in calls)
    assert result.summary["holdout_labels_loaded"] is False


def test_repeat_and_fold_variability_are_distinct(world):
    X, y, a, oof, Adapter, _ = world
    result = compute_diagnostics(oof, X, y, a, "id", model_loader=lambda f: Adapter(), n_repeats=4)
    for fold in [0, 1]:
        values = result.repeats.loc[
            (result.repeats.fold == fold) & (result.repeats.feature == "signal"), "importance"
        ]
        row = result.importance.loc[
            (result.importance.fold == fold) & (result.importance.feature == "signal")
        ].iloc[0]
        assert row.repeat_std == pytest.approx(values.std(ddof=1))
    means = result.importance.loc[
        (result.importance.fold >= 0) & (result.importance.feature == "signal"), "mean"
    ]
    pooled = result.importance.loc[
        (result.importance.fold == -1) & (result.importance.feature == "signal")
    ].iloc[0]
    assert pooled.between_fold_std == pytest.approx(means.std(ddof=1))
    assert np.isnan(pooled.repeat_std)


def test_uniform_subsampling_is_reproducible_on_unique_continuous_targets(world):
    X, y, _, _, Adapter, _ = world
    t1, r1, k1 = regression_permutation_importance(
        Adapter(), X.iloc[:10], y.iloc[:10], max_rows=4, n_repeats=3, seed=7
    )
    t2, r2, k2 = regression_permutation_importance(
        Adapter(), X.iloc[:10], y.iloc[:10], max_rows=4, n_repeats=3, seed=7
    )
    assert k1 == k2
    pd.testing.assert_frame_equal(r1, r2)
    pd.testing.assert_frame_equal(t1, t2)


def test_negative_importance_is_not_clipped():
    X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 100)})
    y = -X.x
    model = SimpleNamespace(predict=lambda frame: frame.x.to_numpy())
    table, _, _ = regression_permutation_importance(model, X, y, n_repeats=8)
    assert table["mean"].iloc[0] < 0


def test_keyed_error_sample_and_sign(world):
    X, y, a, oof, _, _ = world
    deviations = np.array([-2.0, -1.0, 0.0, 1.0, 3.0] * 2)
    oof = oof.copy()
    oof["y_pred_oof"] = y.iloc[:10].to_numpy() - deviations
    result = compute_diagnostics(
        oof.sample(frac=1, random_state=7), X, y, a, "id", permutation=False, error_sample_size=2
    )
    assert result.error_sample.id.tolist() == [4, 9]
    assert set(result.error_sample.error_kind) == {"underprediction"}
    np.testing.assert_array_equal(result.error_sample.signal, result.error_sample.id.map(X.signal))
    empty = compute_diagnostics(oof, X, y, a, "id", permutation=False, error_sample_size=0)
    assert empty.error_sample.empty and not empty.registry_rows


@pytest.mark.parametrize("change", ["holdout", "duplicate", "fold"])
def test_bad_oof_rejected(world, change):
    X, y, a, oof, _, _ = world
    oof = oof.copy()
    if change == "holdout":
        oof.loc[0, "id"] = 10
    elif change == "duplicate":
        oof.loc[0, "id"] = 1
    else:
        oof.loc[0, "fold"] = 1
    with pytest.raises(ValueError):
        align_oof(oof, X, y, a, "id")


def test_saved_fold_prediction_mismatch_rejected(world):
    X, y, a, oof, _, _ = world
    with pytest.raises(ValueError, match="original OOF"):
        compute_diagnostics(
            oof,
            X,
            y,
            a,
            "id",
            model_loader=lambda f: SimpleNamespace(predict=lambda frame: np.zeros(len(frame))),
        )


@pytest.mark.parametrize(
    "kwargs",
    [{"n_repeats": 0}, {"n_repeats": True}, {"max_rows": 0}, {"seed": -1}, {"metric": "roc_auc"}],
)
def test_invalid_permutation_settings(world, kwargs):
    X, y, _, _, Adapter, _ = world
    with pytest.raises(ValueError):
        regression_permutation_importance(Adapter(), X.iloc[:5], y.iloc[:5], **kwargs)


def test_do_not_nest_analysis_in_sealed_artifacts(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(ValueError):
        ensure_analysis_destination(run / "analysis", run)
    release = tmp_path / "release"
    release.mkdir()
    (release / "manifest.json").write_text("{}")
    with pytest.raises(ValueError):
        ensure_analysis_destination(release / "analysis", run)
    ensure_analysis_destination(tmp_path / "analyses/new", run)


def test_standalone_options_do_not_rewrite_or_depend_on_old_analysis_flags():
    from ml_template.evaluation.regression_analysis_run import resolve_options

    cfg = SimpleNamespace(
        seed=42,
        metrics=SimpleNamespace(primary="mae"),
        analysis=SimpleNamespace(
            permutation_importance=False,
            error_sample_size=0,
            permutation_repeats=10,
            permutation_max_rows=50000,
        ),
    )
    options, plots = resolve_options(cfg, target_unit="MPa", plots=False)
    assert options["metric"] == "mae" and options["permutation"] is True
    assert options["error_sample_size"] == 25 and not plots
    assert cfg.analysis.error_sample_size == 0 and cfg.analysis.permutation_importance is False


def test_standalone_preflight_rejects_noninteger_options():
    from ml_template.evaluation.regression_analysis_run import resolve_options

    cfg = SimpleNamespace(
        seed=42,
        metrics=SimpleNamespace(primary="rmse"),
        analysis=SimpleNamespace(permutation_repeats=10, permutation_max_rows=50000),
    )
    with pytest.raises(ValueError):
        resolve_options(cfg, repeats=True)
    with pytest.raises(ValueError):
        resolve_options(cfg, max_rows=1.5)
