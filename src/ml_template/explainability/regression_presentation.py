"""Read-only validation of saved regression explanations; no inference imports."""

from pathlib import Path

import numpy as np
import pandas as pd

from ml_template.explainability.artifacts import fingerprint, read_json, verify_explanation
from ml_template.explainability.regression_result import (
    RegressionTolerance,
    build_regression_explanation,
    global_summary,
)


def load_view(record, summary, manifest, directory, *, max_rows):
    from ml_template.explainability.presentation_data import ExplanationView

    directory = Path(directory)
    if (
        summary.get("schema_version") != 2
        or summary.get("task") != "regression"
        or summary.get("artifact_state") != "complete"
        or summary.get("output_scale") != "prediction_target_units"
        or summary.get("method") != "xgboost_native_tree_shap"
        or summary.get("feature_dependence") != "tree_path_dependent"
        or summary.get("persisted_scoring_batch_verified") is not True
        or summary.get("labels_loaded") is not False
        or not isinstance(summary.get("target_unit"), str)
        or not summary["target_unit"].strip()
        or type(summary.get("n_rows")) is not int
        or not 0 < summary["n_rows"] <= max_rows
    ):
        raise ValueError("unsupported regression explanation semantics or row limit")
    rows = pd.read_parquet(directory / "rows.parquet")
    required = {
        "base_value",
        "y_pred_raw",
        "y_pred_release",
        "y_pred_reconstructed",
        "additivity_residual",
        "release_id",
        "scoring_batch_id",
        "explanation_id",
    }
    declared = summary.get("postprocessing")
    if declared is not None:
        required.add("y_pred_explained")
        required.update(
            name
            for item in summary["interval_columns"].values()
            for name in (item["lower"], item["upper"])
        )
    if not rows.columns.is_unique or set(rows) != required or len(rows) != summary["n_rows"]:
        raise ValueError("unexpected regression explanation row schema")
    keys = rows.index.tolist()
    if not (
        all(isinstance(k, str) for k in keys)
        or all(
            isinstance(k, (int, np.integer)) and not isinstance(k, (bool, np.bool_)) for k in keys
        )
    ):
        raise ValueError("unsupported entity key types")
    for name in ("release_id", "scoring_batch_id", "explanation_id"):
        if not rows[name].eq(summary[name]).all():
            raise ValueError("row identity differs")
    features = pd.read_parquet(directory / "features.parquet")
    contributions = pd.read_parquet(directory / "contributions.parquet")
    if not features.index.equals(rows.index) or not contributions.index.equals(rows.index):
        raise ValueError("explanation entity order differs")
    if (
        not features.columns.equals(contributions.columns)
        or features.shape[1] != summary["n_features"]
    ):
        raise ValueError("estimator feature schema differs")
    verification = read_json(directory / "verification.json")
    if verification.get("passed") is not True or verification.get("labels_loaded") is not False:
        raise ValueError("unverified regression explanation")
    tol = RegressionTolerance(**verification["tolerance"])
    result = build_regression_explanation(
        features,
        contributions,
        rows.base_value,
        rows["y_pred_explained"] if declared is not None else rows.y_pred_raw,
        tolerance=tol,
    )
    if declared is not None:
        from ml_template.models.regression_postprocessing import apply, interval_columns

        point, bounds = apply(
            rows.y_pred_raw.to_numpy(float), declared["parameters"], declared["coverages"]
        )
        if summary.get("explained_prediction_column") != "y_pred_explained" or summary[
            "interval_columns"
        ] != interval_columns(declared["parameters"], declared["coverages"]):
            raise ValueError("explained output or interval schema differs")
        for name, expected_values in {
            "y_pred_release": point,
            "y_pred_explained": point,
            **bounds,
        }.items():
            if not np.allclose(
                rows[name].to_numpy(float),
                expected_values,
                atol=tol.prediction_atol,
                rtol=tol.prediction_rtol,
            ):
                raise ValueError("postprocessed explanation rows differ")
    elif not np.allclose(
        rows.y_pred_release.to_numpy(float),
        rows.y_pred_raw.to_numpy(float),
        atol=tol.prediction_atol,
        rtol=tol.prediction_rtol,
    ):
        raise ValueError("raw release prediction differs")
    for name in ("y_pred_reconstructed", "additivity_residual"):
        if not np.allclose(
            rows[name].to_numpy(float),
            result.rows[name].to_numpy(float),
            atol=tol.prediction_atol,
            rtol=tol.prediction_rtol,
        ):
            raise ValueError("stored reconstruction differs")
    provenance = read_json(directory / "provenance.json")
    if declared is not None and provenance["inference_worker"].get("postprocessing") != declared:
        raise ValueError("correction differs from inference worker evidence")
    binding = dict(provenance["source_binding"])
    claimed = binding.pop("binding_sha256")
    if (
        fingerprint(binding) != claimed
        or claimed != record["binding_sha256"]
        or binding["release_id"] != summary["release_id"]
        or binding["scoring_batch_id"] != summary["scoring_batch_id"]
    ):
        raise ValueError("source binding differs")
    metadata = read_json(directory / "feature_metadata.json")
    if [item["name"] for item in metadata["features"]] != list(features):
        raise ValueError("estimator feature metadata differs")
    varying = summary.get("varying_feature_values", [])
    if not isinstance(varying, list) or not set(varying) <= set(features):
        raise ValueError("invalid varying member feature declaration")
    ranking = pd.read_parquet(directory / "global_summary.parquet")
    expected = global_summary(result)
    if (
        ranking.feature.duplicated().any()
        or set(ranking.feature) != set(expected.feature)
        or len(ranking) != len(expected)
    ):
        raise ValueError("global ranking feature coverage differs")
    actual = ranking.set_index("feature").loc[expected.feature]
    if not np.allclose(
        actual.mean_abs_contribution, expected.mean_abs_contribution, atol=1e-10, rtol=1e-8
    ):
        raise ValueError("global importance differs from contributions")
    if verify_explanation(directory, summary["explanation_id"]) != manifest:
        raise ValueError("explanation changed during presentation loading")
    return ExplanationView(
        record, summary, rows, features, contributions, expected, metadata, verification, manifest
    )
