"""In-memory explanations of a complete explicit release input batch.

Disk-backed scoring-batch verification, execution tracking and reports are a later layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ml_template.explainability.result import (
    ExplanationTolerance,
    MarginExplanation,
    build_margin_explanation,
)

if TYPE_CHECKING:
    from ml_template.scoring.predictor import ReleasedPredictor


@dataclass(frozen=True)
class ReleaseExplanation:
    margin: MarginExplanation
    predictions: pd.DataFrame
    metadata: dict


def explain_release(
    predictor: ReleasedPredictor,
    frame: pd.DataFrame,
    *,
    chunk_size: int | None = None,
    tolerance: ExplanationTolerance | None = None,
) -> ReleaseExplanation:
    """Explain every row; policy is computed ONCE over the entire input batch.

    Pass a normally constructed ReleasedPredictor. This function never disables its
    constructor's release/runtime checks. Returned predictions are newly reproduced,
    not asserted to be a previously persisted scoring-batch artifact.
    """
    tol = ExplanationTolerance() if tolerance is None else tolerance
    if not isinstance(tol, ExplanationTolerance):
        raise TypeError("tolerance must be ExplanationTolerance")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("nonempty batch DataFrame required")
    if chunk_size is None:
        chunk_size = len(frame)
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if not callable(getattr(predictor.model, "explain_margin", None)):
        raise TypeError("released model does not support base-margin explanations")
    predictions = predictor.predict(frame, chunk_size=chunk_size, include_policy=True)
    key = predictor.contract["key_col"]
    required = {key, "p_raw", "p_release", "release_id"}
    if not required <= set(predictions.columns):
        raise ValueError("release predictions are incomplete")
    if predictions[key].isna().any() or predictions[key].duplicated().any():
        raise ValueError("release predictions contain invalid entity keys")
    if not (predictions["release_id"] == predictor.release_id).all():
        raise ValueError("prediction release identity mismatch")
    parts = []
    expected_columns = None
    for start in range(0, len(frame), chunk_size):
        source = frame.iloc[start : start + chunk_size]
        selected = predictor.prepare_features(source)
        part = predictor.model.explain_margin(selected, tolerance=tol)
        if not part.features.index.equals(selected.index):
            raise ValueError("explanation changed entity keys or row order")
        if expected_columns is None:
            expected_columns = part.features.columns
        elif not part.features.columns.equals(expected_columns):
            raise ValueError("estimator features changed between chunks")
        parts.append(part)
    features = pd.concat([part.features for part in parts])
    contributions = pd.concat([part.contributions for part in parts])
    rows = pd.concat([part.rows for part in parts])
    if not pd.Index(predictions[key], name=key).equals(features.index):
        raise ValueError("release predictions differ from explanation key coverage/order")
    result = build_margin_explanation(
        features,
        contributions,
        rows.base_value,
        rows.margin,
        predictions.p_raw.to_numpy(),
        tolerance=tol,
    )
    if not np.allclose(
        rows.p_raw,
        predictions.p_raw,
        atol=tol.probability_atol,
        rtol=tol.probability_rtol,
    ):
        raise ValueError("explanation raw probabilities differ from release predictions")
    return ReleaseExplanation(
        margin=result,
        predictions=predictions.copy(deep=True),
        metadata={
            "schema_version": 1,
            "release_id": predictor.release_id,
            "release_content_sha256": predictor.manifest["content_sha256"],
            "parent_run_id": predictor.info["parent_run_id"],
            "calibration_id": predictor.info["calibration_id"],
            "release_variant": predictor.info["release_variant"],
            "method": result.method,
            "output_scale": result.output_scale,
            "feature_dependence": result.feature_dependence,
            "n_rows": len(features),
            "n_features": len(features.columns),
            "chunk_size": chunk_size,
            "scope": "complete_supplied_batch",
            "persisted_scoring_batch_verified": False,
            "calibrated_probability_attribution": False,
            "policy_attribution": False,
            "labels_loaded": False,
            "limitations": [
                "Contributions explain base-model log-odds, not calibrated probability.",
                "Attribution is not causation or evidence of predictive quality.",
                "Correlated engineered features may share attribution.",
                "Alert-rate decisions require the complete intended decision batch.",
            ],
        },
    )
