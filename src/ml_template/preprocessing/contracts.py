"""Post-Gold validation utility for future adapters. Never fits or transforms."""

import numpy as np
import pandas as pd


def validate_estimator_frame(frame, *, allow_nan=False, require_numeric=True):
    """Validate the final DataFrame BEFORE estimator/tensor conversion.

    Future NN adapters must call this after saved Gold preprocessing. This utility
    is not inserted before existing fitted transformers and does not clean data.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty or not frame.columns.is_unique:
        raise ValueError("estimator input must be a nonempty DataFrame with unique columns")
    for name in frame.columns:
        series = frame[name]
        if not allow_nan and series.isna().any():
            raise ValueError(f"post-Gold missing values: {name}")
        numeric = pd.api.types.is_numeric_dtype(series.dtype)
        if require_numeric and not numeric:
            raise ValueError(f"post-Gold numeric input required: {name}")
        if numeric and not np.isfinite(series.dropna().to_numpy(dtype=float)).all():
            raise ValueError(f"post-Gold infinite numeric values: {name}")
    return frame


def validate_nn_arrays(numeric, categorical, cardinalities):
    """Check the actual CPU arrays immediately before torch tensor conversion.

    Empty branches and empty inference batches are allowed. Missing flags belong
    to the float32 branch, including for categorical-only source inputs.
    """
    if not isinstance(numeric, np.ndarray) or not isinstance(categorical, np.ndarray):
        raise ValueError("NN inputs must be numpy arrays")
    if numeric.ndim != 2 or categorical.ndim != 2:
        raise ValueError("NN inputs must be 2-D")
    if numeric.shape[0] != categorical.shape[0]:
        raise ValueError("NN inputs have different row counts")
    if numeric.shape[1] + categorical.shape[1] == 0:
        raise ValueError("NN inputs have no features")
    if numeric.dtype != np.float32 or categorical.dtype != np.int64:
        raise ValueError("NN input dtypes must be float32 and int64")
    if not np.isfinite(numeric).all():
        raise ValueError("NN numeric input is nonfinite after float32 conversion")
    if not isinstance(cardinalities, (list, tuple)):
        raise ValueError("cardinalities must be a list or tuple")
    if len(cardinalities) != categorical.shape[1]:
        raise ValueError("embedding cardinalities do not match categorical columns")
    for j, size in enumerate(cardinalities):
        if (
            isinstance(size, (bool, np.bool_))
            or not isinstance(size, (int, np.integer))
            or size < 2
        ):
            raise ValueError("embedding cardinalities must be integers >= 2")
        if (categorical[:, j] < 0).any() or (categorical[:, j] >= size).any():
            raise ValueError(f"embedding index out of bounds in categorical column {j}")
