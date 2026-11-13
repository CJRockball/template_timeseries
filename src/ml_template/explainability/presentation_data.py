"""Read-only presentation contracts for saved explanations. No models or labels loaded."""

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ml_template.explainability.artifacts import fingerprint, read_json, verify_explanation


@dataclass(frozen=True)
class ExplanationView:
    record: dict
    summary: dict
    rows: pd.DataFrame
    features: pd.DataFrame
    contributions: pd.DataFrame
    importance: pd.DataFrame
    feature_metadata: dict
    verification: dict
    manifest: dict


def _connection(database):
    path = Path(database).expanduser().resolve()
    if not path.is_file():
        raise ValueError("existing project database required; presentation never creates one")
    return sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)


def list_explanations(database, limit=500):
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("positive limit required")
    with closing(_connection(database)) as conn:
        conn.row_factory = sqlite3.Row
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='explanation_runs'"
        ).fetchone()
        if not exists:
            return []
        rows = conn.execute(
            "SELECT explanation_id,release_id,scoring_batch_id,started_at,n_rows,n_features,artifact_dir "
            "FROM explanation_runs WHERE status='succeeded' ORDER BY started_at DESC,explanation_id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]


def _aligned(frame, index, name):
    if isinstance(frame.index, pd.MultiIndex) or frame.index.hasnans or frame.index.has_duplicates:
        raise ValueError(f"invalid {name} entity keys")
    if len(frame) != len(index) or set(frame.index) != set(index):
        raise ValueError(f"{name} must cover every explanation row exactly once")
    out = frame.loc[index].copy()
    if not out.index.equals(index):
        raise ValueError(f"{name} key type/order mismatch")
    return out


def load_explanation(database, explanation_id, *, max_rows=100000):
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1:
        raise ValueError("positive max_rows required")
    with closing(_connection(database)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM explanation_runs WHERE explanation_id=?", (explanation_id,)
        ).fetchone()
    if not row or row["status"] != "succeeded":
        raise ValueError("explicit succeeded explanation record required")
    record = dict(row)
    directory = Path(record["artifact_dir"]).resolve()
    manifest = verify_explanation(directory, explanation_id)
    if record["content_sha256"] != manifest["content_sha256"]:
        raise ValueError("registry/artifact content fingerprint mismatch")
    summary = read_json(directory / "summary.json")
    for name in ("explanation_id", "release_id", "scoring_batch_id", "n_rows", "n_features"):
        if summary.get(name) != record.get(name):
            raise ValueError(f"registry/summary mismatch: {name}")
    if summary.get("task") == "regression":
        from ml_template.explainability.regression_presentation import load_view

        return load_view(record, summary, manifest, directory, max_rows=max_rows)
    if summary.get("schema_version") != 1 or summary.get("artifact_state") != "complete":
        raise ValueError("unsupported/incomplete explanation summary")
    if (
        summary.get("labels_loaded") is not False
        or summary.get("persisted_scoring_batch_verified") is not True
    ):
        raise ValueError("unsupported explanation evidence")
    if summary.get("output_scale") != "base_model_log_odds":
        raise ValueError("only base-model log-odds contributions supported")
    if type(summary["n_rows"]) is not int or not 0 < summary["n_rows"] <= max_rows:
        raise ValueError("explanation exceeds presentation row limit")
    rows = pd.read_parquet(directory / "rows.parquet")
    if (
        len(rows) != summary["n_rows"]
        or rows.index.hasnans
        or rows.index.has_duplicates
        or isinstance(rows.index, pd.MultiIndex)
    ):
        raise ValueError("invalid explanation row keys/count")
    keys = rows.index.tolist()
    if not (
        all(isinstance(k, str) for k in keys)
        or all(
            isinstance(k, (int, np.integer)) and not isinstance(k, (bool, np.bool_)) for k in keys
        )
    ):
        raise ValueError("entity keys must be consistently strings or integers")
    required = {
        "base_value",
        "margin",
        "p_raw",
        "p_release",
        "release_id",
        "scoring_batch_id",
        "explanation_id",
    }
    allowed = required | {
        "margin_reconstructed",
        "margin_residual",
        "probability_residual",
        "p_raw_saved",
        "p_calibrated",
        "p_calibrated_reproduced",
        "p_calibrated_saved",
        "p_release_reproduced",
        "p_release_saved",
        "policy_flag",
    }
    if (
        rows.columns.duplicated().any()
        or not required <= set(rows.columns)
        or not set(rows.columns) <= allowed
    ):
        raise ValueError("unexpected explanation row schema")
    for name in ("release_id", "scoring_batch_id", "explanation_id"):
        if not (rows[name] == summary[name]).all():
            raise ValueError(f"row identity mismatch: {name}")
    for name in ("base_value", "margin"):
        if not np.isfinite(rows[name].to_numpy(dtype=float)).all():
            raise ValueError("nonfinite margin/baseline")
    for name in rows.columns:
        if name.startswith("p_"):
            values = rows[name].to_numpy(dtype=float)
            if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError("invalid probability")
    if "policy_flag" in rows and (
        rows.policy_flag.isna().any() or not pd.api.types.is_bool_dtype(rows.policy_flag.dtype)
    ):
        raise ValueError("invalid saved policy flags")
    features = _aligned(pd.read_parquet(directory / "features.parquet"), rows.index, "features")
    contributions = _aligned(
        pd.read_parquet(directory / "contributions.parquet"), rows.index, "contributions"
    )
    if (
        features.columns.duplicated().any()
        or len(features.columns) != summary["n_features"]
        or not all(isinstance(n, str) for n in features.columns)
    ):
        raise ValueError("invalid estimator feature schema")
    if not features.columns.equals(contributions.columns):
        raise ValueError("contribution features differ from estimator features")
    values = contributions.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite contributions")
    verification = read_json(directory / "verification.json")
    if verification.get("passed") is not True:
        raise ValueError("saved numerical verification did not pass")
    tolerance = verification.get("tolerance", {})
    margin_atol = tolerance.get("margin_atol", 1e-5)
    margin_rtol = tolerance.get("margin_rtol", 1e-5)
    probability_atol = tolerance.get("probability_atol", 1e-6)
    probability_rtol = tolerance.get("probability_rtol", 1e-5)
    for value in (margin_atol, margin_rtol, probability_atol, probability_rtol):
        if not np.isfinite(value) or value < 0:
            raise ValueError("invalid saved tolerance")
    if not np.allclose(
        rows.base_value.to_numpy() + values.sum(axis=1),
        rows.margin,
        atol=margin_atol,
        rtol=margin_rtol,
    ):
        raise ValueError("stored contributions do not reconstruct margin")
    from scipy.special import expit

    if not np.allclose(
        expit(rows.margin.to_numpy(dtype=float)),
        rows.p_raw,
        atol=probability_atol,
        rtol=probability_rtol,
    ):
        raise ValueError("stored margin does not reproduce raw probability")
    provenance = read_json(directory / "provenance.json")
    binding = dict(provenance["source_binding"])
    claimed = binding.pop("binding_sha256")
    if fingerprint(binding) != claimed or claimed != record["binding_sha256"]:
        raise ValueError("source binding fingerprint mismatch")
    if (
        binding["release_id"] != summary["release_id"]
        or binding["scoring_batch_id"] != summary["scoring_batch_id"]
    ):
        raise ValueError("source binding identity mismatch")
    metadata = read_json(directory / "feature_metadata.json")
    if [item["name"] for item in metadata["features"]] != list(features.columns):
        raise ValueError("feature provenance order differs")
    importance = pd.read_parquet(directory / "global_summary.parquet")
    if importance.feature.duplicated().any() or set(importance.feature) != set(features.columns):
        raise ValueError("global summary feature coverage differs")
    expected = contributions.abs().mean()
    actual = importance.set_index("feature").loc[features.columns, "mean_abs_contribution"]
    if not np.allclose(expected, actual, atol=1e-10, rtol=1e-8):
        raise ValueError("global importance differs from saved contributions")
    if verify_explanation(directory, explanation_id) != manifest:
        raise ValueError("explanation artifacts changed during loading")
    return ExplanationView(
        record,
        summary,
        rows.copy(),
        features,
        contributions,
        importance.sort_values(
            ["mean_abs_contribution", "feature"], ascending=[False, True]
        ).reset_index(drop=True),
        metadata,
        verification,
        manifest,
    )


def selected_entity(view, position):
    if (
        isinstance(position, bool)
        or not isinstance(position, (int, np.integer))
        or not 0 <= position < len(view.rows)
    ):
        raise ValueError("invalid entity selection")
    key = view.rows.index[int(position)]
    contributions = view.contributions.loc[key]
    table = pd.DataFrame(
        {
            "feature": contributions.index,
            "value": view.features.loc[key].tolist(),
            "contribution": contributions.to_numpy(),
        }
    )
    table["abs_contribution"] = table.contribution.abs()
    table = table.sort_values(["abs_contribution", "feature"], ascending=[False, True]).drop(
        columns="abs_contribution"
    )
    return key, view.rows.loc[key], table


def table_records(frame):
    return json.loads(frame.to_json(orient="records", date_format="iso"))
