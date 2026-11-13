"""Versioned regression explanations of complete registered scoring batches."""

import html
import sqlite3
import tempfile
import time
import uuid
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ml_template.explainability.artifacts import (
    file_entries,
    fingerprint,
    seal_explanation,
    sha256,
    verify_explanation,
    write_json,
)
from ml_template.explainability.regression_bridge import run_worker
from ml_template.explainability.regression_result import (
    RegressionTolerance,
    build_regression_explanation,
    global_summary,
)


def inspect(config):
    from ml_template.monitoring.monitor_inputs import load_batch
    from ml_template.scoring.release_artifacts import verify as verify_release

    database = Path(config.registry_path).resolve()
    if not database.is_file():
        raise ValueError("existing project database required")
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as conn:
        batch = load_batch(conn, config.expected_scoring_batch_id)
        found = conn.execute(
            "SELECT artifact_dir FROM model_releases WHERE release_id=?",
            (config.expected_release_id,),
        ).fetchone()
    if (
        batch.task != "regression"
        or batch.release_id != config.expected_release_id
        or not found
        or Path(found[0]).resolve() != Path(config.release_dir).resolve()
        or Path(batch.record["artifact_dir"]).resolve() != Path(config.scoring_dir).resolve()
        or len(batch.source) > config.max_rows
    ):
        raise ValueError("explicit registered batch/release paths, task or row limit differ")
    if not config.target_unit or not config.target_unit.strip():
        raise ValueError("explicit regression target_unit required")
    if batch.summary.get("target_unit") not in (None, config.target_unit):
        raise ValueError("explanation units differ from frozen units")
    release = verify_release(config.release_dir, config.expected_release_id)
    score = Path(config.scoring_dir)
    entries = file_entries(score, exclude=("artifacts.json",))
    snapshots = [name for name in entries if name in {"input.csv", "input.parquet"}]
    if len(snapshots) != 1:
        raise ValueError("one canonical source snapshot required")
    binding = {
        "release_id": batch.release_id,
        "scoring_batch_id": batch.batch_id,
        "release_content_sha256": release["content_sha256"],
        "scoring_manifest_sha256": sha256(score / "artifacts.json"),
        "scoring_files": entries,
        "snapshot_name": snapshots[0],
    }
    binding["binding_sha256"] = fingerprint(binding)
    return batch, binding


def explain_batch(config, *, check_only=False):
    from ml_template.explainability.explain_batch import authoring_runtime, validate_groups

    batch, binding = inspect(config)
    before = authoring_runtime()
    started = time.perf_counter()
    if check_only:
        with tempfile.TemporaryDirectory(prefix="regression-explain-check-") as temporary:
            response = run_worker(config, binding, temporary, check_only=True)
        return {
            "ready": True,
            "task": "regression",
            "n_rows": len(batch.source),
            "runtime_sha256": response["runtime_sha256"],
            "labels_loaded": False,
            "predictions_computed": False,
            "registry_written": False,
            "project_artifacts_written": False,
            "transient_ipc_files_used": True,
        }
    from ml_template.db import explanation_registry as registry

    root = Path(config.output_root)
    root.mkdir(parents=True, exist_ok=True)
    identity = f"explain_{datetime.now(UTC):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:12]}"
    destination = root / identity
    stage = Path(tempfile.mkdtemp(prefix=".regression-explanation-", dir=root))
    registered, published = False, False
    try:
        registry.start(config.registry_path, identity, config.model_dump(), binding, destination)
        registered = True
        with tempfile.TemporaryDirectory(prefix="regression-explain-worker-") as temporary:
            temporary = Path(temporary)
            response = run_worker(config, binding, temporary, check_only=False)
            if (
                response.get("output_scale") != "prediction_target_units"
                or response.get("method") != "xgboost_native_tree_shap"
                or response.get("feature_dependence") != "tree_path_dependent"
            ):
                raise ValueError("worker semantics differ")
            features = pd.read_parquet(temporary / "features.parquet")
            keys = pd.Index(batch.source[batch.key].to_numpy(), name=batch.key)
            if not features.index.equals(keys) or list(features) != response["feature_names"]:
                raise ValueError("worker estimator schema or entity coverage differs")
            if response["n_rows"] != len(keys) or response["n_features"] != len(features.columns):
                raise ValueError("worker row/feature counts differ")
            tolerance = RegressionTolerance(config.prediction_atol, config.prediction_rtol)
            with np.load(temporary / "contributions.npz", allow_pickle=False) as values:
                if set(values.files) != {"contributions", "base_values", "predictions"}:
                    raise ValueError("unexpected contribution array schema")
                result = build_regression_explanation(
                    features,
                    values["contributions"],
                    values["base_values"],
                    values["predictions"],
                    tolerance=tolerance,
                )
            reproduced = pd.read_parquet(temporary / "reproduced_predictions.parquet")
            expected = batch.predictions.set_index(batch.key).loc[keys]
            if (
                not reproduced.index.equals(keys)
                or set(reproduced) != set(expected) - {"scoring_batch_id"}
                or not reproduced.release_id.eq(batch.release_id).all()
            ):
                raise ValueError("reproduced prediction identity/schema differs")
            interval_names = [
                name
                for item in batch.summary.get("interval_columns", {}).values()
                for name in (item["lower"], item["upper"])
            ]
            if response.get("postprocessing") != batch.summary.get("postprocessing"):
                raise ValueError("worker postprocessing differs")
            for name in ("y_pred_raw", "y_pred_release", *interval_names):
                if not np.allclose(
                    reproduced[name],
                    expected[name],
                    atol=config.prediction_atol,
                    rtol=config.prediction_rtol,
                ):
                    raise ValueError("saved prediction parity failed")
            if not np.allclose(
                result.rows.y_pred_raw,
                reproduced[
                    "y_pred_release" if batch.summary.get("postprocessing") else "y_pred_raw"
                ],
                atol=config.prediction_atol,
                rtol=config.prediction_rtol,
            ):
                raise ValueError("SHAP prediction differs from released prediction")
        validate_groups(config, list(features))
        rows = result.rows.copy()
        if batch.summary.get("postprocessing"):
            rows["y_pred_explained"] = rows.pop("y_pred_raw")
            rows["y_pred_raw"] = reproduced.y_pred_raw
            for name in interval_names:
                rows[name] = reproduced[name]
        rows["y_pred_release"] = reproduced.y_pred_release
        for name, value in (
            ("release_id", batch.release_id),
            ("scoring_batch_id", batch.batch_id),
            ("explanation_id", identity),
        ):
            rows[name] = value
        rows.to_parquet(stage / "rows.parquet", index=True)
        features.to_parquet(stage / "features.parquet", index=True)
        result.contributions.to_parquet(stage / "contributions.parquet", index=True)
        ranking = global_summary(result)
        ranking.to_parquet(stage / "global_summary.parquet", index=False)
        if config.groups:
            from ml_template.explainability.summaries import grouped_summary

            grouped_summary(result, config.groups).to_parquet(
                stage / "group_summary.parquet", index=False
            )
        summary = {
            "schema_version": 2,
            "task": "regression",
            "artifact_state": "complete",
            "explanation_id": identity,
            "release_id": batch.release_id,
            "scoring_batch_id": batch.batch_id,
            "target_unit": config.target_unit,
            "n_rows": len(rows),
            "n_features": len(features.columns),
            "n_flagged": None,
            "parent_run_id": batch.summary["parent_run_id"],
            "calibration_id": batch.summary.get("calibration_id"),
            "method": result.method,
            "output_scale": result.output_scale,
            "feature_dependence": result.feature_dependence,
            "predictor_kind": response["predictor_kind"],
            "varying_feature_values": response["varying_feature_values"],
            "persisted_scoring_batch_verified": True,
            "labels_loaded": False,
            "population": "complete_original_scoring_batch",
            "created_at": datetime.now(UTC).isoformat(),
        }
        if batch.summary.get("postprocessing"):
            summary.update(
                postprocessing=batch.summary["postprocessing"],
                interval_columns=batch.summary["interval_columns"],
                explained_prediction_column="y_pred_explained",
            )
        write_json(stage / "summary.json", summary)
        write_json(stage / "config.json", config.model_dump())
        write_json(
            stage / "verification.json", {**result.verification, "saved_batch_parity": response}
        )
        write_json(
            stage / "feature_metadata.json",
            {
                "schema_version": 1,
                "features": [
                    {
                        "name": name,
                        "kind": "estimator_input",
                        "value_semantics": "varies_across_members"
                        if name in response["varying_feature_values"]
                        else "saved_estimator_representation",
                    }
                    for name in features
                ],
            },
        )
        write_json(
            stage / "provenance.json",
            {
                "schema_version": 2,
                "source_binding": binding,
                "binding_sha256": binding["binding_sha256"],
                "inference_worker": response,
                "orchestration_runtime": before,
                "source_manifests_modified": False,
                "units_are_operator_declared": True,
            },
        )
        message = (
            "Contributions reconstruct released point predictions in declared target units. "
            "Affine output correction scales contributions and baseline; its intercept shifts only the baseline. "
            "Saved interval bounds are not SHAP attributions. "
            "These are estimator-space, tree-path-dependent attributions, not causes or "
            "source-feature redistribution. CV contributions and baselines use saved member "
            "weights. Dependence is disabled where fold transforms produce different values. "
            "No target labels were loaded. Units are declared, not inferred."
        )
        (stage / "report.md").write_text(
            "# Regression explanations\n\n" + message + "\n\n" + ranking.to_csv(index=False)
        )
        (stage / "report.html").write_text(
            "<!doctype html><meta charset='utf-8'><h1>Regression explanations</h1><p>"
            + html.escape(message)
            + "</p><p>Units: "
            + html.escape(config.target_unit)
            + "</p>"
            + ranking.to_html(index=False, escape=True)
        )
        if inspect(config)[1] != binding or authoring_runtime() != before:
            raise ValueError("source binding or orchestration code changed during explanation")
        manifest = seal_explanation(stage, identity)
        verify_explanation(stage, identity)
        stage.rename(destination)
        published = True
        registry.finish(
            config.registry_path,
            identity,
            duration_s=time.perf_counter() - started,
            manifest=manifest,
            n_rows=len(rows),
            n_features=len(features.columns),
        )
        return {
            "status": "succeeded",
            "explanation_id": identity,
            "artifact_dir": str(destination),
            "task": "regression",
            "release_id": batch.release_id,
            "scoring_batch_id": batch.batch_id,
            "n_rows": len(rows),
        }
    except Exception as exc:
        failed = destination if published else stage
        if registered:
            registry.finish(
                config.registry_path,
                identity,
                duration_s=time.perf_counter() - started,
                error=str(exc),
                artifact_dir=failed,
            )
        raise
