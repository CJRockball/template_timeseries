"""Frozen regression evaluation using the existing exposure/cache registry."""

import html
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ml_template.db import evaluation_registry as registry
from ml_template.evaluation.evaluation_artifacts import clean, manifest, verify, write_json
from ml_template.evaluation.evaluation_inputs import exposure_warnings
from ml_template.evaluation.regression_inputs import load_holdout, prepare_regression
from ml_template.evaluation.regression_metrics import regression_metrics
from ml_template.models.regression_predictor import continuous


def default_spec_for_run(engine, run_id):
    from ml_template.db import model_run_registry
    from ml_template.evaluation.evaluation_config import EvaluationConfig
    from ml_template.evaluation.regression_config import RegressionEvaluationConfig
    from ml_template.tracking.run_artifacts import resolve_stored_path, verify_artifact

    parent = model_run_registry.get_run(engine, run_id)
    if not parent or parent["status"] != "succeeded":
        raise ValueError("Parent run must exist and have succeeded")
    root = resolve_stored_path(parent["artifact_dir"])
    verify_artifact(root, "config.json")
    task = json.loads((root / "config.json").read_text()).get("task", "binary_classification")
    if task not in {"binary_classification", "regression"}:
        raise ValueError("Unsupported parent task")
    cls = RegressionEvaluationConfig if task == "regression" else EvaluationConfig
    return cls(parent_run_id=run_id)


def assess_regression_criteria(metrics, spec):
    results = []
    for criterion in spec.criteria:
        value = metrics.get(criterion.metric)
        if value is None or not np.isfinite(value):
            passed, status = None, "unavailable"
        else:
            passed = bool(
                value <= criterion.value if criterion.operator == "<=" else value >= criterion.value
            )
            status = "passed" if passed else "failed"
        results.append(
            {**criterion.model_dump(), "observed": value, "passed": passed, "status": status}
        )
    if not results:
        status = "not_assessed"
    elif any(r["status"] == "failed" for r in results):
        status = "failed"
    elif any(r["status"] == "unavailable" for r in results):
        status = "unavailable"
    else:
        status = "passed"
    return {"status": status, "results": results, "automatic_promotion": False}


def _verify_cache(record, identity):
    directory = Path(record["artifact_dir"])
    entries = json.loads((directory / "artifact_manifest.json").read_text())["files"]
    required = {
        "report_data.json",
        "summary.json",
        "spec.json",
        "provenance.json",
        "metrics.csv",
        "acceptance.json",
        "holdout_predictions.parquet",
        "residual_summary.json",
        "report.html",
    }
    if not required <= set(entries):
        raise ValueError("Incomplete cached regression evaluation")
    data = verify(directory)
    summary = data["summary"]
    if (
        summary["identity_hash"] != identity
        or summary["evaluation_id"] != record["evaluation_id"]
        or summary.get("task") != "regression"
    ):
        raise ValueError("Cached regression identity mismatch")
    return data


def evaluate_regression_spec(
    engine,
    cfg,
    *,
    check_only=False,
    output_root=None,
    silver_root=None,
    split_root=None,
    feature_root=None,
    adapter_loader=None,
):
    roots = {"silver_root": silver_root, "split_root": split_root, "feature_root": feature_root}
    from ml_template.evaluation import nn_evaluate as nn

    neural = nn.is_nn_parent(engine, cfg.parent_run_id)
    prepare = nn.prepare_nn_evaluation if neural else prepare_regression
    p = prepare(engine, cfg, **roots)
    cfg = p.spec
    cached = (
        nn.cached(engine, p.identity_hash)
        if neural
        else registry.cached_evaluation(engine, p.identity_hash)
    )
    if cached:
        data = (
            nn.verify_cache(engine, cached, p.identity_hash)
            if neural
            else _verify_cache(cached, p.identity_hash)
        )
        return {
            "evaluation_id": cached["evaluation_id"],
            "status": "succeeded",
            "cached": True,
            "artifact_dir": cached["artifact_dir"],
            "summary": data["summary"],
            "holdout_labels_loaded": False,
            **(
                {
                    "ready": True,
                    "model_type": "nn",
                    "models_loaded": False,
                    "files_written": False,
                    "predictions_computed": False,
                    "release_enabled": False,
                }
                if neural
                else {}
            ),
        }
    prior_fn = nn.exposure_warnings if neural else exposure_warnings
    prior = prior_fn(engine, p)
    if cfg.supersedes_evaluation_id:
        previous = (
            nn.record(engine, cfg.supersedes_evaluation_id)
            if neural
            else registry.get_evaluation(engine, cfg.supersedes_evaluation_id)
        )
        if not previous or previous["parent_run_id"] != cfg.parent_run_id:
            raise ValueError("Superseded evaluation must share the parent")
    if check_only:
        return {
            "parent_run_id": cfg.parent_run_id,
            "task": "regression",
            "identity_hash": p.identity_hash,
            "n_holdout_rows": len(p.hold_keys),
            "prior_exposures": prior,
            "ready": not prior or cfg.acknowledge_prior_exposure,
            "resolved_spec": cfg.model_dump(mode="json"),
            "holdout_labels_loaded": False,
            **(
                {
                    "model_type": "nn",
                    "models_loaded": False,
                    "files_written": False,
                    "predictions_computed": False,
                    "release_enabled": False,
                }
                if neural
                else {}
            ),
        }
    if prior and not cfg.acknowledge_prior_exposure:
        raise ValueError("Holdout previously exposed; document and acknowledge prior exposure")
    if output_root is None:
        from ml_template.paths import ARTIFACTS

        output_root = ARTIFACTS / "evaluations"
    if neural:
        nn.validate_output(p, output_root)
    eid = f"eval_{datetime.now(UTC):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    out = (Path(output_root) / eid).resolve()
    registry.start_evaluation(
        engine,
        evaluation_id=eid,
        identity_hash=p.identity_hash,
        spec=cfg,
        fingerprint=p.fingerprint,
        artifact_dir=str(out),
        prior=prior,
    )
    try:
        if prior_fn(engine, p) != prior:
            raise RuntimeError("Exposure history changed; rerun preflight")
        fresh = prepare(engine, cfg, **roots)
        if fresh.identity_hash != p.identity_hash:
            raise ValueError("Evaluation inputs changed after preflight")
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / "spec.json", cfg.model_dump(mode="json"))
        write_json(out / "provenance.json", p.provenance)
        # Commit exposure BEFORE loading target values; failures remain auditable.
        registry.mark_accessed(engine, eid, cfg.parent_run_id)
        X, y = nn.load_authorized_holdout(engine, p, eid) if neural else load_holdout(p)
        from ml_template.models.predictor_selection import load_evaluation_predictor

        model = load_evaluation_predictor(p, adapter_loader=adapter_loader)
        prediction = continuous(model.predict(X), len(y))
        from ml_template.evaluation.regression_postprocessing import evaluate as postprocess

        raw_prediction = prediction.copy() if neural else None
        prediction, post_columns, interval_rows = postprocess(p, prediction, y)
        if neural:
            post_columns.setdefault("y_pred_raw", raw_prediction)
            if prepare(engine, cfg, **roots).identity_hash != p.identity_hash:
                raise ValueError("NN inputs/calibration changed during evaluation")
        frame = pd.DataFrame(
            {
                p.key_col: p.hold_keys,
                "y_true": y,
                "y_pred": prediction,
                "residual": y - prediction,
                "absolute_error": np.abs(y - prediction),
            }
        )
        frame = frame.assign(**post_columns)
        frame.to_parquet(out / "holdout_predictions.parquet", index=False)
        values = regression_metrics(y, prediction)
        if neural and any(not np.isfinite(value) for value in values.values()):
            raise ValueError("Nonfinite NN evaluation metric")
        metric_rows = [
            {
                "variant": cfg.release_variant,
                "metric": name,
                "value": values.get(name),
                "direction": "max" if name == "r2" else "min",
                "status": "defined" if name in values else "undefined",
                "uncertainty": "not_requested",
            }
            for name in ["rmse", "mae", "r2"]
        ]
        metric_rows.extend(interval_rows)
        acceptance = assess_regression_criteria(values, cfg)
        residual = y - prediction
        residual_summary = {
            "convention": "y_true - y_pred; positive means underprediction",
            "mean": float(np.mean(residual)),
            "std_population": float(np.std(residual)),
            "quantiles": {
                str(q): float(np.quantile(residual, q)) for q in [0.05, 0.25, 0.5, 0.75, 0.95]
            },
        }
        warnings = [
            "Point metrics have no bootstrap confidence intervals; any configured prediction intervals are distinct from metric confidence intervals.",
            "Final and CV-mean predictors are distinct frozen predictors; do not select between them on this holdout.",
            "Residual examples are exploratory, not independent model-selection evidence.",
            "Exposure tracking cannot detect unrecorded manual access or use in another database.",
            *cfg.limitations,
        ]
        if prior:
            warnings.append("Prior exposure exists: this is not untouched first confirmation.")
        if not cfg.criteria:
            warnings.append("No predeclared acceptance criteria: acceptance not assessed.")
        summary = {
            "evaluation_id": eid,
            "identity_hash": p.identity_hash,
            "task": "regression",
            "parent_run_id": cfg.parent_run_id,
            "predictor_kind": cfg.predictor.kind,
            "release_variant": cfg.release_variant,
            "calibration_id": cfg.calibration_id,
            "n_rows": len(y),
            "primary_metric": cfg.primary_metric,
            "primary_metric_direction": "min",
            "primary_value": values[cfg.primary_metric],
            "criteria_status": acceptance["status"],
            "prior_exposure": bool(prior),
            "automatic_promotion": False,
        }
        if cfg.calibration_id is not None:
            summary.update(
                target_unit=cfg.target_unit, interval_coverages=list(cfg.uncertainty.coverages)
            )
        if neural:
            summary.update(
                model_type="nn",
                nn_evaluation_schema=1,
                target_unit=cfg.target_unit,
                interval_coverages=list(cfg.uncertainty.coverages),
            )
        data = clean(
            {
                "summary": summary,
                "spec": cfg.model_dump(mode="json"),
                "provenance": p.provenance,
                "metrics": metric_rows,
                "acceptance": acceptance,
                "residual_summary": residual_summary,
                "warnings": warnings,
                "prior_exposures": prior,
            }
        )
        write_json(out / "summary.json", summary)
        write_json(out / "acceptance.json", acceptance)
        write_json(out / "residual_summary.json", residual_summary)
        write_json(out / "report_data.json", data)
        pd.DataFrame(metric_rows).to_csv(out / "metrics.csv", index=False)
        frame.nlargest(min(25, len(frame)), "absolute_error").to_csv(
            out / "residual_examples.csv", index=False
        )
        body = html.escape(json.dumps(data, indent=2, ensure_ascii=False))
        (out / "report.html").write_text(
            '<!doctype html><meta charset="utf-8"><title>Regression evaluation</title>'
            "<h1>Frozen regression evaluation</h1><pre>" + body + "</pre>",
            encoding="utf-8",
        )
        manifest(out)
        registry.finish_evaluation(engine, eid, rows=data["metrics"])
    except Exception as exc:
        registry.finish_evaluation(engine, eid, error=f"{type(exc).__name__}: {exc}")
        raise
    return {
        "evaluation_id": eid,
        "status": "succeeded",
        "cached": False,
        "artifact_dir": str(out),
        "summary": summary,
        "holdout_labels_loaded": True,
        **(
            {
                "model_type": "nn",
                "models_loaded": True,
                "files_written": True,
                "predictions_computed": True,
                "release_enabled": False,
            }
            if neural
            else {}
        ),
    }
