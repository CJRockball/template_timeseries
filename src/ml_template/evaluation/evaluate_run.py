"""Frozen-model final evaluation. --check reads development data/metadata, not holdout labels."""

import argparse
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ml_template.db import evaluation_registry as registry
from ml_template.evaluation.decision_policy import (
    apply_policy,
    labels,
    policy_metrics,
    probabilities,
)
from ml_template.evaluation.evaluation_artifacts import clean, manifest, verify, write_json
from ml_template.evaluation.evaluation_config import EvaluationConfig, load_evaluation_config
from ml_template.evaluation.evaluation_diagnostics import (
    curve_rows,
    error_rows,
    reliability_rows,
    subgroup_rows,
)
from ml_template.evaluation.evaluation_inputs import exposure_warnings, prepare_evaluation
from ml_template.evaluation.evaluation_statistics import assess_criteria, variant_metrics
from ml_template.evaluation.evaluation_uncertainty import EXCLUDED, bootstrap_metrics
from ml_template.evaluation.final_report import render_report


def _load_inputs(p):
    from ml_template.training.run_inputs import load_run_inputs

    return load_run_inputs(p.run_cfg, **p.roots)


def _load_model(p):
    from ml_template.models.predictor_selection import load_evaluation_predictor

    return load_evaluation_predictor(p)


def _save_predictions(frame, path):
    frame.to_parquet(path, index=False)


def _development_context(p):
    rows = [
        {"source": "selected_parent_cv", **r} for r in p.provenance.get("development_metrics", [])
    ]
    if p.calibration_dir is not None and (p.calibration_dir / "metrics.csv").exists():
        from ml_template.tracking.run_artifacts import verify_artifact

        verify_artifact(p.calibration_dir, "metrics.csv")
        df = pd.read_csv(p.calibration_dir / "metrics.csv")
        rows.extend(
            {"source": "exploratory_calibration_crossfit_not_nested", **r}
            for r in df.loc[
                df.fold.eq(-1) & df.variant.isin(["raw", "calibrated_crossfit"])
            ].to_dict(orient="records")
        )
    return rows


def evaluate_spec(
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
    if getattr(cfg, "task", None) == "regression":
        from ml_template.evaluation.regression_evaluate import evaluate_regression_spec

        return evaluate_regression_spec(
            engine,
            cfg,
            check_only=check_only,
            output_root=output_root,
            silver_root=silver_root,
            split_root=split_root,
            feature_root=feature_root,
            adapter_loader=adapter_loader,
        )
    p = prepare_evaluation(
        engine, cfg, silver_root=silver_root, split_root=split_root, feature_root=feature_root
    )
    cfg = p.spec
    dummy = {"raw": [0.2, 0.8]}
    if cfg.calibration_id:
        dummy["calibrated"] = [0.1, 0.9]
    allowed = variant_metrics([0, 1], dummy, cfg)[cfg.release_variant]
    for c in cfg.criteria:
        if c.metric not in allowed:
            raise ValueError("unknown criterion metric")
        if c.basis == "confidence_bound" and (not cfg.uncertainty.enabled or c.metric in EXCLUDED):
            raise ValueError("confidence bound unavailable")
    cached = registry.cached_evaluation(engine, p.identity_hash)
    if cached:
        data = verify(cached["artifact_dir"])
        if (
            data["summary"]["identity_hash"] != p.identity_hash
            or data["summary"]["evaluation_id"] != cached["evaluation_id"]
        ):
            raise ValueError("cached identity mismatch")
        return {
            "evaluation_id": cached["evaluation_id"],
            "status": "succeeded",
            "cached": True,
            "artifact_dir": cached["artifact_dir"],
            "summary": data["summary"],
            "holdout_labels_loaded": False,
        }
    prior = exposure_warnings(engine, p)
    if cfg.supersedes_evaluation_id:
        previous = registry.get_evaluation(engine, cfg.supersedes_evaluation_id)
        if not previous or previous["parent_run_id"] != cfg.parent_run_id:
            raise ValueError("superseded evaluation must share the parent")
    if check_only:
        return {
            "parent_run_id": cfg.parent_run_id,
            "identity_hash": p.identity_hash,
            "n_holdout_rows": len(p.hold_keys),
            "prior_exposures": prior,
            "ready": not prior or cfg.acknowledge_prior_exposure,
            "resolved_spec": cfg.model_dump(mode="json"),
            "holdout_labels_loaded": False,
        }
    if prior and (not cfg.acknowledge_prior_exposure):
        raise ValueError("holdout previously exposed; document and acknowledge prior exposure")
    if output_root is None:
        from ml_template.paths import ARTIFACTS

        output_root = ARTIFACTS / "evaluations"
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
        if exposure_warnings(engine, p) != prior:
            raise RuntimeError("exposure history changed; rerun preflight")
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / "spec.json", cfg.model_dump(mode="json"))
        write_json(out / "provenance.json", p.provenance)
        registry.mark_accessed(engine, eid, cfg.parent_run_id)
        inputs = _load_inputs(p)
        if (
            inputs.split_id != cfg.expected_split_id
            or inputs.contract.entity_id != p.key_col
            or inputs.contract.target != p.target_col
        ):
            raise ValueError("loaded split or dataset contract mismatch")
        if (
            not inputs.X.index.is_unique
            or inputs.X.index.hasnans
            or (not inputs.y.index.is_unique)
            or inputs.y.index.hasnans
            or (set(inputs.X.index) != set(inputs.y.index))
        ):
            raise ValueError("feature/label key mismatch")
        pd.testing.assert_frame_equal(
            inputs.assignments.reset_index(drop=True), p.assignments.reset_index(drop=True)
        )
        if inputs.assignments[p.key_col].duplicated().any():
            raise ValueError("duplicate split keys")
        X = inputs.X.loc[p.hold_keys]
        if list(X.columns) != p.feature_names:
            raise ValueError("saved input feature order changed")
        y = labels(inputs.y.loc[p.hold_keys].to_numpy(), len(X))
        if adapter_loader is None:
            model = _load_model(p)
        elif getattr(getattr(cfg, "predictor", None), "kind", "final") == "cv_mean":
            from ml_template.models.predictor_selection import load_evaluation_predictor

            model = load_evaluation_predictor(p, adapter_loader=adapter_loader)
        else:
            model = adapter_loader(p.run_cfg, p.run_dir / "model.ubj")
        variants = {"raw": probabilities(model.predict_proba(X))}
        if len(variants["raw"]) != len(y):
            raise ValueError("prediction length mismatch")
        if p.calibrator is not None:
            variants["calibrated"] = probabilities(p.calibrator.transform(variants["raw"]))
            if len(variants["calibrated"]) != len(y):
                raise ValueError("calibrated prediction length mismatch")
        flags = apply_policy(variants[cfg.release_variant], cfg.policy)
        frame = pd.DataFrame(
            {
                p.key_col: p.hold_keys,
                "y_true": y,
                **{f"p_{v}": score for v, score in variants.items()},
            }
        )
        if flags is not None:
            frame["policy_flag"] = flags
        _save_predictions(frame, out / "holdout_predictions.parquet")
        metrics, differences = bootstrap_metrics(y, variants, cfg)
        subgroups = subgroup_rows(X.reset_index(drop=True), y, variants, flags, cfg)
        acceptance = assess_criteria(metrics, cfg)
        warnings = [
            "Intervals condition on this fitted predictor, excluding training, calibration-fitting and selection uncertainty.",
            "IID-row bootstrap requires independent rows; grouped/time-dependent observations need another uncertainty design.",
            "Reliability uses pointwise 95% Wilson intervals, not simultaneous bands; bootstrap confidence is configured separately.",
            "Brier/log loss measure overall probability quality, not calibration alone.",
            "Raw/calibrated comparisons do not authorize selecting a new recipe on the holdout.",
            "Development metrics, subgroup findings and error examples are exploratory.",
            "Exposure tracking cannot detect unrecorded manual access or use in other databases.",
        ] + list(cfg.limitations)
        if prior:
            warnings.append("Prior exposure exists: not untouched first confirmation.")
        if flags is None:
            warnings.append("No policy selected: operational readiness not assessed.")
        if not cfg.criteria:
            warnings.append("No predeclared acceptance criteria: acceptance not assessed.")
        if any(r["status"] == "insufficient_support" for r in subgroups):
            warnings.append("Some slices have insufficient support; their metrics are suppressed.")
        if any(r["n_invalid"] for r in metrics):
            warnings.append(
                "Some bootstrap draws yield undefined metrics; inspect valid/invalid counts."
            )
        summary = {
            "evaluation_id": eid,
            "identity_hash": p.identity_hash,
            "parent_run_id": cfg.parent_run_id,
            "calibration_id": cfg.calibration_id,
            "release_variant": cfg.release_variant,
            "n_rows": len(y),
            "n_positives": int(y.sum()),
            "base_rate": float(y.mean()),
            "primary_metric": cfg.primary_metric,
            "primary_value": next(
                r["value"]
                for r in metrics
                if r["variant"] == cfg.release_variant and r["metric"] == cfg.primary_metric
            ),
            "criteria_status": acceptance["status"],
            "prior_exposure": bool(prior),
            "operational_policy_assessed": flags is not None,
            "automatic_promotion": False,
        }
        data = clean(
            {
                "summary": summary,
                "spec": cfg.model_dump(mode="json"),
                "provenance": {
                    **p.provenance,
                    "prior_exposures": prior,
                    "bootstrap": cfg.uncertainty.model_dump(),
                },
                "warnings": warnings,
                "metrics": metrics,
                "differences": differences,
                "policy_selection": p.provenance.get("policy_selection"),
                "policy_results": []
                if flags is None
                else [
                    {
                        "variant": cfg.release_variant,
                        "kind": cfg.policy.kind,
                        "value": cfg.policy.value,
                        "source": cfg.policy.source,
                        "batch_definition": cfg.policy.batch_definition,
                        **policy_metrics(y, flags),
                    }
                ],
                "reliability": reliability_rows(y, variants, cfg.reliability_bins),
                "subgroups": subgroups,
                "errors": error_rows(frame, X.reset_index(drop=True), flags, cfg),
                "curves": curve_rows(y, variants),
                "acceptance": acceptance,
                "development_context": _development_context(p),
            }
        )
        for name, payload in [
            ("report_data.json", data),
            ("summary.json", summary),
            ("acceptance.json", acceptance),
            ("provenance.json", data["provenance"]),
        ]:
            write_json(out / name, payload)
        for key in [
            "metrics",
            "differences",
            "policy_results",
            "reliability",
            "subgroups",
            "errors",
            "curves",
        ]:
            if data[key]:
                pd.DataFrame(data[key]).to_csv(out / f"{key}.csv", index=False)
            else:
                (out / f"{key}.csv").write_text("status\nnot_requested_or_unavailable\n")
        render_report(data, out)
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
    }


def evaluate_holdout(
    engine, run_id, *, force=False, silver_root=None, split_root=None, adapter_loader=None
):
    if force:
        raise ValueError(
            "destructive force is retired; use a documented revised evaluation specification"
        )
    result = evaluate_spec(
        engine,
        EvaluationConfig(parent_run_id=run_id),
        silver_root=silver_root,
        split_root=split_root,
        adapter_loader=adapter_loader,
    )
    return {
        r["metric"]: r["value"]
        for r in verify(result["artifact_dir"])["metrics"]
        if r["variant"] == "raw"
    }


def main(argv=None):
    from ml_template.evaluation.regression_evaluate import default_spec_for_run

    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--config")
    source.add_argument("--run-id")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = p.parse_args(argv)
    from ml_template.db.connection import get_engine
    from ml_template.tracking.pipeline_logging import setup_logging

    setup_logging()
    cfg = (
        load_evaluation_config(args.config)
        if args.config
        else default_spec_for_run(get_engine(), args.run_id)
    )
    print(
        json.dumps(
            clean(evaluate_spec(get_engine(), cfg, check_only=args.check)),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()

# calibrated-policy update v1
