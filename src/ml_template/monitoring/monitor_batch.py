"""Artifact-consuming operational monitoring commands."""

import argparse
import json
import sqlite3
from pathlib import Path

import pandas as pd

from ml_template.db import monitoring_registry as registry
from ml_template.monitoring.artifacts import new_id, now, seal, write_json
from ml_template.monitoring.drift import compare
from ml_template.monitoring.label_inputs import import_labels, load_labels
from ml_template.monitoring.monitor_config import (
    LabelConfig,
    MonitorConfig,
    ReferenceConfig,
    load_request,
)
from ml_template.monitoring.monitor_inputs import load_batch
from ml_template.monitoring.performance import evaluate_predictions
from ml_template.monitoring.reference import build_reference, load_reference


def quality_report(batch, ranges):
    frame = batch.source
    unknown = set(ranges) - (set(frame.columns) - {batch.key})
    if unknown:
        raise ValueError(f"unknown quality-range features: {sorted(unknown)}")
    result = []
    for name in frame.columns:
        if name == batch.key:
            continue
        series = frame[name]
        item = {
            "feature": name,
            "n_rows": len(series),
            "missing": int(series.isna().sum()),
            "outside_range": None,
        }
        if name in ranges:
            if not pd.api.types.is_numeric_dtype(series.dtype):
                raise ValueError(f"numeric range requested for nonnumeric feature: {name}")
            low, high = ranges[name]
            item.update(
                lower=low, upper=high, outside_range=int(((series < low) | (series > high)).sum())
            )
        result.append(item)
    return {
        "n_rows": len(frame),
        "duplicate_keys": int(frame[batch.key].duplicated().sum()),
        "features": result,
        "scope": "successful batches only; rejected attempts are outside this backend batch",
    }


def findings_for(config, *, drift, quality, performance, reference_n, sampling):
    findings = []

    def add(rule, severity, message, evidence):
        findings.append(
            {
                "rule": rule,
                "severity": severity,
                "message": message,
                "evidence": evidence,
                "action": "investigate; do not automatically refit or change policy",
            }
        )

    enough_rows = quality["n_rows"] >= config.min_rows and reference_n >= config.min_rows
    if not enough_rows:
        add(
            "sample_support",
            "info",
            "Distribution warnings suppressed: insufficient rows",
            {"current": quality["n_rows"], "reference": reference_n, "minimum": config.min_rows},
        )
    for item in quality["features"]:
        if item.get("outside_range"):
            add(
                "quality_range",
                "warning",
                f"Values outside configured range: {item['feature']}",
                item,
            )
    if enough_rows:
        for item in drift:
            if item["js_divergence"] >= config.js_warning:
                add(
                    "distribution_change",
                    "warning",
                    f"Distribution changed: {item['name']}",
                    {"measurement": item, "threshold": config.js_warning},
                )
    if performance is None:
        add("no_labels", "info", "No labels selected; outcome performance is unknown", {})
        return findings
    coverage = performance["mature_label_coverage"]
    if coverage is None or coverage < config.min_label_coverage:
        add(
            "label_coverage",
            "warning",
            "Mature outcomes are incompletely labelled",
            {"coverage": coverage, "required": config.min_label_coverage},
        )
    if sampling != "all_rows":
        add(
            "label_selection",
            "warning",
            "Label collection is selected or unknown",
            {"sampling": sampling},
        )
    if not performance["sufficient_support"]:
        add(
            "outcome_support",
            "info",
            "Performance warnings suppressed: insufficient class support",
            {name: performance[name] for name in ("n_evaluated", "n_positive", "n_negative")},
        )
    eligible = (
        performance["sufficient_support"]
        and coverage is not None
        and coverage >= config.min_label_coverage
        and sampling == "all_rows"
    )
    if eligible:
        metrics = performance["score_metrics"].get("p_release", {})
        brier = metrics.get("brier")
        if config.brier_warning is not None and brier is not None and brier > config.brier_warning:
            add(
                "brier_limit",
                "warning",
                "Release Brier score exceeds configured limit",
                {"value": brier, "limit": config.brier_warning},
            )
        policy = performance.get("policy") or {}
        recall = policy.get("recall")
        if config.recall_floor is not None and recall is not None and recall < config.recall_floor:
            add(
                "recall_limit",
                "warning",
                "Saved-policy recall is below configured floor",
                {"value": recall, "floor": config.recall_floor},
            )
    return findings


def prepare(engine, config):
    batch = load_batch(engine, config.scoring_batch_id)
    from ml_template.monitoring.nn_monitoring_contract import context_for

    nn_context = context_for(batch, config)
    if nn_context is not None:
        batch.summary = {**batch.summary, "nn_context": nn_context}
    reference = load_reference(engine, config.reference_id, batch)
    labels = load_labels(engine, config.label_import_id, batch) if config.label_import_id else None
    quality_report(batch, config.quality_ranges)
    from ml_template.monitoring.regression import validate_options

    validate_options(batch, config, reference=reference, labels=labels)
    return batch, reference, labels


def evaluate(engine, config, *, output_root, check_only=False):
    if check_only:
        batch, _, _ = prepare(engine, config)
        return {
            "ready": True,
            "scoring_batch_id": batch.batch_id,
            "files_written": False,
            "predictions_computed": False,
        }
    identity = new_id("mon")
    directory = Path(output_root).resolve() / identity
    registry.start_run(engine, identity, config, directory)
    release_id = None
    try:
        directory.mkdir(parents=True, exist_ok=False)
        batch, reference, labels = prepare(engine, config)
        release_id = batch.release_id
        quality = quality_report(batch, config.quality_ranges)
        drift = []
        for name, item in reference["profiles"].items():
            scope, column = name.split(":", 1)
            source = batch.source if scope == "input" else batch.predictions
            from ml_template.monitoring.nullable_drift import nullable_compare

            comparator = nullable_compare if item.get("kind") == "nullable_v1" else compare
            drift.append({"name": name, **comparator(source[column], item)})
        performance, sampling = None, None
        if labels:
            sampling = labels["metadata"]["sampling"]
            evaluator = evaluate_predictions
            if batch.task == "regression":
                from ml_template.monitoring.regression import (
                    evaluate_predictions as evaluate_regression,
                )

                evaluator = evaluate_regression
            performance = evaluator(
                batch.predictions,
                labels["frame"],
                batch.key,
                as_of=config.as_of,
                min_labelled=config.min_labelled,
                min_positives=config.min_positives,
                min_negatives=config.min_negatives,
            )
        finder = findings_for
        if batch.task == "regression":
            from ml_template.monitoring.regression import findings as regression_findings

            finder = regression_findings
        findings = finder(
            config,
            drift=drift,
            quality=quality,
            performance=performance,
            reference_n=reference["n_rows"],
            sampling=sampling,
        )
        if batch.task == "regression":
            from ml_template.evaluation.regression_postprocessing import add_monitoring

            add_monitoring(config, batch, labels, performance, findings)
        if reference["scoring_batch_id"] == batch.batch_id:
            findings.append(
                {
                    "rule": "self_reference",
                    "severity": "info",
                    "message": "Current batch is the reference: a plumbing check, not drift evidence",
                    "evidence": {},
                }
            )
        assessment = (
            "review"
            if any(item["severity"] == "warning" for item in findings)
            else "no_configured_warning"
        )
        summary = {
            "schema_version": 1,
            "monitoring_id": identity,
            "scoring_batch_id": batch.batch_id,
            "release_id": batch.release_id,
            "reference_id": config.reference_id,
            "label_import_id": config.label_import_id,
            "as_of": config.as_of.isoformat(),
            "created_at": now(),
            "n_rows": len(batch.source),
            "n_flagged": batch.summary.get("n_flagged"),
            "flagged_fraction": batch.summary.get("flagged_fraction"),
            "duration_s": batch.summary.get("duration_s"),
            "policy_kind": batch.summary["policy"]["kind"],
            "assessment": assessment,
            "label_sampling": sampling,
            "scoring_artifact_hashes": batch.artifact_hashes,
            "limitations": [
                "Descriptive comparisons; no significance claims",
                "No automatic retraining or promotion",
                "Outcome metrics describe the labelled mature subset",
                "A fixed alert budget does not establish model health",
                "Integrity hashes are not signatures",
            ],
        }
        if batch.task == "regression":
            summary.update(
                schema_version=2,
                task="regression",
                target_unit=config.target_unit,
                prediction_kind="continuous",
                limitations=[
                    "Descriptive comparisons, not significance tests",
                    "No automatic retraining or promotion",
                    "Outcome metrics describe the labelled mature subset",
                    "Distribution stability does not establish predictive accuracy",
                    "Units are explicit operator declarations, not inferred conversions",
                    "Integrity hashes are not signatures",
                ],
            )
        if batch.summary.get("model_type") == "nn":
            from ml_template.monitoring.nn_monitoring_contract import assessment_scope

            summary.update(
                model_type="nn",
                nn_context=batch.summary["nn_context"],
                assessment_scope=assessment_scope(
                    config,
                    performance,
                    sampling,
                    reference["n_rows"],
                    len(batch.source),
                    self_reference=reference["scoring_batch_id"] == batch.batch_id,
                ),
            )
        write_json(directory / "config.json", config.model_dump(mode="json"))
        write_json(directory / "quality.json", quality)
        write_json(directory / "drift.json", drift)
        write_json(directory / "performance.json", performance)
        write_json(directory / "findings.json", findings)
        write_json(directory / "summary.json", summary)
        manifest_hash = seal(directory)
        registry.finish_run(
            engine,
            identity,
            release_id=release_id,
            manifest_hash=manifest_hash,
            assessment=assessment,
        )
    except Exception as exc:
        registry.finish_run(
            engine, identity, release_id=release_id, error=f"{type(exc).__name__}: {exc}"
        )
        raise
    return {"monitoring_id": identity, "artifact_dir": str(directory), "assessment": assessment}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", choices=("reference", "labels", "monitor"), required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--database", type=Path, default=Path("data/project.db"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/monitoring"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    cls, action = {
        "reference": (ReferenceConfig, build_reference),
        "labels": (LabelConfig, import_labels),
        "monitor": (MonitorConfig, evaluate),
    }[args.operation]
    config = load_request(args.config, cls)
    database = args.database.resolve()
    if not database.is_file():
        raise ValueError("existing project database is required")
    uri = database.as_uri() + ("?mode=ro" if args.check else "?mode=rw")
    with sqlite3.connect(uri, uri=True) as conn:
        if not args.check:
            registry.ensure_tables(conn)
        result = action(conn, config, output_root=args.output_root, check_only=args.check)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
