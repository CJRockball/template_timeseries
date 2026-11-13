"""Frozen binding hooks for evaluation, packaging, scoring and monitoring."""

from pathlib import Path

import numpy as np

from ml_template.models.regression_postprocessing import (
    apply,
    interval_columns,
    interval_metrics,
    validate_parameters,
)


def attach(engine, spec, provenance):
    from ml_template.training.regression_calibration import load_calibration

    item = load_calibration(
        engine, spec.calibration_id, spec.parent_run_id, provenance, spec.target_unit
    )
    method = item["state"]["parameters"]["correction"]["method"]
    if (spec.release_variant == "recalibrated") != (method != "none"):
        raise ValueError("evaluation variant differs from fitted correction")
    interval_columns(
        item["state"]["parameters"],
        list(spec.uncertainty.coverages) if spec.uncertainty.enabled else [],
    )
    provenance["postprocessing"] = item
    provenance["input_hashes"]["regression_calibration"] = item["sha256"]


def selected(p):
    return list(p.spec.uncertainty.coverages) if p.spec.uncertainty.enabled else []


def evaluate(p, raw, target):
    if "postprocessing" not in p.provenance:
        return raw, {}, []
    state = p.provenance["postprocessing"]["state"]["parameters"]
    point, bounds = apply(raw, state, selected(p))
    measured = interval_metrics(target, bounds, interval_columns(state, selected(p)))
    rows = [
        {
            "variant": p.spec.release_variant,
            "metric": metric + "_" + tag,
            "value": item[metric],
            "direction": "diagnostic",
            "status": "defined",
            "uncertainty": "split_conformal_prediction_interval",
        }
        for tag, item in measured.items()
        for metric in ("empirical_coverage", "mean_width")
    ]
    return point, {"y_pred_raw": raw, **bounds}, rows


def reference_columns(material):
    p = material["prepared"]
    provenance = getattr(p, "provenance", {})
    if "postprocessing" not in provenance:
        return []
    state = provenance["postprocessing"]["state"]["parameters"]
    return [
        "y_pred_raw",
        *[
            name
            for item in interval_columns(state, selected(p)).values()
            for name in (item["lower"], item["upper"])
        ],
    ]


def package(stage, material, info):
    from ml_template.scoring.release_artifacts import read_json, sha256

    p = material["prepared"]
    item = p.provenance.get("postprocessing")
    if item is None:
        return
    source = Path(item["artifact_dir"]) / "calibration.json"
    if sha256(source) != item["sha256"] or read_json(source) != item["state"]:
        raise ValueError("calibration changed after evaluation preflight")
    dest = Path(stage) / "regression_postprocessing.json"
    dest.write_bytes(source.read_bytes())
    if sha256(dest) != item["sha256"]:
        raise ValueError("copied calibration differs")
    info.update(
        target_unit=p.spec.target_unit,
        interval_columns=interval_columns(item["state"]["parameters"], selected(p)),
        postprocessing={
            "schema_version": 1,
            "calibration_id": p.spec.calibration_id,
            "sha256": item["sha256"],
            "coverages": selected(p),
            "parameters": item["state"]["parameters"],
        },
    )


def load_packaged(directory, info, provenance, spec):
    from ml_template.scoring.release_artifacts import read_json, sha256
    from ml_template.training.regression_calibration import fingerprint, prediction_binding

    declared = info.get("postprocessing")
    if declared is None:
        if (
            info.get("calibration_id") is not None
            or info.get("release_variant") != "raw"
            or "postprocessing" in provenance
            or info.get("interval_columns")
            or (Path(directory) / "regression_postprocessing.json").exists()
        ):
            raise ValueError("missing declared postprocessing state")
        return None
    expected = provenance.get("postprocessing")
    if (
        not expected
        or declared["calibration_id"] != info["calibration_id"]
        or spec.get("calibration_id") != info["calibration_id"]
    ):
        raise ValueError("packaged/evaluated calibration identities differ")
    path = Path(directory) / "regression_postprocessing.json"
    state = read_json(path)
    payload = {k: v for k, v in state.items() if k != "payload_sha256"}
    if (
        fingerprint(payload) != state["payload_sha256"]
        or state != expected["state"]
        or state["binding"] != prediction_binding(provenance)
        or sha256(path) != expected["sha256"]
        or declared["sha256"] != expected["sha256"]
        or state["calibration_id"] != info["calibration_id"]
        or state["config"]["target_unit"] != info["target_unit"]
        or spec.get("target_unit") != info["target_unit"]
        or spec.get("release_variant") != info["release_variant"]
        or provenance["input_hashes"].get("regression_calibration") != declared["sha256"]
    ):
        raise ValueError("packaged calibration evidence differs")
    parameters = validate_parameters(state["parameters"])
    if declared["parameters"] != parameters:
        raise ValueError("packaged parameters differ")
    uncertainty = spec["uncertainty"]
    chosen = uncertainty.get("coverages", []) if uncertainty["enabled"] else []
    if (
        declared["coverages"] != chosen
        or info["interval_columns"] != interval_columns(parameters, chosen)
        or (info["release_variant"] == "recalibrated")
        != (parameters["correction"]["method"] != "none")
    ):
        raise ValueError("packaged variant/coverages differ")
    return declared


def add_predictions(frame, declared):
    if declared is None:
        return frame
    point, bounds = apply(
        frame.y_pred_raw.to_numpy(float), declared["parameters"], declared["coverages"]
    )
    result = frame.copy()
    result["y_pred_release"] = point
    for name, values in bounds.items():
        result[name] = values
    return result


def validate_saved_predictions(frame, summary, info):
    from ml_template.monitoring.regression import numeric

    declared = info["postprocessing"]
    parameters = validate_parameters(declared["parameters"])
    if (
        info.get("task") != "regression"
        or info.get("prediction_kind") != "continuous"
        or info["policy"]["kind"] != "none"
        or info["calibration_id"] != declared["calibration_id"]
        or summary.get("postprocessing") != declared
        or summary.get("calibration_id") != info["calibration_id"]
        or summary.get("release_variant") != info["release_variant"]
        or summary.get("target_unit") != info["target_unit"]
        or summary.get("n_flagged") is not None
        or summary.get("prediction_columns") != {"raw": "y_pred_raw", "release": "y_pred_release"}
        or info.get("prediction_columns") != summary.get("prediction_columns")
        or {"p_raw", "p_calibrated", "p_release", "policy_flag"} & set(frame)
    ):
        raise ValueError("saved postprocessing semantics differ")
    definitions = interval_columns(parameters, declared["coverages"])
    if info["interval_columns"] != definitions or summary.get("interval_columns") != definitions:
        raise ValueError("interval output schema differs")
    if not frame.columns.is_unique or len(frame.columns) != 5 + 2 * len(definitions):
        raise ValueError("unexpected interval prediction columns")
    point, bounds = apply(
        numeric(frame.y_pred_raw, "raw prediction"), parameters, declared["coverages"]
    )
    for name, expected in {"y_pred_release": point, **bounds}.items():
        if not np.allclose(expected, numeric(frame[name], name), atol=1e-10, rtol=1e-10):
            raise ValueError("output differs from frozen parameters: " + name)


def add_monitoring(config, batch, labels, performance, findings):
    if not batch.summary.get("postprocessing"):
        return
    if config.target_unit != batch.summary.get("target_unit"):
        raise ValueError("monitoring units differ from frozen interval units")
    if performance is None:
        return
    import pandas as pd

    joined = batch.predictions.merge(
        labels["frame"][[batch.key, "y_true", "eligible_at", "observed_at"]],
        on=batch.key,
        validate="one_to_one",
    )
    cutoff = pd.Timestamp(config.as_of)
    eligible = pd.to_datetime(joined.eligible_at, utc=True) <= cutoff
    known = joined.y_true.notna() & (pd.to_datetime(joined.observed_at, utc=True) <= cutoff)
    chosen = joined.loc[eligible & known]
    performance["interval_metrics"] = interval_metrics(
        chosen.y_true.to_numpy(float), chosen, batch.summary["interval_columns"]
    )
    supported = (
        performance["sufficient_support"]
        and performance["mature_label_coverage"] is not None
        and performance["mature_label_coverage"] >= config.min_label_coverage
        and labels["metadata"]["sampling"] == "all_rows"
    )
    if supported:
        for tag, item in performance["interval_metrics"].items():
            if (
                item["empirical_coverage"] is not None
                and item["empirical_coverage"]
                < item["nominal_coverage"] - config.interval_coverage_slack
            ):
                findings.append(
                    {
                        "rule": "interval_undercoverage",
                        "severity": "warning",
                        "message": "Low empirical interval coverage on mature labelled rows",
                        "evidence": {"level": tag, **item, "slack": config.interval_coverage_slack},
                        "action": "investigate; no automatic recalibration or promotion",
                    }
                )
