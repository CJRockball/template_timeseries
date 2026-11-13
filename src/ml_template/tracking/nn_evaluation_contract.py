"""Read-only NN evaluation evidence validation; no model/evaluator/source imports."""

import hashlib
import json
import math
from pathlib import Path, PurePosixPath

import numpy as np


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def safe(root, name):
    root = Path(root).resolve()
    part = PurePosixPath(name)
    if (
        not name
        or part.is_absolute()
        or ".." in part.parts
        or ("\\" in name)
        or (part.as_posix() != name)
    ):
        raise ValueError("Unsafe frozen artifact path")
    path = root.joinpath(*part.parts)
    if (
        not path.resolve().is_relative_to(root)
        or path.is_symlink()
        or any(p.is_symlink() for p in path.parents if p != root and p.is_relative_to(root))
    ):
        raise ValueError("Unsafe frozen artifact symlink")
    return path


def read_json(path):

    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError("Duplicate frozen JSON key")
            result[k] = v
        return result

    def invalid(v):
        raise ValueError("Nonfinite frozen JSON constant: " + v)

    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_document(root):
    entries = read_json(safe(root, "artifact_manifest.json"))["files"]
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
    if not isinstance(entries, dict) or not required <= set(entries):
        raise ValueError("Incomplete frozen NN evaluation")
    for name, item in entries.items():
        if sha256(safe(root, name)) != item["sha256"]:
            raise ValueError("Frozen NN artifact changed: " + name)
    data = read_json(safe(root, "report_data.json"))
    for key in ("summary", "spec", "provenance", "acceptance", "residual_summary"):
        if data[key] != read_json(safe(root, key + ".json")):
            raise ValueError("Frozen NN report/sidecar differs: " + key)
    return data


def prediction_binding(p):
    return {
        "predictor": p["predictor"],
        "input_hashes": {
            k: v for k, v in p["input_hashes"].items() if k != "regression_calibration"
        },
        "feature_preparation": p["feature_preparation"],
        "code_hashes": {
            k: v
            for k, v in p["code_hashes"].items()
            if k.startswith(("models/", "features/", "preprocessing/", "training/"))
            or k in {"evaluation/regression_inputs.py", "evaluation/regression_postprocessing.py"}
        },
        "versions": p["versions"],
        "python": p["python"],
    }


def interval_state(spec, p):
    u = spec["uncertainty"]
    selected = u.get("coverages", [])
    if (
        type(u["enabled"]) is not bool
        or bool(selected) != u["enabled"]
        or selected != sorted(set(selected))
    ):
        raise ValueError("Invalid frozen interval request")
    if spec.get("calibration_id") is None:
        if (
            selected
            or spec["release_variant"] != "raw"
            or "postprocessing" in p
            or ("regression_calibration" in p["input_hashes"])
        ):
            raise ValueError("Unexpected NN postprocessing")
        return (None, {})
    item = p["postprocessing"]
    state = item["state"]
    c = state["config"]
    parameters = state["parameters"]
    if (
        digest({k: v for k, v in state.items() if k != "payload_sha256"}) != state["payload_sha256"]
        or state["calibration_id"] != spec["calibration_id"]
        or item["calibration_id"] != spec["calibration_id"]
        or (c.get("model_type") != "nn")
        or (c["parent_run_id"] != spec["parent_run_id"])
        or (c["predictor_kind"] != spec["predictor"]["kind"])
        or (c["target_unit"] != spec["target_unit"])
        or (state["binding"] != prediction_binding(p))
        or (item["sha256"] != p["input_hashes"].get("regression_calibration"))
        or (state.get("holdout_labels_loaded") is not False)
    ):
        raise ValueError("Frozen NN calibration binding differs")
    correction = parameters["correction"]
    if (
        set(parameters) != {"schema_version", "correction", "n_calibration", "quantiles"}
        or type(parameters["schema_version"]) is not int
        or parameters["schema_version"] != 1
        or (set(correction) != {"method", "slope", "intercept"})
        or (correction["method"] not in {"none", "bias", "affine"})
        or (not number(correction["slope"]))
        or (not number(correction["intercept"]))
        or (correction["method"] != c["correction"])
        or (correction["method"] in {"none", "bias"} and correction["slope"] != 1)
        or (correction["method"] == "none" and correction["intercept"] != 0)
        or ((spec["release_variant"] == "recalibrated") != (correction["method"] != "none"))
    ):
        raise ValueError("Invalid frozen NN correction")
    n = parameters["n_calibration"]
    quantiles = parameters["quantiles"]
    if type(n) is not int or n < 0 or (not isinstance(quantiles, dict)):
        raise ValueError("Invalid frozen calibration support")
    ordered = []
    for tag, q in quantiles.items():
        level = q["coverage"]
        if not number(level) or not 0 < level < 1:
            raise ValueError("Invalid coverage")
        code = round(level * 1000000)
        rank = ((n + 1) * code + 999999) // 1000000
        if (
            not 0 < code < 1000000
            or abs(code / 1000000 - level) > 1e-12
            or tag != "c" + str(code)
            or (set(q) != {"coverage", "rank", "half_width"})
            or (type(q["rank"]) is not int)
            or (q["rank"] != rank)
            or (not 1 <= rank <= n)
            or (not number(q["half_width"]))
            or (q["half_width"] < 0)
        ):
            raise ValueError("Invalid finite conformal quantile")
        ordered.append((level, q["half_width"]))
    ordered.sort()
    if [x[0] for x in ordered] != c["coverages"] or [x[1] for x in ordered] != sorted(
        x[1] for x in ordered
    ):
        raise ValueError("Non-nested or inconsistent calibrated levels")
    chosen = {}
    for level in selected:
        if not number(level):
            raise ValueError("Invalid selected coverage")
        tag = "c" + str(round(level * 1000000))
        if tag not in quantiles or quantiles[tag]["coverage"] != level:
            raise ValueError("Uncalibrated selected level")
        chosen[tag] = quantiles[tag]
    return (parameters, chosen)


def validate_report(data, record, stored):
    s, cfg, p = (data["summary"], data["spec"], data["provenance"])
    if (
        record["status"] != "succeeded"
        or not record.get("accessed_at")
        or s.get("model_type") != "nn"
        or (type(s.get("nn_evaluation_schema")) is not int)
        or (s.get("nn_evaluation_schema") != 1)
        or (p.get("nn_evaluation_schema") != 1)
        or (s.get("task") != "regression")
        or (cfg.get("task") != "regression")
        or (cfg.get("schema_version") != 2)
        or (type(s["n_rows"]) is not int)
        or (s["n_rows"] < 1)
        or (s.get("automatic_promotion") is not False)
        or (data["acceptance"].get("automatic_promotion") is not False)
    ):
        raise ValueError("Invalid frozen NN evaluation semantics")
    if (
        s["evaluation_id"] != record["evaluation_id"]
        or cfg["parent_run_id"] != record["parent_run_id"]
        or s["parent_run_id"] != record["parent_run_id"]
        or (cfg != json.loads(record["spec_json"]))
        or (digest({"spec": cfg, "provenance": p}) != record["identity_hash"])
        or (s["identity_hash"] != record["identity_hash"])
        or (cfg["expected_silver_dataset_version_id"] != record["silver_id"])
        or (p["holdout_fingerprint"] != record["holdout_fingerprint"])
        or (record.get("calibration_id") != cfg.get("calibration_id"))
        or (s.get("calibration_id") != cfg.get("calibration_id"))
        or (s["release_variant"] != cfg["release_variant"])
    ):
        raise ValueError("Frozen NN registry identity differs")
    prior = json.loads(record["prior_json"])
    if (
        data["prior_exposures"] != prior
        or type(s["prior_exposure"]) is not bool
        or s["prior_exposure"] != bool(prior)
    ):
        raise ValueError("Frozen prior-exposure evidence differs")
    unit = cfg.get("target_unit")
    if (
        not isinstance(unit, str)
        or not unit.strip()
        or unit.strip() in {"target units", "target_units"}
        or (s.get("target_unit") != unit)
    ):
        raise ValueError("Explicit consistent NN units required")
    d = p["predictor"]
    members = d["members"]
    if (
        type(d["schema_version"]) is not int
        or d["schema_version"] != 3
        or d["task"] != "regression"
        or (d["prediction_kind"] != "continuous")
        or (d["model_type"] != "nn")
        or (d["fitted_features"] is not False)
        or (d["weighting"] != "equal")
        or (d["kind"] not in {"final", "cv_mean"})
        or (d["kind"] != cfg["predictor"]["kind"])
        or (d["kind"] != s["predictor_kind"])
        or (
            d["serialization"]
            != {
                "format": "torch",
                "weights_only": True,
                "checkpoint_version": 1,
                "preprocessing": "embedded",
            }
        )
        or (d["serialization"]["weights_only"] is not True)
        or (type(d["serialization"]["checkpoint_version"]) is not int)
        or (not isinstance(members, list))
        or (not members)
        or (d["kind"] == "final" and len(members) != 1)
        or (d["kind"] == "cv_mean" and len(members) < 2)
    ):
        raise ValueError("Frozen NN descriptor differs")
    names = {"config.json", "features.json"}
    model_names = set()
    if d["kind"] == "cv_mean":
        names.add("cv_folds.csv")
    for i, m in enumerate(members):
        name = "model.ubj" if d["kind"] == "final" else f"fold_models/fold_{i}.ubj"
        if (
            m["model_path"] != name
            or m["transformer_path"] is not None
            or (not number(m["weight"]))
            or (abs(m["weight"] - 1 / len(members)) > 1e-15)
            or (
                m["fold_id"] is not None
                if d["kind"] == "final"
                else type(m["fold_id"]) is not int or m["fold_id"] != i
            )
        ):
            raise ValueError("Invalid NN member inventory")
        names.add(name)
        model_names.add(name)
    if (
        set(d["files_sha256"]) != names
        or set(d["training_rows"]) != model_names
        or any(type(v) is not int or v < 1 for v in d["training_rows"].values())
        or any(
            not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
            for v in d["files_sha256"].values()
        )
    ):
        raise ValueError("NN hash/fit-row coverage differs")
    _, chosen = interval_state(cfg, p)
    if s["interval_coverages"] != [v["coverage"] for v in chosen.values()]:
        raise ValueError("Selected interval levels differ")
    metrics = data["metrics"]
    lookup = {r["metric"]: r for r in metrics}
    directions = {"rmse": "min", "mae": "min", "r2": "max"}
    expected = set(directions) | {
        name + "_" + tag for tag in chosen for name in ["empirical_coverage", "mean_width"]
    }
    if len(lookup) != len(metrics) or set(lookup) != expected:
        raise ValueError("Duplicate/incomplete frozen NN metrics")
    for name, row in lookup.items():
        value = row["value"]
        if row["variant"] != cfg["release_variant"]:
            raise ValueError("Metric variant differs")
        if name in directions:
            if row["direction"] != directions[name] or row["uncertainty"] != "not_requested":
                raise ValueError("Point metric semantics differ")
            if value is None:
                if name != "r2" or row["status"] != "undefined":
                    raise ValueError("Undefined error metric")
            elif not number(value) or row["status"] != "defined" or (name != "r2" and value < 0):
                raise ValueError("Invalid point score")
        elif (
            row["direction"] != "diagnostic"
            or row["status"] != "defined"
            or row["uncertainty"] != "split_conformal_prediction_interval"
            or (not number(value))
            or (value < 0)
            or (name.startswith("empirical_coverage_") and value > 1)
        ):
            raise ValueError("Invalid interval metric")
    if (
        s["primary_metric"] != cfg["primary_metric"]
        or s["primary_metric_direction"] != "min"
        or s["primary_value"] != lookup[cfg["primary_metric"]]["value"]
    ):
        raise ValueError("Primary score differs")
    criteria = []
    for criterion in cfg.get("criteria", []):
        value = lookup[criterion["metric"]]["value"]
        passed = (
            None
            if value is None
            else bool(
                value <= criterion["value"]
                if criterion["operator"] == "<="
                else value >= criterion["value"]
            )
        )
        criteria.append(
            {
                **criterion,
                "observed": value,
                "passed": passed,
                "status": "unavailable" if passed is None else "passed" if passed else "failed",
            }
        )
    status = (
        "not_assessed"
        if not criteria
        else "failed"
        if any(r["status"] == "failed" for r in criteria)
        else "unavailable"
        if any(r["status"] == "unavailable" for r in criteria)
        else "passed"
    )
    if (
        data["acceptance"]["results"] != criteria
        or data["acceptance"]["status"] != status
        or s["criteria_status"] != status
    ):
        raise ValueError("Criteria differ from frozen scores")

    def key(r):
        return (r["variant"], r["metric"])

    if sorted(metrics, key=key) != sorted(stored, key=key):
        raise ValueError("Frozen scores differ from registered metrics")
    return data


def validate_predictions(frame, data):
    s, cfg, p = (data["summary"], data["spec"], data["provenance"])
    parameters, chosen = interval_state(cfg, p)
    numeric = {"y_true", "y_pred", "y_pred_raw", "residual", "absolute_error"} | {
        prefix + tag for tag in chosen for prefix in ["pi_lower_", "pi_upper_"]
    }
    keys = [c for c in frame if c not in numeric]
    if (
        not frame.columns.is_unique
        or len(keys) != 1
        or set(frame) != numeric | set(keys)
        or (len(frame) != s["n_rows"])
        or frame[keys[0]].isna().any()
        or frame[keys[0]].duplicated().any()
    ):
        raise ValueError("Saved NN key/schema/support differs")
    encoded = []
    for value in frame[keys[0]]:
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            raise ValueError("Invalid saved key")
        encoded.append(json.dumps([type(value).__name__, value]))
    if (
        digest(
            {
                "silver": cfg["expected_silver_dataset_version_id"],
                "key": keys[0],
                "members": sorted(encoded),
            }
        )
        != p["holdout_fingerprint"]
    ):
        raise ValueError("Saved population fingerprint differs")
    for name in numeric:
        a = frame[name].to_numpy()
        if a.dtype.kind not in "iuf" or not np.isfinite(a).all():
            raise ValueError("Invalid saved numeric values")
    if not np.allclose(frame.residual, frame.y_true - frame.y_pred) or not np.allclose(
        frame.absolute_error, frame.residual.abs()
    ):
        raise ValueError("Saved residuals differ")
    point = frame.y_pred_raw.to_numpy(float)
    if parameters is not None:
        c = parameters["correction"]
        point = point * c["slope"] + c["intercept"]
    if not np.allclose(point, frame.y_pred, atol=1e-10, rtol=1e-10):
        raise ValueError("Saved point correction differs")
    for tag, q in chosen.items():
        if not np.allclose(
            frame["pi_lower_" + tag], point - q["half_width"], atol=1e-10, rtol=1e-10
        ) or not np.allclose(
            frame["pi_upper_" + tag], point + q["half_width"], atol=1e-10, rtol=1e-10
        ):
            raise ValueError("Saved interval arithmetic differs")
    return frame
