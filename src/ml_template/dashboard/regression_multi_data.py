"""Part 2 read-only regression cohorts, comparison, ranking and frozen evidence."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml_template.dashboard import regression_data as base

DIRECTIONS = {"rmse": "min", "mae": "min", "r2": "max"}
COHORT = ["dataset_name", "silver_dataset_version_id", "split_id", "primary_metric", "target_unit"]


def direction(metric):
    if metric not in DIRECTIONS:
        raise ValueError("Unsupported regression metric")
    return DIRECTIONS[metric]


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def all_records(engine):
    if not base.rows(
        engine, "SELECT name FROM sqlite_master WHERE type='table' AND name='model_runs'"
    ):
        return []
    return base.rows(engine, "SELECT * FROM model_runs")


def binary_runs(engine):
    from ml_template.dashboard import experiment_data

    table = experiment_data.load_experiment_runs(engine)
    if table.empty:
        return table
    mask = [base.task_hint(row) == "binary_classification" for row in table.to_dict("records")]
    return table.loc[mask].copy()


def binary_sweep_data(engine):
    from ml_template.dashboard import sweep_data

    launches, runs, params = sweep_data.load_sweep_data(engine)
    if runs.empty:
        return launches, runs, params
    mask = [base.task_hint(row) == "binary_classification" for row in runs.to_dict("records")]
    binary = runs.loc[mask].copy()
    related = set(binary.experiment_run_id.dropna())
    # Preserve launches without registered runs. Mixed-task expected counts cannot be apportioned.
    populated = set(runs.experiment_run_id.dropna())
    launches = launches[
        launches.experiment_run_id.isin(related | (set(launches.experiment_run_id) - populated))
    ].copy()
    mixed = set(runs.loc[~pd.Series(mask, index=runs.index), "experiment_run_id"]) & related
    launches.loc[launches.experiment_run_id.isin(mixed), "n_runs"] = None
    return launches, binary, params[params.run_id.isin(binary.run_id)].copy()


def nonbinary_frame(frame):
    """Classify a loader-provided summary frame without querying another source."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Summary input must be a DataFrame")
    if frame.empty:
        return False
    explicit_binary = False
    for name in ["task", "param::task"]:
        if name in frame:
            values = set(frame[name].dropna().astype(str))
            if values - {"binary_classification"}:
                return True
            explicit_binary = explicit_binary or bool(values)
    if "primary_metric" not in frame:
        if explicit_binary:
            return False
        raise ValueError("Summary frame lacks task/metric metadata")
    metrics = set(frame.primary_metric.dropna().astype(str))
    if metrics & {"rmse", "mae", "r2"} or len(metrics) > 1:
        return True
    if not metrics:
        if explicit_binary:
            return False
        if "primary_value" in frame:
            values = pd.to_numeric(frame.primary_value, errors="coerce")
            if not (values.notna() & np.isfinite(values)).any():
                return False
        raise ValueError("Cannot establish task/metric for scored summary rows")
    binary = {
        "auc",
        "aucpr",
        "roc_auc",
        "pr_auc",
        "brier",
        "log_loss",
        "logloss",
        "binary_logloss",
        "error",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "f2",
    }
    metric = next(iter(metrics))
    if metric in binary or metric.startswith(("precision_at_", "recall_at_")) or explicit_binary:
        return False
    raise ValueError("Unknown summary metric; task semantics cannot be assumed")


def nonbinary_launch(engine, launch, *, frame=None):
    """Frame-first check; registry still verifies real launches when available."""
    if frame is not None and nonbinary_frame(frame):
        return True
    if not launch:
        return False
    if engine is None:
        if frame is None:
            raise ValueError("Supply a validated summary frame or a real registry engine")
        # nonbinary_frame already validated the injected frame; no unchecked bypass.
        return False
    return any(
        base.task_hint(row) != "binary_classification"
        for row in base.rows(
            engine, "SELECT * FROM model_runs WHERE experiment_run_id=?", (launch,)
        )
    )


def _unit(value):
    return (
        value.strip()
        if isinstance(value, str) and value.strip() not in {"", "target_units", "target units"}
        else None
    )


def _recipe(cfg, h, ctx):
    value = copy.deepcopy(cfg)
    for field in ["analysis", "metrics", "seed", "experiment_name", "schema_version"]:
        value.pop(field, None)
    clean = h.get("git_dirty") in (False, 0) and bool(h.get("git_sha"))
    timestamp = pd.to_datetime(h.get("started_at"), utc=True, errors="coerce")
    environment = None
    if "environment.json" in ctx["entries"]:
        base.verify(ctx["root"], ctx["entries"], ["environment.json"])
        environment = base.read_json(base.safe(ctx["root"], "environment.json"))
    verified_code = clean and pd.notna(timestamp) and environment is not None
    code = {
        "git_sha": h.get("git_sha"),
        "git_dirty": h.get("git_dirty"),
        "environment": environment,
    }
    if not verified_code:
        code["unverified_run_id"] = h["run_id"]
    return digest({"recipe": value, "code": code})[:16], verified_code


def inventory(engine):
    records, problems, declarations = [], [], {}
    for h in all_records(engine):
        if base.task_hint(h) != "regression":
            continue
        if h.get("status") != "succeeded":
            records.append(
                {
                    **h,
                    "verified": False,
                    "eligible": False,
                    "task": "regression",
                    "reason": "Not a completed verified run",
                }
            )
            continue
        try:
            ctx = base.context(engine, h["run_id"])
            cfg = ctx["config"]
            primary = h["primary_metric"]
            direction(primary)
            if cfg["metrics"]["primary"] != primary:
                raise ValueError("Snapshot/registry primary metric differs")
            options, mapping, warnings = base.discover(ctx)
            problems.extend(h["run_id"] + ": " + str(w) for w in warnings)
            units = set()
            configured = _unit(cfg.get("analysis", {}).get("regression_target_unit"))
            if configured:
                units.add(configured)
            for directory, entries, _origin in mapping.values():
                base.verify(directory, entries, ["residual_summary.json"])
                declared = _unit(
                    base.read_json(base.safe(directory, "residual_summary.json")).get("target_unit")
                )
                if declared:
                    units.add(declared)
            dataset = (cfg["data"]["dataset_name"], h["silver_dataset_version_id"])
            declarations.setdefault(dataset, set()).update(units)
            recipe, code_verified = _recipe(cfg, h, ctx)
            metrics = base.rows(
                engine,
                "SELECT metric,value,fold FROM run_metrics WHERE run_id=? AND split='cv'",
                (h["run_id"],),
            )
            points = [r["value"] for r in metrics if r["metric"] == primary and r["fold"] == -1]
            score = points[0] if len(points) == 1 else None
            finite = score is not None and np.isfinite(float(score))
            if finite and (
                h.get("primary_value") is None
                or not np.isclose(float(h["primary_value"]), float(score), rtol=0, atol=1e-10)
            ):
                raise ValueError("Registry primary score differs from stored CV metric")
            records.append(
                {
                    **h,
                    "task": "regression",
                    "verified": True,
                    "eligible": finite,
                    "dataset_name": dataset[0],
                    "recipe_id": recipe,
                    "code_verified": code_verified,
                    "primary_value": float(score) if finite else np.nan,
                    "reason": None if finite else "Primary CV metric unavailable",
                    "_ctx": ctx,
                    "_metrics": metrics,
                    "_analyses": options,
                    "_unit_declarations": units,
                }
            )
        except base.ERRORS as exc:
            problems.append(h["run_id"] + ": " + str(exc))
            records.append(
                {
                    **h,
                    "verified": False,
                    "eligible": False,
                    "task": "regression",
                    "reason": str(exc),
                }
            )
    for row in records:
        if not row.get("verified"):
            continue
        units = declarations.get((row["dataset_name"], row["silver_dataset_version_id"]), set())
        if len(units) == 1:
            row["target_unit"] = next(iter(units))
            row["unit_source"] = (
                "consistent declaration for this immutable single-target Silver dataset"
            )
        else:
            row["target_unit"] = None
            row["eligible"] = False
            row["reason"] = (
                "Unknown/conflicting target units; record consistent units with the diagnostic CLI"
            )
    return records, problems


def cohort_options(records):
    tokens = {}
    for row in records:
        if row.get("verified") and row.get("eligible") and row.get("target_unit"):
            values = [row.get(c) for c in COHORT]
            if any(v is None for v in values):
                continue
            token = json.dumps(values, separators=(",", ":"))
            tokens[token] = " | ".join(str(v) for v in values)
    return [
        {"label": label, "value": token}
        for token, label in sorted(tokens.items(), key=lambda item: item[1])
    ]


def select_cohort(records, token):
    if not token:
        return []
    valid = {item["value"] for item in cohort_options(records)}
    if token not in valid:
        raise ValueError("Choose a verified regression cohort")
    values = json.loads(token)
    return [
        r
        for r in records
        if r.get("verified") and all(r.get(c) == v for c, v in zip(COHORT, values, strict=True))
    ]


def compatible(records):
    if len(records) < 2 or len({r["run_id"] for r in records}) != len(records):
        raise ValueError("Select at least two different regression runs")
    if not all(r.get("verified") and r.get("target_unit") for r in records):
        raise ValueError("Only verified runs with known units are comparable")
    if len({tuple(r.get(c) for c in COHORT) for r in records}) != 1:
        raise ValueError("Task, dataset, stored split, primary metric and target units must match")


def aligned_predictions(records):
    aligned, reference = {}, None
    for row in records:
        frame = base.optional_table(row["_ctx"], "oof_predictions.parquet")
        if frame is None:
            raise ValueError("Missing original OOF artifact")
        keys = [c for c in frame if c not in {"fold", "y_pred_oof"}]
        if len(keys) != 1 or frame.columns.duplicated().any():
            raise ValueError("Ambiguous OOF key schema")
        key = keys[0]
        if frame.empty or frame[key].isna().any() or frame[key].duplicated().any():
            raise ValueError("Invalid OOF keys")
        base.finite(frame, ["fold", "y_pred_oof"])
        if (frame.fold % 1 != 0).any() or set(frame.fold) != set(
            range(row["_ctx"]["config"]["split"]["n_folds"])
        ):
            raise ValueError("Invalid OOF folds")
        part = frame.set_index(key).sort_index()
        if reference is not None and (
            part.index.name != reference.index.name
            or not part.index.equals(reference.index)
            or not np.array_equal(part.fold, reference.fold)
        ):
            raise ValueError(
                "OOF key/fold coverage differs; intersection-only comparison forbidden"
            )
        aligned[row["run_id"]] = part
        reference = part
    return aligned


def comparison(engine, run_ids, metric="rmse", baseline=None, analyses=None):
    direction(metric)
    records, warnings = inventory(engine)
    lookup = {r["run_id"]: r for r in records}
    if any(rid not in lookup for rid in run_ids):
        raise ValueError("A run is not in the regression inventory")
    chosen = [lookup[rid] for rid in run_ids]
    compatible(chosen)
    baseline = baseline or run_ids[0]
    if baseline not in run_ids:
        raise ValueError("Baseline must be selected")
    prediction = aligned_predictions(chosen)
    values, folds = {}, []
    for row in chosen:
        pooled = [r["value"] for r in row["_metrics"] if r["fold"] == -1 and r["metric"] == metric]
        value = pooled[0] if len(pooled) == 1 else None
        values[row["run_id"]] = (
            float(value) if value is not None and np.isfinite(float(value)) else None
        )
        folds.extend(
            {**m, "run_id": row["run_id"]}
            for m in row["_metrics"]
            if m["metric"] == metric and m["fold"] >= 0
        )
    reference = values[baseline]
    point = []
    for row in chosen:
        value = values[row["run_id"]]
        delta = value - reference if value is not None and reference is not None else None
        point.append(
            {
                "run_id": row["run_id"],
                "metric": metric,
                "direction": direction(metric),
                "value": value,
                "score_minus_baseline": delta,
                "improvement_vs_baseline": (-delta if direction(metric) == "min" else delta)
                if delta is not None
                else None,
                "unit": "dimensionless" if metric == "r2" else row["target_unit"],
            }
        )
    analyses = analyses or {}
    if set(analyses) - set(run_ids):
        raise ValueError("Analysis provided for an unselected run")
    residuals = {}
    truth = None
    for row in chosen:
        rid = row["run_id"]
        if not analyses.get(rid):
            continue
        loaded = base.load_bundle(engine, rid, analyses[rid])
        if loaded["state"] != "verified" or loaded["analysis"] is None:
            raise ValueError(loaded.get("message", "Invalid explicit analysis"))
        analysis = loaded["analysis"]
        if analysis["summary"]["target_unit"] != row["target_unit"]:
            raise ValueError("Selected analysis units differ")
        frame = analysis["residuals"].set_index(prediction[rid].index.name).sort_index()
        if not frame.index.equals(prediction[rid].index) or not np.array_equal(
            frame.fold, prediction[rid].fold
        ):
            raise ValueError("Analysis OOF alignment differs")
        if truth is not None and not np.array_equal(frame.y_true, truth):
            raise ValueError("Recorded development truths differ")
        truth = frame.y_true.to_numpy()
        residuals[rid] = frame
    return {
        "runs": chosen,
        "point": pd.DataFrame(point),
        "folds": pd.DataFrame(folds),
        "predictions": prediction,
        "residuals": residuals,
        "baseline": baseline,
        "metric": metric,
        "unit": chosen[0]["target_unit"],
        "warnings": warnings,
        "all_analyses_selected": len(residuals) == len(chosen),
    }


def rank_recipes(records):
    usable = [
        r
        for r in records
        if r.get("verified")
        and r.get("eligible")
        and r.get("status") == "succeeded"
        and r.get("target_unit")
    ]
    if not usable:
        return pd.DataFrame()
    if len({tuple(r.get(c) for c in COHORT) for r in usable}) != 1:
        raise ValueError("Select one regression cohort before ranking")
    metric = usable[0]["primary_metric"]
    frame = pd.DataFrame(
        [
            {
                k: r.get(k)
                for k in [
                    "run_id",
                    "experiment_name",
                    "recipe_id",
                    "seed",
                    "primary_value",
                    "started_at",
                    "duration_s",
                    "code_verified",
                ]
            }
            for r in usable
        ]
    )
    seeds = pd.to_numeric(frame.seed, errors="coerce")
    frame = frame[seeds.notna() & np.isfinite(seeds) & (seeds % 1 == 0) & (seeds >= 0)].copy()
    if frame.empty:
        return pd.DataFrame()
    frame["_time"] = pd.to_datetime(frame.started_at, utc=True, errors="coerce")
    frame = frame.sort_values(["_time", "run_id"], ascending=[False, False], kind="stable")
    total = frame.groupby(["experiment_name", "recipe_id"]).size()
    kept = frame.drop_duplicates(["experiment_name", "recipe_id", "seed"])
    board = kept.groupby(["experiment_name", "recipe_id"]).agg(
        n_seeds=("seed", "nunique"),
        mean=("primary_value", "mean"),
        std=("primary_value", "std"),
        min=("primary_value", "min"),
        max=("primary_value", "max"),
        median_duration_s=("duration_s", "median"),
        code_verified=("code_verified", "all"),
    )
    board["excluded_repeats"] = total - kept.groupby(["experiment_name", "recipe_id"]).size()
    board = board.reset_index().sort_values(
        ["mean", "experiment_name", "recipe_id"],
        ascending=[direction(metric) == "min", True, True],
        kind="stable",
    )
    board.insert(0, "rank", range(1, len(board) + 1))
    board["metric"] = metric
    board["direction"] = direction(metric)
    board["target_unit"] = usable[0]["target_unit"]
    board["score_unit"] = "dimensionless" if metric == "r2" else usable[0]["target_unit"]
    return board


def launches(engine):
    if not base.rows(
        engine, "SELECT name FROM sqlite_master WHERE type='table' AND name='experiment_launches'"
    ):
        return []
    return base.rows(
        engine, "SELECT * FROM experiment_launches WHERE kind='sweep' ORDER BY started_at DESC"
    )


def sweep_progress(engine, launch_id):
    all_runs = base.rows(engine, "SELECT * FROM model_runs WHERE experiment_run_id=?", (launch_id,))
    regression = [r for r in all_runs if base.task_hint(r) == "regression"]
    selected = [r for r in launches(engine) if r["experiment_run_id"] == launch_id]
    expected = selected[0].get("n_runs") if selected else None
    return {
        "launch": launch_id,
        "expected_runs_all_tasks": expected,
        "registered_runs_all_tasks": len(all_runs),
        "unregistered_all_tasks": max(0, expected - len(all_runs))
        if expected is not None
        else None,
        "regression_registered": len(regression),
        **{
            s: sum(r["status"] == s for r in regression) for s in ["succeeded", "failed", "running"]
        },
    }


def population_fingerprint(silver_id, key, values):
    encoded = []
    for value in values:
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            raise ValueError("Saved entity keys must be integer or string")
        encoded.append(json.dumps([type(value).__name__, value]))
    return digest({"silver": silver_id, "key": key, "members": sorted(encoded)})


def _load_nn_evaluation(engine, record, directory, predictions):
    from ml_template.tracking.nn_evaluation_contract import (
        read_document,
        validate_predictions,
        validate_report,
    )

    data = read_document(directory)
    stored = base.rows(
        engine,
        "SELECT result_json FROM final_evaluation_metrics WHERE evaluation_id=?",
        (record["evaluation_id"],),
    )
    validate_report(data, record, [json.loads(row["result_json"]) for row in stored])
    frame = None
    if predictions:
        frame = pd.read_parquet(base.safe(directory, "holdout_predictions.parquet"))
        validate_predictions(frame, data)
    return {
        "state": "verified",
        "record": record,
        "data": data,
        "predictions": frame,
        "message": "Frozen NN evidence verified against saved sidecars/registry. No evaluator, model or source dataset was used. Labelled predictions require explicit opt-in.",
    }


def load_regression_evaluation(engine, evaluation_id, predictions=False):
    """Read already registered evidence; never invoke evaluation or mark access."""
    if not evaluation_id:
        return {"state": "empty", "message": "Select an evaluation explicitly"}
    try:
        if not base.rows(
            engine, "SELECT name FROM sqlite_master WHERE type='table' AND name='final_evaluations'"
        ):
            raise ValueError("No saved evaluations")
        records = base.rows(
            engine, "SELECT * FROM final_evaluations WHERE evaluation_id=?", (evaluation_id,)
        )
        if len(records) != 1 or records[0]["status"] != "succeeded":
            raise ValueError("Select a successful saved evaluation")
        record = records[0]
        if not record.get("accessed_at"):
            raise ValueError("Successful evaluation lacks its recorded access timestamp")
        from ml_template.paths import ROOT

        directory = Path(record["artifact_dir"])
        directory = (directory if directory.is_absolute() else ROOT / directory).resolve()
        entries = base.manifest(directory)
        names = [
            "report_data.json",
            "summary.json",
            "spec.json",
            "provenance.json",
            "acceptance.json",
            "residual_summary.json",
            "metrics.csv",
        ]
        base.verify(directory, entries, names)
        data = base.read_json(base.safe(directory, "report_data.json"))
        summary, spec, provenance = data["summary"], data["spec"], data["provenance"]
        if (
            summary != base.read_json(base.safe(directory, "summary.json"))
            or spec != base.read_json(base.safe(directory, "spec.json"))
            or provenance != base.read_json(base.safe(directory, "provenance.json"))
        ):
            raise ValueError("Frozen report/sidecar binding differs")
        if (
            spec != json.loads(record["spec_json"])
            or spec.get("schema_version") != 2
            or spec.get("task") != "regression"
            or summary.get("task") != "regression"
        ):
            raise ValueError("Frozen regression specification differs from registry")
        if (
            summary["evaluation_id"] != evaluation_id
            or summary["parent_run_id"] != record["parent_run_id"]
            or spec["parent_run_id"] != record["parent_run_id"]
        ):
            raise ValueError("Evaluation parent/ID differs")
        if (
            digest({"spec": spec, "provenance": provenance}) != record["identity_hash"]
            or summary["identity_hash"] != record["identity_hash"]
        ):
            raise ValueError("Frozen evaluation identity digest differs")
        if (
            record.get("silver_id") != spec["expected_silver_dataset_version_id"]
            or record.get("holdout_fingerprint") != provenance["holdout_fingerprint"]
        ):
            raise ValueError("Frozen evaluation dataset/population differs")
        if provenance.get("predictor", {}).get("schema_version") == 3:
            return _load_nn_evaluation(engine, record, directory, predictions)
        if (
            record.get("calibration_id") is not None
            or spec.get("calibration_id") is not None
            or spec.get("release_variant") != "raw"
            or spec.get("uncertainty", {}).get("enabled") is not False
        ):
            raise ValueError("Unsupported regression score/uncertainty schema")
        descriptor = provenance["predictor"]
        if (
            descriptor.get("schema_version") != 2
            or descriptor.get("task") != "regression"
            or descriptor.get("prediction_kind") != "continuous"
            or descriptor.get("kind") != spec["predictor"]["kind"]
            or descriptor["kind"] != summary["predictor_kind"]
        ):
            raise ValueError("Frozen predictor semantics differ")
        if summary["n_rows"] < 1 or summary.get("automatic_promotion") is not False:
            raise ValueError("Invalid support/promotion metadata")
        acceptance = base.read_json(base.safe(directory, "acceptance.json"))
        residual_summary = base.read_json(base.safe(directory, "residual_summary.json"))
        if (
            acceptance != data["acceptance"]
            or residual_summary != data["residual_summary"]
            or summary["criteria_status"] != acceptance["status"]
        ):
            raise ValueError("Acceptance/residual sidecars differ")
        metrics = data["metrics"]
        if len({(r["variant"], r["metric"]) for r in metrics}) != len(metrics):
            raise ValueError("Duplicate frozen metrics")
        for row in metrics:
            if (
                row["variant"] != "raw"
                or row["metric"] not in DIRECTIONS
                or row["direction"] != direction(row["metric"])
            ):
                raise ValueError("Frozen metric semantics differ")
            if row["value"] is not None and (
                isinstance(row["value"], bool) or not np.isfinite(float(row["value"]))
            ):
                raise ValueError("Nonfinite frozen metric")
        if not base.rows(
            engine,
            "SELECT name FROM sqlite_master WHERE type='table' AND name='final_evaluation_metrics'",
        ):
            raise ValueError("Frozen metric registry is missing")
        stored = base.rows(
            engine,
            "SELECT result_json FROM final_evaluation_metrics WHERE evaluation_id=?",
            (evaluation_id,),
        )
        stored = [json.loads(r["result_json"]) for r in stored]

        def keyfunc(row):
            return row["variant"], row["metric"]

        if sorted(stored, key=keyfunc) != sorted(metrics, key=keyfunc):
            raise ValueError("Saved report metrics differ from registered frozen metrics")
        primary = [r["value"] for r in metrics if r["metric"] == summary["primary_metric"]]
        if (
            len(primary) != 1
            or primary[0] != summary["primary_value"]
            or summary["primary_metric_direction"] != direction(summary["primary_metric"])
        ):
            raise ValueError("Frozen primary metric differs")
        frame = None
        if predictions:
            name = "holdout_predictions.parquet"
            base.verify(directory, entries, [name])
            frame = pd.read_parquet(base.safe(directory, name))
            key = [c for c in frame if c not in {"y_true", "y_pred", "residual", "absolute_error"}]
            if (
                len(key) != 1
                or len(frame) != summary["n_rows"]
                or frame[key[0]].isna().any()
                or frame[key[0]].duplicated().any()
            ):
                raise ValueError("Saved holdout prediction key/support mismatch")
            if (
                population_fingerprint(
                    spec["expected_silver_dataset_version_id"], key[0], frame[key[0]]
                )
                != record["holdout_fingerprint"]
            ):
                raise ValueError("Saved prediction population fingerprint differs")
            base.finite(frame, ["y_true", "y_pred", "residual", "absolute_error"])
            if not np.allclose(frame.residual, frame.y_true - frame.y_pred) or not np.allclose(
                frame.absolute_error, frame.residual.abs()
            ):
                raise ValueError("Saved prediction residuals differ")
        return {
            "state": "verified",
            "record": record,
            "data": data,
            "predictions": frame,
            "message": "Frozen report metadata verified; no evaluator, model or source dataset was used. Saved labelled prediction artifacts are read only when explicitly enabled.",
        }
    except base.ERRORS as exc:
        return {"state": "invalid", "message": str(exc)}
