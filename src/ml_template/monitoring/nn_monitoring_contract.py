"""NN monitoring lineage and assessment contracts. No inference/runtime invocation."""

from pathlib import Path

from ml_template.monitoring.artifacts import digest, read_json, safe_path


def release_binding(summary, info, manifest, record, release_dir, entries, scoring_manifest_sha256):
    if info.get("model_type") != "nn":
        return summary
    unit = info.get("target_unit")
    if (
        info.get("task") != "regression"
        or info.get("prediction_kind") != "continuous"
        or (not isinstance(unit, str))
        or (not unit.strip())
        or (unit.strip() in {"target units", "target_units"})
        or (summary.get("target_unit") not in {None, unit})
        or (summary.get("parent_run_id") != info["parent_run_id"])
        or (summary.get("calibration_id") != info["calibration_id"])
        or (summary.get("release_variant") != info["release_variant"])
        or (summary.get("prediction_columns") != info["prediction_columns"])
    ):
        raise ValueError("NN scoring/release unit or predictor binding differs")
    descriptor = read_json(Path(release_dir) / "predictor.json")
    if (
        descriptor.get("schema_version") != 3
        or descriptor.get("model_type") != "nn"
        or descriptor.get("kind") != info["predictor_kind"]
    ):
        raise ValueError("NN monitoring requires the frozen NN descriptor")
    frozen = {
        "release_id": record["release_id"],
        "content_sha256": manifest["content_sha256"],
        "parent_run_id": info["parent_run_id"],
        "evaluation_id": info["evaluation_id"],
        "calibration_id": info["calibration_id"],
        "release_variant": info["release_variant"],
        "predictor_kind": info["predictor_kind"],
        "target_unit": unit,
        "prediction_columns": info["prediction_columns"],
        "interval_columns": info.get("interval_columns", {}),
        "release_json_sha256": digest(Path(release_dir) / "release.json"),
        "predictor_json_sha256": digest(Path(release_dir) / "predictor.json"),
        "runtime_json_sha256": digest(Path(release_dir) / "runtime.json"),
    }
    scored = {
        "scoring_batch_id": record["scoring_batch_id"],
        "input_sha256": record["input_sha256"],
        "n_rows": record["n_rows"],
        "predictions_sha256": entries["predictions.parquet"]["sha256"],
        "artifacts_sha256": scoring_manifest_sha256,
    }
    return {
        **summary,
        "model_type": "nn",
        "target_unit": unit,
        "nn_binding": {"schema_version": 1, "release": frozen, "score": scored},
        "unit_provenance": "verified_sealed_release; original score summary was not modified",
    }


def verify_pipeline(root, identity, expected_manifest, batch):
    root = Path(root).expanduser().resolve()
    manifest_path = safe_path(root, "pipeline_manifest.json")
    if digest(manifest_path) != expected_manifest:
        raise ValueError("Pipeline manifest differs from explicit expected SHA256")
    manifest = read_json(manifest_path)
    if (
        manifest.get("schema_version") != 1
        or not isinstance(manifest.get("files"), dict)
        or "pipeline.json" not in manifest["files"]
    ):
        raise ValueError("Incomplete Package 05 envelope")
    actual = {}
    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            raise ValueError("Pipeline symlinks forbidden")
        if p.is_file() and p.name != "pipeline_manifest.json":
            actual[p.relative_to(root).as_posix()] = {
                "sha256": digest(p),
                "bytes": p.stat().st_size,
            }
    if actual != manifest["files"]:
        raise ValueError("Pipeline file hash/size/inventory differs")
    for name in manifest["files"]:
        safe_path(root, name)
    state = read_json(safe_path(root, "pipeline.json"))
    binding = batch.summary["nn_binding"]
    frozen = binding["release"]
    score = binding["score"]
    if (
        root.name != identity
        or state.get("pipeline_id") != identity
        or state.get("status") != "succeeded"
        or (state.get("stage") != "complete")
        or (state.get("predictions_computed") is not True)
        or (state.get("labels_loaded") is not False)
        or (state.get("evaluation_executed") is not False)
        or (state.get("release_built") is not False)
        or (state.get("task") != "regression")
        or (state.get("model_type") != "nn")
        or (state.get("release_id") != frozen["release_id"])
        or (state.get("release_content_sha256") != frozen["content_sha256"])
        or (state.get("target_unit") != frozen["target_unit"])
        or (state.get("parent_run_id") != frozen["parent_run_id"])
        or (state.get("evaluation_id") != frozen["evaluation_id"])
        or (state.get("calibration_id") != frozen["calibration_id"])
        or (state.get("release_variant") != frozen["release_variant"])
        or (state.get("interval_columns") != frozen["interval_columns"])
        or (state.get("scoring_batch_id") != score["scoring_batch_id"])
        or (state.get("predictions_sha256") != score["predictions_sha256"])
        or (state.get("scoring_manifest_sha256") != score["artifacts_sha256"])
        or (state.get("n_rows") != score["n_rows"])
    ):
        raise ValueError("Pipeline/release/scoring identity differs")
    score_dir = Path(state["scoring_dir"]).resolve()
    if (
        not score_dir.is_relative_to(root)
        or score_dir != Path(batch.record["artifact_dir"]).resolve()
    ):
        raise ValueError("Pipeline score directory differs from registry")
    if digest(safe_path(score_dir, "artifacts.json")) != score["artifacts_sha256"]:
        raise ValueError("Pipeline scoring manifest differs")
    if _pipeline_input_format(state) == "canonical":
        if state["input_sha256"] != score["input_sha256"]:
            raise ValueError("Canonical pipeline input binding differs")
    elif _pipeline_input_format(state) == "raw":
        prep = read_json(safe_path(root, "prepared/manifest.json"))
        if (
            prep["input_sha256"] != state["input_sha256"]
            or prep["release_id"] != frozen["release_id"]
            or prep["canonical_sha256"] != score["input_sha256"]
            or (digest(safe_path(root, "prepared/data.parquet")) != prep["canonical_sha256"])
        ):
            raise ValueError("Raw-to-canonical pipeline binding differs")
        names = [
            n
            for n in manifest["files"]
            if n in {"prepared/raw_input.csv", "prepared/raw_input.parquet"}
        ]
        if len(names) != 1 or digest(safe_path(root, names[0])) != prep["input_sha256"]:
            raise ValueError("Raw snapshot binding differs")
    else:
        raise ValueError("Unknown pipeline input format")
    return {
        "schema_version": 1,
        "pipeline_id": identity,
        "pipeline_dir": str(root),
        "pipeline_manifest_sha256": expected_manifest,
        "nn_binding": binding,
        "model_loaded": False,
        "predictions_recomputed": False,
    }


def context_for(batch, config):
    if batch.summary.get("model_type") != "nn":
        if getattr(config, "pipeline_dir", None) is not None:
            raise ValueError("NN context cannot bind a non-NN batch")
        return None
    required = ["pipeline_dir", "expected_pipeline_id", "expected_pipeline_manifest_sha256"]
    if any(getattr(config, name, None) is None for name in required):
        raise ValueError(
            "NN monitoring requires explicit Package 05 pipeline context; use nn_monitor"
        )
    if getattr(config, "target_unit", None) != batch.summary["target_unit"]:
        raise ValueError("Monitoring units differ from frozen NN release")
    return verify_pipeline(
        config.pipeline_dir,
        config.expected_pipeline_id,
        config.expected_pipeline_manifest_sha256,
        batch,
    )


def validate_saved_context(value, batch, *, same_batch=False):
    if batch.summary.get("model_type") != "nn":
        return
    ctx = value.get("nn_context")
    if (
        not isinstance(ctx, dict)
        or ctx.get("schema_version") != 1
        or ctx["nn_binding"]["release"] != batch.summary["nn_binding"]["release"]
    ):
        raise ValueError("Frozen NN reference/labels release context differs")
    if same_batch and ctx["nn_binding"]["score"] != batch.summary["nn_binding"]["score"]:
        raise ValueError("Frozen NN labels scoring context differs")


def assessment_scope(
    config, performance, sampling, reference_n, current_n, *, self_reference=False
):
    if performance is None:
        status = "unknown_no_labels"
    elif performance["n_evaluated"] == 0:
        status = "not_assessed_no_mature_observed_outcomes"
    elif not performance["sufficient_support"]:
        status = "not_assessed_insufficient_outcomes"
    elif (
        performance["mature_label_coverage"] is None
        or performance["mature_label_coverage"] < config.min_label_coverage
    ):
        status = "not_assessed_label_coverage"
    elif sampling != "all_rows":
        status = "not_assessed_selected_or_unknown_sampling"
    else:
        status = "assessed_mature_labelled_subset"
    return {
        "performance_status": status,
        "distribution_status": "not_assessed_insufficient_rows"
        if min(reference_n, current_n) < config.min_rows
        else "not_assessed_self_reference"
        if self_reference
        else "descriptive",
        "self_reference": self_reference,
        "reference_n": reference_n,
        "current_n": current_n,
        "min_rows": config.min_rows,
        "min_labelled": config.min_labelled,
        "prediction_source": "saved_verified_scoring_artifacts",
        "model_loaded": False,
        "predictions_recomputed": False,
        "bias_definition": "mean(y_pred - y_true)",
        "residual_definition": "y_true - y_pred",
        "automatic_promotion": False,
    }


def validate_frozen_view(view):
    s, c = (view["summary"], view["config"])
    ctx = s.get("nn_context")
    if s.get("model_type") != "nn":
        return
    if (
        not isinstance(ctx, dict)
        or ctx.get("schema_version") != 1
        or ctx["pipeline_id"] != c.get("expected_pipeline_id")
        or (ctx["pipeline_manifest_sha256"] != c.get("expected_pipeline_manifest_sha256"))
        or (ctx["nn_binding"]["release"]["release_id"] != s["release_id"])
        or (ctx["nn_binding"]["release"]["target_unit"] != s["target_unit"])
        or (ctx["nn_binding"]["score"]["scoring_batch_id"] != s["scoring_batch_id"])
        or (c["target_unit"] != s["target_unit"])
    ):
        raise ValueError("Frozen NN monitoring context differs")
    from types import SimpleNamespace

    expected = assessment_scope(
        SimpleNamespace(**c),
        view["performance"],
        s["label_sampling"],
        s["assessment_scope"]["reference_n"],
        s["n_rows"],
        self_reference=s["assessment_scope"]["self_reference"],
    )
    if expected != s["assessment_scope"]:
        raise ValueError("Frozen NN assessment scope differs")


def _pipeline_input_format(state):
    """Read the verified Package 05 request; never infer or default its format."""
    request = state.get("request")
    if not isinstance(request, dict):
        raise ValueError("Package 05 pipeline requires a request mapping")
    input_format = request.get("input_format")
    if input_format not in ("canonical", "raw"):
        raise ValueError("Package 05 request.input_format must be canonical or raw")
    return input_format
