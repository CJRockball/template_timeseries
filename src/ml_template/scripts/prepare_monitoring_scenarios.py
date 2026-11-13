"""Create controlled, unlabelled monitoring scenarios from an existing reference batch.

Run from the project root. Reads verified scoring artifacts; no model loading, database writes,
scoring, label generation or source changes. Outputs are new synthetic SOFTWARE TEST inputs.
"""

import argparse
import json
import shutil
import sqlite3
import tempfile
from datetime import UTC, datetime
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", default="score_20261020_144151_eac2f87783b2")
    parser.add_argument("--reference-id", default="ref_20261022_025123_a1cfce75e705")
    parser.add_argument("--database", type=Path, default=Path("data/project.db"))
    parser.add_argument("--output", type=Path, default=Path("data/scoring/monitoring_scenarios"))
    parser.add_argument("--torque-increase", type=float, default=30.0)
    args = parser.parse_args(argv)

    import numpy as np
    import pandas as pd
    import yaml

    from ml_template.monitoring.artifacts import digest
    from ml_template.monitoring.drift import compare
    from ml_template.monitoring.monitor_config import MonitorConfig
    from ml_template.monitoring.monitor_inputs import load_batch
    from ml_template.monitoring.reference import load_reference
    from ml_template.scoring.input_contract import validate_batch

    database = args.database.resolve()
    destination = args.output.resolve()
    if not database.is_file():
        raise ValueError("existing project database required")
    if destination.exists():
        raise ValueError("output already exists; choose a NEW --output directory")
    if not np.isfinite(args.torque_increase) or args.torque_increase <= 0:
        raise ValueError("torque-increase must be finite and positive")
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as conn:
        batch = load_batch(conn, args.batch_id)
        reference = load_reference(conn, args.reference_id, batch)
    if reference["scoring_batch_id"] != batch.batch_id:
        raise ValueError("use the exact batch from which the frozen reference was built")
    if len(batch.source) < 200:
        raise ValueError("use at least 200 reference rows for the default monitoring support rule")
    names = {item["name"]: item for item in batch.contract["source_schema"]}
    if "torque_nm" not in names or names["torque_nm"]["kind"] not in {"integer", "float"}:
        raise ValueError("this AI4I scenario requires numeric torque_nm")
    if "type" not in names or names["type"]["kind"] not in {"category", "string"}:
        raise ValueError("this AI4I scenario requires categorical type")
    type_spec = names["type"]
    allowed = type_spec["categories"] if type_spec["kind"] == "category" else type_spec["values"]
    if "H" not in allowed:
        raise ValueError("the frozen release must allow type H")

    control = batch.source.copy(deep=True)
    torque = control.copy(deep=True)
    values = torque["torque_nm"].to_numpy(dtype=float) + args.torque_increase
    dtype = control["torque_nm"].dtype
    if pd.api.types.is_integer_dtype(dtype):
        bounds = np.iinfo(dtype)
        if ((values % 1 != 0) | (values < bounds.min) | (values > bounds.max)).any():
            raise ValueError("shift would violate the frozen integer torque dtype")
    torque["torque_nm"] = pd.Series(values, index=torque.index).astype(dtype)
    category = control.copy(deep=True)
    if isinstance(category["type"].dtype, pd.CategoricalDtype):
        category["type"] = pd.Categorical(
            ["H"] * len(category),
            categories=category["type"].cat.categories,
            ordered=category["type"].cat.ordered,
        )
    else:
        category["type"] = pd.Series(
            ["H"] * len(category), index=category.index, dtype=control["type"].dtype
        )
    scenarios = {"control": control, "torque_shift": torque, "type_shift": category}
    expectations = {}
    for name, frame in scenarios.items():
        validate_batch(frame, batch.contract)
        measurements = {}
        for feature in ("torque_nm", "type"):
            measurements[feature] = compare(
                frame[feature], reference["profiles"][f"input:{feature}"]
            )["js_divergence"]
        expectations[name] = {"n_rows": len(frame), "input_js_divergence": measurements}
    if expectations["torque_shift"]["input_js_divergence"]["torque_nm"] < 0.1:
        raise ValueError(
            "chosen torque shift is too weak for the default 0.1 warning threshold; increase --torque-increase"
        )
    if expectations["type_shift"]["input_js_divergence"]["type"] < 0.1:
        raise ValueError(
            "all-H scenario is too similar to this reference for the default warning threshold"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".monitoring-scenarios-", dir=destination.parent))
    stamp = datetime.now(UTC).isoformat()
    try:
        for name, frame in scenarios.items():
            path = staging / f"{name}.parquet"
            frame.to_parquet(path, index=False)
            loaded = pd.read_parquet(path)
            validate_batch(loaded, batch.contract)
            pd.testing.assert_frame_equal(frame, loaded)
            config = MonitorConfig(
                scoring_batch_id=f"REPLACE_WITH_NEW_{name.upper()}_SCORING_BATCH_ID",
                reference_id=args.reference_id,
                label_import_id=None,
                as_of=stamp,
            ).model_dump(mode="json")
            (staging / f"monitor_{name}.yaml").write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            metadata = {
                "schema_version": 1,
                "source_system": "synthetic_monitoring_software_test",
                "business_batch_id": f"monitoring_test_{name}",
                "observation_start": None,
                "observation_end": None,
            }
            (staging / f"metadata_{name}.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
        provenance = {
            "schema_version": 1,
            "purpose": "synthetic monitoring SOFTWARE TEST, not prospective or performance evidence",
            "reference_id": args.reference_id,
            "source_scoring_batch_id": batch.batch_id,
            "release_id": batch.release_id,
            "created_at": stamp,
            "source_artifact_hashes": batch.artifact_hashes,
            "keys": "original keys retained; unique within each batch; scenarios are dependent copies, not new independent observations",
            "changes": {
                "control": "no changes",
                "torque_shift": f"torque_nm + {args.torque_increase} for every row; all other columns unchanged",
                "type_shift": "type set to H for every row; all other columns unchanged",
            },
            "labels": "none: do NOT attach original holdout labels to modified scenarios",
            "expected_input_measurements": expectations,
            "output_hashes": {
                f"{name}.parquet": digest(staging / f"{name}.parquet") for name in scenarios
            },
        }
        (staging / "scenario_manifest.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )
        if destination.exists():
            raise ValueError("output appeared during generation; refusing to overwrite")
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "output": str(destination),
                "release_id": batch.release_id,
                "reference_id": args.reference_id,
                "expected_input_measurements": expectations,
                "next": "Score control and shifted files with the SAME existing release in the pinned runtime; insert returned scoring IDs into the corresponding monitor YAMLs; monitor against the SAME frozen reference.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
