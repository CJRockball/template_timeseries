"""Explicit NN monitoring requests; delegates all analysis to the shared backend."""

import argparse
import json
import sqlite3
from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from ml_template.monitoring.monitor_config import LabelConfig, MonitorConfig, ReferenceConfig


class NNContext(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    pipeline_dir: Path
    expected_pipeline_id: str = Field(pattern="^nnpipe_[A-Za-z0-9_]+$")
    expected_pipeline_manifest_sha256: str = Field(pattern="^[0-9a-f]{64}$")


class NNReferenceConfig(ReferenceConfig, NNContext):
    target_unit: str = Field(min_length=1)


class NNLabelConfig(LabelConfig, NNContext):
    target_unit: str = Field(min_length=1)


class NNMonitorConfig(MonitorConfig, NNContext):
    target_unit: str = Field(min_length=1)


class NNTemplateConfig(NNContext):
    scoring_batch_id: str
    target_unit: str = Field(min_length=1)


class ReportConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    monitoring_id: str


def load_request(path, cls):

    class UniqueLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node, deep=False):
        out = {}
        for a, b in node.value:
            key = loader.construct_object(a, deep=deep)
            if key in out:
                raise ValueError("Duplicate NN monitoring request key")
            out[key] = loader.construct_object(b, deep=deep)
        return out

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    return cls.model_validate(yaml.load(Path(path).read_text(), Loader=UniqueLoader))


def validate_output(config, root):
    root = Path(root).expanduser().resolve()
    pipeline = getattr(config, "pipeline_dir", None)
    if pipeline is not None and root.is_relative_to(Path(pipeline).resolve()):
        raise ValueError("Monitoring output cannot be inside immutable pipeline artifacts")
    if any(
        (parent / name).is_file()
        for parent in [root, *root.parents]
        for name in [
            "manifest.json",
            "artifact_manifest.json",
            "artifacts.json",
            "pipeline_manifest.json",
            "release.json",
        ]
    ):
        raise ValueError("Monitoring output cannot be inside sealed artifacts")
    return root


def label_template(engine, config, *, output_root, check_only=False):
    from ml_template.monitoring.artifacts import digest, write_json
    from ml_template.monitoring.monitor_inputs import load_batch
    from ml_template.monitoring.nn_monitoring_contract import context_for

    batch = load_batch(engine, config.scoring_batch_id)
    ctx = context_for(batch, config)
    if ctx is None:
        raise ValueError("NN template requires NN batch")
    root = validate_output(config, output_root)
    path = root / (config.scoring_batch_id + "_labels_template.csv")
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError("Choose a new immutable template output root")
    report = {
        "ready": True,
        "scoring_batch_id": batch.batch_id,
        "n_rows": len(batch.predictions),
        "labels_generated": 0,
        "files_written": False,
    }
    if check_only:
        return report
    root.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            batch.key: batch.predictions[batch.key].to_numpy(),
            "scoring_batch_id": batch.batch_id,
            "y_true": "",
            "observed_at": "",
            "eligible_at": "",
        }
    )
    with path.open("x", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False)
    write_json(
        path.with_suffix(".json"),
        {
            **report,
            "files_written": True,
            "nn_context": ctx,
            "target_unit": config.target_unit,
            "output_sha256": digest(path),
            "instruction": "Fill genuine measured y_true and observed_at only when known. Fill actual timezone-aware eligible_at for every row. Blank eligibility makes this deliberately unready for import.",
        },
    )
    return {**report, "files_written": True, "template_path": str(path)}


def run(engine, operation, config, *, output_root, check_only=True):
    root = validate_output(config, output_root)
    if operation == "template":
        return label_template(engine, config, output_root=root, check_only=check_only)
    if operation == "report":
        from ml_template.usage_dashboard.monitoring_data import load_assessment
        from ml_template.usage_dashboard.nn_monitor_report import bundle_bytes

        view = load_assessment(engine, config.monitoring_id)
        path = root / (config.monitoring_id + ".zip")
        if path.exists():
            raise FileExistsError("Report output exists")
        if check_only:
            return {"ready": True, "files_written": False}
        data = bundle_bytes(view, config.monitoring_id)
        root.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
        return {"report_path": str(path), "files_written": True}
    from ml_template.monitoring.label_inputs import import_labels
    from ml_template.monitoring.monitor_batch import evaluate
    from ml_template.monitoring.reference import build_reference

    return {"reference": build_reference, "labels": import_labels, "monitor": evaluate}[operation](
        engine, config, output_root=root, check_only=check_only
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--operation",
        required=True,
        choices=["reference", "labels", "monitor", "template", "report"],
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--database", type=Path, default=Path("data/project.db"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/monitoring"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    cls = {
        "reference": NNReferenceConfig,
        "labels": NNLabelConfig,
        "monitor": NNMonitorConfig,
        "template": NNTemplateConfig,
        "report": ReportConfig,
    }[args.operation]
    config = load_request(args.config, cls)
    database = args.database.expanduser().resolve()
    if not database.is_file():
        raise FileNotFoundError(database)
    if args.operation == "report":
        print(
            json.dumps(
                run(
                    database,
                    args.operation,
                    config,
                    output_root=args.output_root,
                    check_only=args.check,
                ),
                indent=2,
                allow_nan=False,
            )
        )
        return
    with sqlite3.connect(
        database.as_uri() + ("?mode=ro" if args.check else "?mode=rw"), uri=True
    ) as connection:
        print(
            json.dumps(
                run(
                    connection,
                    args.operation,
                    config,
                    output_root=args.output_root,
                    check_only=args.check,
                ),
                indent=2,
                allow_nan=False,
            )
        )


if __name__ == "__main__":
    main()
