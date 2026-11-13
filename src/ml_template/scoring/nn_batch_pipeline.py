"""Release-first YAML batch pipeline. Inference runs in the release's pinned runtime."""

import argparse
import json
import os
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator

from ml_template.scoring.nn_batch_contract import (
    export_frame,
    key_fingerprint,
    output_columns,
    sha256,
    validate_export_mapping,
    validate_predictions,
    verify_batch,
)

WORKER = "import sys,json,hashlib,io,contextlib,sqlite3\nfrom pathlib import Path\nrequest=json.loads(sys.argv[1]); root=Path(request['runtime_source_root']).resolve()\nsys.path.insert(0,str(root.parent))\nimport ml_template\nif Path(ml_template.__file__).resolve().parent!=root:raise ValueError('Wrong imported runtime source')\n# Known lazy NN check/scoring APIs; future changed runtimes need explicit compatibility review.\nprofile={'scoring/batch.py':'86309894edf0228c966e4ae4728e84cb48c9e950','scoring/prepare_input.py':'8f257278635c77cb411fe53d74bf9aa6e63e0f89','scoring/regression_predictor.py':'0c77b1f41b7901d252a991b07173468a7131ddf5','models/nn_regression_predictor.py':'74d1e6ab0f3628e5945c5df20a900144b5667dc9','scoring/predictor.py':'9d8f31dee4c2be94ab9bcf8b2e1034137772c595','models/predictor_selection.py':'05ec5d6680d1bf77cdae56385d8d15721a6bc199'}\nfor name,expected in profile.items():\n    data=(root/name).read_bytes(); actual=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\\0'+data).hexdigest()\n    if actual!=expected:raise ValueError('Unsupported pinned scoring API: '+name)\nfrom ml_template.scoring.release_artifacts import verify,verify_runtime,read_json,sha256,file_entries\nrelease=Path(request['release_dir']); manifest=verify(release,request['expected_release_id']);verify_runtime(read_json(release/'runtime.json'))\ninfo=read_json(release/'release.json');contract=read_json(release/'contract.json')\nif info.get('model_type')!='nn' or info.get('task')!='regression' or info.get('prediction_kind')!='continuous':raise ValueError('NN regression release required')\nfrom ml_template.scoring.predictor import ReleasedPredictor\nfrom ml_template.scoring.batch import score_batch,read_batch\nfrom ml_template.monitoring.scoring_attempts import parse_metadata\nfrom ml_template.scoring.input_contract import validate_batch\nfrom ml_template.data.canonicalize import canonicalize_frame\nimport numpy as np\nimport pandas as pd\nsource=Path(request['input_path']);before=sha256(source)\nif before!=request['input_sha256']:raise ValueError('Input changed before worker')\nif source.suffix.lower()=='.csv':header=pd.read_csv(source,nrows=0).columns.tolist()\nelif source.suffix.lower()=='.parquet':\n    import pyarrow.parquet as pq\n    header=pq.ParquetFile(source).schema_arrow.names\nelse:raise ValueError('CSV/Parquet required')\nkey=contract['key_col'];required=[key,*[s['name'] for s in contract['source_schema']]]\nif request['input_format']=='raw':\n    spec=contract['input_spec'];mapped=[spec['canonical_names'].get(c,c) for c in header]\n    if len(mapped)!=len(set(mapped)) or set(mapped)!=set(spec['required_columns']) or set(mapped)&set(spec['forbidden_columns']):raise ValueError('Raw input has missing/extra/target columns')\n    raw=pd.read_csv(source,dtype='string') if source.suffix.lower()=='.csv' else pd.read_parquet(source)\n    frame=canonicalize_frame(raw,spec,columns=spec['required_columns'])\nelse:\n    if set(header)!=set(required) or len(header)!=len(set(header)):raise ValueError('Canonical input has missing/extra/target columns')\n    frame=read_batch(source,contract)\nvalidated=validate_batch(frame,contract,enforce_rates=True)\npredictor=ReleasedPredictor(release,expected_release_id=request['expected_release_id']);predictor.prepare_features(frame)\nmetadata=parse_metadata(request.get('batch_metadata',{}))\nencoded=[]\nfor value in validated.index:\n    if isinstance(value,np.generic):value=value.item()\n    if isinstance(value,bool) or not isinstance(value,(str,int)):raise ValueError('Invalid canonical key')\n    encoded.append(json.dumps([type(value).__name__,value]))\nfingerprint=hashlib.sha256(json.dumps(sorted(encoded),separators=(',',':')).encode()).hexdigest()\nif sha256(source)!=before:raise ValueError('Input changed during preflight')\nresult={'ready':True,'release_id':manifest['release_id'],'n_rows':len(frame),'key':key,'key_kind':contract['key_kind'],'key_fingerprint':fingerprint,'input_sha256':before,'models_loaded':False,'predictions_computed':False,'labels_loaded':False,'files_written':False}\nif request['execute']:\n    job=Path(request['job_dir']);canonical=source\n    if request['input_format']=='raw':\n        from ml_template.scoring.prepare_input import main as prepare_input\n        with contextlib.redirect_stdout(io.StringIO()):prepare_input(['--release',str(release),'--input',str(source),'--output-dir',str(job/'prepared'),'--execute'])\n        canonical=job/'prepared/data.parquet';preparation=read_json(job/'prepared/manifest.json')\n        if preparation['input_sha256']!=before or preparation['release_id']!=manifest['release_id'] or preparation['canonical_sha256']!=sha256(canonical):raise ValueError('Preparation snapshot binding differs')\n    if sha256(source)!=before:raise ValueError('Input changed before scoring')\n    from sqlalchemy import create_engine\n    from ml_template import paths\n    paths.LOGS=job/'logs'\n    database=Path(request['database'])\n    engine=create_engine('sqlite://',creator=lambda:sqlite3.connect(database.as_uri()+'?mode=rw',uri=True))\n    if metadata.get('source_system') is None:metadata['source_system']='ml_template.nn_batch_pipeline'\n    if metadata.get('business_batch_id') is None:metadata['business_batch_id']=request['job_id']\n    try:\n        with contextlib.redirect_stdout(io.StringIO()):\n            scored=score_batch(engine,release,canonical,output_root=job/'scores',chunk_size=request.get('chunk_size'),batch_metadata=metadata,check_only=False)\n    finally:engine.dispose()\n    directory=Path(scored['artifact_dir']);summary=read_json(directory/'summary.json');entries=read_json(directory/'artifacts.json')['files']\n    if file_entries(directory,exclude=('artifacts.json',))!=entries:raise ValueError('Scored artifacts differ')\n    if summary['input_sha256']!=sha256(canonical) or summary['release_id']!=manifest['release_id'] or summary['n_rows']!=len(frame):raise ValueError('Scoring identity/support differs')\n    if info.get('postprocessing') is not None:\n        from ml_template.evaluation.regression_postprocessing import validate_saved_predictions\n        validate_saved_predictions(pd.read_parquet(directory/'predictions.parquet'),summary,info)\n    if sha256(source)!=before:raise ValueError('Input changed during scoring')\n    result.update(scored)\n    result.update(predictions_path=str(directory/'predictions.parquet'),predictions_sha256=sha256(directory/'predictions.parquet'),scoring_manifest_sha256=sha256(directory/'artifacts.json'),canonical_sha256=sha256(canonical),models_loaded=True,predictions_computed=True,files_written=True)\nprint('__NN_BATCH_RESULT__'+json.dumps(result,allow_nan=False))\n"


class ExportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    reference: str
    columns: dict[str, str] = Field(default_factory=lambda: {"y_pred_release": "prediction"})


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    release_dir: str
    expected_release_id: str = Field(pattern="^rel_[0-9a-f]{24}$")
    input_path: str
    input_format: Literal["raw", "canonical"] = "canonical"
    runtime_source_root: str
    runtime_python: str = Field(default_factory=lambda: sys.executable)
    database: str = "data/project.db"
    output_root: str = "artifacts/nn_batch_pipelines"
    chunk_size: int | None = Field(default=None, ge=1, strict=True)
    timeout_s: int = Field(default=1800, ge=1, le=86400, strict=True)
    acknowledge_research_scoring: StrictBool = False
    batch_metadata: dict[str, Any] = Field(default_factory=dict)
    export: ExportRequest | None = None

    @field_validator(
        "release_dir",
        "input_path",
        "runtime_source_root",
        "runtime_python",
        "database",
        "output_root",
    )
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError("Explicit nonempty path required")
        return value


def path_for(value, *, follow_symlinks=True):
    from ml_template.paths import ROOT

    path = Path(value).expanduser()
    path = path if path.is_absolute() else ROOT / path
    return path.resolve() if follow_symlinks else path.absolute()


def reference_keys(path, key, key_kind, names):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        header = pd.read_csv(path, nrows=0).columns.tolist()
        if key not in header or len(header) != len(set(header)) or set(header) - {key, *names}:
            raise ValueError("Reference must be key-only or intended output template")
        return pd.read_csv(
            path, usecols=[key], dtype={key: "string" if key_kind == "str" else "int64"}
        )
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        header = pq.ParquetFile(path).schema_arrow.names
        if key not in header or set(header) - {key, *names}:
            raise ValueError("Unexpected reference columns")
        return pd.read_parquet(path, columns=[key])
    raise ValueError("Reference must be CSV/Parquet")


def launch(payload, timeout):
    python = Path(payload["runtime_python"])
    root = Path(payload["runtime_source_root"])
    if (
        not python.is_file()
        or not os.access(python, os.X_OK)
        or (not (root / "__init__.py").is_file())
    ):
        raise ValueError("Supply existing pinned Python and package source root")
    result = subprocess.run(
        [str(python), "-I", "-c", WORKER, json.dumps(payload, allow_nan=False)],
        cwd=root.parent,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode:
        raise RuntimeError(
            "Pinned scoring worker failed; do not change release hashes. " + result.stderr[-12000:]
        )
    lines = [
        line[len("__NN_BATCH_RESULT__") :]
        for line in result.stdout.splitlines()
        if line.startswith("__NN_BATCH_RESULT__")
    ]
    if len(lines) != 1:
        raise ValueError("Worker result protocol failed")
    return json.loads(lines[0])


def run_pipeline(request, *, check_only=True):
    from ml_template.scoring.release_artifacts import read_json, verify

    release = path_for(request.release_dir)
    manifest = verify(release, request.expected_release_id)
    info = read_json(release / "release.json")
    allowed = output_columns(info)
    source = path_for(request.input_path)
    database = path_for(request.database)
    runtime = path_for(request.runtime_source_root)
    python = path_for(request.runtime_python, follow_symlinks=False)
    output = path_for(request.output_root)
    if not source.is_file() or not database.is_file():
        raise FileNotFoundError("Existing unlabelled input and registry required")
    if (
        output == source
        or (output.exists() and (not output.is_dir()))
        or output.is_relative_to(release)
        or output.is_relative_to(runtime)
    ):
        raise ValueError("Output cannot be inside release/runtime source")
    if any(
        (parent / name).is_file()
        for parent in [output, *output.parents]
        for name in [
            "manifest.json",
            "artifact_manifest.json",
            "artifacts.json",
            "release.json",
        ]
    ):
        raise ValueError("Output cannot be inside sealed artifacts")
    ref = None
    ref_hash = None
    if request.export is not None:
        validate_export_mapping(
            request.export.columns, read_json(release / "contract.json")["key_col"], allowed
        )
        ref = path_for(request.export.reference)
        ref_hash = sha256(ref)
    before = sha256(source)
    payload = {
        **request.model_dump(mode="json"),
        "release_dir": str(release),
        "input_path": str(source),
        "database": str(database),
        "runtime_source_root": str(runtime),
        "runtime_python": str(python),
        "input_sha256": before,
        "execute": False,
    }
    checked = launch(payload, request.timeout_s)
    if (
        checked["release_id"] != request.expected_release_id
        or not checked["ready"]
        or checked["predictions_computed"]
    ):
        raise ValueError("Worker preflight identity/state differs")
    if ref is not None:
        keys = reference_keys(
            ref, checked["key"], checked["key_kind"], request.export.columns.values()
        )
        if (
            key_fingerprint(keys[checked["key"]]) != checked["key_fingerprint"]
            or len(keys) != checked["n_rows"]
        ):
            raise ValueError("Reference/input key coverage differs")
        if sha256(ref) != ref_hash:
            raise ValueError("Reference changed during preflight")
    if sha256(source) != before:
        raise ValueError("Input changed during preflight")
    report = {
        **checked,
        "task": "regression",
        "model_type": "nn",
        "target_unit": info["target_unit"],
        "parent_run_id": info["parent_run_id"],
        "evaluation_id": info["evaluation_id"],
        "calibration_id": info["calibration_id"],
        "release_variant": info["release_variant"],
        "interval_columns": info.get("interval_columns", {}),
        "evaluation_executed": False,
        "release_built": False,
    }
    if check_only:
        return {**report, "status": "checked", "pipeline_artifacts_written": False}
    if not request.acknowledge_research_scoring:
        raise ValueError("Execution requires acknowledge_research_scoring:true")
    identity = "nnpipe_" + datetime.now(UTC).strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:12]
    directory = output / identity
    directory.mkdir(parents=True, exist_ok=False)
    state = {
        **report,
        "pipeline_id": identity,
        "status": "running",
        "stage": "scoring",
        "artifact_dir": str(directory),
        "runtime_source_root": str(runtime),
        "runtime_python": str(python),
        "release_content_sha256": manifest["content_sha256"],
        "input_path": str(source),
        "reference_sha256": ref_hash,
        "request": request.model_dump(mode="json"),
        "worker_sha256": __import__("hashlib").sha256(WORKER.encode()).hexdigest(),
    }

    def checkpoint():
        temp = directory / "pipeline.json.tmp"
        temp.write_text(json.dumps(state, indent=2, allow_nan=False))
        temp.replace(directory / "pipeline.json")

    checkpoint()
    try:
        scored = launch(
            {**payload, "execute": True, "job_id": identity, "job_dir": str(directory)},
            request.timeout_s,
        )
        score_dir = Path(scored["artifact_dir"]).resolve()
        if (
            not score_dir.is_relative_to(directory)
            or scored["release_id"] != request.expected_release_id
            or scored.get("status") != "succeeded"
        ):
            raise ValueError("Scoring result identity/path differs")
        summary = verify_batch(score_dir, scored["scoring_manifest_sha256"])
        if (
            summary["input_sha256"] != scored["canonical_sha256"]
            or scored["input_sha256"] != before
        ):
            raise ValueError("Worker canonical/source fingerprint differs")
        if (
            summary["release_id"] != request.expected_release_id
            or summary["scoring_batch_id"] != scored["scoring_batch_id"]
            or summary["n_rows"] != checked["n_rows"]
        ):
            raise ValueError("Scored summary identity differs")
        prediction = score_dir / "predictions.parquet"
        if sha256(prediction) != scored["predictions_sha256"]:
            raise ValueError("Prediction artifact changed")
        frame = pd.read_parquet(prediction)
        validate_predictions(
            frame,
            summary,
            info,
            checked["key"],
            request.expected_release_id,
            scored["scoring_batch_id"],
        )
        if key_fingerprint(frame[checked["key"]]) != checked["key_fingerprint"]:
            raise ValueError("Scored input key coverage differs")
        state.update(
            stage="export",
            models_loaded=True,
            predictions_computed=True,
            scoring_batch_id=scored["scoring_batch_id"],
            attempt_id=scored.get("attempt_id"),
            scoring_dir=str(score_dir),
            predictions_path=str(prediction),
            predictions_sha256=scored["predictions_sha256"],
            scoring_manifest_sha256=scored["scoring_manifest_sha256"],
        )
        checkpoint()
        if ref is not None:
            if sha256(ref) != ref_hash:
                raise ValueError("Reference changed before export")
            keys = reference_keys(
                ref, checked["key"], checked["key_kind"], request.export.columns.values()
            )
            exported = export_frame(
                frame, keys, key=checked["key"], columns=request.export.columns, allowed=allowed
            )
            export_dir = directory / "export"
            export_dir.mkdir()
            destination = export_dir / "predictions.csv"
            with destination.open("x", encoding="utf-8", newline="") as stream:
                exported.to_csv(stream, index=False, float_format="%.17g")
            if sha256(ref) != ref_hash or sha256(prediction) != scored["predictions_sha256"]:
                raise ValueError("Export source changed")
            meta = {
                "schema_version": 1,
                "release_id": request.expected_release_id,
                "scoring_batch_id": scored["scoring_batch_id"],
                "key": checked["key"],
                "columns": request.export.columns,
                "rows": len(exported),
                "target_unit": info["target_unit"],
                "predictions_sha256": scored["predictions_sha256"],
                "reference_sha256": ref_hash,
                "output_sha256": sha256(destination),
            }
            (export_dir / "manifest.json").write_text(json.dumps(meta, indent=2, allow_nan=False))
            state.update(export_path=str(destination), export_sha256=meta["output_sha256"])
        if sha256(source) != before:
            raise ValueError("Input changed during pipeline")
        state.update(status="succeeded", stage="complete", files_written=True)
        checkpoint()
        entries = {
            p.relative_to(directory).as_posix(): {"sha256": sha256(p), "bytes": p.stat().st_size}
            for p in sorted(directory.rglob("*"))
            if p.is_file() and p.name != "pipeline_manifest.json"
        }
        (directory / "pipeline_manifest.json").write_text(
            json.dumps({"schema_version": 1, "files": entries}, indent=2, allow_nan=False)
        )
        return state
    except Exception as exc:
        if state["stage"] == "scoring":
            state.update(models_loaded=None, predictions_computed=None)
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        checkpoint()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)

    class UniqueLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node, deep=False):
        result = {}
        for a, b in node.value:
            key = loader.construct_object(a, deep=deep)
            if key in result:
                raise ValueError("Duplicate pipeline YAML key")
            result[key] = loader.construct_object(b, deep=deep)
        return result

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    request = BatchRequest.model_validate(yaml.load(args.config.read_text(), Loader=UniqueLoader))
    print(json.dumps(run_pipeline(request, check_only=args.check), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
