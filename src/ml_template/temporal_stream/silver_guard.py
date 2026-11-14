"Reject declared temporal inputs in legacy random-split paths. No Arrow dependency."

import json
from pathlib import Path

MESSAGE = (
    "Temporal data requires the forecasting workflow; legacy random/shuffled training is prohibited"
)


def reject_temporal_manifest(manifest):
    if (
        manifest.get("dataset_kind") == "temporal"
        or manifest.get("temporal")
        or manifest.get("temporal_schema")
        or (manifest.get("eligible_for_legacy_training") is False)
    ):
        raise ValueError(MESSAGE)


def reject_temporal_contract(contract):
    candidates = [contract, contract.get("data", {}), contract.get("silver_eda_contract", {})]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if (
            candidate.get("temporal")
            or candidate.get("temporal_schema")
            or candidate.get("dataset_kind") == "temporal"
        ):
            raise ValueError(MESSAGE)


def reject_temporal_path(path):
    path = Path(path)
    manifest = path if path.name == "manifest.json" else path / "manifest.json"
    if manifest.is_file():
        reject_temporal_manifest(json.loads(manifest.read_text(encoding="utf-8")))


def reject_temporal_version(version):
    if isinstance(version, str) and version.startswith("silver_ts_"):
        raise ValueError(MESSAGE)
