"""Deterministic exact and point-in-time enrichment of observation/request rows."""

import hashlib
import json
from dataclasses import dataclass

import pandas as pd

from .contracts import canonical_json, utc
from .store import connection, frame_hash, load_source, setup


@dataclass
class Assembly:
    frame: pd.DataFrame
    manifest: dict


def enrich(db, base, joins):
    """Base and each join use explicit source_id/version references.

    exact requires known_in_advance: true (a caller-attested semantic contract).
    point_in_time selects the latest available_at <= each base origin among
    exact-key matches. Include the applicable target period among the keys.
    """
    out, base_meta = load_source(db, base["source_id"], base["version"])
    provenance = []
    reports = []
    for spec in joins:
        allowed = {
            "source_id",
            "version",
            "on",
            "columns",
            "mode",
            "known_in_advance",
            "available_at",
            "origin",
            "max_missing_fraction",
        }
        if set(spec) - allowed:
            raise ValueError("Unknown join setting")
        right, meta = load_source(db, spec["source_id"], spec["version"])
        on = spec["on"]
        selected = spec["columns"]  # source-column -> output-column
        mode = spec["mode"]
        if not isinstance(on, list) or not on or not isinstance(selected, dict) or not selected:
            raise ValueError("Nonempty join keys and column mapping required")
        if len(set(on)) != len(on) or len(set(selected.values())) != len(selected):
            raise ValueError("Duplicate join/output column")
        if set(selected.values()) & set(out.columns):
            raise ValueError("Enrichment cannot overwrite existing columns")
        if not set(on) <= set(out) or not set([*on, *selected]) <= set(right):
            raise ValueError("Missing join columns")
        if out[on].isna().any().any() or right[on].isna().any().any():
            raise ValueError("Null join key")
        left_types = base_meta["contract"]["columns"]
        # Subsequent joins may only key on original base columns.
        if any(left_types.get(c) != meta["contract"]["columns"][c] for c in on):
            raise ValueError("Join key types differ or are not base columns")
        limit = float(spec.get("max_missing_fraction", 0))
        if not 0 <= limit <= 1:
            raise ValueError("Invalid missing-fraction limit")
        groups = {}
        for record in right.to_dict("records"):
            key = tuple(record[c] for c in on)
            groups.setdefault(key, []).append(record)
        if mode == "exact":
            if spec.get("known_in_advance") is not True:
                raise ValueError("Exact join requires explicit known-in-advance assertion")
            if any(len(rows) != 1 for rows in groups.values()):
                raise ValueError("Right side is not many-to-one")
        elif mode == "point_in_time":
            available, origin = spec["available_at"], spec["origin"]
            if meta["contract"]["columns"].get(available) != "utc_timestamp":
                raise ValueError("available_at must be declared utc_timestamp")
            if left_types.get(origin) != "utc_timestamp":
                raise ValueError("origin must be a base utc_timestamp")
            if right[available].isna().any() or out[origin].isna().any():
                raise ValueError("Null availability/origin")
            if right.duplicated([*on, available]).any():
                raise ValueError("Ambiguous revision at same availability instant")
        else:
            raise ValueError("Unknown join mode")
        additions = {v: [] for v in selected.values()}
        missing = 0
        for row in out.to_dict("records"):
            candidates = groups.get(tuple(row[c] for c in on), [])
            if mode == "point_in_time":
                candidates = [r for r in candidates if utc(r[available]) <= utc(row[origin])]
                record = max(candidates, key=lambda r: utc(r[available])) if candidates else None
            else:
                record = candidates[0] if candidates else None
            if record is None:
                missing += 1
            for src, dest in selected.items():
                additions[dest].append(None if record is None else record[src])
        for dest, values in additions.items():
            out[dest] = pd.Series(values, dtype=object)
        null_fraction = float(out[list(additions)].isna().any(axis=1).mean())
        if null_fraction > limit:
            raise ValueError(f"Join missingness {null_fraction} exceeds {limit}")
        reports.append(
            {
                "unmatched_rows": missing,
                "missing_feature_fraction": null_fraction,
                "output_rows": len(out),
            }
        )
        provenance.append(
            {
                "recipe": spec,
                "source_fingerprint": meta["fingerprint"],
                "source_data_sha256": meta["data_sha256"],
            }
        )
    identity = {
        "format": 1,
        "base": base,
        "base_fingerprint": base_meta["fingerprint"],
        "joins": provenance,
        "output_data_sha256": frame_hash(out),
    }
    assembly_id = "assembly_" + hashlib.sha256(canonical_json(identity).encode()).hexdigest()
    manifest = {
        **identity,
        "assembly_id": assembly_id,
        "rows": len(out),
        "reports": reports,
        "eligible_for_legacy_training": False,
    }
    with connection(db) as conn, conn:
        setup(conn)
        old = conn.execute(
            "SELECT manifest_json FROM temporal_assemblies WHERE assembly_id=?", (assembly_id,)
        ).fetchone()
        if old and json.loads(old[0]) != manifest:
            raise ValueError("Assembly identity conflict")
        if not old:
            conn.execute(
                "INSERT INTO temporal_assemblies VALUES (?,?)",
                (assembly_id, canonical_json(manifest)),
            )
    return Assembly(out, manifest)
