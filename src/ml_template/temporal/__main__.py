"""Explicit CLI, separate from legacy Silver/training commands."""

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import yaml

from .enrichment import enrich
from .store import ingest_csv, load_source


def strict_keys(spec, allowed, required):
    if set(spec) - allowed or not required <= set(spec):
        raise ValueError("Unknown or missing top-level configuration fields")


def export_assembly(result, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".temporal-", dir=output.parent))
    try:
        csv = temp / "data.csv"
        result.frame.to_csv(csv, index=False)
        manifest = dict(result.manifest)
        manifest["csv_sha256"] = hashlib.sha256(csv.read_bytes()).hexdigest()
        (temp / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        # Local single-writer export; do not run concurrent exports to the same destination.
        if output.exists():
            raise FileExistsError(output)
        temp.rename(output)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["ingest", "assemble", "verify"])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    path = args.config.resolve()
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))

    def resolve(value):
        p = Path(value)
        return p if p.is_absolute() else path.parent / p

    if args.command == "ingest":
        strict_keys(
            spec,
            {"database", "source_id", "version", "input", "contract", "provenance"},
            {"database", "source_id", "version", "input", "contract"},
        )
        result = ingest_csv(
            resolve(spec["database"]),
            spec["source_id"],
            spec["version"],
            resolve(spec["input"]),
            spec["contract"],
            spec.get("provenance"),
        )
    elif args.command == "assemble":
        strict_keys(spec, {"database", "base", "joins", "output"}, {"database", "base", "output"})
        output = resolve(spec["output"])
        if output.exists():
            raise FileExistsError(output)
        result = enrich(resolve(spec["database"]), spec["base"], spec.get("joins", []))
        export_assembly(result, output)
        result = result.manifest
    else:
        strict_keys(
            spec, {"database", "source_id", "version"}, {"database", "source_id", "version"}
        )
        _, result = load_source(resolve(spec["database"]), spec["source_id"], spec["version"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
