"CLI: python -m ml_template.temporal_stream ingest|verify|assemble --config FILE."

import argparse
import json
from pathlib import Path

import yaml

from .enrichment import assemble
from .store import ingest_csv, verify_source


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["ingest", "verify", "assemble"])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    path = args.config.resolve()
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))

    def resolve(value):
        p = Path(value)
        return p if p.is_absolute() else path.parent / p

    common = {"database", "batch_size"}
    if args.command == "ingest":
        required = {"database", "source_id", "version", "input", "contract"}
        allowed = common | required | {"provenance"}
    elif args.command == "assemble":
        required = {"database", "base", "output"}
        allowed = common | required | {"joins"}
    else:
        required = {"database", "source_id", "version"}
        allowed = common | required
    if not isinstance(spec, dict) or set(spec) - allowed or (not required <= set(spec)):
        raise ValueError("Unknown or missing configuration field")
    batch = spec.get("batch_size", 5000)
    if args.command == "ingest":
        result = ingest_csv(
            resolve(spec["database"]),
            spec["source_id"],
            spec["version"],
            resolve(spec["input"]),
            spec["contract"],
            spec.get("provenance"),
            batch,
        )
    elif args.command == "assemble":
        result = assemble(
            resolve(spec["database"]),
            spec["base"],
            spec.get("joins", []),
            resolve(spec["output"]),
            batch,
        )
    else:
        result = verify_source(resolve(spec["database"]), spec["source_id"], spec["version"], batch)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
