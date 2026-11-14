"CLI: init, build, verify, summary. Builds no features and executes no models."

import argparse
import json
from pathlib import Path

from .artifacts import verify_plan
from .io import build_from_silver
from .policy import validate_policy


def main(argv=None):
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    bounds = commands.add_parser("bounds")
    bounds.add_argument("--silver-dataset", type=Path, required=True)
    bounds.add_argument("--holdout-periods", type=int, default=28)
    bounds.add_argument("--batch-size", type=int, default=5000)
    init = commands.add_parser("init")
    init.add_argument("--silver-dataset", type=Path, required=True)
    init.add_argument("--database", type=Path, required=True)
    init.add_argument("--output-root", type=Path, required=True)
    init.add_argument("--output", type=Path, required=True)
    init.add_argument("--train-start", required=True)
    init.add_argument("--origin", action="append", required=True)
    init.add_argument("--business-timezone", required=True)
    init.add_argument("--stopping-periods", type=int, default=7)
    init.add_argument("--min-train-observations", type=int, default=28)
    init.add_argument("--holdout-periods", type=int, default=28)
    build = commands.add_parser("build")
    build.add_argument("--config", type=Path, required=True)
    for command in ["verify", "summary"]:
        child = commands.add_parser(command)
        child.add_argument("--plan", type=Path, required=True)
        child.add_argument("--silver-dataset", type=Path)
    args = parser.parse_args(argv)
    if args.command == "bounds":
        from .io import inspect_calendar

        print(
            json.dumps(
                inspect_calendar(args.silver_dataset, args.holdout_periods, args.batch_size),
                indent=2,
            )
        )
        return
    if args.command == "init":
        policy = validate_policy(
            {
                "development_origins": args.origin,
                "train_start": args.train_start,
                "stopping_periods": args.stopping_periods,
                "min_train_observations": args.min_train_observations,
                "holdout_periods": args.holdout_periods,
                "business_timezone": args.business_timezone,
            }
        )
        if not args.database.is_file() or not (args.silver_dataset / "manifest.json").is_file():
            raise FileNotFoundError("Existing Silver directory and project database required")
        config = {
            "silver_dataset": str(args.silver_dataset.resolve()),
            "database": str(args.database.resolve()),
            "output_root": str(args.output_root.resolve()),
            "batch_size": 5000,
            "policy": policy,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        print(str(args.output.resolve()))
        return
    if args.command in {"verify", "summary"}:
        if args.silver_dataset:
            from .io import verify_silver_binding

            result = verify_silver_binding(args.plan, args.silver_dataset)
        else:
            result = verify_plan(args.plan)
        print(
            json.dumps(
                {
                    "plan_id": result["plan_id"],
                    "holdout": result["identity"]["holdout"],
                    "validation": result["validation"],
                },
                indent=2,
            )
        )
        return
    config_file = args.config.resolve()
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if (
        not isinstance(config, dict)
        or set(config) - {"silver_dataset", "database", "output_root", "batch_size", "policy"}
        or (not {"silver_dataset", "database", "output_root", "policy"} <= set(config))
    ):
        raise ValueError("Unknown/missing plan config key")

    def resolve(value):
        value = Path(value)
        return value if value.is_absolute() else config_file.parent / value

    result = build_from_silver(
        resolve(config["silver_dataset"]),
        config["policy"],
        resolve(config["database"]),
        resolve(config["output_root"]),
        config.get("batch_size", 5000),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
