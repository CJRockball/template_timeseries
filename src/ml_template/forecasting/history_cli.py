"History features and naive forecasts: init, predict, score and verify."

import argparse
import json
from pathlib import Path

from .history import recipe
from .history_artifacts import verify_artifact


def main(argv=None):
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    for key in ["plan", "silver-dataset", "database", "output-root", "output"]:
        init.add_argument("--" + key, type=Path, required=True)
    for name in ["predict", "score"]:
        child = commands.add_parser(name)
        child.add_argument("--config", type=Path, required=True)
        if name == "score":
            child.add_argument("--predictions", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "init":
        if (
            not args.database.is_file()
            or not (args.plan / "manifest.json").is_file()
            or (not (args.silver_dataset / "manifest.json").is_file())
        ):
            raise FileNotFoundError("Existing project DB, plan and Silver directory required")
        config = {
            "plan": str(args.plan.resolve()),
            "silver_dataset": str(args.silver_dataset.resolve()),
            "database": str(args.database.resolve()),
            "output_root": str(args.output_root.resolve()),
            "batch_size": 5000,
            "recipe": recipe(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        print(str(args.output.resolve()))
        return
    if args.command == "verify":
        value = verify_artifact(args.artifact)
        print(
            json.dumps(
                {
                    "artifact_id": value["artifact_id"],
                    "kind": value["kind"],
                    "stats": value["stats"],
                },
                indent=2,
            )
        )
        return
    config_file = args.config.resolve()
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    required = {"plan", "silver_dataset", "database", "output_root", "recipe"}
    if (
        not isinstance(config, dict)
        or set(config) - (required | {"batch_size"})
        or (not required <= set(config))
    ):
        raise ValueError("Unknown/missing history config field")

    def resolve(value):
        value = Path(value)
        return value if value.is_absolute() else config_file.parent / value

    from .history_io import predict, score

    values = (resolve(config["plan"]), resolve(config["silver_dataset"]))
    common = (
        recipe(config["recipe"]),
        resolve(config["output_root"]),
        resolve(config["database"]),
        config.get("batch_size", 5000),
    )
    if args.command == "predict":
        result = predict(*values, *common)
    else:
        result = score(*values, args.predictions.resolve(), *common)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
