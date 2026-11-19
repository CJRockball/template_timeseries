"CLI: init, materialize, train, compare and verify. Development only."

import argparse
import json
from pathlib import Path

from .learned_artifacts import verify
from .temporal_samples import options


def load(path):
    import yaml

    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    keys = {
        "plan",
        "silver_dataset",
        "database",
        "output_root",
        "batch_size",
        "recipe",
        "samples",
        "model",
        "seed",
    }
    if not isinstance(config, dict) or set(config) != keys:
        raise ValueError("Unknown/missing temporal XGB config field")
    for name in ["plan", "silver_dataset", "database", "output_root"]:
        value = Path(config[name])
        config[name] = value if value.is_absolute() else path.parent / value
    for name in ["batch_size", "seed"]:
        if type(config[name]) is not int or config[name] < (0 if name == "seed" else 1):
            raise ValueError("Invalid " + name)
    options(config["samples"])
    if not config["database"].is_file():
        raise FileNotFoundError("Existing project DB required")
    config["output_root"].mkdir(parents=True, exist_ok=True)
    return config


def main(argv=None):
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--history-config", type=Path, required=True)
    init.add_argument("--output", type=Path, required=True)
    for name in ["materialize", "train", "compare"]:
        child = sub.add_parser(name)
        child.add_argument("--config", type=Path, required=True)
        if name == "train":
            child.add_argument("--samples", type=Path, required=True)
        if name == "compare":
            child.add_argument("--models", type=Path, required=True)
            child.add_argument("--baseline", type=Path, required=True)
    check = sub.add_parser("verify")
    check.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "init":
        from .history import recipe

        source = args.history_config.resolve()
        old = yaml.safe_load(source.read_text(encoding="utf-8"))

        def resolve(value):
            p = Path(value)
            return str(p.resolve() if p.is_absolute() else (source.parent / p).resolve())

        r = recipe(old["recipe"])
        config = {k: resolve(old[k]) for k in ["plan", "silver_dataset", "database"]}
        config.update(
            {
                "output_root": str(Path(resolve(old["output_root"])) / "xgb"),
                "batch_size": old.get("batch_size", 5000),
                "recipe": r,
                "samples": {
                    "features": [
                        "last_available",
                        *[f"lag_{k}" for k in r["lags"]],
                        *["mean_" + str(w["periods"]) for w in r["rolling"]],
                    ],
                    "max_rows_per_partition": 50000,
                },
                "model": {
                    "objective": "reg:squarederror",
                    "eval_metric": "rmse",
                    "n_estimators": 400,
                    "early_stopping_rounds": 20,
                    "max_depth": 4,
                    "learning_rate": 0.05,
                    "device": "cpu",
                },
                "seed": 42,
            }
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
        print(str(args.output.resolve()))
        return
    if args.command == "verify":
        value = verify(args.artifact)
        print(
            json.dumps(
                {
                    "artifact_id": value["artifact_id"],
                    "kind": value["identity"]["kind"],
                    "inputs": value["identity"]["inputs"],
                },
                indent=2,
            )
        )
        return
    config = load(args.config)
    from .temporal_xgb_io import build_samples, fit, score

    if args.command == "materialize":
        value = build_samples(config)
    elif args.command == "train":
        value = fit(config, args.samples)
    else:
        value = score(config, args.models, args.baseline)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
