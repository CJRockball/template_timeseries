"""Report frozen backtests or explicitly measure a Linux command. No auto promotion."""

import argparse
import json
from pathlib import Path

from .acceptance_store import verify


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    report = sub.add_parser("report")
    for name in ["samples", "models", "scores", "baseline", "output-root", "database"]:
        report.add_argument("--" + name, type=Path, required=True)
    report.add_argument("--resource", type=Path, action="append", default=[])
    report.add_argument("--max-groups", type=int, default=100000)
    check = sub.add_parser("verify")
    check.add_argument("--artifact", type=Path, required=True)
    resource = sub.add_parser("measure")
    resource.add_argument("--label", required=True)
    resource.add_argument("--output-root", type=Path, required=True)
    resource.add_argument("--database", type=Path, required=True)
    resource.add_argument("--watch-dir", type=Path, action="append", default=[])
    resource.add_argument("--interval", type=float, default=0.5)
    resource.add_argument("argv", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command == "verify":
        result = verify(args.artifact)
    elif args.command == "report":
        from .backtest_acceptance import build

        result = build(
            args.samples,
            args.models,
            args.scores,
            args.baseline,
            args.output_root,
            args.database,
            args.resource,
            args.max_groups,
        )
    else:
        from .backtest_resources import measure

        command = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
        result = measure(
            command, args.label, args.output_root, args.database, args.watch_dir, args.interval
        )
    print(json.dumps(result, indent=2))
    if args.command == "measure" and result["command_exit_code"]:
        raise SystemExit(result["command_exit_code"] if result["command_exit_code"] > 0 else 1)


if __name__ == "__main__":
    main()
