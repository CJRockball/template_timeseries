"""External diagnostics for unchanged temporal CLI producers (Python 3.11+)."""

import argparse
import cProfile
import csv
import hashlib
import json
import os
import pstats
import runpy
import sys
import threading
import time
import traceback
from pathlib import Path


def memory_sample():
    result = {}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                key, value, *_ = line.split()
                result[key.rstrip(":") + "_kib"] = int(value)
    except OSError:
        pass
    try:
        for line in Path("/proc/self/io").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in {"rchar", "wchar", "read_bytes", "write_bytes"}:
                result[key] = int(value)
    except OSError:
        pass
    return result


def write_stats(profile, output):
    profile.dump_stats(str(output / "profile.pstats"))
    stats = pstats.Stats(profile)
    records = []
    edges = []
    for (filename, line, function), (
        primitive,
        calls,
        own,
        cumulative,
        callers,
    ) in stats.stats.items():
        identity = str(filename) + ":" + str(line) + "(" + function + ")"
        records.append(
            {
                "function": identity,
                "primitive_calls": primitive,
                "total_calls": calls,
                "self_seconds": own,
                "cumulative_seconds": cumulative,
                "cumulative_per_call_seconds": cumulative / calls if calls else 0,
            }
        )
        for caller, values in callers.items():
            if isinstance(values, tuple) and len(values) == 4:
                cc, nc, tt, ct = values
                edges.append(
                    {
                        "caller": str(caller),
                        "callee": identity,
                        "primitive_calls": cc,
                        "total_calls": nc,
                        "self_seconds": tt,
                        "cumulative_seconds": ct,
                    }
                )
    for name, values in [("functions.csv", records), ("callers.csv", edges)]:
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            fields = list(values[0]) if values else ["function"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(
                sorted(values, key=lambda row: row.get("cumulative_seconds", 0), reverse=True)
            )
    with (output / "hotspots.txt").open("w", encoding="utf-8") as stream:
        view = pstats.Stats(profile, stream=stream)
        stream.write("CUMULATIVE WALL TIME (nested; do not sum rows)\n")
        view.sort_stats("cumulative").print_stats(60)
        stream.write("\nSELF WALL TIME\n")
        view.sort_stats("tottime").print_stats(60)
        stream.write("\nPROJECT FUNCTIONS\n")
        view.sort_stats("cumulative").print_stats("ml_template", 100)
        stream.write("\nCALLERS OF HISTORY / SQL / HASH / SERIALIZATION\n")
        view.print_callers("for_sample|execute|fetch|hash|write_line|dumps|publish|verify")
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["materialize", "train"], default="materialize")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--samples", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=10)
    args = parser.parse_args(argv)
    if not 0.1 <= args.interval <= 3600:
        parser.error("interval must be between 0.1 and 3600 seconds")
    if args.stage == "train" and args.samples is None:
        parser.error("train requires --samples")
    config = args.config.resolve(strict=True)
    import yaml

    settings = yaml.safe_load(config.read_text())
    output = args.output.resolve()
    for key in ["plan", "silver_dataset", "output_root"]:
        path = Path(settings[key])
        path = (config.parent / path).resolve() if not path.is_absolute() else path.resolve()
        if output.is_relative_to(path) or path.is_relative_to(output):
            parser.error("Profile output must be disjoint from " + key)
    if args.samples:
        sample_path = args.samples.resolve()
        if output.is_relative_to(sample_path) or sample_path.is_relative_to(output):
            parser.error("Profile output overlaps samples")
    output.mkdir(parents=True, exist_ok=False)
    command = ["ml_template.forecasting.temporal_xgb", args.stage, "--config", str(config)]
    if args.samples:
        command.extend(["--samples", str(args.samples.resolve())])
    started, cpu_started = time.perf_counter(), time.process_time()
    stop = threading.Event()
    sampling_errors = []

    def observe():
        try:
            with (output / "timeline.jsonl").open("w", encoding="utf-8") as handle:
                while True:
                    elapsed = time.perf_counter() - started
                    sample = {
                        "elapsed_seconds": elapsed,
                        "process_cpu_seconds": time.process_time() - cpu_started,
                        **memory_sample(),
                    }
                    handle.write(json.dumps(sample) + "\n")
                    handle.flush()
                    print(
                        "[profile] elapsed="
                        + str(round(elapsed, 1))
                        + "s RSS="
                        + str(sample.get("VmRSS_kib", "unavailable"))
                        + " KiB",
                        file=sys.stderr,
                        flush=True,
                    )
                    if stop.wait(args.interval):
                        break
        except Exception as error:
            sampling_errors.append(repr(error))

    worker = threading.Thread(target=observe, daemon=True)
    profile = cProfile.Profile()
    original_argv = sys.argv[:]
    error_text = None
    succeeded = False
    worker.start()
    try:
        sys.argv = command
        profile.enable()
        runpy.run_module(command[0], run_name="__main__", alter_sys=True)
        succeeded = True
    except BaseException:
        error_text = traceback.format_exc()
        raise
    finally:
        profile.disable()
        elapsed = time.perf_counter() - started
        cpu = time.process_time() - cpu_started
        sys.argv = original_argv
        stop.set()
        worker.join()
        write_stats(profile, output)
        metadata = {
            "stage": args.stage,
            "command": command,
            "pid": os.getpid(),
            "python": sys.version,
            "elapsed_seconds": elapsed,
            "process_cpu_seconds": cpu,
            "success": succeeded,
            "exception": error_text,
            "sampling_errors": sampling_errors,
            "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            "final_memory": memory_sample(),
            "scope": "Main-thread deterministic function profile; process memory/CPU includes profiler. Native worker threads and subprocesses not function-profiled. Profiling overhead means elapsed time is not an unprofiled benchmark. No phase/origin timestamps inferred.",
            "artifact_policy": "CLI executes normally and may write/register/reuse artifacts. No producer source edited. Profile is not a registered acceptance resource.",
        }
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print("[profile] saved diagnostics: " + str(output), file=sys.stderr)


if __name__ == "__main__":
    main()
