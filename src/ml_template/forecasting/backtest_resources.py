"""Opt-in Linux command measurement; not an automatic training/release runner."""

import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

from .acceptance_store import Publication, canonical, file_hash


def directory_bytes(path):
    path = Path(path)
    total = 0
    if not path.exists():
        return total
    for directory, children, names in os.walk(path, followlinks=False):
        children[:] = [name for name in children if not (Path(directory) / name).is_symlink()]
        for name in names:
            file = Path(directory) / name
            try:
                if not file.is_symlink():
                    total += file.stat().st_size
            except FileNotFoundError:
                continue
    return total


def measure(command, label, output, database, watch=(), interval=0.5):
    if not sys.platform.startswith("linux") or not hasattr(os, "wait4"):
        raise RuntimeError("This measurement wrapper currently supports Linux only")
    if not Path(database).is_file():
        raise FileNotFoundError("Existing project database required")
    if not command or not label or not 0.05 <= interval <= 60:
        raise ValueError("Command, label and interval in [0.05,60] required")
    watched = sorted({str(Path(path).resolve()) for path in watch})
    publication = Publication(output)
    stop = threading.Event()
    peaks = {path: 0 for path in watched}
    failures = {}

    def sample():
        for path in watched:
            try:
                peaks[path] = max(peaks[path], directory_bytes(path))
            except OSError as error:
                key = type(error).__name__
                failures[key] = failures.get(key, 0) + 1

    def poll():
        while not stop.wait(interval):
            sample()

    worker = threading.Thread(target=poll, daemon=True)
    try:
        sample()
        initial_bytes = dict(peaks)
        started = time.monotonic()
        with (publication.stage / "stdout.log").open("wb") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            worker.start()
            try:
                _, status, usage = os.wait4(process.pid, 0)
            except BaseException:
                process.kill()
                _, status, _ = os.wait4(process.pid, 0)
                process.returncode = os.waitstatus_to_exitcode(status)
                raise
            finally:
                stop.set()
                worker.join()
            process.returncode = os.waitstatus_to_exitcode(status)
        elapsed = time.monotonic() - started
        sample()
        directory = Path(__file__).parent
        provenance = {
            "python": platform.python_version(),
            "source_sha256": {
                name: file_hash(directory / name)
                for name in ["backtest_resources.py", "acceptance_store.py"]
            },
        }
        value = {
            "measurement_producer": provenance,
            "label": label,
            "command": command,
            "cwd": str(Path.cwd()),
            "exit_code": process.returncode,
            "elapsed_seconds": elapsed,
            "kernel_reported_child_max_rss_kib": usage.ru_maxrss,
            "rss_scope": "Linux wait4 child-reported ru_maxrss; not a summed concurrent process-tree peak",
            "initial_directory_bytes": initial_bytes,
            "sampled_directory_peak_bytes": peaks,
            "directory_poll_interval_seconds": interval,
            "directory_sampling_errors": failures,
            "platform": platform.platform(),
            "artifact_association": "user-supplied label/command only; not proof of association with later report inputs",
        }
        (publication.stage / "resource.json").write_text(canonical(value) + "\n", encoding="utf-8")
        return {
            **publication.publish(
                "resource", {"label": label, "command": command, "producer": provenance}, database
            ),
            "command_exit_code": process.returncode,
        }
    finally:
        stop.set()
        publication.close()
