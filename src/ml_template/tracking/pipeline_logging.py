"""Logging for the whole ml_template package.

``setup_logging`` attaches a console handler and the pipeline overview log to the
top-level ``ml_template`` logger, so every module logger below it is captured.
``run_log`` adds a per-run file for the duration of one run and removes it again,
so consecutive runs in one process (a sweep) never write into each other's logs.

Modules use fixed logger names, for example
``logging.getLogger("ml_template.data.silver")``, because ``__name__`` is
``__main__`` when a module is started with ``python -m``.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from ml_template.paths import LOGS

BASE_LOGGER_NAME = "ml_template"
DEFAULT_APP_LOG = LOGS / "app" / "pipeline.log"
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _formatter() -> logging.Formatter:
    return logging.Formatter(_FORMAT, datefmt=_DATEFMT)


def _file_handler(path: Path, level: int) -> logging.FileHandler:
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setFormatter(_formatter())
    handler.setLevel(level)
    return handler


def setup_logging(
    level: int = logging.INFO, app_logfile: str | Path | None = None
) -> logging.Logger:
    """Console plus overview log on the package logger. Safe to call repeatedly."""
    base = logging.getLogger(BASE_LOGGER_NAME)
    base.setLevel(level)
    base.propagate = False  # keep records out of the root logger

    # Exact type check: subclasses such as pytest's capture handlers are not our console.
    has_console = any(type(h) is logging.StreamHandler for h in base.handlers)
    if not has_console:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(_formatter())
        console.setLevel(level)
        base.addHandler(console)

    target = Path(app_logfile or DEFAULT_APP_LOG).resolve()
    existing = {
        Path(h.baseFilename).resolve() for h in base.handlers if isinstance(h, logging.FileHandler)
    }
    if target not in existing:
        base.addHandler(_file_handler(target, level))
    return base


@contextmanager
def run_log(path: str | Path, level: int = logging.INFO) -> Iterator[Path]:
    """Write everything logged inside the block to ``path`` as well.

    Works without ``setup_logging``: if the package logger would drop records at
    ``level``, its level is lowered for the block and restored afterwards.
    """
    base = logging.getLogger(BASE_LOGGER_NAME)
    target = Path(path).resolve()
    handler = _file_handler(target, level)
    previous_level = base.level
    if base.getEffectiveLevel() > level:
        base.setLevel(level)
    base.addHandler(handler)
    try:
        yield target
    finally:
        base.removeHandler(handler)
        handler.close()
        base.setLevel(previous_level)


def reset_logging() -> None:
    """Remove and close every handler on the package logger (used by tests)."""
    base = logging.getLogger(BASE_LOGGER_NAME)
    for handler in list(base.handlers):
        base.removeHandler(handler)
        handler.close()
