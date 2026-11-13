"""Git facts recorded with every run: commit and whether the tree was dirty."""

from __future__ import annotations

import subprocess

from ml_template.paths import ROOT


def _run_git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return None


def git_sha(short: bool = False) -> str:
    """Current commit hash, or 'unknown' outside a git checkout."""
    out = _run_git("rev-parse", *(["--short"] if short else []), "HEAD")
    return out or "unknown"


def git_is_dirty() -> bool | None:
    """True if tracked files have uncommitted changes. None if git is unavailable.

    Untracked files are ignored, since data and artifacts are untracked by design.
    """
    out = _run_git("status", "--porcelain", "--untracked-files=no")
    return None if out is None else bool(out)


def git_state() -> dict[str, str | bool | None]:
    """Commit and dirty flag. A run on a dirty tree cannot be rebuilt from its SHA alone."""
    return {"git_sha": git_sha(), "git_dirty": git_is_dirty()}
