"""Track an existing pinned scorer without loading a model in the development process."""

import argparse
import json
import sqlite3
import subprocess
from pathlib import Path

from ml_template.monitoring.scoring_attempts import tracked_score

MARKER = "ML_TEMPLATE_PINNED_SCORE_RESULT:"
CHILD = r"""
import json
import sqlite3
import sys
from pathlib import Path
from ml_template.scoring.batch import score_batch
spec = json.loads(sys.argv[1])
conn = None if spec['check_only'] else sqlite3.connect(Path(spec['database']).as_uri() + '?mode=rw', uri=True)
try:
    result = score_batch(conn,spec['release'],spec['input'],output_root=spec['output_root'],
        chunk_size=spec['chunk_size'],batch_definition=spec['batch_definition'],check_only=spec['check_only'])
finally:
    if conn is not None:
        conn.close()
print('ML_TEMPLATE_PINNED_SCORE_RESULT:' + json.dumps(result,allow_nan=False))
"""


class PinnedScoringError(RuntimeError):
    pass


def run_child(python, spec):
    # Preserve the virtualenv executable path: resolving its symlink loses the venv.
    python = Path(python).expanduser().absolute()
    if not python.is_file():
        raise ValueError("--python must identify an existing pinned environment executable")
    process = subprocess.run(
        [str(python), "-I", "-c", CHILD, json.dumps(spec)],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode:
        raise PinnedScoringError(
            f"Pinned scorer exited {process.returncode}: {(process.stderr or process.stdout)[-3000:]}"
        )
    lines = [line[len(MARKER) :] for line in process.stdout.splitlines() if line.startswith(MARKER)]
    if len(lines) != 1:
        raise PinnedScoringError("Pinned scorer returned no unique structured result")
    result = json.loads(lines[0])
    if not isinstance(result, dict):
        raise PinnedScoringError("Invalid pinned scoring result")
    if spec["check_only"]:
        if result.get("ready") is not True or result.get("files_written") is not False:
            raise PinnedScoringError("Pinned preflight did not confirm a write-free check")
    elif result.get("status") != "succeeded" or not isinstance(result.get("scoring_batch_id"), str):
        raise PinnedScoringError("Pinned scorer did not return a successful batch")
    return result


def score_pinned(
    engine,
    python,
    database,
    release_dir,
    input_path,
    *,
    output_root=None,
    chunk_size=None,
    batch_definition=None,
    check_only=False,
    batch_metadata=None,
):
    database = Path(database).resolve()
    if not database.is_file():
        raise ValueError("existing project database required")

    def operation(
        engine, release_dir, input_path, *, stage_hook=None, batch_metadata=None, check_only=False
    ):
        hook = stage_hook or (lambda stage, batch_id=None: None)
        hook("pinned_process")
        spec = {
            "database": str(database),
            "release": str(Path(release_dir).resolve()),
            "input": str(Path(input_path).resolve()),
            "output_root": str(Path(output_root).resolve()) if output_root else None,
            "chunk_size": chunk_size,
            "batch_definition": batch_definition,
            "check_only": check_only,
        }
        result = run_child(python, spec)
        if not check_only:
            from ml_template.monitoring.monitor_inputs import load_batch

            hook("artifact_verification", result["scoring_batch_id"])
            verified = load_batch(engine, result["scoring_batch_id"])
            if verified.release_id != result.get("release_id"):
                raise ValueError("pinned result/release registry mismatch")
        return result

    return tracked_score(
        operation,
        engine,
        release_dir,
        input_path,
        batch_metadata=batch_metadata,
        check_only=check_only,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--database", type=Path, default=Path("data/project.db"))
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/scores"))
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--batch-definition")
    parser.add_argument("--metadata", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    database = args.database.resolve()
    if not database.is_file():
        raise ValueError("existing project database required")
    conn = None if args.check else sqlite3.connect(database.as_uri() + "?mode=rw", uri=True)
    try:
        result = score_pinned(
            conn,
            args.python,
            database,
            args.release,
            args.input,
            output_root=args.output_root,
            chunk_size=args.chunk_size,
            batch_definition=args.batch_definition,
            check_only=args.check,
            batch_metadata=args.metadata,
        )
    finally:
        if conn is not None:
            conn.close()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
