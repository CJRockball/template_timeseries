"""Prepare or verify an exact historical inference runtime without changing a release."""

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path

VERIFY_CODE = """
import json, sys
from pathlib import Path
import ml_template
from ml_template.scoring.release_artifacts import read_json, verify, verify_runtime, runtime_info
release, root = Path(sys.argv[1]), Path(sys.argv[2])
actual = Path(ml_template.__file__).resolve().parent
if actual != root.resolve():
    raise ValueError(f'Wrong imported source: {actual}; expected {root}')
manifest = verify(release)
expected = read_json(release / 'runtime.json')
try:
    verify_runtime(expected)
except ValueError as exc:
    now = runtime_info()
    packages = {name: {'expected': value, 'actual': now['packages'].get(name)}
                for name, value in expected['packages'].items() if now['packages'].get(name) != value}
    names = set(expected['code_sha256']) | set(now['code_sha256'])
    changed = sorted(name for name in names if expected['code_sha256'].get(name) != now['code_sha256'].get(name))
    raise ValueError(json.dumps({'python_expected': expected['python'], 'python_actual': now['python'],
                                'package_differences': packages, 'changed_source_first_20': changed[:20]})) from exc
print(json.dumps({'ready': True, 'release_id': manifest['release_id'],
                  'runtime': runtime_info(), 'source_root': str(actual),
                  'python': sys.executable}))
"""


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def git(repo, *arguments, input_bytes=None):
    return subprocess.check_output(["git", "-C", str(repo), *arguments], input=input_bytes)


def resolve_commit(repo, commit):
    return (
        git(repo, "rev-parse", "--verify", "--end-of-options", f"{commit}^{{commit}}")
        .decode()
        .strip()
    )


def commit_hashes(repo, commit):
    tree = git(repo, "ls-tree", "-rz", "--full-tree", commit, "--", "src/ml_template")
    entries = []
    for item in tree.split(b"\0"):
        if not item:
            continue
        header, path = item.split(b"\t", 1)
        mode, kind, blob = header.split()
        name = path.decode()
        if not name.endswith(".py"):
            continue
        if mode not in (b"100644", b"100755") or kind != b"blob":
            raise ValueError("symlink/non-file Python sources are unsupported")
        entries.append((name.removeprefix("src/ml_template/"), blob.decode()))
    if not entries:
        return {}
    stream = io.BytesIO(
        git(
            repo,
            "cat-file",
            "--batch",
            input_bytes=("\n".join(blob for _, blob in entries) + "\n").encode(),
        )
    )
    result = {}
    for name, expected_blob in entries:
        header = stream.readline().split()
        if len(header) != 3 or header[0].decode() != expected_blob or header[1] != b"blob":
            raise ValueError("unexpected Git blob response")
        data = stream.read(int(header[2]))
        if stream.read(1) != b"\n":
            raise ValueError("truncated Git blob response")
        result[name] = hashlib.sha256(data).hexdigest()
    return result


def checkout_hashes(checkout):
    root = Path(checkout) / "src/ml_template"
    result = {}
    for path in sorted(root.rglob("*.py")):
        if path.is_symlink():
            raise ValueError("symlink Python sources are unsupported")
        result[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def matching_commit(repo, expected, commit=None, limit=1000):
    if commit:
        candidates = [resolve_commit(repo, commit)]
    else:
        candidates = (
            git(repo, "rev-list", f"--max-count={limit}", "--all", "--", "src/ml_template")
            .decode()
            .splitlines()
        )
    for candidate in candidates:
        if commit_hashes(repo, candidate) == expected:
            return candidate
    raise ValueError(
        "No exact committed source snapshot found; supply the matching commit or recover the original source. No hashes were changed."
    )


def verify_environment(checkout, release, *, timeout=120):
    executable = Path(checkout) / ".venv/bin/python"
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError(f"missing/non-executable pinned Python: {executable}")
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE"):
        env.pop(key, None)
    completed = subprocess.run(
        [
            str(executable),
            "-I",
            "-c",
            VERIFY_CODE,
            str(release),
            str(Path(checkout) / "src/ml_template"),
        ],
        cwd=checkout,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode:
        raise ValueError("Pinned environment does not match: " + completed.stderr[-5000:])
    result = json.loads(completed.stdout)
    if result.get("ready") is not True:
        raise ValueError("runtime verification did not succeed")
    return result


def sync_environment(destination, python_version, timeout):
    environment = os.environ.copy()
    for name in (
        "VIRTUAL_ENV",
        "UV_PROJECT",
        "UV_PROJECT_ENVIRONMENT",
        "UV_PYTHON",
        "PYTHONPATH",
        "PYTHONHOME",
    ):
        environment.pop(name, None)
    environment["UV_PROJECT_ENVIRONMENT"] = str(destination / ".venv")
    subprocess.run(
        ["uv", "sync", "--project", str(destination), "--frozen", "--python", python_version],
        env=environment,
        timeout=timeout,
        check=True,
    )


def prepare(
    repo, release, *, runtime_root=None, checkout=None, commit=None, create=False, timeout=900
):
    from ml_template.scoring.release_artifacts import read_json, verify

    repo, release = Path(repo).resolve(), Path(release).resolve()
    manifest = verify(release)
    runtime = read_json(release / "runtime.json")
    if not re.fullmatch(r"\d+\.\d+\.\d+", runtime.get("python", "")):
        raise ValueError("release must record an exact Python patch version")
    identity = fingerprint(runtime)
    root = Path(runtime_root or repo.parent / ".ml-template-runtimes").expanduser().resolve()
    destination = (
        Path(checkout).expanduser().resolve() if checkout else root / f"rt_{identity[:24]}"
    )
    for target in (root, destination):
        if (
            target in (repo, release)
            or target.is_relative_to(repo)
            or target.is_relative_to(release)
        ):
            raise ValueError(
                "runtime destinations must be outside the development checkout and release"
            )
    pinned_source = destination / "src/ml_template"
    if root == pinned_source or root.is_relative_to(pinned_source):
        raise ValueError("preparation records must remain outside pinned application source")
    if destination.exists() and not destination.is_dir():
        raise ValueError("checkout destination is not a directory")
    existing = destination.exists()
    if existing:
        if checkout_hashes(destination) != runtime["code_sha256"]:
            raise ValueError("existing checkout has different source; it will not be overwritten")
        resolved = resolve_commit(destination, "HEAD")
    else:
        resolved = matching_commit(repo, runtime["code_sha256"], commit)
        for name in ("pyproject.toml", "uv.lock"):
            git(repo, "cat-file", "-e", f"{resolved}:{name}")
    executable = destination / ".venv/bin/python"
    env_exists = executable.is_file() and os.access(executable, os.X_OK)
    verified = (
        verify_environment(destination, release, timeout=min(timeout, 120)) if env_exists else None
    )
    if verified is not None and fingerprint(verified["runtime"]) != identity:
        raise ValueError("verified runtime fingerprint differs")
    result = {
        "ready": verified is not None,
        "created": False,
        "runtime_sha256": identity,
        "release_id": manifest["release_id"],
        "commit": resolved,
        "source_matches_commit": commit_hashes(destination if existing else repo, resolved)
        == runtime["code_sha256"],
        "checkout": str(destination),
        "yaml_values": {
            "runtime_python": str(executable),
            "runtime_source_root": str(destination / "src/ml_template"),
        },
        "next_action": "reuse" if verified is not None else "create_with_explicit_flag",
    }
    if not create:
        return result
    if not env_exists:
        if (destination / ".venv").exists() or (destination / ".venv").is_symlink():
            raise ValueError(
                "incomplete existing .venv will not be repaired or overwritten automatically"
            )
        if not existing:
            destination.parent.mkdir(parents=True, exist_ok=True)
            git(repo, "worktree", "add", "--detach", str(destination), resolved)
        try:
            sync_environment(destination, runtime["python"], timeout)
            verified = verify_environment(destination, release, timeout=min(timeout, 120))
        except Exception as exc:
            exc.add_note(
                f"Historical checkout retained for inspection: {destination}. No release files were edited."
            )
            raise
        result["created"] = True
    if (
        checkout_hashes(destination) != runtime["code_sha256"]
        or fingerprint(verified["runtime"]) != identity
    ):
        raise ValueError("source/runtime changed during preparation")
    verify(release, manifest["release_id"])
    records = root / "records"
    records.mkdir(parents=True, exist_ok=True)
    record = records / f"rt_{identity[:24]}_{uuid.uuid4().hex[:12]}.json"
    record.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "verified": True,
                "created_at": datetime.now(UTC).isoformat(),
                "runtime_sha256": identity,
                "release_id": manifest["release_id"],
                "commit": resolved,
                "checkout": str(destination),
                "yaml_values": result["yaml_values"],
            },
            indent=2,
        )
        + "\n"
    )
    return {**result, "ready": True, "next_action": "reuse", "preparation_manifest": str(record)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--checkout", type=Path)
    parser.add_argument("--commit")
    parser.add_argument("--timeout", type=int, default=900)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--create", action="store_true")
    args = parser.parse_args(argv)
    if args.repo is None:
        from ml_template.paths import ROOT

        args.repo = ROOT
    if args.timeout < 1:
        parser.error("timeout must be positive")
    result = prepare(
        args.repo,
        args.release,
        runtime_root=args.runtime_root,
        checkout=args.checkout,
        commit=args.commit,
        create=args.create,
        timeout=args.timeout,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
