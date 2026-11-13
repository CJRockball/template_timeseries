"""Content-addressed releases. Integrity checks are not authentication/signatures."""

import hashlib
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath

PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "xgboost",
    "torch",
    "pyarrow",
    "pydantic",
    "joblib",
    "PyYAML",
)


def canonical(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)


def read_json(path):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError(f"duplicate JSON key: {key}")
            out[key] = value
        return out

    def invalid(value):
        raise ValueError(f"invalid JSON constant: {value}")

    return json.loads(
        Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=invalid
    )


def write_json(path, data):
    Path(path).write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inside(root, name):
    root = Path(root).resolve()
    part = PurePosixPath(name)
    if (
        not name
        or part.is_absolute()
        or ".." in part.parts
        or "\\" in name
        or part.as_posix() != name
    ):
        raise ValueError("unsafe artifact path")
    path = root.joinpath(*part.parts)
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("unsafe artifact symlink")
    return path


def file_entries(root, *, exclude=("manifest.json",)):
    root = Path(root)
    entries = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("package symlinks are unsupported")
        name = path.relative_to(root).as_posix()
        if path.is_file() and name not in exclude:
            entries[name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    return entries


def seal(root):
    payload = {"schema_version": 1, "files": file_entries(root)}
    digest = hashlib.sha256(canonical(payload).encode()).hexdigest()
    manifest = {**payload, "content_sha256": digest, "release_id": f"rel_{digest[:24]}"}
    write_json(Path(root) / "manifest.json", manifest)
    return manifest


def verify(root, expected_release_id=None):
    root = Path(root).resolve()
    if (root / "manifest.json").is_symlink():
        raise ValueError("unsafe manifest symlink")
    manifest = read_json(root / "manifest.json")
    if (
        set(manifest) != {"schema_version", "files", "content_sha256", "release_id"}
        or manifest["schema_version"] != 1
    ):
        raise ValueError("unsupported release manifest")
    payload = {"schema_version": 1, "files": manifest["files"]}
    digest = hashlib.sha256(canonical(payload).encode()).hexdigest()
    if manifest["content_sha256"] != digest or manifest["release_id"] != f"rel_{digest[:24]}":
        raise ValueError("release content identity mismatch")
    if expected_release_id is not None and expected_release_id != manifest["release_id"]:
        raise ValueError("unexpected release identity")
    for name in manifest["files"]:
        inside(root, name)
    if file_entries(root) != manifest["files"]:
        raise ValueError("release artifact missing, modified or unlisted")
    required = {
        "release.json",
        "config.json",
        "contract.json",
        "feature_recipe.json",
        "runtime.json",
        "evidence.json",
    }
    if "predictor.json" in manifest["files"]:
        from ml_template.models.predictor_selection import required_predictor_files

        descriptor = read_json(root / "predictor.json")
        required.update(required_predictor_files(descriptor))
        required.add("predictor.json")
        for name, expected_hash in descriptor["files_sha256"].items():
            if manifest["files"].get(name, {}).get("sha256") != expected_hash:
                raise ValueError(f"predictor descriptor/hash binding mismatch: {name}")
    else:
        # Legacy final-model releases retain their original mandatory model file.
        required.add("model.ubj")
    if not required <= manifest["files"].keys():
        raise ValueError("incomplete release package")
    return manifest


def runtime_info():
    import ml_template

    root = Path(ml_template.__file__).resolve().parent
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not-installed"
    return {
        "python": platform.python_version(),
        "packages": versions,
        "code_sha256": {
            p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob("*.py"))
        },
    }


def verify_runtime(expected):
    if runtime_info() != expected:
        raise ValueError(
            "runtime/code differs from release; restore the pinned source and locked environment"
        )
