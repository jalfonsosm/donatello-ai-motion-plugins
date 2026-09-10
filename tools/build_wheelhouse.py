#!/usr/bin/env python3
"""Build a redistributable, platform-specific Kimodo wheel bundle.

This script intentionally produces only Kimodo and SOMA-X wheels. Heavy
third-party packages (Torch, NumPy, SciPy, etc.) remain official binary-wheel
downloads at install time, so the plugin release stays reasonably small while
the end user's machine never has to invoke a native compiler.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


KIMODO_REPOSITORY = "https://github.com/nv-tlabs/kimodo.git"
KIMODO_REVISION = "1aece8c124d73d255ceff5086d983b844c9f4e94"
SOMA_REPOSITORY = "https://github.com/NVlabs/SOMA-X.git"
SOMA_REVISION = "d29dbe5a3f5a0b2632ecac91e8d5125f243a7e36"
SCHEMA_VERSION = 1


def platform_key(python: Path) -> str:
    system = {"Darwin": "darwin", "Linux": "linux", "Windows": "windows"}.get(platform.system())
    machine = platform.machine().lower()
    machine = {"aarch64": "arm64", "amd64": "x86_64", "x64": "x86_64"}.get(machine, machine)
    if not system or machine not in {"arm64", "x86_64"}:
        raise RuntimeError(f"Unsupported build host: {platform.system()} {platform.machine()}")
    version = subprocess.check_output([str(python), "-c", "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')"], text=True).strip()
    return f"{system}-{machine}-cp{version}"


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def checkout(destination: Path, repository: str, revision: str, paths: list[str]) -> Path:
    """Fetch only packaging sources, never model/demo assets tracked upstream."""
    run(["git", "clone", "--depth", "1", "--filter=blob:none", "--no-checkout", repository, str(destination)])
    run(["git", "-C", str(destination), "sparse-checkout", "init", "--no-cone"])
    run(["git", "-C", str(destination), "sparse-checkout", "set", "--no-cone", *paths])
    run(["git", "-C", str(destination), "checkout", "--detach", revision])
    return destination


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def one_wheel(directory: Path, prefix: str) -> Path:
    wheels = sorted(directory.glob(f"{prefix}-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected exactly one {prefix} wheel, found: {wheels}")
    return wheels[0]


def build(python: Path, output: Path) -> None:
    key = platform_key(python)
    output.mkdir(parents=True, exist_ok=True)
    # These are build-machine tools only. They do not enter the wheelhouse or
    # the end-user runtime, which installs the resulting wheels directly.
    run([str(python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    with tempfile.TemporaryDirectory(prefix="flatrig_kimodo_wheelhouse_") as temporary:
        root = Path(temporary)
        soma = checkout(
            root / "soma-x", SOMA_REPOSITORY, SOMA_REVISION,
            ["soma/", "setup.cfg", "pyproject.toml", "README.md", "LICENSE"],
        )
        kimodo = checkout(
            root / "kimodo", KIMODO_REPOSITORY, KIMODO_REVISION,
            ["kimodo/", "setup.py", "pyproject.toml", "README.md", "LICENSE"],
        )
        environment = dict(os.environ)
        environment["SKIP_MOTION_CORRECTION_IN_SETUP"] = "1"
        for source in (soma, kimodo):
            run([
                str(python), "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
                str(source), "--wheel-dir", str(output),
            ], env=environment)
        shutil.copy2(kimodo / "LICENSE", output / "KIMODO_LICENSE.txt")
        shutil.copy2(soma / "LICENSE", output / "SOMA_X_LICENSE.txt")

    soma_wheel = one_wheel(output, "py_soma_x")
    kimodo_wheel = one_wheel(output, "kimodo")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "platform": key,
        "kimodo_revision": KIMODO_REVISION,
        "soma_revision": SOMA_REVISION,
        "motion_correction": False,
        "wheels": [
            {"role": "soma", "filename": soma_wheel.name, "sha256": digest(soma_wheel)},
            {"role": "kimodo", "filename": kimodo_wheel.name, "sha256": digest(kimodo_wheel)},
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Built {key} wheelhouse at {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, default=Path(sys.executable), help="CPython used for the target ABI")
    parser.add_argument("--output", type=Path, help="Default: wheelhouse/<host-platform>-cpXY")
    args = parser.parse_args()
    target = args.output or Path(__file__).resolve().parents[1] / "wheelhouse" / platform_key(args.python)
    build(args.python.resolve(), target.resolve())


if __name__ == "__main__":
    main()
