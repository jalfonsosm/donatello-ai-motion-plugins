"""Build kimodo.cpp's `kmd-generate`/`kmd-inspect` for the current platform
and stage a relocatable, ready-to-run bundle plus a SHA-256 manifest under
`native/<platform>/`. See `native/README.md` for the full contract.

Usage:
    python tools/build_native.py --output native/darwin-arm64
    python tools/build_native.py --output native/linux-x86_64 --enable-vulkan
    python tools/build_native.py --output native/windows-x86_64 --enable-vulkan

This never runs on an end user's machine -- only at release-build time (a
contributor's machine, or CI). The end user only ever receives the resulting
`native/<platform>/` directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_KIMODO_CPP_REPOSITORY = "https://github.com/localai-org/kimodo.cpp.git"
_DEFAULT_REVISION = "568b0253f346fbe369587c7dae73d58594a14c90"
_SCHEMA_VERSION = 1


def _run(command: list[str], **kwargs) -> None:
    print("+", " ".join(str(part) for part in command))
    subprocess.run(command, check=True, **kwargs)


def _native_platform_key() -> str:
    system = {"Darwin": "darwin", "Linux": "linux", "Windows": "windows"}[platform.system()]
    machine = {"aarch64": "arm64", "amd64": "x86_64", "x64": "x86_64"}.get(platform.machine().lower(), platform.machine().lower())
    return f"{system}-{machine}"


def _configure_and_build(source_dir: Path, build_dir: Path, *, enable_vulkan: bool) -> None:
    args = [
        "cmake", "-S", str(source_dir), "-B", str(build_dir), "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release", "-DKIMODO_BUILD_TESTS=OFF",
        f"-DKIMODO_ENABLE_VULKAN={'ON' if enable_vulkan else 'OFF'}",
    ]
    _run(args)
    _run(["cmake", "--build", str(build_dir), "--target", "kmd-generate", "kmd-inspect"])


def _macos_bundle(build_dir: Path, output_dir: Path) -> list[Path]:
    exe_suffix = ""
    binaries = ["kmd-generate", "kmd-inspect"]
    staged: list[Path] = []
    for name in binaries:
        src = build_dir / (name + exe_suffix)
        dst = output_dir / (name + exe_suffix)
        shutil.copy2(src, dst)
        staged.append(dst)

    # Copy every versioned dylib the binaries actually load, renamed to the
    # exact short name (`libX.0.dylib`, no full `.0.20.2` suffix) their own
    # LC_LOAD_DYLIB commands reference -- `otool -L` is the source of truth.
    seen: set[str] = set()
    for binary in staged:
        result = subprocess.run(["otool", "-L", str(binary)], check=True, capture_output=True, text=True)
        for line in result.stdout.splitlines()[1:]:
            line = line.strip()
            if not line.startswith("@rpath/"):
                continue
            short_name = line.split()[0].removeprefix("@rpath/")
            if short_name in seen:
                continue
            seen.add(short_name)
            candidates = sorted(build_dir.rglob(short_name.replace(".0.dylib", ".*.dylib")))
            if not candidates:
                candidates = sorted(build_dir.rglob(short_name))
            if not candidates:
                raise RuntimeError(f"Could not locate {short_name} under {build_dir}")
            dst = output_dir / short_name
            shutil.copy2(candidates[0], dst)
            staged.append(dst)
            subprocess.run(["install_name_tool", "-id", f"@rpath/{short_name}", str(dst)], check=True)

    for binary in staged[:2]:
        result = subprocess.run(["otool", "-l", str(binary)], check=True, capture_output=True, text=True)
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("path ") and "(offset" in line:
                path = line.split()[1]
                if path != "@executable_path":
                    subprocess.run(["install_name_tool", "-delete_rpath", path, str(binary)], check=False)
        subprocess.run(["install_name_tool", "-add_rpath", "@executable_path", str(binary)], check=False)
    return staged


def _linux_bundle(build_dir: Path, output_dir: Path) -> list[Path]:
    binaries = ["kmd-generate", "kmd-inspect"]
    staged: list[Path] = []
    for name in binaries:
        dst = output_dir / name
        shutil.copy2(build_dir / name, dst)
        staged.append(dst)
    for so_path in build_dir.rglob("*.so*"):
        if so_path.is_symlink():
            continue
        dst = output_dir / so_path.name
        shutil.copy2(so_path, dst)
        staged.append(dst)
    for binary in staged:
        subprocess.run(["patchelf", "--set-rpath", "$ORIGIN", str(binary)], check=True)
    return staged


def _windows_bundle(build_dir: Path, output_dir: Path) -> list[Path]:
    staged: list[Path] = []
    for name in ("kmd-generate.exe", "kmd-inspect.exe"):
        candidates = list(build_dir.rglob(name))
        if not candidates:
            raise RuntimeError(f"Could not find {name} under {build_dir}")
        dst = output_dir / name
        shutil.copy2(candidates[0], dst)
        staged.append(dst)
    for dll_path in build_dir.rglob("*.dll"):
        dst = output_dir / dll_path.name
        shutil.copy2(dll_path, dst)
        staged.append(dst)
    return staged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="native/<platform> directory to write")
    parser.add_argument("--kimodo-cpp-revision", default=_DEFAULT_REVISION)
    parser.add_argument("--enable-vulkan", action="store_true", default=(platform.system() != "Darwin"))
    parser.add_argument("--source-dir", type=Path, default=None, help="Reuse an existing checkout instead of cloning")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="kimodo_cpp_build_") as temp:
        source_dir = args.source_dir or (Path(temp) / "kimodo.cpp")
        if args.source_dir is None:
            _run(["git", "clone", "--recurse-submodules", _KIMODO_CPP_REPOSITORY, str(source_dir)])
            _run(["git", "-C", str(source_dir), "checkout", "--detach", args.kimodo_cpp_revision])
            _run(["git", "-C", str(source_dir), "submodule", "update", "--init", "--recursive"])
        ggml_revision = subprocess.run(
            ["git", "-C", str(source_dir / "ggml"), "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
        ).stdout.strip()

        build_dir = Path(temp) / "build"
        _configure_and_build(source_dir, build_dir, enable_vulkan=args.enable_vulkan)

        system = platform.system()
        if system == "Darwin":
            staged = _macos_bundle(build_dir, args.output)
        elif system == "Linux":
            staged = _linux_bundle(build_dir, args.output)
        elif system == "Windows":
            staged = _windows_bundle(build_dir, args.output)
        else:
            raise RuntimeError(f"Unsupported build platform: {system}")

    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "platform": _native_platform_key(),
        "kimodo_cpp_revision": args.kimodo_cpp_revision,
        "ggml_revision": ggml_revision,
        "vulkan_enabled": args.enable_vulkan,
        "files": [
            {
                "filename": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "executable": path.name.startswith("kmd-"),
            }
            for path in sorted(staged, key=lambda p: p.name)
        ],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} ({len(manifest['files'])} files)")


if __name__ == "__main__":
    main()
