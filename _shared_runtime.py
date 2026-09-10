"""Managed-runtime helpers shared by every engine in this plugin.

Kimodo and AnyTop are both heavy PyTorch backends. Rather than each installing
its own multi-gigabyte virtual environment, every engine in this project
installs into **one shared venv** (`shared_runtime_root()`), so PyTorch and
its common dependencies are downloaded and stored exactly once. Engine-owned
material that must stay separable -- model weights, source checkouts, and the
per-engine "is my stuff installed" marker -- still lives under that engine's
own `context.cache_dir`, so removing one engine's cache (Donatello's "Manage
Storage & Models" panel deletes by plugin id) never disturbs another engine's
weights or the shared interpreter itself.
"""

from __future__ import annotations

import hashlib
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

_PERCENT_RE = re.compile(r"(?<!\d)(\d{1,3}(?:\.\d+)?)\s*%")


def shared_runtime_root(cache_dir: Path) -> Path:
    """The one venv every engine installs into.

    `cache_dir` is an engine's own per-plugin cache dir (an engine calls this
    with its own `context.cache_dir`, i.e.
    `<app cache>/local_ai/plugin_cache/motion/<plugin_id>/`). Its parent is
    the shared `motion/` plugin-cache root that Donatello's storage panel
    already scans one level down, so the shared venv shows up there too
    (as `_donatello_shared`) instead of being invisible to cleanup.
    """
    return Path(cache_dir).parent / "_donatello_shared"


def runtime_python(runtime: Path) -> Path:
    return runtime / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def platform_key() -> str:
    system = {"Darwin": "darwin", "Linux": "linux", "Windows": "windows"}.get(platform.system())
    machine = platform.machine().lower()
    machine = {"aarch64": "arm64", "amd64": "x86_64", "x64": "x86_64"}.get(machine, machine)
    if not system or machine not in {"arm64", "x86_64"}:
        return "unsupported"
    return f"{system}-{machine}-cp{sys.version_info.major}{sys.version_info.minor}"


def matches_sha256(path: Path, expected: str) -> bool:
    if not path.is_file():
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == expected


def bootstrap_python() -> Path:
    candidates = [Path(sys.executable)]
    for name in ("python3.12", "python3.11", "python3.10", "python3"):
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))
    for candidate in candidates:
        if candidate.is_file() and _can_create_venv(candidate):
            return candidate
    raise RuntimeError(
        "Donatello AI Motion setup could not find a Python interpreter able to create an isolated environment."
    )


def _can_create_venv(python: Path) -> bool:
    probe = subprocess.run([str(python), "-c", "import venv"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return probe.returncode == 0


def ensure_venv(runtime: Path, progress, check_cancelled, progress_range: tuple[float, float] = (0.0, 0.02)) -> Path:
    """Create the shared venv if it doesn't exist yet. Idempotent and safe to
    call from every engine's own `generate()` -- whichever engine runs first
    creates it; later engines just reuse it."""
    python = runtime_python(runtime)
    if not python.is_file():
        check_cancelled()
        runtime.mkdir(parents=True, exist_ok=True)
        progress(progress_range[0], "Preparing the shared Donatello AI Motion Python runtime")
        bootstrap = bootstrap_python()
        run(
            [str(bootstrap), "-m", "venv", str(runtime / "venv")],
            "Creating the shared Donatello AI Motion runtime", runtime=runtime,
            progress=progress, progress_range=progress_range, check_cancelled=check_cancelled,
        )
    return python


def run(
    command: list[str], label: str, *, runtime: Path | None = None,
    progress=None, progress_range: tuple[float, float] | None = None,
    check_cancelled=None, extra_environment: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> None:
    """Run an interruptible child process and forward useful CLI progress."""
    environment = os.environ.copy()
    if runtime is not None:
        environment["PATH"] = str(runtime_python(runtime).parent) + os.pathsep + environment.get("PATH", "")
        # Gated/large weights must be reclaimable from Donatello's Plugins
        # storage group, not left in a user-global Hugging Face/Torch cache.
        environment["HF_HOME"] = str(runtime / "huggingface")
        environment["HUGGINGFACE_HUB_CACHE"] = str(runtime / "huggingface" / "hub")
        environment["TORCH_HOME"] = str(runtime / "torch")
    if extra_environment:
        environment.update(extra_environment)
    process = subprocess.Popen(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=environment, bufsize=1, cwd=str(cwd) if cwd is not None else None,
    )
    lines: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        try:
            if process.stdout is not None:
                for line in process.stdout:
                    lines.put(line)
        finally:
            lines.put(None)

    threading.Thread(target=read_output, daemon=True).start()
    output: list[str] = []
    done = False
    last_fraction = -1.0
    if progress is not None and progress_range is not None:
        # pip, git and CMake usually emit a stage name but no trustworthy
        # whole-stage percentage. Tell the host to use its travelling bar
        # until the command itself reports a real percentage.
        progress(None, label)
    try:
        while not done:
            if check_cancelled is not None:
                check_cancelled()
            try:
                line = lines.get(timeout=0.25)
            except queue.Empty:
                continue
            if line is None:
                done = True
                continue
            output.append(line)
            if len(output) > 200:
                output.pop(0)
            if progress is not None and progress_range is not None:
                match = _PERCENT_RE.search(line)
                if match:
                    percent = max(0.0, min(100.0, float(match.group(1)))) / 100.0
                    fraction = progress_range[0] + percent * (progress_range[1] - progress_range[0])
                    if fraction > last_fraction:
                        progress(fraction, f"{label}: {line.strip()[:160]}")
                        last_fraction = fraction
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        raise
    return_code = process.wait()
    if return_code != 0:
        detail = "".join(output)[-3000:].strip()
        raise RuntimeError(f"{label} failed.{' ' + detail if detail else ''}")


def pip_install(
    python: Path, packages: list[str], label: str, *, runtime: Path,
    progress=None, progress_range: tuple[float, float] | None = None,
    check_cancelled=None, only_binary: bool = True,
) -> None:
    """Idempotent pip install into the shared venv. `only_binary` keeps the
    installer native-compiler-free; callers with a pure-Python sdist-only
    dependency (no wheel published) must install it separately first, since a
    package already satisfied in the venv is left alone regardless of this
    flag."""
    command = [str(python), "-m", "pip", "install", "--prefer-binary", *packages]
    if only_binary:
        command.insert(4, "--only-binary=:all:")
    run(command, label, runtime=runtime, progress=progress, progress_range=progress_range, check_cancelled=check_cancelled)
