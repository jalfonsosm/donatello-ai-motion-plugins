"""Kimodo adapter for Donatello's generic AI Motion API.

The host knows nothing about Kimodo. This plugin installs its dependencies
into the *shared* Donatello AI Motion venv (`_shared_runtime`) on first
generation, then runs Kimodo in that separate Python process. Only the
model weights and Kimodo/SOMA-X source checkout are private to this engine
(under its own `context.cache_dir`); the interpreter and its common
dependencies (PyTorch, etc.) are shared with the other engines in this
project so they aren't downloaded and installed twice.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from flatrig_private.local_ai.motion_plugin_api import (
    MotionContext,
    MotionPlugin,
    MotionRequest,
    MotionResult,
)

# Sibling `_shared_*` modules aren't on sys.path by default: the host loads
# this file via `importlib.util.spec_from_file_location`, which does not add
# its own directory to sys.path the way a normally-imported package would.
_PLUGIN_DIR = str(Path(__file__).resolve().parent)
if _PLUGIN_DIR not in sys.path:
    sys.path.insert(0, _PLUGIN_DIR)

import _shared_runtime as runtime


_KIMODO_REPOSITORY = "https://github.com/nv-tlabs/kimodo.git"
_KIMODO_REVISION = "1aece8c124d73d255ceff5086d983b844c9f4e94"
_SOMA_REPOSITORY = "https://github.com/NVlabs/SOMA-X.git"
_SOMA_REVISION = "d29dbe5a3f5a0b2632ecac91e8d5125f243a7e36"
_RUNTIME_VERSION = 7  # bumped: dependencies now install into the shared venv, not a per-engine one; +pygltflib
_BUNDLE_SCHEMA_VERSION = 1
_BINARY_DEPENDENCIES = (
    "torch", "hydra-core>=1.3", "omegaconf>=2.3", "numpy>=1.23", "scipy>=1.10",
    "transformers==5.1.0", "urllib3>=2.6.3", "boto3", "peft>=0.18", "einops>=0.7", "pygltflib>=1.16",
    "tqdm>=4.0", "packaging>=21.0", "pydantic>=2.0", "filelock>=3.20.3",
    "gradio>=6.8.0", "gradio_client>=1.0", "trimesh>=3.21.7", "scenepic>=1.1.0",
    "pillow>=9.0", "av>=16.1.0", "bvhio",
)


class KimodoPlugin(MotionPlugin):
    plugin_id = "kimodo"
    label = "Kimodo"
    description = "Text-to-motion. Its isolated local runtime installs automatically on first use."
    version = "0.3.0"
    license_id = "NVIDIA Open Model License"
    license_url = "https://github.com/nv-tlabs/kimodo"
    requires_consent = True
    experimental = True
    setup_on_generate = True

    def form_schema(self) -> list[dict[str, Any]]:
        return [
            {"id": "model", "label": "Kimodo model", "type": "select", "default": "Kimodo-SOMA-RP-v1.1", "options": ["Kimodo-SOMA-RP-v1.1", "Kimodo-SOMA-SEED-v1.1"]},
            {"id": "steps", "label": "Diffusion steps", "type": "number", "default": 42, "min": 1, "max": 200, "step": 1},
        ]

    def is_installed(self, cache_root: Path) -> bool:
        from flatrig_private.local_ai.motion_plugins import plugin_cache_directory
        return _install_is_ready(plugin_cache_directory(cache_root, self.plugin_id))

    def generate(self, request: MotionRequest, context: MotionContext) -> MotionResult:
        context.check_cancelled()
        shared = runtime.shared_runtime_root(context.cache_dir)
        python = runtime.ensure_venv(shared, context.progress, context.check_cancelled, progress_range=(0.0, 0.02))
        _ensure_dependencies(python, shared, context)
        npz = context.output_dir / "kimodo_motion.npz"
        bvh = context.output_dir / "kimodo_motion.bvh"
        model = str(request.extra["model"])
        context.progress(0.5, "Starting Kimodo generation")
        command = [
            str(python), "-m", "kimodo.scripts.generate", request.prompt,
            "--model", model, "--duration", str(request.duration_seconds),
            "--output", str(npz), "--diffusion_steps", str(int(request.extra["steps"])),
            "--seed", str(request.seed),
        ]
        if not _install_has_motion_correction(context.cache_dir):
            # The optional native extension is only enabled by a verified
            # platform bundle. Kimodo's CLI supports this documented opt-out.
            command.append("--no-postprocess")
        runtime.run(
            command, "Kimodo generation", runtime=shared,
            progress=context.progress, progress_range=(0.5, 0.9),
            check_cancelled=context.check_cancelled,
        )
        context.check_cancelled()
        context.progress(0.9, "Converting Kimodo motion to SOMA BVH")
        runtime.run(
            [str(python), "-m", "kimodo.scripts.motion_convert", str(npz), str(bvh)],
            "Kimodo BVH conversion", runtime=shared,
            progress=context.progress, progress_range=(0.9, 0.99),
            check_cancelled=context.check_cancelled,
        )
        if not bvh.is_file():
            raise RuntimeError("Kimodo did not produce a BVH output.")
        context.check_cancelled()
        context.progress(0.99, "Exporting GLB")
        # GLB, not FBX: the retarget pipeline's 3D-source-export step reads
        # the source through Blender, and Blender's FBX importer only
        # supports *binary* FBX, which this project's pure-Python exporter
        # does not produce. BVH itself is not an accepted *source* format
        # for that step either (only an accepted output) -- GLB is. This
        # conversion needs `pygltflib`, which lives in the shared venv, not
        # in this host process -- run it there as a subprocess.
        glb_path = context.output_dir / "kimodo_motion.glb"
        runtime.run(
            [str(python), str(_plugin_root() / "_bvh_to_glb_driver.py"), str(_plugin_root()), str(bvh), str(glb_path)],
            "Converting BVH to GLB", runtime=shared, progress=context.progress,
            progress_range=(0.99, 1.0), check_cancelled=context.check_cancelled,
        )
        if not glb_path.is_file():
            raise RuntimeError("Kimodo did not produce a GLB output.")
        context.progress(1.0, "Kimodo GLB exported")
        return MotionResult(
            glb_path,
            {"source_skeleton": "soma", "fps": request.fps, "model": model},
            artifact_kind="animation_3d",
        )


def _install_marker(cache_dir: Path) -> Path:
    return cache_dir / "installed.json"


def _install_is_ready(cache_dir: Path) -> bool:
    marker = _install_marker(cache_dir)
    if not marker.is_file():
        return False
    try:
        return int(json.loads(marker.read_text(encoding="utf-8")).get("runtime_version", 0)) == _RUNTIME_VERSION
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _install_metadata(cache_dir: Path) -> dict[str, Any]:
    try:
        data = json.loads(_install_marker(cache_dir).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _install_has_motion_correction(cache_dir: Path) -> bool:
    return _install_metadata(cache_dir).get("motion_correction") == "prebuilt"


def _ensure_dependencies(python: Path, shared: Path, context: MotionContext) -> None:
    cache_dir = context.cache_dir
    if _install_is_ready(cache_dir):
        return

    context.check_cancelled()
    context.progress(0.02, "Installing Kimodo runtime dependencies (first use only)")
    runtime.run(
        [str(python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        "Preparing Kimodo installer", runtime=shared, progress=context.progress,
        progress_range=(0.02, 0.05), check_cancelled=context.check_cancelled,
    )
    # hydra-core (a binary dependency below) pins antlr4-python3-runtime==4.9.*,
    # and that package has never published a wheel for the 4.9.x series (only
    # 4.10+ did) -- so `--only-binary=:all:` makes the whole batch install
    # unresolvable. Install it first, explicitly allowed to build from its
    # sdist: it is a pure-Python parser runtime with no C extension, so this
    # still never invokes a native compiler.
    runtime.run(
        [
            str(python), "-m", "pip", "install",
            "--no-binary", "antlr4-python3-runtime", "antlr4-python3-runtime==4.9.3",
        ],
        "Installing Kimodo's pure-Python ANTLR runtime", runtime=shared, progress=context.progress,
        progress_range=(0.05, 0.06), check_cancelled=context.check_cancelled,
    )
    runtime.pip_install(
        python, list(_BINARY_DEPENDENCIES), "Installing Kimodo binary dependencies",
        runtime=shared, progress=context.progress, progress_range=(0.06, 0.2), check_cancelled=context.check_cancelled,
    )
    bundle = _platform_bundle()
    if bundle is not None:
        _install_platform_bundle(python, shared, cache_dir, bundle, context.progress, context.check_cancelled)
        motion_correction = "prebuilt" if bundle.get("motion_correction") else "disabled"
    else:
        _install_source_runtime(python, shared, cache_dir, context.progress, context.check_cancelled)
        motion_correction = "disabled"
    context.check_cancelled()
    _install_marker(cache_dir).write_text(json.dumps({
        "runtime_version": _RUNTIME_VERSION,
        "platform": runtime.platform_key(),
        "motion_correction": motion_correction,
    }), encoding="utf-8")


def _plugin_root() -> Path:
    return Path(__file__).resolve().parent


def _platform_bundle() -> dict[str, Any] | None:
    """Return a verified local wheel bundle for this OS/CPU/Python ABI."""
    bundle_dir = _plugin_root() / "wheelhouse" / runtime.platform_key()
    manifest_path = bundle_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict) or manifest.get("schema_version") != _BUNDLE_SCHEMA_VERSION:
        return None
    if manifest.get("platform") != runtime.platform_key() or not isinstance(manifest.get("wheels"), list):
        return None
    roles = set()
    for item in manifest["wheels"]:
        if not isinstance(item, dict):
            return None
        filename, digest, role = item.get("filename"), item.get("sha256"), item.get("role")
        if not isinstance(filename, str) or Path(filename).name != filename:
            return None
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            return None
        if role not in {"kimodo", "soma", "motion_correction"} or not runtime.matches_sha256(bundle_dir / filename, digest):
            return None
        roles.add(role)
    required_roles = {"kimodo", "soma"}
    has_correction = "motion_correction" in roles
    if not required_roles.issubset(roles) or bool(manifest.get("motion_correction")) != has_correction:
        return None
    return manifest


def _install_platform_bundle(python: Path, shared: Path, cache_dir: Path, bundle: dict[str, Any], progress, check_cancelled) -> None:
    bundle_dir = _materialize_platform_bundle(cache_dir, bundle)
    wheels = {item["role"]: bundle_dir / item["filename"] for item in bundle["wheels"]}
    installs = [("soma", (0.2, 0.22)), ("kimodo", (0.22, 0.24))]
    if "motion_correction" in wheels:
        installs.append(("motion_correction", (0.24, 0.25)))
    for role, span in installs:
        runtime.run(
            [str(python), "-m", "pip", "install", "--no-index", "--no-deps", str(wheels[role])],
            f"Installing prebuilt Kimodo {role} wheel", runtime=shared, progress=progress,
            progress_range=span, check_cancelled=check_cancelled,
        )


def _materialize_platform_bundle(cache_dir: Path, bundle: dict[str, Any]) -> Path:
    """Copy verified release wheels into the removable managed plugin cache."""
    import shutil
    source_dir = _plugin_root() / "wheelhouse" / runtime.platform_key()
    target_dir = cache_dir / "bundles" / runtime.platform_key()
    target_dir.mkdir(parents=True, exist_ok=True)
    for item in bundle["wheels"]:
        source = source_dir / item["filename"]
        target = target_dir / item["filename"]
        if not runtime.matches_sha256(target, item["sha256"]):
            shutil.copy2(source, target)
        if not runtime.matches_sha256(target, item["sha256"]):
            raise RuntimeError(f"The copied Kimodo wheel did not match its manifest: {item['filename']}")
    shutil.copy2(source_dir / "manifest.json", target_dir / "manifest.json")
    return target_dir


def _install_source_runtime(python: Path, shared: Path, cache_dir: Path, progress, check_cancelled) -> None:
    """Fallback with no native compilation; a release bundle is preferred."""
    soma = _source_checkout(cache_dir, "soma-x", _SOMA_REPOSITORY, _SOMA_REVISION, progress, (0.2, 0.22), check_cancelled)
    kimodo = _source_checkout(cache_dir, "kimodo", _KIMODO_REPOSITORY, _KIMODO_REVISION, progress, (0.22, 0.23), check_cancelled)
    runtime.run(
        [str(python), "-m", "pip", "install", "--no-deps", "--no-build-isolation", str(soma)],
        "Installing SOMA runtime without native compilation", runtime=shared,
        progress=progress, progress_range=(0.23, 0.24), check_cancelled=check_cancelled,
    )
    runtime.run(
        [str(python), "-m", "pip", "install", "--no-deps", "--no-build-isolation", str(kimodo)],
        "Installing Kimodo without native compilation", runtime=shared,
        progress=progress, progress_range=(0.24, 0.25), check_cancelled=check_cancelled,
        extra_environment={"SKIP_MOTION_CORRECTION_IN_SETUP": "1"},
    )


def _source_checkout(cache_dir: Path, name: str, repository: str, revision: str, progress, span, check_cancelled) -> Path:
    """Keep a patchable upstream checkout inside the plugin's managed cache."""
    source = cache_dir / "source" / name
    if not (source / ".git").is_dir():
        source.parent.mkdir(parents=True, exist_ok=True)
        runtime.run(
            ["git", "clone", "--depth", "1", repository, str(source)],
            f"Downloading {name} source", progress=progress,
            progress_range=span, check_cancelled=check_cancelled,
        )
    runtime.run(
        ["git", "-C", str(source), "checkout", "--detach", revision],
        f"Pinning {name} source", progress=progress,
        progress_range=span, check_cancelled=check_cancelled,
    )
    return source


ENGINES = [KimodoPlugin]
