"""Kimodo adapter for Donatello's generic AI Motion API, via kimodo.cpp
(https://github.com/localai-org/kimodo.cpp, Apache-2.0/MIT) instead of
NVIDIA's Python reference implementation.

Why the switch: NVIDIA's Python kimodo needs the full fp16
`meta-llama/Meta-Llama-3-8B-Instruct` (~16GB) loaded in PyTorch, with no
quantization path available, and cannot run at all on an 8GB-VRAM machine.
kimodo.cpp is a from-scratch GGML/C++ reimplementation of the same models
(motion diffusion *and* the LLM2Vec bidirectional text encoder -- a real
engineering feat, since LLM2Vec strips Llama-3's causal attention mask,
which a stock llama.cpp fork does not support) with a layer-streaming knob
(`KIMODO_TEXT_LAYER_CHUNK`) that keeps only a few of the text encoder's 32
transformer layers resident in VRAM/RAM at once -- the actual mechanism that
makes an 8GB card workable. (It is not weight quantization: kimodo.cpp's own
README says "quantised models are not implemented yet", so the GGUF download
is comparable in size to the original fp16 weights.)

This adapter ships prebuilt native binaries per platform under
`native/<platform>/` (built by `tools/build_native.py`, see
`.github/workflows/build-native.yml` for the cross-platform CI matrix), so an
end user never needs a C++ toolchain -- matching this project's existing
"never invokes a native compiler on an end-user's machine" rule.
"""

from __future__ import annotations

import json
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
_PLUGIN_DIR = Path(__file__).resolve().parent
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

import _shared_runtime as runtime

_MOTION_REPO = "LocalAI-io/Kimodo-SOMA-RP-v1.1-GGML"
_MOTION_FILENAME = "models/kimodo-soma-rp-v1.1-f32.gguf"
_TEXT_REPO = "LocalAI-io/Llama-3-Kimodo-GGML"
_TEXT_BUNDLE_PATTERN = "generated/llm2vec-text-bundle/*"
_TEXT_BUNDLE_SUBDIR = "generated/llm2vec-text-bundle"
_NATIVE_MANIFEST_SCHEMA_VERSION = 1
_GLB_RUNTIME_VERSION = 1
# kimodo.cpp's CLI takes a frame count directly, not a duration/fps pair;
# 30 fps matches the original NVIDIA Python plugin's own default.
_FPS = 30.0
_GLB_DEPENDENCIES = ("numpy", "pygltflib>=1.16")

# Apache-2.0: copied from kimodo.cpp's src/skeleton.hpp, which itself is
# copied from NVIDIA Kimodo's Apache-2.0 kimodo/skeleton/definitions.py --
# the fixed 30-joint "soma30" control skeleton every SOMA RP/SEED checkpoint
# predicts (parent-local rest offsets, in metres).
_SOMA30_NAMES = [
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "Jaw",
    "LeftEye", "RightEye", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "LeftHandThumbEnd", "LeftHandMiddleEnd", "RightShoulder", "RightArm", "RightForeArm",
    "RightHand", "RightHandThumbEnd", "RightHandMiddleEnd", "LeftLeg", "LeftShin", "LeftFoot",
    "LeftToeBase", "RightLeg", "RightShin", "RightFoot", "RightToeBase",
]
_SOMA30_PARENTS = [
    -1, 0, 1, 2, 3, 4, 5, 6, 6, 6, 3, 10, 11, 12, 13, 13, 3, 16, 17, 18, 19, 19, 0, 22, 23, 24, 0, 26, 27, 28,
]
_SOMA30_OFFSETS = [
    (0.0, 0.0, 0.0), (-0.00013727, 0.0500376256, -0.00053726669),
    (-1.86574103e-9, 0.0712530139, -0.000298248546), (-5.75188398e-9, 0.0755006305, -0.00815970992),
    (-0.00181676517, 0.263112953, -0.00553348292), (-2.85102231e-8, 0.0770939664, 0.0230258546),
    (-4.5975437e-8, 0.0612891595, 0.0195370861), (2.63687901e-5, 0.0047559225, 0.0309494062),
    (0.0320638079, 0.0538020513, 0.0758688308), (-0.0322244017, 0.05361869, 0.0755823359),
    (0.0162165175, 0.232371641, 0.0511341324), (0.149198457, 2.19397873e-8, -0.0550232576),
    (0.287393078, 2.50268389e-9, -2.58787737e-5), (0.270939812, -7.06625108e-9, 2.60897248e-5),
    (0.122686267, -0.0322017573, 0.0483306876), (0.190119595, -0.00312878387, -0.000339570373),
    (-0.0138011824, 0.231803086, 0.0521415786), (-0.150371962, 1.17387901e-7, -0.0554560437),
    (-0.287366393, 1.87628082e-8, -2.59709359e-5), (-0.271336198, -1.16767401e-9, 2.61269368e-5),
    (-0.122642483, -0.0321145448, 0.0480403904), (-0.190005945, -0.00306615542, -0.0003157343),
    (0.10043214, -0.0843452671, 0.0259565473), (-1e-8, -0.432217537, -0.00802912805),
    (1e-8, -0.421550959, -0.0348152298), (0.0, -0.0505947206, 0.132315294),
    (-0.10047278, -0.0829525995, 0.0262031695), (1e-8, -0.433622059, -0.00805555828),
    (2e-8, -0.421173943, -0.0347839785), (-3.42907669e-9, -0.0507960932, 0.132841956),
]


class KimodoPlugin(MotionPlugin):
    plugin_id = "kimodo"
    label = "Kimodo"
    description = (
        "Text-to-motion (native C++/GGML runtime, CPU/Metal/Vulkan). "
        "Its runtime and weights install automatically on first use."
    )
    version = "0.4.0"
    license_id = "NVIDIA Open Model License"
    license_url = "https://github.com/nv-tlabs/kimodo"
    requires_consent = True
    experimental = True
    setup_on_generate = True

    def form_schema(self) -> list[dict[str, Any]]:
        return [
            {"id": "steps", "label": "Diffusion steps", "type": "number", "default": 42, "min": 1, "max": 200, "step": 1},
        ]

    def is_installed(self, cache_root: Path) -> bool:
        from flatrig_private.local_ai.motion_plugins import plugin_cache_directory
        cache_dir = plugin_cache_directory(cache_root, self.plugin_id)
        return _native_binary_dir() is not None and _weights_are_ready(cache_dir)

    def generate(self, request: MotionRequest, context: MotionContext) -> MotionResult:
        context.check_cancelled()
        context.progress(0.0, "Preparing Kimodo's native runtime")
        native_dir = _native_binary_dir()
        if native_dir is None:
            raise RuntimeError(
                f"No prebuilt Kimodo native binary for this platform ({runtime.native_platform_key()})."
            )
        binary = native_dir / ("kmd-generate.exe" if sys.platform == "win32" else "kmd-generate")

        motion_gguf, text_bundle_dir = _ensure_weights(context)
        context.check_cancelled()

        shared = runtime.shared_runtime_root(context.cache_dir)
        python = runtime.ensure_venv(shared, context.progress, context.check_cancelled, progress_range=(0.65, 0.67))
        _ensure_glb_dependencies(python, shared, context)

        prompt_path = context.output_dir / "prompt.txt"
        prompt_path.write_text(request.prompt, encoding="utf-8")
        raw_dir = context.output_dir / "raw"
        frames = max(1, round(request.duration_seconds * _FPS))
        steps = int(request.extra["steps"])

        context.progress(0.75, "Sampling Kimodo motion")
        runtime.run(
            [
                str(binary), str(motion_gguf), str(text_bundle_dir), str(prompt_path),
                str(frames), str(steps), str(request.seed), str(raw_dir),
            ],
            "Kimodo generation", cwd=native_dir, progress=context.progress,
            progress_range=(0.75, 0.95), check_cancelled=context.check_cancelled,
        )
        context.check_cancelled()
        context.progress(0.95, "Exporting GLB")
        glb_path = context.output_dir / "kimodo_motion.glb"
        skeleton_args: list[str] = []
        for index, name in enumerate(_SOMA30_NAMES):
            offset = _SOMA30_OFFSETS[index]
            skeleton_args += [name, str(_SOMA30_PARENTS[index]), str(offset[0]), str(offset[1]), str(offset[2])]
        runtime.run(
            [
                str(python), str(_PLUGIN_DIR / "_kimodo_native_to_glb_driver.py"), str(_PLUGIN_DIR),
                str(raw_dir / "root_positions.f32"), str(raw_dir / "local_rotations_xyzw.f32"),
                str(glb_path), str(_FPS), str(len(_SOMA30_NAMES)), *skeleton_args,
            ],
            "Converting native output to GLB", runtime=shared, progress=context.progress,
            progress_range=(0.95, 1.0), check_cancelled=context.check_cancelled,
        )
        if not glb_path.is_file():
            raise RuntimeError("Kimodo did not produce a GLB output.")
        context.progress(1.0, "Kimodo GLB exported")
        return MotionResult(
            glb_path,
            {"source_skeleton": "soma", "fps": _FPS},
            artifact_kind="animation_3d",
        )


def _native_binary_dir() -> Path | None:
    key = runtime.native_platform_key()
    candidate = _PLUGIN_DIR / "native" / key
    manifest_path = candidate / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict) or manifest.get("schema_version") != _NATIVE_MANIFEST_SCHEMA_VERSION:
        return None
    if manifest.get("platform") != key or not isinstance(manifest.get("files"), list):
        return None
    for item in manifest["files"]:
        if not isinstance(item, dict):
            return None
        filename, digest = item.get("filename"), item.get("sha256")
        if not isinstance(filename, str) or Path(filename).name != filename:
            return None
        if not runtime.matches_sha256(candidate / filename, str(digest)):
            return None
    return candidate


def _weights_marker(cache_dir: Path) -> Path:
    return cache_dir / "weights" / ".complete"


def _weights_are_ready(cache_dir: Path) -> bool:
    return _weights_marker(cache_dir).is_file()


def _ensure_weights(context: MotionContext) -> tuple[Path, Path]:
    cache_dir = context.cache_dir
    weights_dir = cache_dir / "weights"
    motion_gguf = weights_dir / "motion.gguf"
    text_bundle_dir = weights_dir / _TEXT_BUNDLE_SUBDIR
    if _weights_are_ready(cache_dir):
        return motion_gguf, text_bundle_dir

    context.check_cancelled()
    hf_cache = cache_dir / "huggingface"
    if not motion_gguf.is_file():
        context.progress(0.05, "Downloading Kimodo SOMA motion checkpoint (~1.1GB, ungated)")
        _download_hf_file(context, hf_cache, _MOTION_REPO, _MOTION_FILENAME, motion_gguf, progress_range=(0.05, 0.2))
    context.check_cancelled()
    if not text_bundle_dir.is_dir() or not any(text_bundle_dir.glob("*.gguf")):
        context.progress(0.2, "Downloading Kimodo's native LLM2Vec text encoder (~15GB, subject to Meta's Llama 3 terms)")
        runtime.run(
            [sys.executable, str(_PLUGIN_DIR / "_kimodo_weights_download_driver.py"),
             _TEXT_REPO, _TEXT_BUNDLE_PATTERN, str(weights_dir)],
            "Downloading Kimodo text encoder", hf_cache_dir=hf_cache, progress=context.progress,
            progress_range=(0.2, 0.65), check_cancelled=context.check_cancelled,
        )
    context.check_cancelled()
    _weights_marker(cache_dir).parent.mkdir(parents=True, exist_ok=True)
    _weights_marker(cache_dir).write_text("ok", encoding="utf-8")
    return motion_gguf, text_bundle_dir


def _download_hf_file(
    context: MotionContext, hf_cache: Path, repo_id: str, filename: str, destination: Path,
    *, progress_range: tuple[float, float],
) -> None:
    runtime.run(
        [sys.executable, str(_PLUGIN_DIR / "_kimodo_weights_download_driver.py"), repo_id, filename, str(destination.parent)],
        "Downloading Kimodo motion checkpoint", hf_cache_dir=hf_cache, progress=context.progress,
        progress_range=progress_range, check_cancelled=context.check_cancelled,
    )
    downloaded = destination.parent / filename
    if downloaded.is_file() and downloaded != destination:
        destination.parent.mkdir(parents=True, exist_ok=True)
        downloaded.replace(destination)


def _glb_deps_marker(cache_dir: Path) -> Path:
    return cache_dir / "glb_deps_installed.json"


def _ensure_glb_dependencies(python: Path, shared: Path, context: MotionContext) -> None:
    marker = _glb_deps_marker(context.cache_dir)
    if marker.is_file():
        try:
            if int(json.loads(marker.read_text(encoding="utf-8")).get("runtime_version", 0)) == _GLB_RUNTIME_VERSION:
                return
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    context.check_cancelled()
    context.progress(0.67, "Installing GLB export dependencies (first use only)")
    runtime.pip_install(
        python, list(_GLB_DEPENDENCIES), "Installing GLB export dependencies",
        runtime=shared, progress=context.progress, progress_range=(0.67, 0.75), check_cancelled=context.check_cancelled,
    )
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"runtime_version": _GLB_RUNTIME_VERSION}), encoding="utf-8")


ENGINES = [KimodoPlugin]
