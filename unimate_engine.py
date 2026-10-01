"""UniMate adapter for Donatello's generic AI Motion API.

UniMate (https://github.com/Friedrich-M/UniMate, MIT; SIGGRAPH Asia 2026) is
a text-conditioned flow-matching model that animates *a given skeleton* of
any topology -- quadrupeds, birds, insects, snakes, humanoids, rigged
objects. Unlike every other engine here it does not bring its own rig: the
host hands it the session character's skeleton (``request.target_rig``,
declared by ``rig_input = "session_character"``) and the result is a GLB on
those very bones, so it appends to the character without a mapping review.

Generation runs in the shared venv through ``_unimate_generate_driver.py``,
which imports only UniMate's MIT model/sampler/text-encoder code. UniMate's
own dataset loader and preprocessing import a third-party ``Motion`` package
with no license; this project does not install or import it --
``_unimate_skeleton.py`` reimplements the rig canonicalization, topology
features and feature decoding inference needs.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

from flatrig_private.local_ai.motion_plugin_api import (
    MotionContext,
    MotionPlugin,
    MotionRequest,
    MotionResult,
)

_PLUGIN_DIR = str(Path(__file__).resolve().parent)
if _PLUGIN_DIR not in sys.path:
    sys.path.insert(0, _PLUGIN_DIR)

import _shared_runtime as runtime

_UNIMATE_REPOSITORY = "https://github.com/Friedrich-M/UniMate.git"
_UNIMATE_REVISION = "2c5b384715aa63d8639b1ed7eb74bfe614570c7a"
_WEIGHTS_REPO = "Linzhan/UniMate"
_WEIGHTS_REVISION = "92710b30abc0a7708c9f280f38fd7e65448c4f95"
_RUN = "unimate_uniml3d_f60_v2"
_CHECKPOINT = f"{_RUN}/checkpoints/checkpoint_step_100000.pt"
_TEXT_ENCODER_REPO = "google/flan-t5-base"
_TEXT_ENCODER_FILES = ("config.json", "spiece.model", "tokenizer.json", "tokenizer_config.json",
                       "special_tokens_map.json", "model.safetensors")
_RUNTIME_VERSION = 2

# What UniMate's own requirements.txt lists for inference, minus everything
# only its data pipeline, training or renderers use (bpy, CLIP, spaCy,
# accelerate, tyro, the unlicensed Motion package). torch/numpy stay
# unpinned so a venv another engine already populated is reused as-is.
_DEPENDENCIES = (
    # torch is installed separately via ensure_torch (CUDA/ROCm/CPU index);
    # listing it here would let a bare PyPI install shadow the accelerator build.
    "numpy", "scipy", "einops>=0.8", "torch_geometric>=2.6", "torchdiffeq>=0.2.5",
    "transformers>=4.57", "sentencepiece", "rich", "pygltflib>=1.16",
)

_FPS = 30.0            # UniMate's feature scale is tied to 30 fps
_SEGMENT_FRAMES = 60   # one generation window
_OVERLAP = 10          # frames shared by consecutive segments

_STATS = {
    "truebones": "Creature or animal",
    "mixamo": "Humanoid",
    "objaverse": "Object or anything else",
}


class UniMatePlugin(MotionPlugin):
    plugin_id = "unimate"
    label = "UniMate"
    description = (
        "Animates the character in your session from a text prompt -- creatures, birds, insects, "
        "snakes, humanoids or rigged objects. Its isolated runtime installs on first use."
    )
    version = "0.1.0"
    license_id = "MIT"
    license_url = "https://github.com/Friedrich-M/UniMate"
    requires_consent = True
    experimental = True
    setup_on_generate = True
    requires_prompt = True
    rig_input = "session_character"

    def form_schema(self) -> list[dict[str, Any]]:
        return [
            {"id": "body", "label": "The character is", "type": "select", "default": "truebones",
             "options": [{"value": key, "label": label} for key, label in _STATS.items()]},
            {"id": "guidance", "label": "Follow the prompt (1 = loosely, 6 = strictly)", "type": "number",
             "default": 3.0, "min": 1.0, "max": 6.0, "step": 0.5},
        ]

    def is_installed(self, cache_root: Path) -> bool:
        return (_cache_dir_from_root(cache_root) / "installed.json").is_file()

    def generate(self, request: MotionRequest, context: MotionContext) -> MotionResult:
        context.check_cancelled()
        if not request.target_rig:
            raise ValueError("Load a character in the 3D tab first: UniMate animates the session's character.")
        shared = runtime.shared_runtime_root(context.cache_dir)
        python = runtime.ensure_venv(shared, context.progress, context.check_cancelled, progress_range=(0.0, 0.03))
        _ensure_dependencies(python, shared, context)
        checkout = _ensure_checkout(context)
        run_dir = _ensure_weights(context)

        rig_path = context.output_dir / "target_rig.json"
        rig_path.write_text(json.dumps(request.target_rig), encoding="utf-8")
        prompts = _segment_prompts(request.prompt, request.duration_seconds)
        glb_path = context.output_dir / "unimate_motion.glb"
        report_path = context.output_dir / "unimate_report.json"
        repo_root = Path(__file__).resolve().parent
        context.progress(0.6, f"Animating {request.target_rig.get('source_name') or 'the character'}")
        runtime.run(
            [
                str(python), str(repo_root / "_unimate_generate_driver.py"),
                "--shared-lib-parent", str(repo_root),
                "--exp-dir", str(run_dir),
                "--checkpoint", str(run_dir.parent / _CHECKPOINT),
                "--rig", str(rig_path),
                "--prompts", json.dumps(prompts),
                "--stats", str(request.extra.get("body") or "truebones"),
                "--cfg-scale", str(float(request.extra.get("guidance") or 3.0)),
                "--overlap", str(_OVERLAP),
                "--seed", str(int(request.seed)),
                "--fps", str(_FPS),
                "--output", str(glb_path),
                "--report", str(report_path),
            ],
            "UniMate sampling", runtime=shared, hf_cache_dir=_hf_home(context),
            progress=context.progress,
            progress_range=(0.6, 1.0), check_cancelled=context.check_cancelled,
            extra_environment={
                "PYTHONPATH": str(checkout), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            },
            cwd=checkout,
        )
        if not glb_path.is_file():
            raise RuntimeError("UniMate did not produce a GLB output.")
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
        context.progress(1.0, "UniMate GLB exported")
        return MotionResult(
            glb_path,
            {
                "fps": _FPS,
                "segments": len(prompts),
                "frames": report.get("frames"),
                "target_rig": request.target_rig.get("source_name"),
                "same_skeleton_as_target": True,
                "joint_labels": report.get("joint_labels"),
                "joints_following_parent": report.get("joints_following_parent"),
                "positions_cross_check": report.get("positions_cross_check"),
            },
            artifact_kind="animation_3d",
        )


def _segment_prompts(prompt: str, duration_seconds: float) -> list[str]:
    """One prompt per 2-second window, chained in order.

    Several lines are several actions ("walks forward", "stops", "roars");
    the duration decides how many windows there are, the last action
    repeating when there are more windows than lines.
    """
    actions = [line.strip() for line in str(prompt or "").splitlines() if line.strip()]
    if not actions:
        raise ValueError("Describe the motion.")
    frames = max(1.0, float(duration_seconds)) * _FPS
    extra = max(0.0, frames - _SEGMENT_FRAMES)
    count = max(len(actions), 1 + math.ceil(extra / (_SEGMENT_FRAMES - _OVERLAP)))
    count = min(count, 12)
    return [actions[min(index, len(actions) - 1)] for index in range(count)]


def _cache_dir_from_root(cache_root: Path) -> Path:
    from flatrig_private.local_ai.motion_plugins import plugin_cache_directory
    return plugin_cache_directory(cache_root, UniMatePlugin.plugin_id)


def _ensure_dependencies(python: Path, shared: Path, context: MotionContext) -> None:
    marker = context.cache_dir / "installed.json"
    if marker.is_file():
        try:
            if int(json.loads(marker.read_text(encoding="utf-8")).get("runtime_version", 0)) == _RUNTIME_VERSION:
                return
        except (OSError, ValueError):
            pass
    context.check_cancelled()
    flavor = runtime.ensure_torch(
        python, shared, context.progress, context.check_cancelled, progress_range=(0.03, 0.12),
    )
    context.progress(0.12, f"Installing UniMate's Python dependencies (torch={flavor})")
    runtime.pip_install(
        python, list(_DEPENDENCIES), "Installing UniMate dependencies",
        runtime=shared, progress=context.progress, progress_range=(0.12, 0.2),
        check_cancelled=context.check_cancelled,
    )
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"runtime_version": _RUNTIME_VERSION, "torch_flavor": flavor}), encoding="utf-8")


def _ensure_checkout(context: MotionContext) -> Path:
    checkout = context.cache_dir / "source" / "unimate"
    if not (checkout / ".git").is_dir():
        context.check_cancelled()
        context.progress(0.2, "Downloading UniMate source")
        checkout.parent.mkdir(parents=True, exist_ok=True)
        runtime.run(
            ["git", "clone", _UNIMATE_REPOSITORY, str(checkout)],
            "Downloading UniMate source", progress=context.progress,
            progress_range=(0.2, 0.25), check_cancelled=context.check_cancelled,
        )
    runtime.run(
        ["git", "-C", str(checkout), "checkout", "--detach", _UNIMATE_REVISION],
        "Pinning UniMate source", progress=context.progress,
        progress_range=(0.25, 0.26), check_cancelled=context.check_cancelled,
    )
    return checkout


def _hf_home(context: MotionContext) -> Path:
    """This engine's own Hugging Face cache, so the text encoder is removed
    together with the engine's other downloads."""
    return context.cache_dir / "huggingface"


def _ensure_weights(context: MotionContext) -> Path:
    """The checkpoint run folder; also caches the text encoder.

    UniMate's T5 wrapper only accepts the Hub id (``google/flan-t5-base``),
    so the encoder goes into a Hub-layout cache the sampler reads offline.
    """
    weights_dir = context.cache_dir / "weights"
    run_dir = weights_dir / _RUN
    marker = weights_dir / ".complete"
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == _WEIGHTS_REVISION:
        return run_dir
    context.check_cancelled()
    from huggingface_hub import hf_hub_download, snapshot_download
    context.progress(0.27, "Downloading the UniMate checkpoint (~1.2 GB, first use only)")
    for filename in (f"{_RUN}/config.json", f"{_RUN}/dataset_stats.npy", _CHECKPOINT):
        context.check_cancelled()
        hf_hub_download(_WEIGHTS_REPO, filename, revision=_WEIGHTS_REVISION, local_dir=str(weights_dir))
    context.progress(0.5, "Downloading the text encoder (flan-t5-base, ~1 GB, first use only)")
    snapshot_download(
        _TEXT_ENCODER_REPO, allow_patterns=list(_TEXT_ENCODER_FILES),
        cache_dir=str(_hf_home(context) / "hub"),
    )
    marker.write_text(_WEIGHTS_REVISION, encoding="utf-8")
    return run_dir


ENGINES = [UniMatePlugin]
