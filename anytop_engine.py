"""AnyTop adapter for Donatello's generic AI Motion API.

AnyTop (https://github.com/Anytop2025/Anytop, MIT) is topology-conditioned,
not text-conditioned: its text encoder only embeds each skeleton's own joint
*names* for cross-topology alignment, there is no free-text prompt control at
inference time. `requires_prompt = False` on this class tells the host that;
the actual "what to generate" control is the `object_type` selector below,
covering AnyTop's pretrained Truebones Zoo skeleton categories (mostly
non-human creatures -- birds, insects, reptiles, quadrupeds -- which is a
useful complement to Kimodo's human/SOMA-only text-to-motion).

This adapter does not install or import the `Motion` package AnyTop's own
`sample/generate.py` depends on for its positions -> BVH-rotations step
(https://github.com/inbar-2344/Motion, no license file at all, forked from
Daniel Holden's research code). `_anytop_generate_driver.py` in this project
reimplements that one step from scratch (see `_shared_positions_to_rotations.py`,
`_shared_motion_recover.py`) and is what actually runs at generation time.
"""

from __future__ import annotations

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

_ANYTOP_REPOSITORY = "https://github.com/Anytop2025/Anytop.git"
_ANYTOP_REVISION = "e780d1575ca0121f29bb53821b309cf564156a95"
_ANYTOP_WEIGHTS_REPO = "inbar2344/AnyTop"
_RUNTIME_VERSION = 2  # bumped: +pygltflib

# AnyTop's own dependency list (environment.yaml) trimmed to what generation
# actually needs: no moviepy/imageio-ffmpeg (this adapter never renders the
# upstream debug .mp4), no pymel (Maya-only visualization script, unused),
# no nvidia-*-cu12 packages (CUDA-only; irrelevant off-CUDA and simply
# unavailable as wheels on macOS/CPU). `torch`/`transformers`/`scipy`/`numpy`
# are intentionally left unpinned here -- they are already installed (and
# pinned) by whichever engine's `generate()` ran first in this shared venv;
# re-requesting them unpinned lets pip verify what's already there instead of
# fighting kimodo's own pins.
_BINARY_DEPENDENCIES = (
    "torch", "transformers", "numpy", "scipy", "spacy", "num2words", "pygltflib>=1.16",
)

#: object_type -> AnyTop checkpoint subset folder prefix. "millipeds_snakes"
#: is one combined checkpoint covering both the MILLIPEDS and SNAKES
#: Truebones categories -- there is no separate "snakes" checkpoint upstream.
_OBJECT_TYPES: dict[str, str] = {
    # Flying
    "Bat": "flying", "Dragon": "flying", "Bird": "flying", "Buzzard": "flying",
    "Eagle": "flying", "Giantbee": "flying", "Parrot": "flying", "Parrot2": "flying",
    "Pigeon": "flying", "Pteranodon": "flying", "Tukan": "flying",
    # Bipeds (non-human bipedal creatures)
    "Ostrich": "bipeds", "Flamingo": "bipeds", "Raptor": "bipeds", "Raptor2": "bipeds",
    "Raptor3": "bipeds", "Trex": "bipeds", "Chicken": "bipeds", "Tyranno": "bipeds",
    # Milliped/insect/arachnid
    "Cricket": "millipeds_snakes", "SpiderG": "millipeds_snakes", "Scorpion": "millipeds_snakes",
    "Isopetra": "millipeds_snakes", "FireAnt": "millipeds_snakes", "Crab": "millipeds_snakes",
    "Centipede": "millipeds_snakes", "Roach": "millipeds_snakes", "Ant": "millipeds_snakes",
    "HermitCrab": "millipeds_snakes", "Scorpion-2": "millipeds_snakes", "Spider": "millipeds_snakes",
    # Snakes
    "Anaconda": "millipeds_snakes", "KingCobra": "millipeds_snakes",
    # Quadrupeds
    "Cat": "quadropeds", "Hippopotamus": "quadropeds", "Comodoa": "quadropeds", "Camel": "quadropeds",
    "Bear": "quadropeds", "Buffalo": "quadropeds", "BrownBear": "quadropeds", "Coyote": "quadropeds",
    "Crocodile": "quadropeds", "Elephant": "quadropeds", "Deer": "quadropeds", "Fox": "quadropeds",
    "Gazelle": "quadropeds", "Goat": "quadropeds", "Jaguar": "quadropeds", "Lynx": "quadropeds",
    "Tricera": "quadropeds", "Stego": "quadropeds", "SandMouse": "quadropeds", "Raindeer": "quadropeds",
    "Puppy": "quadropeds", "PolarBear": "quadropeds", "Monkey": "quadropeds", "Mammoth": "quadropeds",
    "Alligator": "quadropeds", "Hamster": "quadropeds", "Hound": "quadropeds", "Leapord": "quadropeds",
    "Lion": "quadropeds", "PolarBearB": "quadropeds", "Rat": "quadropeds", "Rhino": "quadropeds",
    "SabreToothTiger": "quadropeds", "Skunk": "quadropeds", "Turtle": "quadropeds",
}

_CHECKPOINT_DIR_BY_SUBSET = {
    "flying": "flying_model_dataset_truebones_bs_16_latentdim_128/model000229999.pt",
    "bipeds": "bipeds_model_dataset_truebones_bs_16_latentdim_128/model000329999.pt",
    "millipeds_snakes": "millipeds_snakes_model_dataset_truebones_bs_16_latentdim_128/model000349999.pt",
    "quadropeds": "quadropeds_model_dataset_truebones_bs_16_latentdim_128/model000189999.pt",
}


class AnyTopPlugin(MotionPlugin):
    plugin_id = "anytop"
    label = "AnyTop"
    description = (
        "Motion for non-human creatures (birds, insects, reptiles, quadrupeds), by skeleton "
        "category rather than text -- its isolated runtime installs automatically on first use."
    )
    version = "0.1.0"
    license_id = "MIT"
    license_url = "https://github.com/Anytop2025/Anytop"
    requires_consent = True
    experimental = True
    setup_on_generate = True
    requires_prompt = False

    def form_schema(self) -> list[dict[str, Any]]:
        options = [
            {"value": name, "label": f"{name} ({subset.replace('_', ' + ')})"}
            for name, subset in sorted(_OBJECT_TYPES.items())
        ]
        return [{"id": "object_type", "label": "Creature", "type": "select", "default": "Dragon", "options": options}]

    def is_installed(self, cache_root: Path) -> bool:
        marker = _cache_dir_from_root(cache_root) / "installed.json"
        return marker.is_file()

    def generate(self, request: MotionRequest, context: MotionContext) -> MotionResult:
        context.check_cancelled()
        shared = runtime.shared_runtime_root(context.cache_dir)
        python = runtime.ensure_venv(shared, context.progress, context.check_cancelled, progress_range=(0.0, 0.03))
        _ensure_dependencies(python, shared, context)
        checkout = _ensure_checkout(context)
        weights_dir = _ensure_weights(context)

        object_type = str(request.extra["object_type"])
        subset = _OBJECT_TYPES.get(object_type)
        if subset is None:
            raise ValueError(f"Unknown AnyTop creature '{object_type}'.")
        model_path = weights_dir / "checkpoints" / _CHECKPOINT_DIR_BY_SUBSET[subset]
        cond_path = weights_dir / "dataset" / "truebones" / "zoo" / "truebones_processed" / "cond.npy"
        if not model_path.is_file() or not cond_path.is_file():
            raise RuntimeError("AnyTop checkpoint download is incomplete.")

        repo_root = Path(__file__).resolve().parent
        # GLB, not FBX: the retarget pipeline's 3D-source-export step reads
        # the source through Blender, and Blender's FBX importer only
        # supports *binary* FBX, which this project's pure-Python exporter
        # does not produce. BVH is not an accepted *source* format for that
        # step either (only an accepted output) -- GLB is.
        glb_path = context.output_dir / "anytop_motion.glb"
        context.progress(0.65, f"Sampling AnyTop motion for {object_type}")
        runtime.run(
            [
                str(python), str(repo_root / "_anytop_generate_driver.py"),
                "--shared-lib-parent", str(repo_root),
                "--output", str(glb_path),
                "--model_path", str(model_path),
                "--object_type", object_type,
                "--cond_path", str(cond_path),
                "--seed", str(int(request.seed)),
                "--motion_length", str(max(0.5, float(request.duration_seconds))),
                "--device", "0",
            ],
            "AnyTop sampling", runtime=shared, progress=context.progress,
            progress_range=(0.65, 1.0), check_cancelled=context.check_cancelled,
            extra_environment={"PYTHONPATH": str(checkout)}, cwd=checkout,
        )
        if not glb_path.is_file():
            raise RuntimeError("AnyTop did not produce a GLB output.")
        context.progress(1.0, "AnyTop GLB exported")
        return MotionResult(
            glb_path,
            {"object_type": object_type, "subset": subset, "fps": 20.0},
            artifact_kind="animation_3d",
        )


def _cache_dir_from_root(cache_root: Path) -> Path:
    from flatrig_private.local_ai.motion_plugins import plugin_cache_directory
    return plugin_cache_directory(cache_root, AnyTopPlugin.plugin_id)


def _ensure_dependencies(python: Path, shared: Path, context: MotionContext) -> None:
    marker = context.cache_dir / "installed.json"
    if marker.is_file():
        try:
            import json
            if int(json.loads(marker.read_text(encoding="utf-8")).get("runtime_version", 0)) == _RUNTIME_VERSION:
                return
        except (OSError, ValueError):
            pass
    context.check_cancelled()
    context.progress(0.03, "Installing AnyTop's Python dependencies (first use only)")
    runtime.pip_install(
        python, list(_BINARY_DEPENDENCIES), "Installing AnyTop dependencies",
        runtime=shared, progress=context.progress, progress_range=(0.03, 0.15), check_cancelled=context.check_cancelled,
    )
    import json
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"runtime_version": _RUNTIME_VERSION}), encoding="utf-8")


def _ensure_checkout(context: MotionContext) -> Path:
    checkout = context.cache_dir / "source" / "anytop"
    if not (checkout / ".git").is_dir():
        context.check_cancelled()
        context.progress(0.15, "Downloading AnyTop source")
        checkout.parent.mkdir(parents=True, exist_ok=True)
        runtime.run(
            ["git", "clone", _ANYTOP_REPOSITORY, str(checkout)],
            "Downloading AnyTop source", progress=context.progress,
            progress_range=(0.15, 0.25), check_cancelled=context.check_cancelled,
        )
    runtime.run(
        ["git", "-C", str(checkout), "checkout", "--detach", _ANYTOP_REVISION],
        "Pinning AnyTop source", progress=context.progress,
        progress_range=(0.25, 0.26), check_cancelled=context.check_cancelled,
    )
    return checkout


def _ensure_weights(context: MotionContext) -> Path:
    weights_dir = context.cache_dir / "weights"
    marker = weights_dir / ".complete"
    if marker.is_file():
        return weights_dir
    context.check_cancelled()
    context.progress(0.30, "Downloading AnyTop pretrained checkpoints (~120MB, ungated)")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=_ANYTOP_WEIGHTS_REPO, local_dir=str(weights_dir))
    marker.write_text("ok", encoding="utf-8")
    return weights_dir
