"""Driver run inside the shared venv to sample one UniMate motion for one rig.

Invoked with ``PYTHONPATH`` pointing at a pinned UniMate checkout
(https://github.com/Friedrich-M/UniMate, MIT). It imports only UniMate's
model, flow sampler, text encoder, config and batch collation -- none of
which touch the unlicensed ``Motion`` package (``Animation`` /
``Quaternions``) that UniMate's dataset loader and data pipeline import at
module scope. Everything that would have gone through those modules (rig
canonicalization, topology features, feature decoding) is this project's
own ``_unimate_skeleton.py``.

Writes the GLB plus a small JSON report (joint labels, dropped joints, the
positions cross-check) next to it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-lib-parent", required=True)
    parser.add_argument("--exp-dir", required=True, help="Checkpoint run folder (config.json, dataset_stats.npy).")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rig", required=True, help="Host target_rig descriptor (JSON).")
    parser.add_argument("--prompts", required=True, help="JSON list of prompts, one per segment.")
    parser.add_argument("--stats", default="truebones", help="Which dataset's normalization statistics to use.")
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--overlap", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50, help="Fixed Euler steps per segment.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--dump-features", default="", help="Also save the raw (T, J, 12) features (.npy).")
    return parser.parse_args()


def _device(name: str):
    import torch
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _configure_accelerator(device) -> None:
    """Turn on the cheap CUDA/HIP speed knobs UniMate's sampling benefits from."""
    import torch
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def _autocast_context(device):
    """fp16 on NVIDIA CUDA, bf16 on HIP (same rule as FlatRig's image path)."""
    import contextlib
    import torch
    if device.type != "cuda":
        return contextlib.nullcontext()
    hip = bool(getattr(getattr(torch, "version", None), "hip", None))
    dtype = torch.bfloat16 if hip and hasattr(torch, "bfloat16") else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def main() -> None:
    args = _parse_args()
    sys.path.insert(0, args.shared_lib_parent)
    import numpy as np
    import torch

    import _shared_glb as glb
    import _unimate_skeleton as skeleton

    from unimate.configs.schema import MainConfig
    from unimate.dataset.mixture.collate import mixture_batch_collate
    from unimate.inference.generate import ClassifierFreeSampleModel
    from unimate.inference.motion_inbetweening import inbetween_sample_ode
    from unimate.models.factory import create_model, create_transport
    from unimate.models.text_encoder.factory import create_text_encoder
    from unimate.training.ema import EMAModel
    from unimate.utils.text_emb_cache import pool, sequences_from_hidden

    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2 ** 32))
    device = _device(args.device)
    _configure_accelerator(device)
    exp_dir = Path(args.exp_dir)
    config = MainConfig.from_json(str(exp_dir / "config.json"))
    max_joints = int(config.dataset.max_joints)
    max_frames = int(config.dataset.max_motion_length)

    rig = skeleton.prepare_rig(json.loads(Path(args.rig).read_text(encoding="utf-8")), max_joints)
    prompts = [str(p) for p in json.loads(args.prompts) if str(p).strip()]
    if not prompts:
        raise SystemExit("No prompt given.")
    print(f"Rig: {len(rig.names)} joints used ({len(rig.dropped)} leaf joints follow their parent).")

    # The host turns any "NN%" in this output into its progress bar; the
    # library's own loading bars would read as "100%" before sampling starts.
    from transformers.utils import logging as transformers_logging
    transformers_logging.disable_progress_bar()
    print(f"Loading the text encoder ({config.model.text_encoder_version})...")
    encoder = create_text_encoder(
        encoder_type=config.model.text_encoder_type,
        encoder_version=config.model.text_encoder_version,
        device=str(device), pool=False,
    )

    def encode(texts):
        with torch.no_grad():
            inputs = encoder.tokenize(list(texts))
            hidden = encoder(inputs)
        return sequences_from_hidden(hidden.detach().cpu(), inputs["attention_mask"].cpu())

    joint_names_emb = np.stack([pool(tokens) for tokens in encode(rig.clean_names)])
    captions = [{"caption_emb": pool(tokens), "caption_tokens": tokens} for tokens in encode(prompts)]
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    stats = np.load(exp_dir / "dataset_stats.npy", allow_pickle=True).item()
    if args.stats not in stats:
        raise SystemExit(f"Unknown statistics '{args.stats}'; available: {sorted(stats)}")
    stat = stats[args.stats]
    count = len(rig.names)
    mean = np.zeros((count, 12))
    std = np.ones((count, 12))
    mean[0], std[0] = stat["mean_root"], stat["std_root"]
    mean[1:], std[1:] = stat["mean_local"], stat["std_local"]

    tpose = (skeleton.tpose_features(rig.tpose) - mean) / std
    parent_feats = tpose.copy()
    for joint, parent in enumerate(rig.parents):
        if parent >= 0:
            parent_feats[joint] = tpose[parent]
    topology = skeleton.topology_condition(rig.parents)

    def condition(caption):
        sample = {
            "motion": np.zeros((max_frames, count, 12)),
            "max_motion_length": max_frames,
            "motion_length": max_frames,
            "max_joints": count,
            "parents": np.asarray(rig.parents),
            "edge_indexs": topology["edge_indexs"],
            "tpos_first_frame": tpose,
            "tpos_first_frame_parents": parent_feats,
            "offsets": rig.tpose - np.array([rig.tpose[p] if p >= 0 else np.zeros(3) for p in rig.parents]),
            "joint_graph_dist": topology["joint_graph_dist"],
            "joint_relations": topology["joint_relations"],
            "joint_depths": topology["joint_depths"],
            "spectral_feats": topology["spectral_feats"],
            "joint_names_emb": joint_names_emb,
            "object_type": "donatello",
            "start_idx": 0,
            "mean": mean,
            "std": std,
            **caption,
        }
        _, cond = mixture_batch_collate([sample])
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in cond.items()}

    print("Loading the UniMate checkpoint...")
    model = create_model(dataset_config=config.dataset, model_config=config.model)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("model_state_dict", state))
    if config.training.use_ema and "ema_state_dict" in state:
        ema = EMAModel(parameters=model.parameters(), decay=config.training.ema_decay, use_ema_warmup=True)
        ema.load_state_dict(state["ema_state_dict"])
        ema.copy_to(model.parameters())
    del state
    model.to(device).eval()
    # Run on the rig's own joint count instead of the checkpoint's padding
    # (71). Padded joints are masked out of every attention, so the output is
    # the same (checked: max difference 7e-6 on one denoiser call) while the
    # joint-axis attention shrinks quadratically -- about 14x faster on a
    # 22-joint quadruped. Only these two attributes carry the padded size.
    for module in model.modules():
        if hasattr(module, "max_joints"):
            module.max_joints = count
        if type(module).__name__ == "FinalLayer":
            module.joint = count
    transport = create_transport(training_config=config.training)
    shape = (1, count, 12, max_frames)
    overlap = max(1, min(int(args.overlap), max_frames - 1))

    class Progress(torch.nn.Module):
        """Prints a percentage per network call; the host turns it into its bar."""

        def __init__(self, inner, total):
            super().__init__()
            self.inner, self.total, self.calls = inner, max(1, total), 0
            self.cond_mask_prob = getattr(inner, "cond_mask_prob", 0.0)

        def forward(self, x, timesteps, cond=None, **kwargs):
            out = self.inner(x, timesteps, cond, **kwargs)
            self.calls += 1
            print(f"Sampling {min(100.0, 100.0 * self.calls / self.total):.0f}%", flush=True)
            return out

    guided = ClassifierFreeSampleModel(model, cfg_scale=args.cfg_scale) if args.cfg_scale > 1.0 else model
    tracked = Progress(guided, args.steps * len(captions))

    # Every segment goes through UniMate's fixed-step replacement sampler:
    # the first with nothing pinned (plain Euler on the flow ODE), each later
    # one with its first ``overlap`` frames pinned to the previous tail --
    # UniMate's own motion-expansion scheme. Euler keeps the cost predictable
    # (steps x 2 network calls with guidance); the adaptive dopri5 default
    # took over 25 minutes on a laptop CPU.
    print(f"Sampling {len(captions)} segment(s) on {device.type}...", flush=True)
    segments = []
    with torch.no_grad(), _autocast_context(device):
        previous = None
        for caption in captions:
            known = torch.zeros(shape, device=device)
            keep = torch.zeros((1, 1, 1, max_frames), dtype=torch.bool, device=device)
            if previous is not None:
                known[..., :overlap] = previous[..., -overlap:]
                keep[..., :overlap] = True
            previous = inbetween_sample_ode(
                sample_model=tracked, transport=transport, cond=condition(caption),
                x1_known=known, keep_mask=keep, motion_shape=shape,
                num_steps=args.steps, device=device,
            )
            segments.append(previous if not segments else previous[..., overlap:])
    sample = torch.cat(segments, dim=-1).float()
    features = sample[0, :count].cpu().permute(2, 0, 1).numpy() * std[None] + mean[None]
    if args.dump_features:
        np.save(args.dump_features, features)

    decoded = skeleton.decode_features(features, rig.parents, rig.tpose)
    mismatch = float(np.abs(decoded.positions - decoded.ric_positions).max())
    print(f"Positions cross-check (rotations vs position channels): max {mismatch:.4f} body diameters / 2.")
    animation = skeleton.gltf_animation(rig, decoded)
    glb.write_glb_from_quaternions(
        Path(args.output), animation.names, animation.parents, animation.offsets,
        animation.quaternions_xyzw, args.fps, root_translation=animation.root_translation,
    )
    Path(args.report).write_text(json.dumps({
        "frames": int(features.shape[0]),
        "fps": args.fps,
        "joints_used": rig.names,
        "joint_labels": dict(zip(rig.names, rig.clean_names)),
        "joints_following_parent": rig.dropped,
        "positions_cross_check": mismatch,
        "device": device.type,
    }, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
