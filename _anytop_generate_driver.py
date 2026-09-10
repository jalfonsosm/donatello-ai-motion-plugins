"""Driver run inside the shared venv (as a subprocess, cwd = the AnyTop
checkout) to sample one AnyTop motion and write it straight to GLB.

This is a fork of AnyTop's own `sample/generate.py` (MIT-licensed, Copyright
(c) Anytop2025 contributors, https://github.com/Anytop2025/Anytop) trimmed to
a single object_type / single repetition, with everything downstream of the
sampled positions tensor replaced: the original script converts positions to
BVH rotations via a third-party `Motion` package
(`git+https://github.com/inbar-2344/Motion.git`) that carries no license of
its own. This project does not install or import that package; the
`positions -> local rotations -> GLB` step below is this project's own
implementation (`_shared_positions_to_rotations.py`,
`_shared_motion_recover.py`, `_shared_glb.py`).

Invoked with cwd set to the AnyTop checkout root, so `from utils...`,
`from model...`, `from data_loaders...` resolve exactly as upstream's own
`python -m sample.generate` does. Two extra flags this project adds
(`--shared-lib-parent`, `--output`) are stripped from argv before AnyTop's
own `generate_args()` parses it, since that parser errors on unknown flags.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _pop_flag(name: str) -> str:
    index = sys.argv.index(name)
    value = sys.argv[index + 1]
    del sys.argv[index:index + 2]
    return value


def main() -> None:
    shared_lib_parent = _pop_flag("--shared-lib-parent")
    output_path = _pop_flag("--output")
    sys.path.insert(0, shared_lib_parent)
    import _shared_glb as glb
    import _shared_motion_recover as motion_recover
    from _shared_positions_to_rotations import positions_to_local_euler_xyz

    import numpy as np
    import torch

    from utils.fixseed import fixseed
    from utils.parser_util import generate_args
    from utils.model_util import create_model_and_diffusion_general_skeleton, load_model
    from utils import dist_util
    from data_loaders.tensors import truebones_batch_collate
    from data_loaders.truebones.truebones_utils.get_opt import get_opt
    from model.conditioners import T5Conditioner

    args = generate_args()
    fixseed(args.seed)
    dist_util.setup_dist(args.device)
    device = dist_util.dev()
    opt = get_opt(device)

    cond_dict = np.load(args.cond_path, allow_pickle=True).item()
    object_type = args.object_type[0]
    n_frames = max(1, int(round(args.motion_length * opt.fps)))

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion_general_skeleton(args)
    print(f"Loading checkpoint from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location="cpu")
    load_model(model, state_dict)
    print(f"Loading T5 text encoder on {device}...")
    t5_conditioner = T5Conditioner(
        name=args.t5_name, finetune=False, word_dropout=0.0, normalize_text=False, device=str(device),
    )
    model.to(device)
    model.eval()

    parents = cond_dict[object_type]["parents"]
    n_joints = len(parents)
    mean = cond_dict[object_type]["mean"]
    std = cond_dict[object_type]["std"]
    tpos_first_frame = (cond_dict[object_type]["tpos_first_frame"] - mean) / (std + 1e-6)
    tpos_first_frame = np.nan_to_num(tpos_first_frame)
    joints_names = cond_dict[object_type]["joints_names"]
    names_tokens = t5_conditioner.tokenize(joints_names)
    joints_names_embs = t5_conditioner(names_tokens).detach().cpu().numpy()

    batch = [
        np.zeros((n_frames, n_joints, model.feature_len)), n_frames, parents, tpos_first_frame,
        cond_dict[object_type]["offsets"], _temporal_mask(args.temporal_window, n_frames),
        cond_dict[object_type]["joints_graph_dist"], cond_dict[object_type]["joint_relations"],
        object_type, joints_names_embs, 0, mean, std, model.max_joints,
    ]
    _, model_kwargs = truebones_batch_collate([batch])

    print("Sampling...")
    sample = diffusion.p_sample_loop(
        model, (1, model.max_joints, model.feature_len, n_frames), clip_denoised=False,
        model_kwargs=model_kwargs, skip_timesteps=0, init_image=None, progress=True,
        dump_steps=None, noise=None, const_noise=False,
    )
    motion = sample[0, :n_joints].cpu().permute(2, 0, 1).numpy() * std + mean  # (n_frames, n_joints, feature_len)

    checkout = Path.cwd()
    positions = motion_recover.recover_world_positions(motion, checkout)  # (n_frames, n_joints, 3)
    offsets = np.asarray(cond_dict[object_type]["offsets"], dtype=np.float64)
    local_euler, root_translation = positions_to_local_euler_xyz(positions, [int(p) for p in parents], offsets)

    glb.write_glb_from_euler_xyz(
        Path(output_path), list(joints_names), [int(p) for p in parents],
        [tuple(map(float, row)) for row in offsets],
        [[tuple(map(float, local_euler[f, j])) for j in range(n_joints)] for f in range(n_frames)],
        fps=float(opt.fps),
        root_translation=[tuple(map(float, row)) for row in root_translation],
    )
    print(f"Wrote {output_path}")


def _temporal_mask(window: int, max_len: int):
    """AnyTop's own `create_temporal_mask_for_window`
    (`data_loaders/truebones/data/dataset.py`, MIT) -- copied here directly
    (rather than imported) because importing that module also imports the
    unlicensed `Motion` package transitively at module scope."""
    import torch
    margin = window // 2
    mask = torch.zeros(max_len + 1, max_len + 1)
    mask[:, 0] = 1
    for i in range(max_len + 1):
        mask[i, max(0, i - margin):min(max_len + 1, i + margin + 2)] = 1
    return mask


if __name__ == "__main__":
    main()
