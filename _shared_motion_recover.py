"""Recover world-space joint positions from AnyTop's root-relative-with-
root-velocity ("BVH RIC") diffusion output feature.

This is this project's own from-scratch numpy/SciPy reimplementation of the
root-recovery step used across HumanML3D-lineage motion representations
(root channel = [absolute 6D heading, XZ linear velocity, height, ...],
integrated to a world path via cumulative sum) -- it does not import or
derive from any third-party IK/BVH library. `rotation_6d_to_matrix_np` is
imported directly from AnyTop's own (MIT-licensed) `utils/rotation_conversions`
module at call time, since that conversion is AnyTop's own utility code, not
part of the unlicensed dependency this project avoids.

The root's per-frame world rotation and XZ velocity direction convention was
not independently re-derived from a spec; `_ROOT_VELOCITY_IN_LOCAL_FRAME`
below is written so it can be flipped and empirically re-checked against a
real generated sample (a walking/locomoting sample should produce a smooth,
bounded root path) if the recovered path looks physically implausible.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# Root feature velocity is stored in the root's own current-heading frame
# (its "local"/body frame); recovering world position means rotating that
# velocity INTO world space before integrating. `True` uses `r_rot.apply(...)`
# (local -> world); `False` uses `r_rot.inv().apply(...)`. See module
# docstring -- calibrate this against one real sample if root motion looks
# wrong (e.g. sliding backwards, or not translating at all for a "walks
# forward" style output).
_ROOT_VELOCITY_IN_LOCAL_FRAME = True

_rotation_6d_to_matrix_np = None


def _load_rotation_6d_to_matrix_np(anytop_checkout: Path):
    global _rotation_6d_to_matrix_np
    if _rotation_6d_to_matrix_np is not None:
        return _rotation_6d_to_matrix_np
    module_path = Path(anytop_checkout) / "utils" / "rotation_conversions.py"
    spec = importlib.util.spec_from_file_location("_anytop_rotation_conversions", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    _rotation_6d_to_matrix_np = module.rotation_6d_to_matrix_np
    return _rotation_6d_to_matrix_np


def recover_root_rotation_and_position(root_features: np.ndarray, anytop_checkout: Path) -> tuple[Rotation, np.ndarray]:
    """`root_features`: (n_frames, 13) -- the root joint's own feature
    channel of AnyTop's sampled motion tensor. Layout (AnyTop's own,
    matching its `data_loaders.truebones` root feature convention):
    [0]=height-delta placeholder (unused here), [3:9]=6D absolute root
    heading, [9]=local X velocity, [11]=local Z velocity, [1]=world height.
    """
    rotation_6d_to_matrix_np = _load_rotation_6d_to_matrix_np(anytop_checkout)
    root_rotation = Rotation.from_matrix(rotation_6d_to_matrix_np(root_features[:, 3:9]))

    n_frames = root_features.shape[0]
    local_velocity = np.zeros((n_frames, 3))
    local_velocity[1:, 0] = root_features[:-1, 9]
    local_velocity[1:, 2] = root_features[:-1, 11]
    world_velocity = (
        root_rotation.apply(local_velocity) if _ROOT_VELOCITY_IN_LOCAL_FRAME
        else root_rotation.inv().apply(local_velocity)
    )
    world_position = np.cumsum(world_velocity, axis=0)
    world_position[:, 1] = root_features[:, 1]
    return root_rotation, world_position


def recover_world_positions(motion: np.ndarray, anytop_checkout: Path) -> np.ndarray:
    """`motion`: (n_frames, n_joints, 13), AnyTop's de-normalized sampled
    motion tensor (root joint first). Returns (n_frames, n_joints, 3) world
    positions."""
    root_rotation, root_world_position = recover_root_rotation_and_position(motion[:, 0, :], anytop_checkout)
    local_positions = motion[:, 1:, :3]
    n_frames, n_other_joints, _ = local_positions.shape
    flat = local_positions.reshape(-1, 3)
    # Repeat each frame's root rotation once per joint in that frame, so a
    # batched `Rotation.apply` rotates every joint by its own frame's root
    # orientation (matches the original per-frame-broadcast semantics).
    per_vector_rotation = root_rotation[np.repeat(np.arange(n_frames), n_other_joints)]
    rotated_flat = (
        per_vector_rotation.apply(flat) if _ROOT_VELOCITY_IN_LOCAL_FRAME
        else per_vector_rotation.inv().apply(flat)
    )
    rotated = rotated_flat.reshape(n_frames, n_other_joints, 3)
    rotated[:, :, 0] += root_world_position[:, 0:1]
    rotated[:, :, 2] += root_world_position[:, 2:3]
    return np.concatenate([root_world_position[:, None, :], rotated], axis=1)
