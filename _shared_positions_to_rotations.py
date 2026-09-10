"""Recover per-joint local rotations from a sequence of world-space joint
positions, given a known rest hierarchy (parents + rest-pose bone offsets).

This is an original implementation written for this project -- it does not
use, wrap, or derive from any third-party inverse-kinematics/BVH library.
Some upstream motion-generation research code (e.g. models trained to output
joint *positions* rather than rotations) ships with a bundled IK-fitting
helper of unclear or absent licensing; rather than depend on that, this module
solves the same well-known problem (recovering local joint rotations from
positions plus a known rest pose) from first principles, using only SciPy's
`Rotation.align_vectors` (Kabsch/SVD optimal-rotation fit) applied per joint.

Algorithm, per frame, walked top-down (root before children):
  1. Compute each joint's rest-pose world position once, by forward-kinematics
     over the rest offsets at identity rotation (`_rest_world_positions`).
  2. For joint J with direct children C1..Cn, find the single rotation R that
     best rotates J's rest-pose child directions onto J's *current-frame*
     child directions (both expressed in world space) -- this R is exactly
     J's world-space orientation for that frame. With one child this is an
     exact closed-form alignment; with several, `align_vectors` returns the
     least-squares-optimal single rotation (this is why the fit is closed
     form and needs no iterative refinement).
  3. J's local rotation (what BVH/FBX actually store) is
     `parent_world_rotation^-1 * world_rotation[J]`.
  4. A leaf joint (no children) has nothing to align to; its local rotation
     is left at identity, so it simply inherits its parent's final world
     orientation -- the same convention BVH's own "End Site" leaves use (no
     rotation channels at all).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def _rest_world_positions(parents: list[int], offsets: np.ndarray) -> np.ndarray:
    """Forward-kinematics of the rest pose at identity rotation: each joint's
    world position is simply the cumulative sum of offsets up to the root."""
    n = len(parents)
    world = np.zeros((n, 3), dtype=np.float64)
    for index in range(n):
        parent = parents[index]
        world[index] = offsets[index] if parent < 0 else world[parent] + offsets[index]
    return world


def _children_of(parents: list[int]) -> list[list[int]]:
    children: list[list[int]] = [[] for _ in parents]
    for index, parent in enumerate(parents):
        if parent >= 0:
            children[parent].append(index)
    return children


def positions_to_local_euler_xyz(
    positions: np.ndarray, parents: list[int], offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
        positions: (n_frames, n_joints, 3) world-space joint positions.
        parents: parent index per joint, -1 for the root. Must be topologically
            ordered (every joint's parent has a strictly smaller index) --
            the convention every BVH/FBX hierarchy and this project's own
            callers already use.
        offsets: (n_joints, 3) rest-pose local offset of each joint from its
            parent (root's own offset is typically the origin).

    Returns:
        (local_euler_degrees_xyz, root_translation):
        - local_euler_degrees_xyz: (n_frames, n_joints, 3) local Euler-XYZ
          (intrinsic, degrees) rotation per joint per frame.
        - root_translation: (n_frames, 3) the root's world position per
          frame, i.e. `positions[:, 0, :]`, returned for convenience since
          callers need it to animate root locomotion alongside rotation.
    """
    n_frames, n_joints, _ = positions.shape
    parents_array = list(parents)
    children = _children_of(parents_array)
    rest_world = _rest_world_positions(parents_array, offsets)

    rest_child_dirs: list[np.ndarray] = []
    for index in range(n_joints):
        kids = children[index]
        if not kids:
            rest_child_dirs.append(np.zeros((0, 3)))
            continue
        vectors = rest_world[kids] - rest_world[index]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        rest_child_dirs.append(np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 1e-9))

    local_euler = np.zeros((n_frames, n_joints, 3), dtype=np.float64)
    identity = Rotation.identity()
    for frame_index in range(n_frames):
        frame_positions = positions[frame_index]
        world_rotations: list[Rotation] = [identity] * n_joints
        for index in range(n_joints):
            kids = children[index]
            if kids:
                vectors = frame_positions[kids] - frame_positions[index]
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                current_dirs = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 1e-9)
                rest_dirs = rest_child_dirs[index]
                valid = (norms[:, 0] > 1e-9) & (np.linalg.norm(rest_dirs, axis=1) > 1e-9)
                if np.any(valid):
                    fitted, _rssd = Rotation.align_vectors(current_dirs[valid], rest_dirs[valid])
                    world_rotations[index] = fitted
                elif parents_array[index] >= 0:
                    world_rotations[index] = world_rotations[parents_array[index]]
            elif parents_array[index] >= 0:
                # Leaf: no direction to fit against, inherit the parent's
                # world orientation (local rotation identity).
                world_rotations[index] = world_rotations[parents_array[index]]

            parent = parents_array[index]
            local = world_rotations[index] if parent < 0 else world_rotations[parent].inv() * world_rotations[index]
            local_euler[frame_index, index] = local.as_euler("xyz", degrees=True)

    return local_euler, positions[:, 0, :].copy()
