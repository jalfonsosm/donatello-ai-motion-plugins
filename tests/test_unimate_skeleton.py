"""NumPy-only checks for the UniMate rig/feature code (no PyTorch needed).

Run: python -m pytest tests/
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _unimate_skeleton as sk  # noqa: E402


def _quadruped_rig():
    """A small Z-up quadruped facing -Y (Blender's front), host-style."""
    joints = [
        ("hips", None, (0, 0.4, 1.0)), ("chest", "hips", (0, -0.4, 1.0)),
        ("neck", "chest", (0, -0.7, 1.3)), ("head", "neck", (0, -0.9, 1.5)),
        ("tail", "hips", (0, 0.9, 1.0)),
    ]
    for side, x in (("l", 0.2), ("r", -0.2)):
        joints += [
            (f"front_upper_{side}", "chest", (x, -0.4, 0.9)), (f"front_lower_{side}", f"front_upper_{side}", (x, -0.4, 0.5)),
            (f"front_foot_{side}", f"front_lower_{side}", (x, -0.45, 0.05)),
            (f"back_upper_{side}", "hips", (x, 0.4, 0.9)), (f"back_lower_{side}", f"back_upper_{side}", (x, 0.45, 0.5)),
            (f"back_foot_{side}", f"back_lower_{side}", (x, 0.4, 0.05)),
        ]
    parts = {"spine": ["hips", "chest"], "head": ["neck", "head"], "tail": ["tail"]}
    for side, sign in (("l", "pos"), ("r", "neg")):
        parts[f"front_{sign}"] = [f"front_upper_{side}", f"front_lower_{side}", f"front_foot_{side}"]
        parts[f"back_{sign}"] = [f"back_upper_{side}", f"back_lower_{side}", f"back_foot_{side}"]
    return {
        "joints": [{"name": n, "parent": p, "head": list(h)} for n, p, h in joints],
        "forward": [0, -1, 0], "up": [0, 0, 1], "parts": parts,
    }


def _rotation(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def test_canonical_rotation_is_proper_and_maps_forward_and_up():
    rotation = sk.canonical_rotation([0, -1, 0.1], [0, 0, 1])
    assert np.isclose(np.linalg.det(rotation), 1.0)
    assert np.allclose(rotation @ [0, 0, 1], [0, 1, 0])
    assert (rotation @ np.array([0, -1, 0]))[2] > 0.99


def test_prepare_rig_canonicalizes_grounds_and_labels():
    rig = sk.prepare_rig(_quadruped_rig(), max_joints=71)
    assert rig.names[0] == "hips" and rig.parents[0] == -1
    assert all(p < i for i, p in enumerate(rig.parents))          # breadth-first
    assert np.isclose(rig.tpose[:, 1].min(), 0.0)                 # grounded
    assert np.allclose(rig.tpose[0, [0, 2]], 0.0)                 # root centred
    head = rig.tpose[rig.names.index("head")]
    tail = rig.tpose[rig.names.index("tail")]
    assert head[2] > tail[2]                                      # faces +Z
    labels = dict(zip(rig.names, rig.clean_names))
    assert labels["front_upper_l"] == "Left Upper Arm"
    assert labels["back_lower_r"] == "Right Shin"
    assert labels["head"] == "Head" and labels["tail"] == "Tail"
    # Facing +Z with Y up, +X is the character's left: no mirroring on the way in.
    assert rig.tpose[rig.names.index("front_upper_l")][0] > 0


def test_prune_drops_shortest_leaves_first():
    rig = sk.prepare_rig(_quadruped_rig(), max_joints=15)
    assert len(rig.names) == 15
    assert set(rig.dropped) <= {"head", "tail", "front_foot_l", "front_foot_r", "back_foot_l", "back_foot_r"}


def test_topology_matches_unimate_conventions():
    topo = sk.topology_condition([-1, 0, 1, 1])
    assert topo["joint_relations"][1, 2] == sk.EDGE_TYPES["child"]
    assert topo["joint_relations"][2, 3] == sk.EDGE_TYPES["sibling"]
    assert topo["joint_relations"][3, 3] == sk.EDGE_TYPES["end_effector"]
    assert topo["joint_graph_dist"][0, 3] == 2
    assert topo["spectral_feats"].shape == (4, sk.MAX_FREQS)
    assert topo["edge_indexs"].shape == (2, 6)


def _encode(local, parents, tpose, root):
    """UniMate's 12-channel features for an identity-rest skeleton (facing kept identity)."""
    frames, count = local.shape[:2]
    offsets = np.array([tpose[j] - tpose[p] if p >= 0 else np.zeros(3) for j, p in enumerate(parents)])
    world = np.zeros_like(local)
    positions = np.zeros((frames, count, 3))
    for j, p in enumerate(parents):
        if p < 0:
            world[:, j], positions[:, j] = local[:, j], root
        else:
            world[:, j] = world[:, p] @ local[:, j]
            positions[:, j] = positions[:, p] + np.einsum("tij,j->ti", world[:, p], offsets[j])
    features = np.zeros((frames, count, 12))
    features[:, :, 0:3] = positions
    features[:, :, 0] -= root[:, None, 0]
    features[:, :, 2] -= root[:, None, 2]
    for j, p in enumerate(parents):
        if p >= 0:
            features[:, j, 3:9] = np.concatenate([local[:, p, :, 0], local[:, p, :, 1]], -1)
    features[:, 0, 3:9] = (1, 0, 0, 0, 1, 0)
    features[:-1, 0, 9] = root[1:, 0] - root[:-1, 0]
    features[:-1, 0, 11] = root[1:, 2] - root[:-1, 2]
    features[:, 0, 1] = root[:, 1]
    return features, positions


def test_decode_recovers_a_known_animation():
    rig = sk.prepare_rig(_quadruped_rig(), max_joints=71)
    frames, count = 12, len(rig.names)
    rng = np.random.default_rng(3)
    local = np.stack([np.stack([_rotation(rng.normal(size=3), 0.4 * rng.normal()) for _ in range(count)])
                      for _ in range(frames)])
    root = np.stack([[0.0, rig.tpose[0, 1], 0.05 * t] for t in range(frames)])
    features, positions = _encode(local, rig.parents, rig.tpose, root)

    decoded = sk.decode_features(features, rig.parents, rig.tpose)

    assert np.allclose(decoded.positions, positions, atol=1e-6)
    assert np.allclose(decoded.ric_positions, positions, atol=1e-6)
    has_child = {p for p in rig.parents if p >= 0}
    for j in has_child:
        assert np.allclose(decoded.local_rotations[:, j], local[:, j], atol=1e-6)


def test_gltf_output_reproduces_the_motion_in_host_space():
    rig = sk.prepare_rig(_quadruped_rig(), max_joints=71)
    frames, count = 6, len(rig.names)
    local = np.tile(np.eye(3), (frames, count, 1, 1))
    local[:, rig.names.index("neck")] = _rotation([1, 0, 0], 0.5)
    root = np.stack([[0.0, rig.tpose[0, 1], 0.1 * t] for t in range(frames)])
    features, positions = _encode(local, rig.parents, rig.tpose, root)
    animation = sk.gltf_animation(rig, sk.decode_features(features, rig.parents, rig.tpose))

    def rotate(q, v):
        x, y, z, w = q
        u = np.array([x, y, z])
        return v + 2 * np.cross(u, np.cross(u, v) + w * v)

    # glTF FK of the written file, then back to the host's Z-up world.
    world_rot = [None] * count
    world_pos = np.zeros((frames, len(animation.names), 3))
    for t in range(frames):
        for j, p in enumerate(animation.parents):
            q = np.asarray(animation.quaternions_xyzw[t][j])
            if p < 0:
                world_rot[j] = [q]
                world_pos[t, j] = animation.root_translation[t]
            else:
                world_pos[t, j] = world_pos[t, p] + _apply_chain(world_rot[p], np.asarray(animation.offsets[j]), rotate)
                world_rot[j] = world_rot[p] + [q]
    host = world_pos @ sk.HOST_TO_GLTF  # inverse of HOST_TO_GLTF (orthonormal)
    expected = ((positions - rig.shift) / rig.scale) @ rig.rotation
    order = [animation.names.index(n) for n in rig.names]
    assert np.allclose(host[:, order], expected, atol=1e-5)


def _apply_chain(quats, vector, rotate):
    for q in reversed(quats):
        vector = rotate(q, vector)
    return vector
