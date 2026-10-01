"""Skeleton side of the UniMate engine: everything except the network.

UniMate (https://github.com/Friedrich-M/UniMate, MIT) conditions on one rig
and generates motion *for that rig*. Its own preprocessing reads assets
through Blender and converts rotations with a third-party ``Motion`` package
that carries no license. This module rebuilds the parts inference needs from first
principles with NumPy alone, so it can be unit-tested without PyTorch:

* ``prepare_rig``: the host's rig descriptor -> UniMate's canonical T-pose
  (facing +Z, Y up, leaf-to-leaf diameter 2, root centred on XZ, grounded),
  in breadth-first joint order, plus the cleaned anatomical joint names the
  model's text encoder expects.
* ``topology_condition``: the graph features ``cond.npy`` would carry
  (edge relations, clamped topology distances, depths, edge index and
  Laplacian eigenvectors), matching ``unimate/utils/topology_utils.py``.
* ``decode_features``: UniMate's 12-channel output -> root trajectory and
  per-joint local rotations, with a positions cross-check.
* ``gltf_animation``: those rotations -> the host rig's own bones in glTF
  space, ready for ``_shared_glb.write_glb_from_quaternions``.

Representation (see UniMate's ``unimate/utils/motion_utils.py``): per joint
``[RIFKE position(3) | rotation 6D(6) | local velocity(3)]``. Rotations are
deltas from the T-pose expressed in the T-pose's world frame, so a skeleton
whose rest orientations are all identity (bones defined by offsets only, the
way this module builds it) reads them directly as local rotations. Slot ``j``
stores the rotation of ``parents[j]`` ("HML order"); the root slot stores the
facing rotation instead, and the root trajectory is integrated from velocity.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

TARGET_DIAMETER = 2.0
MAX_PATH_LEN = 5
MAX_FREQS = 8
EDGE_TYPES = {
    "self": 0, "parent": 1, "child": 2, "sibling": 3,
    "no_relation": 4, "end_effector": 5,
}

# Host body-plan parts -> UniMate's cleaned vocabulary (see UniMate's
# data_process/joint_annotation/vocab.py). Quadruped forelegs use the arm
# words and hind legs the leg words, as the Truebones rigs it trained on do.
_FORELEG_WORDS = ("Upper Arm", "Forearm", "Hand", "Finger")
_HINDLEG_WORDS = ("Thigh", "Shin", "Foot", "Toe", "Toe End")
_WING_WORDS = ("Shoulder", "Wing", "Wing", "Wing", "Feather")


@dataclass
class PreparedRig:
    """One rig in UniMate's canonical frame.

    ``order`` lists the kept joints in breadth-first order (root first);
    ``parents`` indexes into it. ``world_heads`` are the host's rest joint
    positions for *every* rig joint, keyed by name, in the host's world.
    """

    names: list[str]
    clean_names: list[str]
    parents: list[int]
    tpose: np.ndarray                  # (J, 3) canonical rest positions
    rotation: np.ndarray               # (3, 3) host world -> canonical
    scale: float
    shift: np.ndarray                  # (3,) added after rotate+scale
    all_names: list[str]
    all_parents: list[int]             # into all_names, BFS order, -1 = root
    world_heads: dict[str, np.ndarray]
    dropped: list[str] = field(default_factory=list)


def _unit(vector: Sequence[float]) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(array))
    if length < 1e-9:
        raise ValueError("Degenerate direction vector.")
    return array / length


def canonical_rotation(forward: Sequence[float], up: Sequence[float]) -> np.ndarray:
    """Rotation taking the host's forward to +Z and up to +Y.

    Built from the two directions only, so it is always a proper rotation:
    a rig can never be mirrored on the way in.
    """
    up_axis = _unit(up)
    forward_axis = np.asarray(forward, dtype=np.float64)
    forward_axis = _unit(forward_axis - up_axis * float(forward_axis @ up_axis))
    right = np.cross(up_axis, forward_axis)
    return np.stack([right, up_axis, forward_axis])


def bfs_order(names: Sequence[str], parent_of: dict[str, str | None]) -> list[str]:
    children: dict[str | None, list[str]] = {}
    for name in names:
        parent = parent_of.get(name)
        children.setdefault(parent if parent in parent_of else None, []).append(name)
    roots = children.get(None, [])
    if len(roots) != 1:
        raise ValueError(f"UniMate needs a single-rooted skeleton; found {len(roots)} roots.")
    order, queue = [], deque(roots)
    while queue:
        name = queue.popleft()
        order.append(name)
        queue.extend(children.get(name, []))
    return order


def _depths(parents: Sequence[int]) -> np.ndarray:
    depth = np.zeros(len(parents), dtype=np.int64)
    for index, parent in enumerate(parents):
        depth[index] = 0 if parent < 0 else depth[parent] + 1
    return depth


def _prune_to(order: list[str], parent_of: dict[str, str | None],
              heads: dict[str, np.ndarray], limit: int) -> tuple[list[str], list[str]]:
    """Drop leaf joints until at most ``limit`` remain.

    The shortest terminal bones go first (finger and toe tips, face
    details): they carry the least visible motion, and a dropped joint still
    follows its parent rigidly in the output. The rig's own leaves are used
    up before any bone that only became a leaf by pruning, so trimming never
    eats a whole chain (a head, then its neck) while other tips remain.
    """
    kept = list(order)
    dropped: list[str] = []
    original_leaves = {n for n in order if not any(parent_of.get(c) == n for c in order)}
    while len(kept) > limit:
        has_child = {parent_of[n] for n in kept if parent_of.get(n) in kept}
        leaves = [n for n in kept if n not in has_child and parent_of.get(n) in kept]
        leaves = [n for n in leaves if n in original_leaves] or leaves
        if not leaves:
            break
        victim = min(leaves, key=lambda n: float(np.linalg.norm(heads[n] - heads[parent_of[n]])))
        kept.remove(victim)
        dropped.append(victim)
    return kept, dropped


def _geodesic_diameter(positions: np.ndarray, parents: Sequence[int]) -> float:
    adjacency: list[list[tuple[int, float]]] = [[] for _ in parents]
    for child, parent in enumerate(parents):
        if parent >= 0:
            length = float(np.linalg.norm(positions[child] - positions[parent]))
            adjacency[parent].append((child, length))
            adjacency[child].append((parent, length))

    def farthest(start: int) -> tuple[int, float]:
        distance = {start: 0.0}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for other, length in adjacency[node]:
                if other not in distance:
                    distance[other] = distance[node] + length
                    queue.append(other)
        node = max(distance, key=distance.get)
        return node, distance[node]

    end, _ = farthest(0)
    _, diameter = farthest(end)
    return diameter


def clean_joint_names(order: Sequence[str], parent_of: dict[str, str | None],
                      parts: dict[str, Sequence[str]], heads: dict[str, np.ndarray],
                      side: np.ndarray) -> list[str]:
    """Anatomical label per joint from the host's body-plan parts.

    The host labels chains by geometry (``spine``, ``head``, ``tail``,
    ``front_pos`` ...; ``pos`` is the rig's left, ``side = up x forward``).
    Unlabelled joints inherit a coarse word from where they hang.
    """
    label: dict[str, str] = {}
    for part, chain in parts.items():
        chain = [name for name in chain if name in parent_of]
        if not chain:
            continue
        side_word = "Left" if part.endswith("_pos") else "Right" if part.endswith("_neg") else ""
        kind = part.rsplit("_", 1)[0] if side_word else part
        if kind == "face":
            side_word = ""  # a jaw hangs on the midline whichever half it leans into
        # A five-bone foreleg starts at the clavicle; shorter ones at the arm.
        clavicle = kind == "front" and len(chain) >= 5
        for index, name in enumerate(chain):
            if kind == "spine":
                word = "Hips" if index == 0 else ("Ribcage" if index == len(chain) - 1 and len(chain) > 2 else "Spine")
            elif kind == "head":
                word = "Head" if index == len(chain) - 1 else "Neck"
            elif kind == "tail":
                word = "Tail"
            elif kind == "face":
                word = "Jaw"
            elif kind == "wing":
                word = _WING_WORDS[min(index, len(_WING_WORDS) - 1)]
            elif kind == "front":
                position = min(index - clavicle, len(_FORELEG_WORDS) - 1)
                word = "Shoulder" if clavicle and index == 0 else _FORELEG_WORDS[position]
            elif kind in {"back", "leg"}:
                word = _HINDLEG_WORDS[min(index, len(_HINDLEG_WORDS) - 1)]
            else:
                word = "Bone"
            label[name] = f"{side_word} {word}".strip()

    names = []
    for name in order:
        if name in label:
            names.append(label[name])
            continue
        # Nearest labelled ancestor decides the family; side from geometry.
        parent = parent_of.get(name)
        while parent is not None and parent not in label:
            parent = parent_of.get(parent)
        if parent is None:
            names.append("Root" if parent_of.get(name) is None else "Body")
            continue
        base = label[parent].replace("Left ", "").replace("Right ", "")
        word = {"Head": "Head End", "Neck": "Head", "Hips": "Body", "Spine": "Body",
                "Ribcage": "Body", "Tail": "Tail"}.get(base, base)
        offset = float(heads[name] @ side)
        side_word = "" if word in {"Head End", "Head", "Body", "Tail"} else ("Left " if offset >= 0 else "Right ")
        names.append(f"{side_word}{word}")
    return names


def prepare_rig(rig: dict[str, Any], max_joints: int) -> PreparedRig:
    """Canonicalize the host's ``target_rig`` descriptor.

    ``rig['joints']``: ``[{name, parent, head: [x, y, z]}]`` in the host's
    world; ``rig['forward']`` / ``rig['up']``: the character's anatomical
    frame in that world; ``rig['parts']``: body-plan chains by part name.
    """
    joints = [j for j in rig.get("joints") or [] if j.get("name")]
    if len(joints) < 2:
        raise ValueError("The character needs at least two bones to animate.")
    parent_of = {str(j["name"]): (str(j["parent"]) if j.get("parent") else None) for j in joints}
    for name, parent in list(parent_of.items()):
        if parent is not None and parent not in parent_of:
            parent_of[name] = None
    heads = {str(j["name"]): np.asarray(j["head"], dtype=np.float64) for j in joints}
    all_names = bfs_order(list(parent_of), parent_of)
    all_parents = [all_names.index(parent_of[n]) if parent_of[n] else -1 for n in all_names]

    kept, dropped = _prune_to(all_names, parent_of, heads, max_joints)
    parents = [kept.index(parent_of[n]) if parent_of[n] else -1 for n in kept]

    rotation = canonical_rotation(rig["forward"], rig["up"])
    side = np.cross(_unit(rig["up"]), rotation[2])  # the rig's left, in host space
    world = np.stack([heads[n] for n in kept])
    rotated = world @ rotation.T
    diameter = _geodesic_diameter(rotated, parents)
    if not np.isfinite(diameter) or diameter < 1e-6:
        raise ValueError("The character's skeleton has no measurable size.")
    scale = TARGET_DIAMETER / diameter
    scaled = rotated * scale
    shift = np.array([-scaled[0, 0], -scaled[:, 1].min(), -scaled[0, 2]])
    tpose = scaled + shift

    clean = clean_joint_names(kept, parent_of, rig.get("parts") or {}, heads, side)
    return PreparedRig(
        names=kept, clean_names=clean, parents=parents, tpose=tpose,
        rotation=rotation, scale=scale, shift=shift,
        all_names=all_names, all_parents=all_parents, world_heads=heads, dropped=dropped,
    )


def topology_condition(parents: Sequence[int]) -> dict[str, np.ndarray]:
    """``cond.npy``'s graph fields, as UniMate's topology_utils computes them."""
    count = len(parents)
    children: list[list[int]] = [[] for _ in range(count)]
    for child, parent in enumerate(parents):
        if parent >= 0:
            children[parent].append(child)

    relations = np.full((count, count), EDGE_TYPES["no_relation"], dtype=np.int64)
    for i in range(count):
        for j in range(count):
            if i == j:
                relations[i, j] = EDGE_TYPES["self"]
            elif parents[j] == i:
                relations[i, j] = EDGE_TYPES["child"]
            elif parents[i] == j and parents[i] != -1:
                relations[i, j] = EDGE_TYPES["parent"]
            elif parents[i] != -1 and parents[j] == parents[i]:
                relations[i, j] = EDGE_TYPES["sibling"]
        if not children[i]:
            relations[i, i] = EDGE_TYPES["end_effector"]

    adjacency = np.zeros((count, count))
    for child, parent in enumerate(parents):
        if parent >= 0:
            adjacency[child, parent] = adjacency[parent, child] = 1.0
    distances = np.full((count, count), MAX_PATH_LEN, dtype=np.int64)
    for start in range(count):
        seen = {start: 0}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if seen[node] >= MAX_PATH_LEN:
                continue
            for other in np.flatnonzero(adjacency[node]):
                if int(other) not in seen:
                    seen[int(other)] = seen[node] + 1
                    queue.append(int(other))
        for node, distance in seen.items():
            distances[start, node] = min(distance, MAX_PATH_LEN)

    edges = [(p, c) for c, p in enumerate(parents) if p >= 0]
    edge_index = np.array(
        [[p for p, _ in edges] + [c for _, c in edges], [c for _, c in edges] + [p for p, _ in edges]],
        dtype=np.int64,
    )

    # Symmetric normalized Laplacian, smallest non-trivial eigenvectors, L2
    # normalized and zero-padded to MAX_FREQS (topology_utils defaults).
    degree = adjacency.sum(axis=1)
    inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(np.maximum(degree, 1e-12)), 0.0)
    laplacian = np.eye(count) - inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]
    _, vectors = np.linalg.eigh(laplacian)
    k = min(count - 1, MAX_FREQS)
    spectral = np.real(vectors[:, 1:1 + k])
    norms = np.linalg.norm(spectral, axis=0)
    spectral = spectral / np.where(norms > 1e-12, norms, 1.0)
    if k < MAX_FREQS:
        spectral = np.pad(spectral, ((0, 0), (0, MAX_FREQS - k)))

    return {
        "joint_relations": relations,
        "joint_graph_dist": distances,
        "joint_depths": _depths(parents),
        "edge_indexs": edge_index,
        "spectral_feats": spectral.astype(np.float32),
    }


def tpose_features(tpose: np.ndarray) -> np.ndarray:
    """The T-pose condition: positions, identity 6D rotation, zero velocity."""
    features = np.zeros((tpose.shape[0], 12))
    features[:, :3] = tpose
    features[:, 3:9] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    return features


def rotation_6d_to_matrix(cont6d: np.ndarray) -> np.ndarray:
    """UniMate's decoder (``rotation_conversions.rotation_6d_to_matrix_np``)."""
    x = cont6d[..., 0:3] / np.linalg.norm(cont6d[..., 0:3], axis=-1, keepdims=True)
    z = np.cross(x, cont6d[..., 3:6])
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=-1)


@dataclass
class DecodedMotion:
    root_positions: np.ndarray   # (T, 3) canonical
    local_rotations: np.ndarray  # (T, J, 3, 3) parent-relative, identity rest
    positions: np.ndarray        # (T, J, 3) canonical, from the rotations (FK)
    ric_positions: np.ndarray    # (T, J, 3) canonical, from the position channels


def decode_features(features: np.ndarray, parents: Sequence[int], tpose: np.ndarray) -> DecodedMotion:
    """Invert UniMate's 12-channel representation for one sample."""
    frames, count, _ = features.shape
    facing = rotation_6d_to_matrix(features[:, 0, 3:9])                      # (T, 3, 3)
    root = np.zeros((frames, 3))
    root[1:, [0, 2]] = features[:-1, 0, [9, 11]]
    root = np.einsum("tji,tj->ti", facing, root)                              # facing^-1 @ v
    root = np.cumsum(root, axis=0)
    root[:, 1] = features[:, 0, 1]

    local = np.tile(np.eye(3), (frames, count, 1, 1))
    slots = rotation_6d_to_matrix(features[:, :, 3:9])
    for child, parent in enumerate(parents):
        if parent >= 0:
            local[:, parent] = slots[:, child]

    offsets = np.array([tpose[j] - tpose[p] if p >= 0 else np.zeros(3) for j, p in enumerate(parents)])
    world_rot = np.zeros_like(local)
    positions = np.zeros((frames, count, 3))
    for joint, parent in enumerate(parents):
        if parent < 0:
            world_rot[:, joint] = local[:, joint]
            positions[:, joint] = root
        else:
            world_rot[:, joint] = world_rot[:, parent] @ local[:, joint]
            positions[:, joint] = positions[:, parent] + np.einsum("tij,j->ti", world_rot[:, parent], offsets[joint])

    ric = np.einsum("tji,tkj->tki", facing, features[:, :, 0:3])
    ric[:, :, 0] += root[:, None, 0]
    ric[:, :, 2] += root[:, None, 2]
    ric[:, 0] = root
    return DecodedMotion(root, local, positions, ric)


# glTF is Y-up with -Z forward-ish conventions; the host world is Blender's
# Z-up. Blender's glTF importer maps glTF (x, y, z) to Blender (x, -z, y), so
# writing host coordinates out needs the inverse: (x, y, z) -> (x, z, -y).
HOST_TO_GLTF = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


def _matrix_to_quaternion_xyzw(matrices: np.ndarray) -> np.ndarray:
    m = matrices
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    quat = np.zeros(m.shape[:-2] + (4,))
    w = np.sqrt(np.maximum(0.0, 1.0 + trace)) / 2.0
    x = np.sqrt(np.maximum(0.0, 1.0 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2])) / 2.0
    y = np.sqrt(np.maximum(0.0, 1.0 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2])) / 2.0
    z = np.sqrt(np.maximum(0.0, 1.0 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2])) / 2.0
    quat[..., 0] = np.copysign(x, m[..., 2, 1] - m[..., 1, 2])
    quat[..., 1] = np.copysign(y, m[..., 0, 2] - m[..., 2, 0])
    quat[..., 2] = np.copysign(z, m[..., 1, 0] - m[..., 0, 1])
    quat[..., 3] = w
    return quat / np.linalg.norm(quat, axis=-1, keepdims=True)


@dataclass
class GltfAnimation:
    names: list[str]
    parents: list[int]
    offsets: list[tuple[float, float, float]]
    quaternions_xyzw: list[list[tuple[float, float, float, float]]]
    root_translation: list[tuple[float, float, float]]


def gltf_animation(rig: PreparedRig, decoded: DecodedMotion) -> GltfAnimation:
    """Generated motion on the host's *own* bones, in glTF space.

    Every bone of the input rig is written (dropped leaves stay rigid on their
    parent), under its own name, with the host's rest offsets -- so the result
    shares the character's skeleton and appends without a mapping review.
    """
    to_host = rig.rotation.T                       # canonical -> host world
    to_gltf = HOST_TO_GLTF @ to_host               # canonical -> glTF
    frames = decoded.local_rotations.shape[0]
    index_of = {name: i for i, name in enumerate(rig.names)}

    host_local = np.einsum("ij,tkjl,ml->tkim", to_gltf, decoded.local_rotations, to_gltf)
    identity = np.array([0.0, 0.0, 0.0, 1.0])
    kept_quats = _matrix_to_quaternion_xyzw(host_local)

    offsets, quats = [], np.zeros((frames, len(rig.all_names), 4))
    for index, name in enumerate(rig.all_names):
        parent = rig.all_parents[index]
        head = rig.world_heads[name]
        base = rig.world_heads[rig.all_names[parent]] if parent >= 0 else np.zeros(3)
        offsets.append(tuple(float(v) for v in HOST_TO_GLTF @ (head - base)))
        quats[:, index] = kept_quats[:, index_of[name]] if name in index_of else identity

    host_root = ((decoded.root_positions - rig.shift) / rig.scale) @ to_host.T
    root = host_root @ HOST_TO_GLTF.T
    return GltfAnimation(
        names=list(rig.all_names),
        parents=list(rig.all_parents),
        offsets=offsets,
        quaternions_xyzw=[[tuple(float(v) for v in quats[t, j]) for j in range(len(rig.all_names))]
                          for t in range(frames)],
        root_translation=[tuple(float(v) for v in row) for row in root],
    )
