"""Binary glTF (.glb) skeletal-animation export, shared by every engine in
this project.

GLB is the format both engines return, not FBX or BVH: Donatello's retarget
pipeline always imports the source through Blender first, and Blender's FBX
importer only accepts *binary* FBX (this project deliberately does not
implement that from scratch), while BVH is accepted there as an *output*
format only, never as a source. glTF's binary layout is a well-documented
open standard, and `pygltflib` (MIT) makes writing it straightforward. Uses
`pygltflib` (MIT) and, for `write_glb_from_euler_xyz` only, `scipy` (BSD-3)
for the Euler->quaternion conversion.
"""

from __future__ import annotations

from pathlib import Path


def write_glb_from_euler_xyz(
    path: Path, names: list[str], parents: list[int], offsets: list[tuple[float, float, float]],
    euler_degrees_xyz: "list[list[tuple[float, float, float]]]", fps: float,
    root_translation: "list[tuple[float, float, float]] | None" = None,
) -> None:
    """Same contract as `write_glb_from_quaternions`, taking each joint's
    local rotation as Euler-XYZ degrees per frame instead.
    `euler_degrees_xyz[frame][joint] = (x, y, z)`.
    """
    from scipy.spatial.transform import Rotation

    n_frames = len(euler_degrees_xyz)
    n_joints = len(names)
    quaternions_by_joint = [
        Rotation.from_euler(
            "xyz", [euler_degrees_xyz[frame][joint] for frame in range(n_frames)], degrees=True,
        ).as_quat()
        for joint in range(n_joints)
    ]  # quaternions_by_joint[joint] is (n_frames, 4) xyzw; transpose to [frame][joint] below
    quaternions_by_frame = [
        [quaternions_by_joint[joint][frame] for joint in range(n_joints)] for frame in range(n_frames)
    ]
    write_glb_from_quaternions(path, names, parents, offsets, quaternions_by_frame, fps, root_translation=root_translation)


def write_glb_from_quaternions(
    path: Path, names: list[str], parents: list[int], offsets: list[tuple[float, float, float]],
    quaternions_xyzw: "list[list[tuple[float, float, float, float]]]", fps: float,
    root_translation: "list[tuple[float, float, float]] | None" = None,
) -> None:
    """Write a skinned GLB straight from an in-memory joint hierarchy and
    per-frame local rotations. `names`/`parents`/`offsets` describe the rest
    pose (root offset is unused when `root_translation` overrides it); each
    joint's local rotation is `quaternions_xyzw[frame][joint] = (x, y, z, w)`.
    When `root_translation` is given (one (x, y, z) per frame, world space),
    the root joint (index 0) also animates translation, so locomotion is
    preserved instead of only being visible as in-place rotation.

    A plain hierarchy of glTF `Node`s imports as generic Empties, not bones --
    Blender's glTF importer (used by the retarget pipeline) only creates a
    real Armature from a `skin` object referencing the joint nodes, which
    this always adds. No mesh is deformed here, but Blender still *positions*
    each bone from its inverse bind matrix even without one, so this
    computes a real one per joint (the inverse of that joint's accumulated
    rest-pose world translation) rather than identity -- identity collapses
    every bone to the origin, which silently produces a structurally-valid
    but geometrically-degenerate armature (this shipped once; retargeting it
    "worked" in the sense of not crashing, matched zero joints, and was easy
    to misread as an unrelated-topology mapping-review gate instead).

    glTF MAT4 accessors are 16 floats in column-major order; a translation's
    (tx, ty, tz) therefore lands at flat indices [12, 13, 14]. Building the
    matrix as `numpy.eye(4)` with `m[3, 0:3] = (tx, ty, tz)` and flattening
    row-major (numpy's default) puts them at exactly those indices, so this
    reads as an ordinary affine matrix despite the row/column-major mismatch
    between numpy's default and glTF's.
    """
    import numpy as np
    import pygltflib

    n_frames = len(quaternions_xyzw)
    n_joints = len(names)
    times = np.arange(n_frames, dtype="<f4") / float(fps)

    buffer_blob = b""
    buffer_views: list[pygltflib.BufferView] = []
    accessors: list[pygltflib.Accessor] = []

    def add_accessor(data, with_bounds: bool = True, component_type: int | None = None, dtype: str = "<f4") -> int:
        nonlocal buffer_blob
        data = np.ascontiguousarray(data, dtype=dtype)
        byte_offset = len(buffer_blob)
        raw = data.tobytes()
        buffer_blob += raw
        while len(buffer_blob) % 4 != 0:
            buffer_blob += b"\x00"
        buffer_views.append(pygltflib.BufferView(buffer=0, byteOffset=byte_offset, byteLength=len(raw)))
        bv_index = len(buffer_views) - 1
        acc_type = "SCALAR" if data.ndim == 1 else {3: "VEC3", 4: "VEC4", 16: "MAT4"}[data.shape[1]]
        mins = maxs = None
        if with_bounds:
            mins = data.min(axis=0).tolist() if data.ndim > 1 else [float(data.min())]
            maxs = data.max(axis=0).tolist() if data.ndim > 1 else [float(data.max())]
        accessors.append(pygltflib.Accessor(
            bufferView=bv_index, componentType=component_type or pygltflib.FLOAT, count=len(data),
            type=acc_type, min=mins, max=maxs,
        ))
        return len(accessors) - 1

    time_accessor = add_accessor(times)

    nodes: list[pygltflib.Node] = []
    for index, name in enumerate(names):
        translation = list(root_translation[0]) if index == 0 and root_translation is not None else list(offsets[index])
        nodes.append(pygltflib.Node(name=name, translation=translation))
    for index, parent in enumerate(parents):
        if parent >= 0:
            if nodes[parent].children is None:
                nodes[parent].children = []
            nodes[parent].children.append(index)

    samplers: list[pygltflib.AnimationSampler] = []
    channels: list[pygltflib.AnimationChannel] = []

    if root_translation is not None:
        translation_accessor = add_accessor(np.asarray(root_translation, dtype="<f4"))
        samplers.append(pygltflib.AnimationSampler(input=time_accessor, output=translation_accessor))
        channels.append(pygltflib.AnimationChannel(
            sampler=len(samplers) - 1, target=pygltflib.AnimationChannelTarget(node=0, path="translation"),
        ))

    for joint_index in range(n_joints):
        quaternions = np.asarray(
            [quaternions_xyzw[frame][joint_index] for frame in range(n_frames)], dtype="<f4",
        )
        rotation_accessor = add_accessor(quaternions)
        samplers.append(pygltflib.AnimationSampler(input=time_accessor, output=rotation_accessor))
        channels.append(pygltflib.AnimationChannel(
            sampler=len(samplers) - 1, target=pygltflib.AnimationChannelTarget(node=joint_index, path="rotation"),
        ))

    # Identity inverse-bind matrices would place every bone at the origin --
    # Blender's glTF importer uses them (not just the node translations) to
    # position an armature's bones even with no mesh attached. Each joint's
    # inverse bind matrix must be the inverse of its own rest-pose *world*
    # transform: a pure translation by minus its accumulated rest position
    # (forward-kinematics of `offsets` down the parent chain; every rest
    # orientation here is identity, so "inverse of a translation" is just
    # negating it -- no rotation/scale component to invert).
    rest_world = [(0.0, 0.0, 0.0)] * n_joints
    for index, parent in enumerate(parents):
        base = rest_world[parent] if parent >= 0 else (0.0, 0.0, 0.0)
        ox, oy, oz = offsets[index]
        rest_world[index] = (base[0] + ox, base[1] + oy, base[2] + oz)
    inverse_bind = np.tile(np.eye(4, dtype="<f4"), (n_joints, 1, 1))
    for index, (x, y, z) in enumerate(rest_world):
        inverse_bind[index, 3, 0:3] = (-x, -y, -z)  # translation row; see write_glb_from_quaternions docstring
    inverse_bind_matrices_accessor = add_accessor(inverse_bind.reshape(n_joints, 16), with_bounds=False)

    root_indices = [index for index, parent in enumerate(parents) if parent < 0]
    skin = pygltflib.Skin(
        joints=list(range(n_joints)), inverseBindMatrices=inverse_bind_matrices_accessor,
        skeleton=root_indices[0] if len(root_indices) == 1 else None,
    )
    gltf = pygltflib.GLTF2(
        asset=pygltflib.Asset(version="2.0", generator="Donatello AI Motion Plugins"),
        scenes=[pygltflib.Scene(nodes=root_indices)],
        scene=0,
        nodes=nodes,
        skins=[skin],
        animations=[pygltflib.Animation(samplers=samplers, channels=channels)],
        buffers=[pygltflib.Buffer(byteLength=len(buffer_blob))],
        bufferViews=buffer_views,
        accessors=accessors,
    )
    gltf.set_binary_blob(buffer_blob)
    gltf.save(str(path))
