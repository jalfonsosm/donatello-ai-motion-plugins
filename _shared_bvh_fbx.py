"""BVH / ASCII-FBX / GLB skeletal animation conversion, shared by every
engine in this project.

The BVH parser/writer and the ASCII FBX writer are pure Python, no
third-party library. `write_glb_from_euler_xyz` uses `pygltflib` (MIT) --
glTF's binary layout is a well-documented open standard (unlike binary FBX,
which Donatello's retarget pipeline requires but this project deliberately
does not implement from scratch); GLB is the format both engines actually
return, since Blender's FBX importer -- used by the retarget pipeline's 3D-
source-export step -- only accepts *binary* FBX, and BVH is not an accepted
source format there at all (only an accepted *output*).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Joint:
    name: str
    parent: int
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    channels: list[str] = field(default_factory=list)


def tokens(text: str) -> list[str]:
    return re.findall(r"[{}]|[^\s{}]+", text)


def parse_bvh(path: Path) -> tuple[list[Joint], list[list[float]], float]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    hierarchy, motion = raw.split("MOTION", 1)
    items, cursor, joints = tokens(hierarchy), 0, []
    if items and items[0] == "HIERARCHY":
        cursor = 1

    def take() -> str:
        nonlocal cursor
        value = items[cursor]; cursor += 1
        return value

    def joint(parent: int) -> None:
        nonlocal cursor
        kind, name = take(), take()
        if kind not in {"ROOT", "JOINT"}: raise ValueError(f"Unexpected BVH token {kind}")
        index = len(joints); joints.append(Joint(name, parent));
        if take() != "{": raise ValueError("Expected BVH joint block")
        while items[cursor] != "}":
            key = take()
            if key == "OFFSET": joints[index].offset = (float(take()), float(take()), float(take()))
            elif key == "CHANNELS": joints[index].channels = [take() for _ in range(int(take()))]
            elif key == "JOINT": cursor -= 1; joint(index)
            elif key == "End":
                take(); take()
                depth = 1
                while depth:
                    token = take(); depth += token == "{"; depth -= token == "}"
            else: raise ValueError(f"Unexpected BVH hierarchy field {key}")
        take()
    joint(-1)
    lines = [line.strip() for line in motion.splitlines() if line.strip()]
    frames = int(lines[0].split(":", 1)[1]); fps = 1.0 / float(lines[1].split(":", 1)[1])
    values = [list(map(float, line.split())) for line in lines[2:2 + frames]]
    return joints, values, fps


def rotation_order(channels: list[str]) -> int:
    order = "".join(channel[0].upper() for channel in channels if channel.endswith("rotation"))
    return {"XYZ": 0, "XZY": 1, "YZX": 2, "YXZ": 3, "ZXY": 4, "ZYX": 5}.get(order, 0)


def _channel_property(channel: str) -> str | None:
    if channel.endswith("position"):
        return "Lcl Translation"
    if channel.endswith("rotation"):
        return "Lcl Rotation"
    return None


def fbx(joints: list[Joint], frames: list[list[float]], fps: float) -> str:
    """`frames[i]` holds one flat float per joint channel, in the same order
    as each joint's declared `channels` (concatenated joint by joint) --
    exactly the layout a parsed BVH's MOTION block uses.

    A joint animates "Lcl Translation" and/or "Lcl Rotation" depending on
    which of those its own channels actually cover -- typically rotation
    only, except a BVH ROOT, which also carries position channels for
    locomotion.
    """
    models = [1000 + index for index in range(len(joints))]
    lines = ["; FBX 7.4.0 project file", "FBXHeaderExtension:  {", "  FBXHeaderVersion: 1003", "  FBXVersion: 7400", "}", "Definitions:  {", f"  Count: {len(joints)}", "}", "Objects:  {"]
    for index, joint in enumerate(joints):
        x, y, z = joint.offset
        lines += [f'  Model: {models[index]}, "Model::{joint.name}", "LimbNode" {{', "    Version: 232", "    Properties70:  {",
                  f'      P: "Lcl Translation", "Lcl Translation", "", "A",{x},{y},{z}',
                  '      P: "Lcl Rotation", "Lcl Rotation", "", "A",0,0,0',
                  f'      P: "RotationOrder", "enum", "", "",{rotation_order(joint.channels)}', "    }", "  }"]
    stack, layer, node_base, curve_base = 9000, 9001, 10000, 20000
    lines += [f'  AnimationStack: {stack}, "AnimStack::Take 001", "" {{}}', f'  AnimationLayer: {layer}, "AnimLayer::BaseLayer", "" {{}}']
    offsets, at = [], 0
    for joint in joints: offsets.append(at); at += len(joint.channels)
    times = [round(frame / fps * 46186158000) for frame in range(len(frames))]

    # Per joint, the per-axis values animated for each property ("Lcl
    # Translation" / "Lcl Rotation") it actually declares channels for.
    animated: list[dict[str, dict[str, list[float]]]] = []
    for index, joint in enumerate(joints):
        by_property: dict[str, dict[str, list[float]]] = {}
        for channel_index, channel in enumerate(joint.channels):
            property_name = _channel_property(channel)
            if property_name is None:
                continue
            axes = by_property.setdefault(property_name, {"X": [], "Y": [], "Z": []})
            axis = channel[0].upper()
            for frame in frames:
                axes[axis].append(frame[offsets[index] + channel_index])
        animated.append(by_property)

    curve_node_id: dict[tuple[int, str], int] = {}
    next_id = node_base
    for index, by_property in enumerate(animated):
        for property_name in by_property:
            curve_node_id[(index, property_name)] = next_id
            lines.append(f'  AnimationCurveNode: {next_id}, "AnimCurveNode::{joints[index].name}_{property_name}", "" {{}}')
            next_id += 1

    curve_id: dict[tuple[int, str, str], int] = {}
    next_curve = curve_base
    for index, by_property in enumerate(animated):
        for property_name, axes in by_property.items():
            for axis in "XYZ":
                curve_id[(index, property_name, axis)] = next_curve
                data = axes[axis] or [0.0] * len(frames)
                lines += [f'  AnimationCurve: {next_curve}, "AnimCurve::{joints[index].name}_{property_name}_{axis}", "" {{', "    Default: 0", "    KeyVer: 4008", f"    KeyTime: *{len(times)} {{ a: " + ",".join(map(str, times)) + "}", f"    KeyValueFloat: *{len(data)} {{ a: " + ",".join(f"{v:.8g}" for v in data) + "}", "  }"]
                next_curve += 1

    lines += ["}", "Connections:  {"]
    for index, joint in enumerate(joints):
        lines.append(f'  C: "OO",{models[index]},{0 if joint.parent < 0 else models[joint.parent]}')
    lines.append(f'  C: "OO",{layer},{stack}')
    for (index, property_name), node in curve_node_id.items():
        lines += [f'  C: "OO",{node},{layer}', f'  C: "OP",{node},{models[index]},"{property_name}"']
        for axis in "XYZ":
            lines.append(f'  C: "OP",{curve_id[(index, property_name, axis)]},{node},"d|{axis}"')
    return "\n".join(lines + ["}", ""])


def _joints_and_frames_from_euler_xyz(
    names: list[str], parents: list[int], offsets: list[tuple[float, float, float]],
    euler_degrees_xyz: "list[list[tuple[float, float, float]]]",
    root_translation: "list[tuple[float, float, float]] | None",
) -> tuple[list[Joint], list[list[float]]]:
    root_channels = ["Xposition", "Yposition", "Zposition"] if root_translation is not None else []
    joints = [
        Joint(
            name, parents[index], offsets[index],
            (root_channels if index == 0 else []) + ["Xrotation", "Yrotation", "Zrotation"],
        )
        for index, name in enumerate(names)
    ]
    # Flat channel-value list in the *per-joint* layout both `fbx()` and
    # `bvh_text()` expect (channels concatenated joint by joint): the root's
    # slice is [Xpos, Ypos, Zpos,] Xrot, Yrot, Zrot (translation only when
    # `root_translation` is given), every other joint's is just its rotation.
    frames = []
    for frame_index in range(len(euler_degrees_xyz)):
        frame = []
        if root_translation is not None:
            frame.extend(root_translation[frame_index])
        for joint_index in range(len(joints)):
            frame.extend(euler_degrees_xyz[frame_index][joint_index])
        frames.append(frame)
    return joints, frames


def write_fbx_from_euler_xyz(
    path: Path, names: list[str], parents: list[int], offsets: list[tuple[float, float, float]],
    euler_degrees_xyz: "list[list[tuple[float, float, float]]]", fps: float,
    root_translation: "list[tuple[float, float, float]] | None" = None,
) -> None:
    """Write an FBX straight from an in-memory joint hierarchy + per-frame
    Euler-XYZ rotations (degrees), skipping a BVH round-trip entirely.

    Every joint gets a 3-channel Xrotation/Yrotation/Zrotation layout. When
    `root_translation` is given (one (x, y, z) per frame, world space), the
    root joint (index 0) additionally animates Xposition/Yposition/Zposition
    so locomotion is preserved instead of only being visible as in-place
    rotation. `euler_degrees_xyz[frame][joint] = (x, y, z)`.
    """
    joints, frames = _joints_and_frames_from_euler_xyz(names, parents, offsets, euler_degrees_xyz, root_translation)
    Path(path).write_text(fbx(joints, frames, fps), encoding="utf-8")


def _bvh_hierarchy_lines(joints: list[Joint]) -> list[str]:
    children: list[list[int]] = [[] for _ in joints]
    for index, joint in enumerate(joints):
        if joint.parent >= 0:
            children[joint.parent].append(index)

    def write_joint(index: int, depth: int, is_root: bool) -> list[str]:
        joint = joints[index]
        indent = "\t" * depth
        kind = "ROOT" if is_root else "JOINT"
        lines = [f"{indent}{kind} {joint.name}", f"{indent}{{"]
        x, y, z = joint.offset
        lines.append(f"{indent}\tOFFSET {x:.6f} {y:.6f} {z:.6f}")
        lines.append(f"{indent}\tCHANNELS {len(joint.channels)} " + " ".join(joint.channels))
        kids = children[index]
        if kids:
            for child in kids:
                lines.extend(write_joint(child, depth + 1, False))
        else:
            lines.append(f"{indent}\tEnd Site")
            lines.append(f"{indent}\t{{")
            lines.append(f"{indent}\t\tOFFSET 0.000000 0.000000 0.000000")
            lines.append(f"{indent}\t}}")
        lines.append(f"{indent}}}")
        return lines

    roots = [index for index, joint in enumerate(joints) if joint.parent < 0]
    lines = ["HIERARCHY"]
    for root in roots:
        lines.extend(write_joint(root, 0, True))
    return lines


def bvh_text(joints: list[Joint], frames: list[list[float]], fps: float) -> str:
    """Inverse of `parse_bvh`: write a standard-format BVH text file."""
    lines = _bvh_hierarchy_lines(joints)
    lines += ["MOTION", f"Frames: {len(frames)}", f"Frame Time: {1.0 / fps:.6f}"]
    for frame in frames:
        lines.append(" ".join(f"{value:.6f}" for value in frame))
    return "\n".join(lines) + "\n"


def write_bvh_from_euler_xyz(
    path: Path, names: list[str], parents: list[int], offsets: list[tuple[float, float, float]],
    euler_degrees_xyz: "list[list[tuple[float, float, float]]]", fps: float,
    root_translation: "list[tuple[float, float, float]] | None" = None,
) -> None:
    """Same in-memory contract as `write_fbx_from_euler_xyz`, writing a plain
    BVH file instead. Blender's FBX importer (used by the retarget pipeline's
    3D-source-export step) only supports *binary* FBX, which this project's
    pure-Python exporter does not produce; BVH has no such ASCII/binary split
    and both engines' output goes through this path for that reason.
    """
    joints, frames = _joints_and_frames_from_euler_xyz(names, parents, offsets, euler_degrees_xyz, root_translation)
    Path(path).write_text(bvh_text(joints, frames, fps), encoding="utf-8")


def write_glb_from_euler_xyz(
    path: Path, names: list[str], parents: list[int], offsets: list[tuple[float, float, float]],
    euler_degrees_xyz: "list[list[tuple[float, float, float]]]", fps: float,
    root_translation: "list[tuple[float, float, float]] | None" = None,
) -> None:
    """Same in-memory contract as `write_fbx_from_euler_xyz`/
    `write_bvh_from_euler_xyz`, writing a binary glTF (.glb) instead -- the
    format Donatello's retarget pipeline actually accepts from this project's
    engines (see module docstring).
    """
    import numpy as np
    import pygltflib
    from scipy.spatial.transform import Rotation

    n_frames = len(euler_degrees_xyz)
    n_joints = len(names)
    times = (np.arange(n_frames, dtype="<f4") / float(fps))

    buffer_blob = b""
    buffer_views: list[pygltflib.BufferView] = []
    accessors: list[pygltflib.Accessor] = []

    def add_accessor(data, with_bounds: bool = True) -> int:
        nonlocal buffer_blob
        data = np.ascontiguousarray(data, dtype="<f4")
        byte_offset = len(buffer_blob)
        raw = data.tobytes()
        buffer_blob += raw
        while len(buffer_blob) % 4 != 0:
            buffer_blob += b"\x00"
        buffer_views.append(pygltflib.BufferView(buffer=0, byteOffset=byte_offset, byteLength=len(raw)))
        bv_index = len(buffer_views) - 1
        if data.ndim == 1:
            acc_type = "SCALAR"
        else:
            acc_type = {3: "VEC3", 4: "VEC4", 16: "MAT4"}[data.shape[1]]
        mins = maxs = None
        if with_bounds:
            mins = data.min(axis=0).tolist() if data.ndim > 1 else [float(data.min())]
            maxs = data.max(axis=0).tolist() if data.ndim > 1 else [float(data.max())]
        accessors.append(pygltflib.Accessor(
            bufferView=bv_index, componentType=pygltflib.FLOAT, count=len(data),
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
        euler_for_joint = [euler_degrees_xyz[frame][joint_index] for frame in range(n_frames)]
        quaternions = Rotation.from_euler("xyz", euler_for_joint, degrees=True).as_quat()  # (n_frames, 4) xyzw
        rotation_accessor = add_accessor(quaternions)
        samplers.append(pygltflib.AnimationSampler(input=time_accessor, output=rotation_accessor))
        channels.append(pygltflib.AnimationChannel(
            sampler=len(samplers) - 1, target=pygltflib.AnimationChannelTarget(node=joint_index, path="rotation"),
        ))

    # A plain hierarchy of transform nodes imports as generic Empties, not an
    # animatable skeleton -- Blender's glTF importer (used by the retarget
    # pipeline) only creates an Armature (with real bones) from a `skin`
    # object referencing the joint nodes. There is no mesh being deformed
    # here, so the inverse bind matrices don't affect anything visually;
    # bone rest positions come from the node hierarchy's own translations
    # (already set above), identity is only a formally valid placeholder.
    identity_matrices = np.tile(np.eye(4, dtype="<f4"), (n_joints, 1, 1)).reshape(n_joints, 16)
    inverse_bind_matrices_accessor = add_accessor(identity_matrices, with_bounds=False)

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


def write_glb_from_bvh(path: Path, joints: list[Joint], frames: list[list[float]], fps: float) -> None:
    """Convert an already-parsed BVH (`parse_bvh`'s return shape: a flat
    per-frame list of one float per channel, channels concatenated joint by
    joint) straight to GLB, for engines whose upstream tool already produces
    BVH text (e.g. Kimodo's own `motion_convert` step)."""
    offsets_by_channel: list[int] = []
    at = 0
    for joint in joints:
        offsets_by_channel.append(at)
        at += len(joint.channels)

    euler_degrees_xyz: list[list[tuple[float, float, float]]] = []
    root_translation: list[tuple[float, float, float]] | None = None
    has_root_translation = any(channel.endswith("position") for channel in joints[0].channels)
    if has_root_translation:
        root_translation = []

    for frame in frames:
        frame_euler: list[tuple[float, float, float]] = []
        for index, joint in enumerate(joints):
            base = offsets_by_channel[index]
            rotation_by_axis = {"X": 0.0, "Y": 0.0, "Z": 0.0}
            position_by_axis = {"X": 0.0, "Y": 0.0, "Z": 0.0}
            for channel_index, channel in enumerate(joint.channels):
                value = frame[base + channel_index]
                if channel.endswith("rotation"):
                    rotation_by_axis[channel[0].upper()] = value
                elif channel.endswith("position"):
                    position_by_axis[channel[0].upper()] = value
            frame_euler.append((rotation_by_axis["X"], rotation_by_axis["Y"], rotation_by_axis["Z"]))
            if index == 0 and root_translation is not None:
                root_translation.append((position_by_axis["X"], position_by_axis["Y"], position_by_axis["Z"]))
        euler_degrees_xyz.append(frame_euler)

    write_glb_from_euler_xyz(
        path, [joint.name for joint in joints], [joint.parent for joint in joints],
        [joint.offset for joint in joints], euler_degrees_xyz, fps, root_translation=root_translation,
    )
