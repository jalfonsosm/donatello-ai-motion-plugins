"""Driver run inside the shared venv (where `pygltflib` is installed) to
convert kimodo.cpp's raw `kmd-generate` output (two flat float32 files) to
GLB. The host process running `kimodo_engine.py`'s `generate()` does not have
`pygltflib` available -- only the shared runtime does -- so this conversion
cannot happen in-process there.

Usage: python _kimodo_native_to_glb_driver.py ROOT_POSITIONS.f32
    LOCAL_ROTATIONS_XYZW.f32 OUTPUT.glb FPS JOINT_COUNT NAME0 PARENT0 OX0 OY0
    OZ0 [NAME1 PARENT1 OX1 OY1 OZ1 ...]

The skeleton (names/parents/rest offsets) is passed on the command line
rather than hardcoded here, so this driver stays reusable for any future
kimodo.cpp checkpoint (SOMA/SMPL-X/G1) -- `kimodo_engine.py` is the one place
that knows which skeleton a given motion GGUF predicts.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path


def _read_f32(path: Path) -> list[float]:
    data = path.read_bytes()
    count = len(data) // 4
    return list(struct.unpack(f"<{count}f", data[: count * 4]))


def main() -> None:
    shared_lib_parent = sys.argv[1]
    sys.path.insert(0, shared_lib_parent)
    import _shared_glb as glb

    root_positions_path = Path(sys.argv[2])
    rotations_path = Path(sys.argv[3])
    output_path = Path(sys.argv[4])
    fps = float(sys.argv[5])
    joint_count = int(sys.argv[6])

    rest = sys.argv[7:]
    if len(rest) != joint_count * 5:
        raise ValueError(f"Expected {joint_count} joints x 5 fields (name, parent, ox, oy, oz), got {len(rest)} values.")
    names: list[str] = []
    parents: list[int] = []
    offsets: list[tuple[float, float, float]] = []
    for index in range(joint_count):
        base = index * 5
        names.append(rest[base])
        parents.append(int(rest[base + 1]))
        offsets.append((float(rest[base + 2]), float(rest[base + 3]), float(rest[base + 4])))

    root_positions_flat = _read_f32(root_positions_path)
    rotations_flat = _read_f32(rotations_path)
    n_frames = len(root_positions_flat) // 3
    if len(rotations_flat) != n_frames * joint_count * 4:
        raise ValueError(
            f"local_rotations_xyzw.f32 has {len(rotations_flat)} floats; expected "
            f"{n_frames} frames x {joint_count} joints x 4 (quaternion) = {n_frames * joint_count * 4}."
        )

    root_translation = [
        (root_positions_flat[frame * 3], root_positions_flat[frame * 3 + 1], root_positions_flat[frame * 3 + 2])
        for frame in range(n_frames)
    ]
    quaternions_xyzw = [
        [
            tuple(rotations_flat[(frame * joint_count + joint) * 4 : (frame * joint_count + joint) * 4 + 4])
            for joint in range(joint_count)
        ]
        for frame in range(n_frames)
    ]

    glb.write_glb_from_quaternions(
        output_path, names, parents, offsets, quaternions_xyzw, fps, root_translation=root_translation,
    )
    print(f"Wrote {output_path} ({n_frames} frames, {joint_count} joints)")


if __name__ == "__main__":
    main()
