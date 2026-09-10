"""Tiny driver run inside the shared venv (where `pygltflib` is installed) to
convert a BVH file to GLB. The host process that runs `kimodo_engine.py`'s
`generate()` does not have `pygltflib` available -- only the per-plugin
shared runtime does -- so this conversion cannot happen in-process there.

Usage: python _bvh_to_glb_driver.py <input.bvh> <output.glb>
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    shared_lib_parent, bvh_path, glb_path = sys.argv[1], sys.argv[2], sys.argv[3]
    sys.path.insert(0, shared_lib_parent)
    import _shared_bvh_fbx as bvh_fbx

    joints, frames, fps = bvh_fbx.parse_bvh(Path(bvh_path))
    bvh_fbx.write_glb_from_bvh(Path(glb_path), joints, frames, fps)


if __name__ == "__main__":
    main()
