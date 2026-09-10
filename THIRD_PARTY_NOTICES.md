# Third-party notices

## Kimodo / SOMA-X (NVIDIA)

The Kimodo and SOMA-X wheels created by `tools/build_wheelhouse.py` are derived from NVIDIA source releases under Apache-2.0. Every generated platform wheelhouse carries copies of their respective licenses as `KIMODO_LICENSE.txt` and `SOMA_X_LICENSE.txt`.

The wheelhouse intentionally excludes PyTorch and other transitive packages. They are obtained as their official binary wheels at installation time and retain their own licenses and notices. Kimodo model weights are never distributed by this plugin; users obtain them from the official provider subject to the applicable NVIDIA model terms.

## AnyTop (MIT)

`anytop_engine.py` clones [Anytop2025/Anytop](https://github.com/Anytop2025/Anytop) (MIT License, Copyright (c) Anytop2025 contributors) at a pinned revision and runs its model/diffusion/sampling code unmodified. `_anytop_generate_driver.py` is a fork of that project's own `sample/generate.py`, trimmed to one object type/repetition and rewritten for CPU/MPS/CUDA portability instead of a hardcoded CUDA device. AnyTop's pretrained checkpoints (`inbar2344/AnyTop` on Hugging Face) are MIT-licensed and ungated.

**Not used, installed, or vendored:** AnyTop's own `sample/generate.py` depends on a third-party `Motion` package (`git+https://github.com/inbar-2344/Motion.git`, itself forked from `sigal-raab/Motion`, based on Daniel Holden's motion-synthesis research code at theorangeduck.com) for converting sampled joint positions to BVH rotations. Neither that package nor the repository it was forked from carries a license file of any kind, which fails this project's requirement to depend only on code with a permissive, explicitly granted license. This project's `_shared_positions_to_rotations.py` and `_shared_motion_recover.py` solve the same well-known problem (recovering local joint rotations, and a world-space root path, from positions plus a known rest pose) from first principles using only SciPy, without importing, wrapping, or deriving from that package or from Holden's code.

`_shared_positions_to_rotations.py`'s per-joint rotation fit uses `scipy.spatial.transform.Rotation.align_vectors` (BSD-3-Clause, part of SciPy).
