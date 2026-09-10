# Third-party notices

## kimodo.cpp / ggml (Apache-2.0 / MIT)

`native/<platform>/` vendors prebuilt binaries of [localai-org/kimodo.cpp](https://github.com/localai-org/kimodo.cpp) (Apache-2.0) and its pinned [ggml](https://github.com/ggml-org/ggml) submodule (MIT) -- a from-scratch GGML/C++ reimplementation of NVIDIA's Kimodo, built by `tools/build_native.py` and committed directly (source is never vendored, only the built binaries + their SHA-256 manifest). kimodo.cpp's own `src/skeleton.hpp` -- copied into `kimodo_engine.py` as `_SOMA30_NAMES`/`_SOMA30_PARENTS`/`_SOMA30_OFFSETS` -- states it was itself "copied from NVIDIA Kimodo's Apache-2.0 `kimodo/skeleton/definitions.py`"; both source and destination are Apache-2.0.

Kimodo's motion checkpoints (`LocalAI-io/Kimodo-SOMA-RP-v1.1-GGML` on Hugging Face) are subject to the NVIDIA Open Model License. Its text encoder is LLM2Vec over Meta Llama 3 (`LocalAI-io/Llama-3-Kimodo-GGML`); though the GGUF conversion is hosted ungated, the underlying weights remain subject to Meta's Llama 3 license terms as well. Neither is distributed by this plugin -- both download directly from their respective Hugging Face repos at generation time, after the user accepts the applicable terms there.

## AnyTop (MIT)

`anytop_engine.py` clones [Anytop2025/Anytop](https://github.com/Anytop2025/Anytop) (MIT License, Copyright (c) Anytop2025 contributors) at a pinned revision and runs its model/diffusion/sampling code unmodified. `_anytop_generate_driver.py` is a fork of that project's own `sample/generate.py`, trimmed to one object type/repetition and rewritten for CPU/MPS/CUDA portability instead of a hardcoded CUDA device. AnyTop's pretrained checkpoints (`inbar2344/AnyTop` on Hugging Face) are MIT-licensed and ungated.

**Not used, installed, or vendored:** AnyTop's own `sample/generate.py` depends on a third-party `Motion` package (`git+https://github.com/inbar-2344/Motion.git`, itself forked from `sigal-raab/Motion`, based on Daniel Holden's motion-synthesis research code at theorangeduck.com) for converting sampled joint positions to BVH rotations. Neither that package nor the repository it was forked from carries a license file of any kind, which fails this project's requirement to depend only on code with a permissive, explicitly granted license. This project's `_shared_positions_to_rotations.py` and `_shared_motion_recover.py` solve the same well-known problem (recovering local joint rotations, and a world-space root path, from positions plus a known rest pose) from first principles using only SciPy, without importing, wrapping, or deriving from that package or from Holden's code.

`_shared_positions_to_rotations.py`'s per-joint rotation fit uses `scipy.spatial.transform.Rotation.align_vectors` (BSD-3-Clause, part of SciPy).
