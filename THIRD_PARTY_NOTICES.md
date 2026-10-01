# Third-party notices

## kimodo.cpp / ggml (Apache-2.0 / MIT)

`native/<platform>/` vendors prebuilt binaries of [localai-org/kimodo.cpp](https://github.com/localai-org/kimodo.cpp) (Apache-2.0) and its pinned [ggml](https://github.com/ggml-org/ggml) submodule (MIT) -- a from-scratch GGML/C++ reimplementation of NVIDIA's Kimodo, built by `tools/build_native.py` and committed directly (source is never vendored, only the built binaries + their SHA-256 manifest). kimodo.cpp's own `src/skeleton.hpp` -- copied into `kimodo_engine.py` as `_SOMA30_NAMES`/`_SOMA30_PARENTS`/`_SOMA30_OFFSETS` -- states it was itself "copied from NVIDIA Kimodo's Apache-2.0 `kimodo/skeleton/definitions.py`"; both source and destination are Apache-2.0.

Kimodo's motion checkpoints (`LocalAI-io/Kimodo-SOMA-RP-v1.1-GGML` on Hugging Face) are subject to the NVIDIA Open Model License. Its text encoder is LLM2Vec over Meta Llama 3 (`LocalAI-io/Llama-3-Kimodo-GGML`); though the GGUF conversion is hosted ungated, the underlying weights remain subject to Meta's Llama 3 license terms as well. Neither is distributed by this plugin -- both download directly from their respective Hugging Face repos at generation time, after the user accepts the applicable terms there.

## UniMate (MIT)

`unimate_engine.py` clones [Friedrich-M/UniMate](https://github.com/Friedrich-M/UniMate) (MIT License, Copyright (c) the UniMate authors) at a pinned commit and runs its model, flow-matching sampler, T5 text-encoder wrapper, config schema and batch collation unmodified. Its checkpoints (`Linzhan/UniMate` on Hugging Face, MIT) and the `google/flan-t5-base` text encoder (Apache-2.0) are downloaded at a pinned revision on first use; neither is redistributed by this project.

UniMate's weights were trained on UniML3D, whose sources keep their own terms: Adobe Mixamo's terms of use, the per-object licenses of Objaverse-XL, and the Truebones Zoo license. Review them before shipping generated motion commercially.

**Not used, installed, or vendored:** UniMate's dataset loader, conditioning and data pipeline import a third-party `Motion` package (`git+https://github.com/inbar-2344/Motion.git`, based on Daniel Holden's research code) that carries no license file, plus `bpy` for its own preprocessing. This project imports none of those modules. `_unimate_skeleton.py` reimplements, with NumPy only and from the formulas documented in UniMate's sources, the rig canonicalization, the topology features and the decoding of UniMate's motion representation.
