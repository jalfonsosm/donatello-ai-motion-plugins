# Donatello AI Motion Plugins

A collection of local 3D motion-generation engines for Donatello's **Plugins** tab, each implementing Donatello's generic AI Motion API (`doc/ENGINE_PLUGINS.md` in the main repo). Bundling them in one project lets them share infrastructure -- one managed Python venv for the pieces that need PyTorch, and one GLB exporter -- instead of each installing its own multi-gigabyte environment.

| Engine | `plugin_id` | What it generates | License |
| --- | --- | --- | --- |
| [Kimodo](https://github.com/nv-tlabs/kimodo) (via [kimodo.cpp](https://github.com/localai-org/kimodo.cpp)) | `kimodo` | Human motion from a text prompt (SOMA skeleton) | NVIDIA Open Model License |
| [AnyTop](https://github.com/Anytop2025/Anytop) | `anytop` | Motion for ~60 non-human creatures (birds, insects, reptiles, quadrupeds), picked by species rather than text | MIT |

Each engine appears as its own selectable entry in the Plugins tab (`kimodo_engine.py`, `anytop_engine.py`) -- there is no dropdown-within-one-plugin, since Donatello's Plugins tab already lists every discovered engine separately.

## Install

Copy this project folder to Donatello's **Plugins → Open Plugins Folder** folder (motion). That is the complete installation for both engines -- Kimodo's own runtime is a prebuilt native binary already committed under `native/<platform>/` (see below), nothing to compile.

AnyTop needs PyTorch; on first AnyTop generation it creates a **shared virtual environment** at `local_ai/plugin_cache/motion/_donatello_shared/`. Kimodo only needs that same shared venv for a much lighter reason -- converting its native output to GLB needs `pygltflib`+`numpy`, nothing else -- so if AnyTop already installed it there, Kimodo just reuses it. Each engine keeps its own model weights under its own `local_ai/plugin_cache/motion/<engine>/`, so removing one engine's cache from **Manage Storage & Models** never touches the other engine's weights or the shared interpreter.

## Kimodo

Runs [kimodo.cpp](https://github.com/localai-org/kimodo.cpp) (Apache-2.0), a from-scratch GGML/C++ reimplementation of NVIDIA Kimodo, instead of NVIDIA's own Python reference implementation. Two reasons:

1. **Hardware fit.** NVIDIA's Python Kimodo needs the full fp16 `meta-llama/Meta-Llama-3-8B-Instruct` (~16GB) loaded in PyTorch, with no quantization path -- it simply does not run on an 8GB-VRAM card. kimodo.cpp's GGML runtime supports the same models on CPU/Metal/Vulkan, with a layer-streaming knob (`KIMODO_TEXT_LAYER_CHUNK`) that keeps only a few of the text encoder's 32 transformer layers resident at once. That streaming, not quantization, is what actually makes an 8GB card workable -- kimodo.cpp's own README says quantised models aren't implemented yet, so the GGUF download is comparable in size to the original fp16 weights (~15GB for the text encoder), not smaller.
2. **A real engineering gap it closes.** Kimodo's text encoder is [LLM2Vec](https://github.com/McGill-NLP/llm2vec): it strips Llama-3's causal attention mask to turn a decoder-only LLM into a bidirectional embedder. A stock `llama.cpp` fork does not support that out of the box; kimodo.cpp built native GGML support for it specifically.

Kimodo's text encoder is still, transitively, Llama-3-derived and subject to Meta's terms; the GGUF bundle download is ungated on `LocalAI-io`'s Hugging Face org, but review the model card before redistributing anything derived from it.

### Prebuilt native binaries, not a Python subprocess

`native/<platform>/` (see `native/README.md`) holds a ready-to-run `kmd-generate`/`kmd-inspect` bundle per platform, built once by `tools/build_native.py` (CI: `.github/workflows/build-native.yml`) and committed directly -- a few MB, small enough that this needs no release-asset indirection. `kimodo_engine.py` verifies the current platform's bundle against its `manifest.json` (SHA-256 per file) before running anything from it. **No C++ toolchain is ever needed on an end user's machine.**

At generation time the engine: runs the platform's `kmd-generate` binary with the prompt (native, no Python involved); downloads the SOMA motion checkpoint (~1.1GB) and the LLM2Vec text-encoder GGUF bundle (~15GB, both ungated on `LocalAI-io`) via `huggingface_hub`, already available in Donatello's own Python; and converts kimodo.cpp's raw quaternion/position output straight to a skinned GLB via the shared venv's `pygltflib`.

## AnyTop

AnyTop is **topology-conditioned, not text-conditioned**: its T5 encoder only embeds each skeleton's own joint names for cross-topology alignment, so there is no free-text prompt control at inference time (`requires_prompt = False`). The "what to generate" control is a `Creature` selector covering AnyTop's ~60 pretrained Truebones Zoo skeleton categories, grouped into its five checkpoint subsets (flying / bipeds / quadrupeds / millipeds+snakes / all). Its pretrained checkpoints are small (~120 MB total, ungated) and downloaded automatically from `inbar2344/AnyTop` on Hugging Face.

AnyTop's own `sample/generate.py` converts sampled joint positions to BVH rotations via a third-party [`Motion`](https://github.com/inbar-2344/Motion) package that carries **no license file at all** (forked from Daniel Holden's research code, itself informally licensed). This project does not install, import, or vendor that package. `_anytop_generate_driver.py` reuses AnyTop's own MIT-licensed model/diffusion/sampling code unmodified, and replaces only that one conversion step with this project's own implementation:

- `_shared_positions_to_rotations.py` -- an original per-joint rotation-fitting algorithm (closed-form, via SciPy's `Rotation.align_vectors`), not derived from any third-party IK library.
- `_shared_motion_recover.py` -- this project's own reimplementation of the standard HumanML3D-lineage root-position recovery step (root feature -> world path via cumulative sum), using SciPy quaternions instead of the unlicensed package's `Quaternions` class.

## Output and retargeting

Both engines return their result as a skinned `.glb` `animation_3d` artifact -- not FBX or BVH. Donatello's retarget pipeline ("Animate → Add Animation") always converts its source through Blender first, and Blender's FBX importer only accepts *binary* FBX (this project deliberately does not implement that from scratch); BVH is accepted there as an output format only, never as a source. A bare glTF node hierarchy also isn't enough -- Blender's importer only creates a real Armature from a proper `skin` object referencing the joint nodes, which `_shared_glb.py`'s GLB writer sets up (identity inverse-bind matrices, since no mesh is being deformed).

Kimodo's SOMA skeleton is recognized by the direct-retarget semantic bypass; AnyTop's creature skeletons are arbitrary topologies and retarget through the generic GMR path (expect a manual mapping-review prompt for two unrelated creature topologies -- that gate is by design, not a bug).

## License

This project's own code is Apache-2.0. kimodo.cpp (Apache-2.0) and its `ggml` submodule (MIT) are vendored as prebuilt binaries only, never their source. Kimodo model weights and generated outputs remain subject to NVIDIA's (and, transitively, Meta's) terms. AnyTop code and model weights are MIT (Anytop2025 contributors); see `THIRD_PARTY_NOTICES.md`.
