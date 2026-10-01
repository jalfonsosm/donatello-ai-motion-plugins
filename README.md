# Donatello AI Motion Plugins

A collection of local 3D motion-generation engines for Donatello's **Plugins** tab, each implementing Donatello's generic AI Motion API (`doc/ENGINE_PLUGINS.md` in the main repo). Bundling them in one project lets them share infrastructure -- one managed Python venv for the pieces that need PyTorch, and one GLB exporter -- instead of each installing its own multi-gigabyte environment.

| Engine | `plugin_id` | What it generates | License |
| --- | --- | --- | --- |
| [Kimodo](https://github.com/nv-tlabs/kimodo) (via [kimodo.cpp](https://github.com/localai-org/kimodo.cpp)) | `kimodo` | Human motion from a text prompt (SOMA skeleton) | NVIDIA Open Model License |
| [UniMate](https://github.com/Friedrich-M/UniMate) | `unimate` | Motion from a text prompt for **the character in your session**, any skeleton: quadrupeds, birds, insects, snakes, humanoids, rigged objects | MIT (code and weights) |

Each engine appears as its own selectable entry in the Plugins tab (`kimodo_engine.py`, `unimate_engine.py`) -- there is no dropdown-within-one-plugin, since Donatello's Plugins tab already lists every discovered engine separately.

## Install

Copy this project folder to Donatello's **Plugins → Open Plugins Folder** folder (motion). That is the complete installation for both engines -- Kimodo's own runtime is a prebuilt native binary already committed under `native/<platform>/` (see below), nothing to compile.

UniMate needs PyTorch; on its first generation it creates a **shared virtual environment** at `local_ai/plugin_cache/motion/_donatello_shared/`. Kimodo only needs that same shared venv for a much lighter reason -- converting its native output to GLB needs `pygltflib`+`numpy`, nothing else -- so if UniMate already installed it there, Kimodo just reuses it. Each engine keeps its own model weights under its own `local_ai/plugin_cache/motion/<engine>/`, so removing one engine's cache from **Manage Storage & Models** never touches the other engine's weights or the shared interpreter.

## Kimodo

Runs [kimodo.cpp](https://github.com/localai-org/kimodo.cpp) (Apache-2.0), a from-scratch GGML/C++ reimplementation of NVIDIA Kimodo, instead of NVIDIA's own Python reference implementation. Two reasons:

1. **Hardware fit.** NVIDIA's Python Kimodo needs the full fp16 `meta-llama/Meta-Llama-3-8B-Instruct` (~16GB) loaded in PyTorch, with no quantization path -- it simply does not run on an 8GB-VRAM card. kimodo.cpp's GGML runtime supports the same models on CPU/Metal/Vulkan, with a layer-streaming knob (`KIMODO_TEXT_LAYER_CHUNK`) that keeps only a few of the text encoder's 32 transformer layers resident at once. That streaming, not quantization, is what actually makes an 8GB card workable -- kimodo.cpp's own README says quantised models aren't implemented yet, so the GGUF download is comparable in size to the original fp16 weights (~15GB for the text encoder), not smaller.
2. **A real engineering gap it closes.** Kimodo's text encoder is [LLM2Vec](https://github.com/McGill-NLP/llm2vec): it strips Llama-3's causal attention mask to turn a decoder-only LLM into a bidirectional embedder. A stock `llama.cpp` fork does not support that out of the box; kimodo.cpp built native GGML support for it specifically.

Kimodo's text encoder is still, transitively, Llama-3-derived and subject to Meta's terms; the GGUF bundle download is ungated on `LocalAI-io`'s Hugging Face org, but review the model card before redistributing anything derived from it.

### Prebuilt native binaries, not a Python subprocess

`native/<platform>/` (see `native/README.md`) holds a ready-to-run `kmd-generate`/`kmd-inspect` bundle per platform, built once by `tools/build_native.py` (CI: `.github/workflows/build-native.yml`) and committed directly -- a few MB, small enough that this needs no release-asset indirection. `kimodo_engine.py` verifies the current platform's bundle against its `manifest.json` (SHA-256 per file) before running anything from it. **No C++ toolchain is ever needed on an end user's machine.**

At generation time the engine: runs the platform's `kmd-generate` binary with the prompt (native, no Python involved); downloads the SOMA motion checkpoint (~1.1GB) and the LLM2Vec text-encoder GGUF bundle (~15GB, both ungated on `LocalAI-io`) via `huggingface_hub`, already available in Donatello's own Python; and converts kimodo.cpp's raw quaternion/position output straight to a skinned GLB via the shared venv's `pygltflib`.

## UniMate

[UniMate](https://linzhanmou.com/unimate/) (SIGGRAPH Asia 2026) is a text-conditioned flow-matching transformer that generates motion for a **given** skeleton of any topology, trained on UniML3D (Truebones Zoo, Mixamo and rigged Objaverse-XL). It is the one engine here that does not bring its own skeleton: it declares `rig_input = "session_character"`, so Donatello hands it the character loaded in the session (deform bones, rest positions, confirmed front and its geometric body-plan labels) and it returns a GLB **on those very bones**. "Animate -> Add Animation" then transfers it with the direct retarget -- no joint-mapping review, which is what made creature animation impractical with species-conditioned engines.

The prompt can hold several lines; each is one action and they play in order, chained with UniMate's own motion-expansion scheme (each 2-second window pins its first frames to the previous window's tail). The duration decides how many windows there are. Options: what the character is (picks UniMate's normalization statistics: creature, humanoid or object) and how strictly to follow the prompt (classifier-free guidance).

**Hardware.** The denoiser is 74M parameters plus `flan-t5-base` as text encoder. Sampling runs on the rig's own joint count rather than the checkpoint's 71-joint padding (padded joints are masked anyway; same output to 1e-5) with a fixed 50-step Euler integration, so a 2-second clip takes about a minute and a half on a laptop CPU, seconds on a GPU. First use downloads the source (pinned commit), the `unimate_uniml3d_f60_v2` checkpoint (~1.2 GB, pinned revision) and `flan-t5-base` (~1 GB) into the engine's own cache.

**No unlicensed code.** UniMate's dataset loader and preprocessing import, at module scope, a license-less third-party `Motion` package (`Animation`, `Quaternions`). `_unimate_generate_driver.py` imports only UniMate's MIT model, flow sampler, text encoder, config and batch collation; `_unimate_skeleton.py` is this project's own NumPy implementation of everything those modules would have done:

- rig canonicalization (facing +Z, Y up, leaf-to-leaf diameter 2, root centred and grounded, breadth-first joint order), built from the character's forward/up only, so a rig can never be mirrored;
- the cleaned anatomical joint vocabulary the text encoder expects (`Left Thigh`, `Tail`, `Neck`, ...), from Donatello's geometric body-plan parts rather than bone names (generated rigs name their bones `joint_N`);
- the topology condition (edge relations, clamped graph distances, depths, symmetric-Laplacian eigenvectors);
- decoding UniMate's 12-channel representation (RIFKE positions, T-pose-relative 6D rotations in parent order, root velocity) back to local rotations, cross-checked against the position channels (the report records the result);
- writing the result on the character's own bones and rest offsets, in glTF space.

Rigs above 71 joints lose their shortest leaf bones first (those then follow their parent rigidly). Known limits: the checkpoint is the authors' "preview" release and some motions or skeletons still fail; a clip does not loop by itself.

## Output and retargeting

Every engine returns its result as a skinned `.glb` `animation_3d` artifact -- not FBX or BVH. Donatello's retarget pipeline ("Animate → Add Animation") always converts its source through Blender first, and Blender's FBX importer only accepts *binary* FBX (this project deliberately does not implement that from scratch); BVH is accepted there as an output format only, never as a source. A bare glTF node hierarchy also isn't enough -- Blender's importer only creates a real Armature from a proper `skin` object referencing the joint nodes, which `_shared_glb.py`'s GLB writer sets up (identity inverse-bind matrices, since no mesh is being deformed).

Kimodo's SOMA skeleton is recognized by the direct-retarget semantic bypass. UniMate's output shares the session character's bone names and rest positions, so the retarget pairs every bone by name and goes direct (measured on the bundled quadruped: 22/22 bones, no mapping review).

## License

This project's own code is Apache-2.0. kimodo.cpp (Apache-2.0) and its `ggml` submodule (MIT) are vendored as prebuilt binaries only, never their source. Kimodo model weights and generated outputs remain subject to NVIDIA's (and, transitively, Meta's) terms. UniMate code and model weights are MIT (the UniMate authors); its training data keeps its own terms -- see `THIRD_PARTY_NOTICES.md`.
