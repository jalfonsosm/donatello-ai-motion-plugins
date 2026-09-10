# Donatello AI Motion Plugins

A collection of local 3D motion-generation engines for Donatello's **Plugins** tab, each implementing Donatello's generic AI Motion API (`doc/ENGINE_PLUGINS.md` in the main repo). Bundling them in one project lets them share a single managed Python runtime instead of each installing its own multi-gigabyte PyTorch environment.

| Engine | `plugin_id` | What it generates | License |
| --- | --- | --- | --- |
| [Kimodo](https://github.com/nv-tlabs/kimodo) | `kimodo` | Human motion from a text prompt (SOMA skeleton) | NVIDIA Open Model License |
| [AnyTop](https://github.com/Anytop2025/Anytop) | `anytop` | Motion for ~60 non-human creatures (birds, insects, reptiles, quadrupeds), picked by species rather than text | MIT |

Each engine appears as its own selectable entry in the Plugins tab (`kimodo_engine.py`, `anytop_engine.py`) -- there is no dropdown-within-one-plugin, since Donatello's Plugins tab already lists every discovered engine separately.

## Install

Copy this project folder to Donatello's **Plugins → Open Plugins Folder** folder (motion). That is the complete installation for both engines.

On first generation, whichever engine runs first creates **one shared virtual environment** at `local_ai/plugin_cache/motion/_donatello_shared/` and installs PyTorch and other common dependencies there; the other engine reuses it instead of installing its own copy. Each engine keeps its own model weights and source checkout under its own `local_ai/plugin_cache/motion/<engine>/`, so removing one engine's cache from **Manage Storage & Models** never touches the other engine's weights or the shared interpreter.

## Kimodo

Runs Kimodo's official Python CLI, converts the generated motion to SOMA BVH, then writes an ASCII FBX with this project's own pure-Python exporter (`_shared_bvh_fbx.py`). No Blender, `bpy`, node-graph, or web-service dependency.

Kimodo's text encoder uses a gated Hugging Face model (`meta-llama/Meta-Llama-3-8B-Instruct`, via its LLM2Vec fallback encoder). If the machine is not already authenticated with a Hugging Face account that has accepted that model's terms, Kimodo reports the provider's authentication requirement; this legal access step cannot be automated by Donatello or this plugin.

### Native-free installation and platform bundles

The installer never invokes a native compiler on an end-user's machine. It installs third-party dependencies exclusively as binary wheels (with one deliberate carve-out: `antlr4-python3-runtime` has never published a wheel for the `4.9.x` series `hydra-core` requires, so that one pure-Python package installs from its sdist -- no compiler involved either way) and, when the release contains a matching `wheelhouse/<platform>/manifest.json`, verifies its SHA-256 hashes, copies its Kimodo and SOMA-X wheels into the managed cache, and installs them offline. Supported bundle keys include `darwin-arm64-cp311`, `linux-x86_64-cp311`, and `windows-x86_64-cp311`.

Build the current host's bundle with `python tools/build_wheelhouse.py --python /path/to/python3.11`; attach its `wheelhouse/<platform>` directory to the release plugin folder. The builder deliberately does not package model weights or large third-party wheels such as Torch.

On Apple Silicon Macs, the generated bundle is intentionally native-free. Kimodo's upstream `MotionCorrection` extension cannot be compiled because it uses Intel-only SSE/AVX instructions, so the adapter calls Kimodo with `--no-postprocess`. Generation remains available; the only trade-off is that Kimodo's optional motion-correction pass (for example, foot-skate cleanup) is not applied.

## AnyTop

AnyTop is **topology-conditioned, not text-conditioned**: its T5 encoder only embeds each skeleton's own joint names for cross-topology alignment, so there is no free-text prompt control at inference time (`requires_prompt = False`). The "what to generate" control is a `Creature` selector covering AnyTop's ~60 pretrained Truebones Zoo skeleton categories, grouped into its five checkpoint subsets (flying / bipeds / quadrupeds / millipeds+snakes / all). Its pretrained checkpoints are small (~120 MB total, ungated) and downloaded automatically from `inbar2344/AnyTop` on Hugging Face.

AnyTop's own `sample/generate.py` converts sampled joint positions to BVH rotations via a third-party [`Motion`](https://github.com/inbar-2344/Motion) package that carries **no license file at all** (forked from Daniel Holden's research code, itself informally licensed). This project does not install, import, or vendor that package. `_anytop_generate_driver.py` reuses AnyTop's own MIT-licensed model/diffusion/sampling code unmodified, and replaces only that one conversion step with this project's own implementation:

- `_shared_positions_to_rotations.py` -- an original per-joint rotation-fitting algorithm (closed-form, via SciPy's `Rotation.align_vectors`), not derived from any third-party IK library.
- `_shared_motion_recover.py` -- this project's own reimplementation of the standard HumanML3D-lineage root-position recovery step (root feature -> world path via cumulative sum), using SciPy quaternions instead of the unlicensed package's `Quaternions` class.
- `_shared_bvh_fbx.py` -- the same pure-Python FBX writer Kimodo uses, extended to write directly from an in-memory joint hierarchy (no BVH round-trip needed for AnyTop's output) and to animate root translation (locomotion), not just rotation.

## Output and retargeting

Both engines return their FBX as an `animation_3d` artifact. Donatello can inspect the animated source skeleton in the central viewer, then opens a native save dialog. Import the saved file with **Animate → Add Animation** to judge it retargeted onto a selected character rig. Kimodo's SOMA skeleton is recognized by the direct-retarget semantic bypass; AnyTop's creature skeletons are arbitrary topologies and retarget through the generic GMR path.

## License

This project's own code is Apache-2.0. Kimodo code, model weights, and generated outputs remain subject to NVIDIA's terms. AnyTop code and model weights are MIT (Anytop2025 contributors); see `THIRD_PARTY_NOTICES.md`.
