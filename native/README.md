# Prebuilt native binaries

Each `native/<platform>/` directory is a **committed, ready-to-run** bundle for Kimodo's native [kimodo.cpp](https://github.com/localai-org/kimodo.cpp) (Apache-2.0) runtime: the `kmd-generate`/`kmd-inspect` executables plus every shared library they load, already relocated so they work from any install location. Nothing in this directory is built on an end user's machine — that is the entire point of committing binaries here instead of source.

`<platform>` is `<os>-<cpu>`, e.g. `darwin-arm64`, `linux-x86_64`, `windows-x86_64` (no Python ABI tag: these are plain native binaries, no CPython C extension involved). `kimodo_engine.py` picks the bundle matching the current machine and verifies every file's SHA-256 against that directory's `manifest.json` before running anything from it.

## Rebuilding a bundle

```sh
python tools/build_native.py --output native/<platform>
```

This clones [localai-org/kimodo.cpp](https://github.com/localai-org/kimodo.cpp) (with its pinned `ggml` submodule) at the revision recorded in the existing `manifest.json` (or `--kimodo-cpp-revision` to bump it), configures and builds `kmd-generate`/`kmd-inspect` with CMake, copies every dylib/so/dll the binaries actually load, fixes up their install names/rpaths to be relocatable (`@executable_path` / `$ORIGIN` / bundled next to the `.exe`), and writes a fresh `manifest.json` with SHA-256 hashes for everything.

Backend selection:
- **macOS**: CPU + BLAS (Accelerate) + Metal. GGML enables Metal automatically on Apple platforms; no Vulkan/MoltenVK dependency.
- **Linux / Windows**: CPU + Vulkan (built with the LunarG Vulkan SDK in CI; gracefully falls back to CPU at runtime on a machine with no Vulkan-capable GPU/driver). Vulkan is what actually matters for the 8GB-VRAM case this exists for -- `KIMODO_TEXT_LAYER_CHUNK` streams the text encoder's 32 transformer layers through VRAM a few at a time instead of requiring all of them resident, which is the real fix for a modest GPU (kimodo.cpp does not implement weight quantization).

`.github/workflows/build-native.yml` runs this same script across macOS/Linux/Windows runners and opens a PR updating `native/` when `localai-org/kimodo.cpp`'s pinned revision changes.

## Why binaries in git, not a release-asset wheelhouse

Each platform's bundle is only a few MB (the executables are ~200KB each; the GGML shared libraries are the bulk of it) -- small enough that committing them directly is simpler than the release-asset indirection Kimodo's old Python-wheel installer used, and it means `git clone` alone is a complete install.
