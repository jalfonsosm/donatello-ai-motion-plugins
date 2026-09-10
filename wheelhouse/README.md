# Platform wheelhouses

Each release may embed a directory named `<os>-<cpu>-cp<python ABI>`, for example `darwin-arm64-cp311`.

It contains `manifest.json`, a Kimodo wheel, a `py-soma-x` wheel, and the Apache-2.0 notices copied by the builder. A future bundle may additionally carry a `motion_correction` wheel. The manifest records SHA-256 digests and whether that verified native wheel is present. The adapter refuses a missing or mismatched bundle and safely falls back to the source-only, native-compilation-free installer.

Build a bundle on its native target with:

```sh
python tools/build_wheelhouse.py --python /path/to/python3.11
```

Release bundles must include Apache-2.0 notices for Kimodo and SOMA-X. Do not add Kimodo model weights to this directory: they have separate NVIDIA model terms and are downloaded after the user has accepted them.
