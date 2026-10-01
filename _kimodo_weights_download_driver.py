"""Driver run as a subprocess of the *host's own* Python interpreter (which
already has `huggingface_hub` -- this download needs nothing from the shared
venv) so `_shared_runtime.run()`'s stdout-line progress scraping gets real
per-file progress out of huggingface_hub's own tqdm bars, the same way it
already does for pip/git output.

Usage: python _kimodo_weights_download_driver.py REPO_ID ALLOW_PATTERN LOCAL_DIR
"""

from __future__ import annotations

import sys


def main() -> None:
    repo_id, allow_pattern, local_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id, allow_patterns=[allow_pattern], local_dir=local_dir)
    print(f"Downloaded {repo_id} ({allow_pattern}) to {local_dir}")


if __name__ == "__main__":
    main()
