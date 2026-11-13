#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "$project_root"
# Use a Linux-native cache: NTFS rename/locking failed during the first install.
export UV_CACHE_DIR="${MCSE_UV_CACHE_DIR:-/tmp/mcse-uv-cache-01a1abd4}"
export UV_LINK_MODE=copy
export UV_HTTP_TIMEOUT=300
export UV_HTTP_RETRIES=2
export UV_CONCURRENT_DOWNLOADS=2
uv=.tools/uv-x86_64-unknown-linux-gnu/uv
[[ -x "$uv" && -x .venv/bin/python ]] || { echo 'Run bootstrap_wsl.sh first.' >&2; exit 1; }
"$uv" pip install --python .venv/bin/python \
    torch==2.5.1+cu121 torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
"$uv" pip install --python .venv/bin/python -r requirements-inference.txt
"$uv" pip check --python .venv/bin/python
mkdir -p artifacts/environment
"$uv" pip freeze --python .venv/bin/python > artifacts/environment/requirements-resolved.txt
