#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "$project_root"
export UV_CACHE_DIR="${MCSE_UV_CACHE_DIR:-/tmp/mcse-uv-cache-01a1abd4}"
export UV_LINK_MODE=copy
export UV_HTTP_TIMEOUT=120
uv=.tools/uv-x86_64-unknown-linux-gnu/uv
command -v gcc >/dev/null || { echo 'PESQ needs gcc and matching Python development headers.' >&2; exit 1; }
"$uv" pip install --python .venv/bin/python -c requirements-inference.lock.txt \
    openai-whisper==20250625 pystoi==0.4.1 Cython==3.0.12 wheel==0.45.1 matplotlib==3.8.4
# Build against the existing pinned NumPy, not an unrelated isolated NumPy ABI.
"$uv" pip install --python .venv/bin/python -c requirements-inference.lock.txt --no-build-isolation pesq==0.0.4
"$uv" pip check --python .venv/bin/python
"$uv" pip freeze --python .venv/bin/python > artifacts/environment/requirements-g1-resolved.txt
