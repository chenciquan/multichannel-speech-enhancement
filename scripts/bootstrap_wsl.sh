#!/usr/bin/env bash
# One project-local interpreter/environment. No sudo, shell profile or driver changes.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "$project_root"
export UV_CACHE_DIR="$project_root/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$project_root/.tools/python"
export UV_LINK_MODE=copy
uv_version=0.9.7
uv_archive="$project_root/.tools/uv-$uv_version.tar.gz"
uv_sha256=b26fcc8dfa1c39b5a5613445af3be3eefda45d9a39359bee271eafe34913583e
mkdir -p -- .tools .cache/uv artifacts/environment
if [[ ! -x .tools/uv-x86_64-unknown-linux-gnu/uv ]]; then
    curl --fail --silent --show-error --location --retry 2 --connect-timeout 20 \
        "https://github.com/astral-sh/uv/releases/download/$uv_version/uv-x86_64-unknown-linux-gnu.tar.gz" \
        --output "$uv_archive"
    printf '%s  %s\n' "$uv_sha256" "$uv_archive" | sha256sum --check -
    tar -xzf "$uv_archive" -C .tools
fi
uv=.tools/uv-x86_64-unknown-linux-gnu/uv
"$uv" --version
if [[ -e .venv && ! -f .venv/pyvenv.cfg ]]; then
    echo 'Refusing to overwrite an existing .venv directory.' >&2
    exit 1
fi
if [[ ! -f .venv/pyvenv.cfg ]]; then
    "$uv" venv --no-python-downloads --python /usr/bin/python3.12 .venv
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3,12), sys.version; print(sys.version)'
echo 'Project-local Python is ready. Dependency installation is a separate recorded step.'

