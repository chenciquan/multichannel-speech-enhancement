#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "$project_root"
commit=9c194032659a3047a4b4c244a9f541f4d34b1682
repo=third_party/McNet
mkdir -p third_party artifacts/environment
if [[ -e "$repo" && ! -d "$repo/.git" ]]; then
    echo 'Refusing to replace an existing non-Git directory.' >&2
    exit 1
fi
if [[ ! -d "$repo/.git" ]]; then
    git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/Audio-WestlakeU/McNet.git "$repo"
fi
actual="$(git -C "$repo" rev-parse HEAD)"
if [[ "$actual" != "$commit" ]]; then
    echo "Unexpected upstream commit: $actual; expected $commit. No checkout or reset performed." >&2
    exit 1
fi
# Our resumable .download file is intentionally untracked. Git itself refuses
# sparse checkout if another untracked file would be overwritten.
if [[ -n "$(git -C "$repo" status --porcelain --untracked-files=no)" ]]; then
    echo 'Upstream checkout has local changes; refusing to alter its sparse selection.' >&2
    exit 1
fi
git -c http.lowSpeedLimit=1024 -c http.lowSpeedTime=60 -C "$repo" sparse-checkout set src config
checkpoint="$repo/model_checkpoints/offline/epoch494_criteria18.78_sdr18.78.ckpt"
if [[ ! -f "$checkpoint" ]]; then
    mkdir -p "$repo/model_checkpoints/offline"
    curl --fail --silent --show-error --location --continue-at - --retry 2 --connect-timeout 20 --max-time 300 \
        "https://raw.githubusercontent.com/Audio-WestlakeU/McNet/$commit/model_checkpoints/offline/epoch494_criteria18.78_sdr18.78.ckpt" \
        --output "$checkpoint.download"
    # The model loader independently verifies the exact official Git blob.
    mv --no-clobber -- "$checkpoint.download" "$checkpoint"
fi
git -C "$repo" rev-parse HEAD
sha256sum "$checkpoint" \
    | tee artifacts/environment/mcnet_checkpoint.sha256
echo 'Original upstream checkout fetched without source edits; license scope remains recorded separately.'
