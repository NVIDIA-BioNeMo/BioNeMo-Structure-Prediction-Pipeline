#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Dedicated folding openfold-cli kernel image build backend.
#
# Operator usage goes through:
#   ./containers/scripts/build.sh folding openfold-cli [docker-build-args...]
#
# Direct backend usage:
#   ./containers/folding/openfold-cli/build.sh [image] [docker-build-args...]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
IMAGE="${1:-bspp-orchestration:folding-openfold-cli}"
if [[ $# -gt 0 ]]; then
  shift
fi
read -r -a ENGINE <<< "${CONTAINER_ENGINE:-docker}"
CONTEXT="$HERE/context"
DIST="$HERE/dist"

[[ -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "folding openfold-cli image builds require a clean committed source checkout" >&2
  exit 2
}
command -v "${ENGINE[0]}" >/dev/null || { echo "${ENGINE[0]} is required" >&2; exit 127; }

SOURCE_COMMIT="$(git -C "$REPO" rev-parse HEAD)"
SOURCE_DATE_EPOCH="$(git -C "$REPO" show -s --format=%ct "$SOURCE_COMMIT")"
export SOURCE_DATE_EPOCH
cp "$REPO/containers/scripts/install-orchestration-source.sh" "$HERE/"
rm -rf "$CONTEXT"
mkdir -p "$CONTEXT/wheels" "$DIST"

uv --directory "$REPO" build --offline --wheel \
  --package bspp-orchestration-contract --out-dir "$CONTEXT/wheels"
uv --directory "$REPO" build --offline --wheel \
  --package bspp-orchestration-runtime --out-dir "$CONTEXT/wheels"
uv --directory "$REPO" build --offline --wheel \
  --package bspp-orchestration-control --out-dir "$CONTEXT/wheels"
CONTRACT_WHEEL="$(find "$CONTEXT/wheels" -name 'bspp_orchestration_contract-*.whl' -print -quit)"
CONTROL_WHEEL="$(find "$CONTEXT/wheels" -name 'bspp_orchestration_control-*.whl' -print -quit)"
RUNTIME_WHEEL="$(find "$CONTEXT/wheels" -name 'bspp_orchestration_runtime-*.whl' -print -quit)"
CONTRACT_SHA="$(sha256sum "$CONTRACT_WHEEL" | cut -d' ' -f1)"
CONTROL_SHA="$(sha256sum "$CONTROL_WHEEL" | cut -d' ' -f1)"
RUNTIME_SHA="$(sha256sum "$RUNTIME_WHEEL" | cut -d' ' -f1)"
LOCK_SHA="$(sha256sum "$HERE/image-lock.json" | cut -d' ' -f1)"

# Verify the base image digest is not the placeholder.
BASE_DIGEST="$(jq -r '.base_image.linux_amd64_digest' "$HERE/image-lock.json")"
PLACEHOLDER_SHA="$(printf '0%.0s' {1..64})"
[[ "$BASE_DIGEST" != "sha256:${PLACEHOLDER_SHA}" ]] || {
  echo "image-lock base_image.linux_amd64_digest is still a placeholder; pin a real digest before the first build" >&2
  exit 2
}

# Read wired pins from image-lock.json.
OPENFOLD_SOURCE_COMMIT="$(jq -r '.openfold_source_commit' "$HERE/image-lock.json")"
PDBFIXER_REF="$(jq -r '.pdbfixer_ref' "$HERE/image-lock.json")"
TORCH_SPEC="$(jq -r '.torch.spec' "$HERE/image-lock.json")"
TORCH_INDEX_URL="$(jq -r '.torch.index_url' "$HERE/image-lock.json")"
S5CMD_SHA="$(jq -r '.s5cmd.artifact_sha256' "$HERE/image-lock.json")"
S5CMD_VERSION="$(jq -r '.s5cmd.version' "$HERE/image-lock.json")"

# Python 3.12 is provisioned inside the Dockerfile via deadsnakes PPA (the
# base image nvidia/cuda:12.1.1-devel-ubuntu22.04 ships Python 3.10). The
# build-time RUN assertion and the image-smoke.py Python 3.12 check verify the
# provisioned interpreter. No pre-build base-image gate is needed.

jq -n \
  --arg source_commit "$SOURCE_COMMIT" \
  --arg image_lock_sha256 "$LOCK_SHA" \
  --arg contract_wheel_sha256 "$CONTRACT_SHA" \
  --arg runtime_wheel_sha256 "$RUNTIME_SHA" \
  --arg control_wheel_sha256 "$CONTROL_SHA" \
  --arg openfold_source_commit "$OPENFOLD_SOURCE_COMMIT" \
  --arg cuda_version "$(jq -r '.base_image.cuda_version' "$HERE/image-lock.json")" \
  '{schema_version: 1, source_commit: $source_commit, image_lock_sha256: $image_lock_sha256,
    contract_wheel_sha256: $contract_wheel_sha256, runtime_wheel_sha256: $runtime_wheel_sha256, control_wheel_sha256: $control_wheel_sha256,
    openfold_source_commit: $openfold_source_commit, cuda_version: $cuda_version}' \
  > "$CONTEXT/folding-openfold-cli-image.json"

"${ENGINE[@]}" build \
  "$@" \
  --network "${CONTAINER_BUILD_NETWORK:-default}" --platform linux/amd64 \
  --build-arg "SOURCE_DATE_EPOCH=$SOURCE_DATE_EPOCH" \
  --build-arg "BASE_DIGEST=$BASE_DIGEST" \
  --build-arg "OPENFOLD_SOURCE_COMMIT=$OPENFOLD_SOURCE_COMMIT" \
  --build-arg "PDBFIXER_REF=$PDBFIXER_REF" \
  --build-arg "TORCH_SPEC=$TORCH_SPEC" \
  --build-arg "TORCH_INDEX_URL=$TORCH_INDEX_URL" \
  --build-arg "S5CMD_VERSION=$S5CMD_VERSION" \
  --build-arg "S5CMD_SHA256=$S5CMD_SHA" \
  --tag "$IMAGE" "$HERE"
IMAGE_ID="$("${ENGINE[@]}" image inspect --format '{{.Id}}' "$IMAGE")"
jq -n \
  --arg image "$IMAGE" --arg image_id "$IMAGE_ID" --arg source_commit "$SOURCE_COMMIT" \
  --arg image_lock_sha256 "$LOCK_SHA" --arg contract_wheel_sha256 "$CONTRACT_SHA" \
  --arg runtime_wheel_sha256 "$RUNTIME_SHA" \
  --arg control_wheel_sha256 "$CONTROL_SHA" \
  '{image: $image, image_id: $image_id, source_commit: $source_commit,
    image_lock_sha256: $image_lock_sha256, contract_wheel_sha256: $contract_wheel_sha256,
    runtime_wheel_sha256: $runtime_wheel_sha256, control_wheel_sha256: $control_wheel_sha256}' > "$DIST/build-record.json"
echo "$DIST/build-record.json"
