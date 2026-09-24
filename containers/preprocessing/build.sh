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

# Dedicated preprocessing image build backend.
#
# Operator usage goes through:
#   ./containers/scripts/build.sh preprocessing [docker-build-args...]
#
# Direct backend usage:
#   ./containers/preprocessing/build.sh [image] [docker-build-args...]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
IMAGE="${1:-bspp-preprocessing-runtime:local}"
if [[ $# -gt 0 ]]; then
  shift
fi
read -r -a ENGINE <<< "${CONTAINER_ENGINE:-docker}"
CONTEXT="$HERE/context"
DIST="$HERE/dist"

[[ -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "preprocessing image builds require a clean committed source checkout" >&2
  exit 2
}
command -v "${ENGINE[0]}" >/dev/null || { echo "${ENGINE[0]} is required" >&2; exit 127; }

SOURCE_COMMIT="$(git -C "$REPO" rev-parse HEAD)"
SOURCE_DATE_EPOCH="$(git -C "$REPO" show -s --format=%ct "$SOURCE_COMMIT")"
export SOURCE_DATE_EPOCH
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
# The rsync pin is documentation-only authority (like the folding runtime): the
# actual rsync package is resolved by `pixi install --locked` from the committed
# pixi.lock inside the Dockerfile, and the cross-bound invariant is asserted
# statically by tests/test_preprocessing_image_definition.py.
RSYNC_VERSION="$(jq -r '.rsync.version' "$HERE/image-lock.json")"
S5CMD_SHA="$(jq -r '.s5cmd.artifact_sha256' "$HERE/image-lock.json")"
S5CMD_VERSION="$(jq -r '.s5cmd.version' "$HERE/image-lock.json")"
PIXI_URL="$(jq -r '.pixi.artifact_url' "$HERE/image-lock.json")"
PIXI_SHA="$(jq -r '.pixi.artifact_sha256' "$HERE/image-lock.json")"
MMSEQS_URL="$(jq -r '.mmseqs.artifact_url' "$HERE/image-lock.json")"
MMSEQS_SHA="$(jq -r '.mmseqs.artifact_sha256' "$HERE/image-lock.json")"

jq -n \
  --arg source_commit "$SOURCE_COMMIT" \
  --arg image_lock_sha256 "$LOCK_SHA" \
  --arg contract_wheel_sha256 "$CONTRACT_SHA" \
  --arg runtime_wheel_sha256 "$RUNTIME_SHA" \
  --arg control_wheel_sha256 "$CONTROL_SHA" \
  --arg colabfold_version "$(jq -r '.colabfold.version' "$HERE/image-lock.json")" \
  --arg mmseqs_version "$(jq -r '.mmseqs.source_commit' "$HERE/image-lock.json")" \
  --arg rsync_version "$RSYNC_VERSION" \
  --arg cuda_version "$(jq -r '.base_image.cuda_version' "$HERE/image-lock.json")" \
  '{schema_version: 1, source_commit: $source_commit, image_lock_sha256: $image_lock_sha256,
    contract_wheel_sha256: $contract_wheel_sha256, runtime_wheel_sha256: $runtime_wheel_sha256, control_wheel_sha256: $control_wheel_sha256,
    colabfold_version: $colabfold_version, mmseqs_version: $mmseqs_version,
    rsync_version: $rsync_version, cuda_version: $cuda_version}' \
  > "$CONTEXT/preprocessing-runtime-image.json"

"${ENGINE[@]}" build \
  "$@" \
  --network "${CONTAINER_BUILD_NETWORK:-default}" --platform linux/amd64 \
  --build-arg "SOURCE_DATE_EPOCH=$SOURCE_DATE_EPOCH" \
  --build-arg "PIXI_URL=$PIXI_URL" \
  --build-arg "PIXI_SHA256=$PIXI_SHA" \
  --build-arg "MMSEQS_URL=$MMSEQS_URL" \
  --build-arg "MMSEQS_SHA256=$MMSEQS_SHA" \
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
