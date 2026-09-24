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

# Dedicated postprocessing image build backend.
#
# Operator usage goes through:
#   ./containers/scripts/build.sh postprocessing [docker-build-args...]
#
# Direct backend usage:
#   ./containers/postprocessing/build.sh [image] [docker-build-args...]
#
# Bakes the orchestration source (Contract + Control + Runtime wheels at the
# pinned commit) into the generic postprocessing image, matching the dedicated
# preprocessing/folding builders. The variant pins (BASE_IMAGE, PIXI_MANIFEST,
# CUDA_VERSION, CUDA_COMPAT_*) come from containers/variants/postprocessing.env.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
IMAGE="${1:-bspp-orchestration:postprocessing}"
if [[ $# -gt 0 ]]; then
  shift
fi
read -r -a ENGINE <<< "${CONTAINER_ENGINE:-docker}"
CONTEXT="$HERE/context"
DIST="$HERE/dist"
VARIANT_ENV="$REPO/containers/variants/postprocessing.env"

[[ -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "postprocessing image builds require a clean committed source checkout" >&2
  exit 2
}
command -v "${ENGINE[0]}" >/dev/null || { echo "${ENGINE[0]} is required" >&2; exit 127; }
[[ -f "$VARIANT_ENV" ]] || { echo "postprocessing variant env missing: $VARIANT_ENV" >&2; exit 2; }

# shellcheck disable=SC1090
source "$VARIANT_ENV"
: "${VARIANT_TAG:?unset}" "${BASE_IMAGE:?unset}" "${PIXI_MANIFEST:?unset}" \
  "${CUDA_VERSION:?unset}" "${PLATFORMS:?unset}"

HOST_ARCH="$(dpkg --print-architecture 2>/dev/null || uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/')"
if [[ ",${PLATFORMS}," != *",linux/${HOST_ARCH},"* ]]; then
    echo "ERROR: variant postprocessing does not declare linux/${HOST_ARCH} in PLATFORMS=${PLATFORMS}" >&2
    exit 1
fi
[[ -f "$REPO/$PIXI_MANIFEST" ]] || { echo "ERROR: ${PIXI_MANIFEST} not found." >&2; exit 1; }

SOURCE_COMMIT="$(git -C "$REPO" rev-parse HEAD)"
SOURCE_DATE_EPOCH="$(git -C "$REPO" show -s --format=%ct "$SOURCE_COMMIT")"
export SOURCE_DATE_EPOCH
rm -rf "$CONTEXT"
mkdir -p "$CONTEXT/wheels" "$DIST"

uv --directory "$REPO" build --offline --wheel \
  --package bspp-orchestration-contract --out-dir "$CONTEXT/wheels"
uv --directory "$REPO" build --offline --wheel \
  --package bspp-orchestration-control --out-dir "$CONTEXT/wheels"
uv --directory "$REPO" build --offline --wheel \
  --package bspp-orchestration-runtime --out-dir "$CONTEXT/wheels"
CONTRACT_WHEEL="$(find "$CONTEXT/wheels" -name 'bspp_orchestration_contract-*.whl' -print -quit)"
CONTROL_WHEEL="$(find "$CONTEXT/wheels" -name 'bspp_orchestration_control-*.whl' -print -quit)"
RUNTIME_WHEEL="$(find "$CONTEXT/wheels" -name 'bspp_orchestration_runtime-*.whl' -print -quit)"
CONTRACT_SHA="$(sha256sum "$CONTRACT_WHEEL" | cut -d' ' -f1)"
CONTROL_SHA="$(sha256sum "$CONTROL_WHEEL" | cut -d' ' -f1)"
RUNTIME_SHA="$(sha256sum "$RUNTIME_WHEEL" | cut -d' ' -f1)"
MANIFEST_SHA="$(sha256sum "$REPO/$PIXI_MANIFEST" | cut -d' ' -f1)"
VARIANT_SHA="$(sha256sum "$VARIANT_ENV" | cut -d' ' -f1)"

jq -n \
  --arg source_commit "$SOURCE_COMMIT" \
  --arg pixi_manifest_sha256 "$MANIFEST_SHA" \
  --arg variant_env_sha256 "$VARIANT_SHA" \
  --arg contract_wheel_sha256 "$CONTRACT_SHA" \
  --arg control_wheel_sha256 "$CONTROL_SHA" \
  --arg runtime_wheel_sha256 "$RUNTIME_SHA" \
  --arg cuda_version "$CUDA_VERSION" \
  '{schema_version: 1, source_commit: $source_commit, pixi_manifest_sha256: $pixi_manifest_sha256,
    variant_env_sha256: $variant_env_sha256, contract_wheel_sha256: $contract_wheel_sha256,
    control_wheel_sha256: $control_wheel_sha256, runtime_wheel_sha256: $runtime_wheel_sha256,
    cuda_version: $cuda_version}' \
  > "$CONTEXT/postprocessing-runtime-image.json"

# Forward optional toolkit pin overrides; unset means the Dockerfile ARG
# defaults (public PDBeurope nvidia-postproc at the pinned commit) apply unchanged.
TOOLKIT_PIN_ARGS=()
if [[ -n "${TOOLKIT_BRANCH:-}" ]]; then
    TOOLKIT_PIN_ARGS+=(--build-arg "TOOLKIT_BRANCH=${TOOLKIT_BRANCH}")
fi
if [[ -n "${TOOLKIT_REF:-}" ]]; then
    TOOLKIT_PIN_ARGS+=(--build-arg "TOOLKIT_REF=${TOOLKIT_REF}")
fi

# Forward a base-image digest only when the variant env declares one; never
# emit an empty BASE_DIGEST that would silently override a caller-supplied
# `--build-arg BASE_DIGEST=...` (docker's last --build-arg wins).
BASE_DIGEST_ARG=()
if [[ -n "${BASE_DIGEST:-}" ]]; then
    BASE_DIGEST_ARG+=(--build-arg "BASE_DIGEST=${BASE_DIGEST}")
fi

"${ENGINE[@]}" build \
  "$@" \
  --network "${CONTAINER_BUILD_NETWORK:-default}" --platform linux/amd64 \
  --build-arg "SOURCE_DATE_EPOCH=$SOURCE_DATE_EPOCH" \
  "${TOOLKIT_PIN_ARGS[@]+"${TOOLKIT_PIN_ARGS[@]}"}" \
  --build-arg "TOOLKIT_REPO=${TOOLKIT_REPO:-https://github.com/PDBeurope/AFDB-Integration-Kit.git}" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  "${BASE_DIGEST_ARG[@]+"${BASE_DIGEST_ARG[@]}"}" \
  --build-arg "PIXI_MANIFEST=${PIXI_MANIFEST}" \
  --build-arg "CUDA_VERSION=${CUDA_VERSION}" \
  --build-arg "CUDA_COMPAT_PACKAGE=${CUDA_COMPAT_PACKAGE:-}" \
  --build-arg "CUDA_COMPAT_DIR=${CUDA_COMPAT_DIR:-}" \
  --build-arg "S5CMD_SHA256=${S5CMD_SHA256:-}" \
  -f "$REPO/containers/Dockerfile" \
  -t "$IMAGE" "$REPO"
IMAGE_ID="$("${ENGINE[@]}" image inspect --format '{{.Id}}' "$IMAGE")"
jq -n \
  --arg image "$IMAGE" --arg image_id "$IMAGE_ID" --arg source_commit "$SOURCE_COMMIT" \
  --arg pixi_manifest_sha256 "$MANIFEST_SHA" --arg variant_env_sha256 "$VARIANT_SHA" \
  --arg contract_wheel_sha256 "$CONTRACT_SHA" --arg control_wheel_sha256 "$CONTROL_SHA" \
  --arg runtime_wheel_sha256 "$RUNTIME_SHA" \
  '{image: $image, image_id: $image_id, source_commit: $source_commit,
    pixi_manifest_sha256: $pixi_manifest_sha256, variant_env_sha256: $variant_env_sha256,
    contract_wheel_sha256: $contract_wheel_sha256, control_wheel_sha256: $control_wheel_sha256,
    runtime_wheel_sha256: $runtime_wheel_sha256}' > "$DIST/build-record.json"
echo "$DIST/build-record.json"
