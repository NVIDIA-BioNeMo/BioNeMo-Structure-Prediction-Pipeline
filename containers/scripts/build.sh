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

# =============================================================================
# Build a container variant locally.
#
# Usage:
#   ./containers/scripts/build.sh postprocessing
#   ./containers/scripts/build.sh preprocessing
#   ./containers/scripts/build.sh folding <runtime|colabfold|openfold-cli|bioir|all>
#   ./containers/scripts/build.sh postprocessing --squashfs
#   ./containers/scripts/build.sh preprocessing --squashfs /path.sqsh
#   ./containers/scripts/build.sh folding runtime --squashfs /path.sqsh
#   ./containers/scripts/build.sh folding all
#   ./containers/scripts/build.sh postprocessing --no-cache
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

VARIANT="${1:?Usage: $0 <postprocessing|preprocessing|folding> [--squashfs [path]] [docker-build-args...]}"
shift

FOLDING_IMAGE=""
if [[ "$VARIANT" == "folding" ]]; then
    FOLDING_IMAGE="${1:?Usage: $0 folding <runtime|colabfold|openfold-cli|bioir|all> [--squashfs [path]] [docker-build-args...]}"
    shift
fi

if [[ "$VARIANT" == "preprocessing" ]]; then
    VARIANT_TAG="preprocessing"
    IMAGE_NAME="bspp-orchestration:preprocessing"
elif [[ "$VARIANT" == "folding" ]]; then
    VARIANT_TAG="folding"
elif [[ "$VARIANT" == "postprocessing" ]]; then
    VARIANT_TAG="postprocessing"
    IMAGE_NAME="bspp-orchestration:postprocessing"
else
    echo "ERROR: Unknown variant '${VARIANT}'. Available: postprocessing preprocessing 'folding <runtime|colabfold|openfold-cli|bioir|all>'" >&2
    exit 1
fi

# Collect docker build args (split --squashfs out).
SQUASHFS=""
SQUASHFS_PATH=""
BUILD_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --squashfs)
            SQUASHFS=1
            if [[ "${2:-}" && "${2:-}" != --* ]]; then
                SQUASHFS_PATH="$2"; shift
            fi
            ;;
        *) BUILD_ARGS+=("$1") ;;
    esac
    shift
done

if [[ "$VARIANT" == "preprocessing" ]]; then
    echo "==> Building ${IMAGE_NAME} with the dedicated preprocessing image definition"
    "${REPO_ROOT}/containers/preprocessing/build.sh" \
        "$IMAGE_NAME" \
        "${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"}"
elif [[ "$VARIANT" == "postprocessing" ]]; then
    echo "==> Building ${IMAGE_NAME} with the dedicated postprocessing image definition"
    "${REPO_ROOT}/containers/postprocessing/build.sh" \
        "$IMAGE_NAME" \
        "${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"}"
elif [[ "$VARIANT" == "folding" ]]; then
    FOLDING_IMAGES=(runtime colabfold openfold-cli bioir)
    if [[ "$FOLDING_IMAGE" == "all" ]]; then
        for img in "${FOLDING_IMAGES[@]}"; do
            echo "==> Building bspp-orchestration:folding-${img} with the dedicated folding ${img} image definition"
            "${REPO_ROOT}/containers/folding/${img}/build.sh" \
                "bspp-orchestration:folding-${img}" \
                "${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"}"
        done
    else
        echo "==> Building bspp-orchestration:folding-${FOLDING_IMAGE} with the dedicated folding ${FOLDING_IMAGE} image definition"
        "${REPO_ROOT}/containers/folding/${FOLDING_IMAGE}/build.sh" \
            "bspp-orchestration:folding-${FOLDING_IMAGE}" \
            "${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"}"
    fi
fi

if [[ -n "$SQUASHFS" ]]; then
    if [[ "$VARIANT" == "folding" ]]; then
        if [[ "$FOLDING_IMAGE" == "all" ]]; then
            if [[ -n "$SQUASHFS_PATH" ]]; then
                echo "ERROR: --squashfs with an explicit path is incompatible with 'folding all'; omit the path to get per-image .sqsh names" >&2
                exit 2
            fi
            for img in runtime colabfold openfold-cli bioir; do
                OUTPUT="${REPO_ROOT}/bspp-orchestration-folding-${img}.sqsh"
                echo "==> Converting to squashfs: ${OUTPUT}"
                enroot import -o "${OUTPUT}" dockerd://"bspp-orchestration:folding-${img}"
                echo "Done: ${OUTPUT}"
            done
        else
            OUTPUT="${SQUASHFS_PATH:-${REPO_ROOT}/bspp-orchestration-folding-${FOLDING_IMAGE}.sqsh}"
            echo "==> Converting to squashfs: ${OUTPUT}"
            enroot import -o "${OUTPUT}" dockerd://"bspp-orchestration:folding-${FOLDING_IMAGE}"
            echo "Done: ${OUTPUT}"
        fi
    else
        OUTPUT="${SQUASHFS_PATH:-${REPO_ROOT}/bspp-orchestration-${VARIANT_TAG}.sqsh}"
        echo "==> Converting to squashfs: ${OUTPUT}"
        enroot import -o "${OUTPUT}" dockerd://"${IMAGE_NAME}"
        echo "Done: ${OUTPUT}"
    fi
fi
