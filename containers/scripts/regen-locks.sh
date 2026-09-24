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
# Regenerate pixi locks for container manifests.
#
# This replaces the old uv requirements lock path. Each variant points at a
# pixi manifest via containers/variants/<variant>.env:PIXI_MANIFEST.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

if ! command -v pixi >/dev/null 2>&1; then
    echo "ERROR: pixi is required. Install from https://pixi.sh/ or use the container build." >&2
    exit 1
fi

for variant_env in containers/variants/*.env; do
    variant="$(basename "${variant_env}" .env)"
    (
        # shellcheck disable=SC1090
        source "${variant_env}"
        : "${PIXI_MANIFEST:?unset}" "${CUDA_VERSION:?unset}"

        export CONDA_OVERRIDE_CUDA="${CUDA_VERSION}"
        echo "==> locking ${variant}: ${PIXI_MANIFEST}"
        pixi lock --manifest-path "${PIXI_MANIFEST}"
    )
done

echo "Done. Commit containers/pyprojects/*.lock if pixi writes per-manifest lockfiles."
