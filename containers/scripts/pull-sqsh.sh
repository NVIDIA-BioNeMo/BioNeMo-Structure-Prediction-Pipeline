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
# Pull a container variant from a configured OCI registry and convert to
# squashfs for use with enroot/pyxis on SLURM clusters.
#
# Usage:
#   sbatch containers/scripts/pull-sqsh.sh             # uses IMAGE_TAG from .env
#   sbatch containers/scripts/pull-sqsh.sh postprocessing  # override tag via $1
#   sbatch containers/scripts/pull-sqsh.sh preprocessing
#   sbatch containers/scripts/pull-sqsh.sh folding-<image>   # e.g. folding-runtime
#
# Slurm account/partition are site-specific and passed by the operator on the
# sbatch command line, for example:
#   sbatch --partition=<partition> --account=<account> \
#       containers/scripts/pull-sqsh.sh preprocessing
#
# Private registry auth:
#   Enroot reads netrc-style credentials from ~/.config/enroot/.credentials.
#   Store a registry token with read_registry scope as:
#     machine <registry-host> login <user> password <token>
#
# Environment:
#   BSPP_REGISTRY          (required) OCI registry host, e.g.
#                           registry.example.com
#   BSPP_IMAGE_REPOSITORY  (optional) image repository name;
#                           default: bspp-orchestration
#   BSPP_REGISTRY_IMPORT   (optional) registry host[:port] used only to build
#                           the enroot docker:// reference; defaults to
#                           BSPP_REGISTRY. Set it when the cluster's enroot
#                           cannot use the push reference form: enroot 3.4.1
#                           rejects registry:port/repo:tag at parse time and
#                           misroutes repo@digest auth to the default
#                           registry; the same registry serves /v2/ on the
#                           plain host, so dropping the port both parses and
#                           authenticates (enroot matches `machine <host>`
#                           credentials).
#   BASE_DIR                (optional) output base directory;
#                           default: ${SLURM_SUBMIT_DIR:-$PWD}
#   CONTAINER_DIR           (optional) containers directory;
#                           default: ${BASE_DIR}/containers
#   CONTAINER_IMAGE         (optional) full output .sqsh path; overrides the
#                           derived default
# =============================================================================

#SBATCH --job-name=bspp_pull
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail

LOCAL_SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "${BSPP_ORCH:-}/containers/scripts/resolve-repo-root.sh" ]]; then
    source "${BSPP_ORCH}/containers/scripts/resolve-repo-root.sh"
elif [[ -f "${SLURM_SUBMIT_DIR:-}/containers/scripts/resolve-repo-root.sh" ]]; then
    source "${SLURM_SUBMIT_DIR}/containers/scripts/resolve-repo-root.sh"
else
    source "${LOCAL_SCRIPT_DIR}/resolve-repo-root.sh"
fi
REPO_ROOT="$(resolve_orchestration_repo_root)"
CONTAINERS_DIR="${REPO_ROOT}/containers"

# Load runtime config
if [[ -f "${CONTAINERS_DIR}/.env" ]]; then
    set -a; source "${CONTAINERS_DIR}/.env"; set +a
fi

BSPP_REGISTRY="${BSPP_REGISTRY:?BSPP_REGISTRY is required; set it to your OCI registry host (e.g. registry.example.com)}"
BSPP_IMAGE_REPOSITORY="${BSPP_IMAGE_REPOSITORY:-bspp-orchestration}"
IMAGE_TAG="${1:-${IMAGE_TAG:-postprocessing}}"
BSPP_REGISTRY_IMPORT="${BSPP_REGISTRY_IMPORT:-${BSPP_REGISTRY}}"
IMAGE="${BSPP_REGISTRY_IMPORT}/${BSPP_IMAGE_REPOSITORY}:${IMAGE_TAG}"
ENROOT_IMAGE="docker://${IMAGE}"
BASE_DIR="${BASE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
CONTAINER_DIR="${CONTAINER_DIR:-${BASE_DIR}/containers}"
OUTPUT="${CONTAINER_IMAGE:-${CONTAINER_DIR}/bspp-orchestration-${IMAGE_TAG}.sqsh}"
LOCAL_BASE="${SLURM_TMPDIR:-/tmp/${USER}/bspp-enroot-import-${SLURM_JOB_ID:-$$}}"

export ENROOT_TRANSFER_TIMEOUT="${ENROOT_TRANSFER_TIMEOUT:-1800}"
export ENROOT_TEMP_PATH="${ENROOT_TEMP_PATH:-${LOCAL_BASE}/enroot-tmp}"
export ENROOT_CACHE_PATH="${ENROOT_CACHE_PATH:-${LOCAL_BASE}/enroot-cache}"
export LC_ALL="${LC_ALL:-C}"

echo "=============================================="
echo "BSPP container pull"
echo "=============================================="
echo "Job ID:  ${SLURM_JOB_ID:-local}"
echo "Node:    $(hostname)"
echo "Image:   ${IMAGE}"
echo "Enroot:  ${ENROOT_IMAGE}"
echo "Output:  ${OUTPUT}"
echo "Temp:    ${ENROOT_TEMP_PATH}"
echo "Cache:   ${ENROOT_CACHE_PATH}"
echo "Start:   $(date)"
echo "=============================================="

mkdir -p "$(dirname "${OUTPUT}")"
mkdir -p "${ENROOT_TEMP_PATH}" "${ENROOT_CACHE_PATH}"

# Remove stale image so enroot does a clean import
rm -f "${OUTPUT}"

echo "Importing ${IMAGE} ..."
enroot import -o "${OUTPUT}" "${ENROOT_IMAGE}"

echo "Done: ${OUTPUT} ($(du -h "${OUTPUT}" | cut -f1))"
echo "SHA-256: $(sha256sum "${OUTPUT}" | cut -d' ' -f1)"
echo "Finished: $(date)"
