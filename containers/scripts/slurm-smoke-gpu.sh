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
# Run the container GPU smoke test on a SLURM GPU node and save JSON evidence.
# =============================================================================
#
# Usage:
#   IMAGE_TAG=postprocessing sbatch --account=<account> containers/scripts/slurm-smoke-gpu.sh
#
# The Slurm account is supplied by the submitter (``sbatch --account=<account>``)
# and is not compiled into this script. The job fails closed if the account env
# var is absent.
# =============================================================================

#SBATCH --job-name=bspp_smoke
# Partition is not pinned here: submit with `sbatch --partition=<partition>`
# (or set SBATCH_PARTITION in the submission environment).
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1

set -euo pipefail

: "${SLURM_JOB_ACCOUNT:?SLURM_JOB_ACCOUNT is required; submit with sbatch --account=<account>}"

LOCAL_SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "${BSPP_ORCH:-}/containers/scripts/resolve-repo-root.sh" ]]; then
    source "${BSPP_ORCH}/containers/scripts/resolve-repo-root.sh"
elif [[ -f "${ORCH_DIR:-}/containers/scripts/resolve-repo-root.sh" ]]; then
    source "${ORCH_DIR}/containers/scripts/resolve-repo-root.sh"
elif [[ -f "${SLURM_SUBMIT_DIR:-}/containers/scripts/resolve-repo-root.sh" ]]; then
    source "${SLURM_SUBMIT_DIR}/containers/scripts/resolve-repo-root.sh"
else
    source "${LOCAL_SCRIPT_DIR}/resolve-repo-root.sh"
fi
REPO_ROOT="$(resolve_orchestration_repo_root)"
CONTAINERS_DIR="${REPO_ROOT}/containers"

if [[ -f "${CONTAINERS_DIR}/.env" ]]; then
    set -a; source "${CONTAINERS_DIR}/.env"; set +a
fi

IMAGE_TAG="${IMAGE_TAG:-postprocessing}"
BASE_DIR="${BASE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
CONTAINER_DIR="${CONTAINER_DIR:-${BASE_DIR}/containers}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${CONTAINER_DIR}/bspp-orchestration-${IMAGE_TAG}.sqsh}"
SMOKE_DIR="${SMOKE_DIR:-${REPO_ROOT}/containers/smoke-results}"
SMOKE_JSON="${SMOKE_JSON:-${SMOKE_DIR}/${IMAGE_TAG}-${SLURM_JOB_ID:-local}-${SLURM_ARRAY_TASK_ID:-0}.json}"

TOOLKIT_DIR="${TOOLKIT_DIR:-${REPO_ROOT}/../bspp/AFDB-Integration-Kit}"
ORCH_DIR="${ORCH_DIR:-${REPO_ROOT}}"

MOUNTS="${SMOKE_MOUNTS:-/lustre:/lustre}"
MOUNTS+=",${TOOLKIT_DIR}:/workspace/AFDB-Integration-Kit"
MOUNTS+=",${ORCH_DIR}:/workspace/bspp-orchestration"

mkdir -p "$(dirname "${SMOKE_JSON}")"

echo "BSPP container smoke"
echo "  job:       ${SLURM_JOB_ID:-local}"
echo "  node:      $(hostname)"
echo "  image:     ${CONTAINER_IMAGE}"
echo "  tag:       ${IMAGE_TAG}"
echo "  json:      ${SMOKE_JSON}"
echo "  mounts:    ${MOUNTS}"

srun \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${MOUNTS}" \
    --no-container-mount-home \
    /usr/local/bin/entrypoint.sh bspp-container-smoke-gpu "${SMOKE_JSON}"
