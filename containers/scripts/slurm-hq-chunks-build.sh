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

# Build and optionally validate high-quality chunks inside the BSPP container.

#SBATCH --job-name=bspp_hq_chunks
# Partition is not pinned here: submit with `sbatch --partition=<partition>`
# (or set SBATCH_PARTITION in the submission environment).
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00

set -euo pipefail

require_env() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        echo "Missing required environment variable: ${name}" >&2
        exit 64
    fi
}

require_env RUN_ROOT
require_env SELECTED_IDS
require_env MODEL_TAR_INDEX
require_env STAGING_ROOT
require_env CHUNKS_DIR
require_env CONTAINER_IMAGE
require_env EVIDENCE_DIR

ORCH_DIR="${ORCH_DIR:-${BSPP_ORCH:-}}"
if [[ -z "${ORCH_DIR}" ]]; then
    LOCAL_SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    if [[ -f "${SLURM_SUBMIT_DIR:-}/containers/scripts/resolve-repo-root.sh" ]]; then
        source "${SLURM_SUBMIT_DIR}/containers/scripts/resolve-repo-root.sh"
    else
        source "${LOCAL_SCRIPT_DIR}/resolve-repo-root.sh"
    fi
    ORCH_DIR="$(resolve_orchestration_repo_root)"
fi

CHUNK_SIZE="${CHUNK_SIZE:-1000}"
RUN_VALIDATE="${RUN_VALIDATE:-1}"
VALIDATE_SAMPLE_LIMIT="${VALIDATE_SAMPLE_LIMIT:-20}"

mkdir -p "${EVIDENCE_DIR}" "${STAGING_ROOT}" "${CHUNKS_DIR}"

CONTAINER_MOUNTS="${RUN_ROOT}:${RUN_ROOT},${ORCH_DIR}:/workspace/bspp-orchestration"
if [[ -n "${LOCAL_TAR_ROOT:-}" ]]; then
    CONTAINER_MOUNTS="${CONTAINER_MOUNTS},${LOCAL_TAR_ROOT}:${LOCAL_TAR_ROOT}"
fi

run_orchestration_cli() {
    local cli_args=("$@")
    srun \
        --container-image="${CONTAINER_IMAGE}" \
        --container-mounts="${CONTAINER_MOUNTS}" \
        --no-container-mount-home \
        /usr/local/bin/entrypoint.sh \
        bash -lc '
            set -euo pipefail
            PY=/opt/bspp-orchestration-env/.pixi/envs/default/bin/python
            test -x "$PY"
            export PYTHONPATH=/workspace/bspp-orchestration/packages/orchestration-contract/src:/workspace/bspp-orchestration/packages/orchestration-runtime/src${PYTHONPATH:+:${PYTHONPATH}}
            "$PY" -c "from bspp.orchestration.runtime.cli import cli; cli()" "$@"
        ' bash "${cli_args[@]}"
}

BUILD_ARGS=(
    hq-chunks build
    --selected-ids "${SELECTED_IDS}"
    --model-tar-index "${MODEL_TAR_INDEX}"
    --staging-root "${STAGING_ROOT}"
    --chunks-dir "${CHUNKS_DIR}"
    --chunk-size "${CHUNK_SIZE}"
    --write-report "${EVIDENCE_DIR}/build"
)

if [[ -n "${LOCAL_TAR_ROOT:-}" ]]; then
    BUILD_ARGS+=(--local-tar-root "${LOCAL_TAR_ROOT}")
fi

echo "BSPP HQ chunk build"
echo "  job:             ${SLURM_JOB_ID:-local}"
echo "  image:           ${CONTAINER_IMAGE}"
echo "  selected ids:    ${SELECTED_IDS}"
echo "  model tar index: ${MODEL_TAR_INDEX}"
echo "  local tar root:  ${LOCAL_TAR_ROOT:-<index-relative>}"
echo "  staging root:    ${STAGING_ROOT}"
echo "  chunks dir:      ${CHUNKS_DIR}"
echo "  evidence:        ${EVIDENCE_DIR}"
echo "  orch dir:        ${ORCH_DIR}"
echo "  chunk size:      ${CHUNK_SIZE}"

run_orchestration_cli "${BUILD_ARGS[@]}"
echo "BSPP_HQ_CHUNKS_BUILD_PASS"

if [[ "${RUN_VALIDATE}" == "1" ]]; then
    VALIDATE_ARGS=(
        validate hq-chunks
        --chunks-dir "${CHUNKS_DIR}"
        --selected-ids "${SELECTED_IDS}"
        --model-tar-index "${MODEL_TAR_INDEX}"
        --chunk-size "${CHUNK_SIZE}"
        --sample-limit "${VALIDATE_SAMPLE_LIMIT}"
        --write-report "${EVIDENCE_DIR}/validate"
        --strict
    )
    run_orchestration_cli "${VALIDATE_ARGS[@]}"
    echo "BSPP_HQ_CHUNKS_VALIDATE_PASS"
fi
