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

# Submit and monitor orchestration-native HQ chunk packaging.
#
# Required:
#   RUN_ROOT, SELECTED_IDS, MODEL_TAR_INDEX, STAGING_ROOT, CHUNKS_DIR,
#   CONTAINER_IMAGE
#
# Optional:
#   EVIDENCE_DIR, REPORT, ORCH_DIR, LOCAL_TAR_ROOT, CHUNK_SIZE=1000,
#   RUN_VALIDATE=1, VALIDATE_SAMPLE_LIMIT=20, SLURM_ACCOUNT and SLURM_PARTITION
#   (required for submission).

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
require_env SLURM_ACCOUNT
require_env SLURM_PARTITION

LOCAL_SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -n "${ORCH_DIR:-}" ]]; then
    BSPP_ORCH_ROOT="${ORCH_DIR}"
elif [[ -n "${BSPP_ORCH:-}" ]]; then
    BSPP_ORCH_ROOT="${BSPP_ORCH}"
else
    source "${LOCAL_SCRIPT_DIR}/resolve-repo-root.sh"
    BSPP_ORCH_ROOT="$(resolve_orchestration_repo_root)"
fi

UTC_STAMP="${UTC_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
EVIDENCE_DIR="${EVIDENCE_DIR:-${RUN_ROOT}/evidence/hq_chunks/hq_chunks_${UTC_STAMP}}"
REPORT="${REPORT:-}"
CHUNK_SIZE="${CHUNK_SIZE:-1000}"
RUN_VALIDATE="${RUN_VALIDATE:-1}"
VALIDATE_SAMPLE_LIMIT="${VALIDATE_SAMPLE_LIMIT:-20}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
SLURM_PARTITION="${SLURM_PARTITION:-}"

mkdir -p "${EVIDENCE_DIR}"

append_report() {
    if [[ -n "${REPORT}" ]]; then
        tee -a "${REPORT}"
    else
        cat
    fi
}

{
    echo
    echo "## BSPP HQ chunk packaging"
    echo "- UTC start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "- Evidence dir: ${EVIDENCE_DIR}"
    echo "- Selected IDs: ${SELECTED_IDS}"
    echo "- Model tar index: ${MODEL_TAR_INDEX}"
    echo "- Local tar root: ${LOCAL_TAR_ROOT:-<index-relative>}"
    echo "- Staging root: ${STAGING_ROOT}"
    echo "- Chunks dir: ${CHUNKS_DIR}"
    echo "- Container: ${CONTAINER_IMAGE}"
    echo "- Orchestration source: ${BSPP_ORCH_ROOT}"
    echo "- Chunk size: ${CHUNK_SIZE}"
    echo "- Run validation: ${RUN_VALIDATE}"
} | append_report

{
    echo "selected_id_rows=$(grep -cv '^[[:space:]]*$' "${SELECTED_IDS}")"
    echo "model_tar_index_rows=$(( $(wc -l < "${MODEL_TAR_INDEX}") - 1 ))"
    echo "orchestration_commit=$(git -C "${BSPP_ORCH_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "orchestration_dirty=$(git -C "${BSPP_ORCH_ROOT}" status --short 2>/dev/null | wc -l || echo unknown)"
} | tee "${EVIDENCE_DIR}/preflight.txt"

out_path="${EVIDENCE_DIR}/hq_chunks_%j.out"
err_path="${EVIDENCE_DIR}/hq_chunks_%j.err"
job_id="$(
    sbatch \
        --parsable \
        --account="${SLURM_ACCOUNT}" \
        --partition="${SLURM_PARTITION}" \
        --export=ALL,RUN_ROOT="${RUN_ROOT}",SELECTED_IDS="${SELECTED_IDS}",MODEL_TAR_INDEX="${MODEL_TAR_INDEX}",STAGING_ROOT="${STAGING_ROOT}",CHUNKS_DIR="${CHUNKS_DIR}",CONTAINER_IMAGE="${CONTAINER_IMAGE}",EVIDENCE_DIR="${EVIDENCE_DIR}/job",ORCH_DIR="${BSPP_ORCH_ROOT}",LOCAL_TAR_ROOT="${LOCAL_TAR_ROOT:-}",CHUNK_SIZE="${CHUNK_SIZE}",RUN_VALIDATE="${RUN_VALIDATE}",VALIDATE_SAMPLE_LIMIT="${VALIDATE_SAMPLE_LIMIT}" \
        --output="${out_path}" \
        --error="${err_path}" \
        "${LOCAL_SCRIPT_DIR}/slurm-hq-chunks-build.sh"
)"
out_path="${out_path//%j/${job_id}}"
err_path="${err_path//%j/${job_id}}"
echo "Submitted HQ chunk job: ${job_id}" | tee "${EVIDENCE_DIR}/hq_chunks_job_id.txt" | append_report

while true; do
    date -u +"%Y-%m-%dT%H:%M:%SZ"
    squeue -j "${job_id}" -o "%.18i %.12P %.36j %.8u %.2t %.12M %.10l %.6D %R" || true
    if ! squeue -h -j "${job_id}" | grep -q .; then
        break
    fi
    sleep "${MONITOR_INTERVAL_SECONDS:-30}"
done

sacct -j "${job_id}" \
    --format=JobIDRaw,JobName%48,Partition,State,ExitCode,Elapsed,MaxRSS,MaxVMSize,AllocCPUS,ReqMem,NodeList%40 \
    -P | tee "${EVIDENCE_DIR}/hq_chunks_sacct.txt"

state="$(sacct -j "${job_id}" -X --format=State,ExitCode -n -P | tail -n 1)"
{
    echo "--- hq_chunks job ${job_id}: ${state} ---"
    echo "--- ${out_path} ---"
    [[ -f "${out_path}" ]] && tail -n 240 "${out_path}" || echo "missing"
    echo "--- ${err_path} ---"
    [[ -f "${err_path}" ]] && tail -n 240 "${err_path}" || echo "missing"
} | tee "${EVIDENCE_DIR}/hq_chunks_log_tail.txt"

{
    echo "- HQ chunk job: ${job_id}"
    echo "- Evidence: ${EVIDENCE_DIR}"
    echo "- UTC end: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | append_report

if ! echo "${state}" | grep -qE '^COMPLETED\|0:0$'; then
    echo "BSPP_HQ_CHUNKS_FAILED"
    exit 1
fi
if ! grep -q "BSPP_HQ_CHUNKS_BUILD_PASS" "${out_path}"; then
    echo "BSPP_HQ_CHUNKS_BUILD_MARKER_MISSING"
    exit 1
fi
if [[ "${RUN_VALIDATE}" == "1" ]] && ! grep -q "BSPP_HQ_CHUNKS_VALIDATE_PASS" "${out_path}"; then
    echo "BSPP_HQ_CHUNKS_VALIDATE_MARKER_MISSING"
    exit 1
fi

echo "BSPP_HQ_CHUNKS_PASS"
echo "EVIDENCE_DIR=${EVIDENCE_DIR}"
