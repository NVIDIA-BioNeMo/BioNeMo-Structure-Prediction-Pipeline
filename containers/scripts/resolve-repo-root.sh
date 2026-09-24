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

# Resolve the orchestration repo root from a SLURM-safe source.
#
# Under sbatch, BASH_SOURCE[0] points at Slurm's spool copy, not the original
# script path. Prefer explicit/submission context before falling back to the
# local script path for non-SLURM use.

resolve_orchestration_repo_root() {
    if [[ -n "${BSPP_ORCH:-}" && -f "${BSPP_ORCH}/containers/scripts/resolve-repo-root.sh" ]]; then
        cd "${BSPP_ORCH}" && pwd
        return
    fi
    if [[ -n "${ORCH_DIR:-}" && -f "${ORCH_DIR}/containers/scripts/resolve-repo-root.sh" ]]; then
        cd "${ORCH_DIR}" && pwd
        return
    fi
    if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/containers/scripts/resolve-repo-root.sh" ]]; then
        cd "${SLURM_SUBMIT_DIR}" && pwd
        return
    fi

    local source_path="${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}"
    local script_dir
    script_dir="$(cd "$(dirname "${source_path}")" && pwd)"
    cd "${script_dir}/../.." && pwd
}
