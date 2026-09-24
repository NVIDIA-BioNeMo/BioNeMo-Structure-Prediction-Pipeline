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
# Container entrypoint for bspp-orchestration.
#
# Baked-source model: the orchestration source is baked into the image and is
# the default; a bind-mount is a dev override. This script activates the baked
# pixi environment, optionally exposes CUDA compatibility libraries, resolves
# the toolkit (override-first, baked-fallback), and resolves the orchestration
# source (baked-first, dev-mount override).
# =============================================================================
set -euo pipefail

source /opt/pixi-activate.sh

if [[ "${BSPP_CUDA_COMPAT:-off}" != "off" && -n "${BSPP_CUDA_COMPAT_DIR:-}" ]]; then
    if [[ -d "${BSPP_CUDA_COMPAT_DIR}" ]]; then
        export LD_LIBRARY_PATH="${BSPP_CUDA_COMPAT_DIR}:${LD_LIBRARY_PATH:-}"
    elif [[ "${BSPP_CUDA_COMPAT}" == "force" ]]; then
        echo "[entrypoint] requested CUDA compat dir missing: ${BSPP_CUDA_COMPAT_DIR}" >&2
        exit 1
    fi
fi

# Strict editable mode keeps metadata inside the environment rather than
# writing back into mounted host source trees.
INSTALL_FLAGS=(
    --no-deps
    --root-user-action=ignore
    --config-settings editable_mode=strict
)

# =============================================================================
# Toolkit resolution — override-first, baked-fallback, fail-closed.
#
# 1. Explicit mount override at /workspace/AFDB-Integration-Kit or
#    /workspace/afdb-toolkit wins when valid.
# 2. If no valid mount, fall back to baked /opt/afdb-toolkit with
#    provenance validation.
# 3. Fail with an actionable error when neither source is valid.
# =============================================================================

_check_toolkit_override() {
    local candidates=()
    local any_mount_exists=0
    for pkg in /workspace/AFDB-Integration-Kit /workspace/afdb-toolkit; do
        if [[ -d "${pkg}" ]]; then
            any_mount_exists=1
        fi
        if [[ -f "${pkg}/pyproject.toml" ]]; then
            candidates+=("${pkg}")
        fi
    done

    # If a recognized mount path exists but was excluded (no pyproject.toml),
    # fail closed rather than silently falling through to baked.
    #
    # NOTE for operators: a directory bind-mounted at /workspace/AFDB-Integration-Kit
    # or /workspace/afdb-toolkit that is NOT a valid toolkit source (no
    # pyproject.toml) — e.g. a stale bind mount, or an unrelated/CI volume — aborts
    # container startup here. This is intentional (an explicit but malformed override
    # must not silently fall back to the baked toolkit), but it means those two paths
    # are reserved: mount a valid toolkit checkout there, or nothing.
    if (( ${#candidates[@]} == 0 )); then
        if (( any_mount_exists == 1 )); then
            echo "[entrypoint] mount override present but missing pyproject.toml; expecting a valid toolkit mount at /workspace/AFDB-Integration-Kit or /workspace/afdb-toolkit" >&2
            exit 1
        fi
        return 1
    fi

    # Deduplicate multiple mounts that resolve to the same real path.
    if (( ${#candidates[@]} > 1 )); then
        local first_realpath
        first_realpath="$(readlink -f "${candidates[0]}")"
        for pkg in "${candidates[@]:1}"; do
            if [[ "$(readlink -f "${pkg}")" != "${first_realpath}" ]]; then
                echo "[entrypoint] multiple AFDB toolkit mounts found: ${candidates[*]}" >&2
                echo "[entrypoint] mount exactly one of /workspace/AFDB-Integration-Kit or /workspace/afdb-toolkit" >&2
                exit 1
            fi
        done
    fi

    local toolkit_root="${candidates[0]}"

    # Validate override has required files.
    if [[ ! -f "${toolkit_root}/scripts/production_pipeline.py" ]]; then
        echo "[entrypoint] mount override missing production_pipeline.py: ${toolkit_root}/scripts/production_pipeline.py" >&2
        exit 1
    fi
    if [[ ! -f "${toolkit_root}/pyproject.toml" ]]; then
        echo "[entrypoint] mount override missing pyproject.toml: ${toolkit_root}" >&2
        exit 1
    fi

    export BSPP_TOOLKIT_ROOT="${toolkit_root}"
    export BSPP_TOOLKIT_SOURCE="override"
    export PYTHONPATH="${toolkit_root}:${PYTHONPATH:-}"
    export BSPP_TOOLKIT_PROVENANCE_COMMIT=""

    if [[ "${BSPP_INSTALL_TOOLKIT_EDITABLE:-off}" == "on" ]]; then
        python -m pip install "${INSTALL_FLAGS[@]}" -e "${toolkit_root}" >/dev/null
    fi

    return 0
}

_check_baked_toolkit() {
    local baked_root="/opt/afdb-toolkit"
    # Public fail-closed default; build overrides bind BSPP_EXPECTED_TOOLKIT_COMMIT
    # to the actual built ref. The internal child inherits that value.
    local expected_commit="${BSPP_EXPECTED_TOOLKIT_COMMIT:-e2fa757aa0cb2cec8e4a8382627fcbbca7599556}"

    if [[ ! -d "${baked_root}" ]]; then
        echo "[entrypoint] no AFDB toolkit found: no valid mount override at /workspace/AFDB-Integration-Kit or /workspace/afdb-toolkit, and baked toolkit at /opt/afdb-toolkit is missing or invalid" >&2
        exit 1
    fi

    local provenance_file="${baked_root}/provenance.json"
    if [[ ! -f "${provenance_file}" ]]; then
        echo "[entrypoint] baked toolkit provenance missing: ${provenance_file}" >&2
        exit 1
    fi

    # Validate the pinned commit matches the expected value.
    local actual_commit
    actual_commit=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('commit',''))" "${provenance_file}" 2>/dev/null || true)
    if [[ -z "${actual_commit}" ]]; then
        echo "[entrypoint] baked toolkit provenance unparseable or missing commit: ${provenance_file}" >&2
        exit 1
    fi
    if [[ "${actual_commit}" != "${expected_commit}" ]]; then
        echo "[entrypoint] baked toolkit provenance invalid: expected commit ${expected_commit}, found ${actual_commit}" >&2
        exit 1
    fi

    if [[ ! -f "${baked_root}/scripts/production_pipeline.py" ]]; then
        echo "[entrypoint] baked toolkit missing required file: ${baked_root}/scripts/production_pipeline.py" >&2
        exit 1
    fi

    local ipsae_bin="${baked_root}/ipsae_cpp"
    if [[ ! -x "${ipsae_bin}" ]]; then
        echo "[entrypoint] baked toolkit iPSAE not executable: ${ipsae_bin}" >&2
        exit 1
    fi

    export BSPP_TOOLKIT_ROOT="${baked_root}"
    export BSPP_TOOLKIT_SOURCE="baked"
    export BSPP_TOOLKIT_PROVENANCE_COMMIT="${actual_commit}"
    export PYTHONPATH="${baked_root}:${PYTHONPATH:-}"

    # Baked toolkit is immutable — never editable-install.
    return 0
}

if ! _check_toolkit_override; then
    _check_baked_toolkit
fi

# Orchestration source resolution — baked-first, dev-mount override.
# The shared helper installs the baked wheels by default; a bind-mount is a
# dev override gated by BSPP_ORCHESTRATION_DEV_MOUNT=1.
source /usr/local/bin/install-orchestration-source.sh

exec "$@"
