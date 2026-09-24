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

# Shared install-mode helper for phase container entrypoints.
#
# Baked-first model: the orchestration source (Contract + Control + Runtime
# wheels) is baked into the image and is the default. A bind-mount at
# /workspace/bspp-orchestration is a DEV override that must be explicitly
# requested with BSPP_ORCHESTRATION_DEV_MOUNT=1.
#
# Mode selection:
#   BSPP_INSTALL_MODE_SKIP=1            -> baked (opt-out; benchmark determinism)
#   BSPP_ORCHESTRATION_DEV_MOUNT != 1   -> baked (default); a present mount
#                                           without the dev flag fails closed
#   BSPP_ORCHESTRATION_DEV_MOUNT == 1   -> override (editable install); the
#                                           mounted commit must equal the baked
#                                           commit unless
#                                           BSPP_ORCHESTRATION_DEV_MOUNT_ALLOW_MISMATCH=1
#
# Exports BSPP_ORCHESTRATION_SOURCE ("override" or "baked") and, when a
# non-empty commit is determined, BSPP_ORCHESTRATION_PROVENANCE_COMMIT.

_INSTALL_ORCH_ROOT="${BSPP_INSTALL_ORCH_ROOT:-/workspace/bspp-orchestration}"
_ORCH_CONTRACT_TOML="${_INSTALL_ORCH_ROOT}/packages/orchestration-contract/pyproject.toml"
_ORCH_CONTROL_TOML="${_INSTALL_ORCH_ROOT}/packages/orchestration-control/pyproject.toml"
_ORCH_RUNTIME_TOML="${_INSTALL_ORCH_ROOT}/packages/orchestration-runtime/pyproject.toml"
_IMAGE_JSON="${BSPP_IMAGE_JSON:-/opt/bspp/folding-runtime-image.json}"

# --- Provenance helpers ---

_read_baked_commit() {
    if [[ -f "$_IMAGE_JSON" ]]; then
        python -c "
import json, sys
try:
    v = json.load(open('${_IMAGE_JSON}')).get('source_commit', '')
    sys.stdout.write(v if isinstance(v, str) else '')
except Exception:
    pass
" 2>/dev/null
    fi
}

_read_override_commit() {
    # Try git first (available in openfold-cli image which installs git).
    # Use safe.directory to avoid "dubious ownership" on bind mounts owned by
    # a different UID than the container's root user.
    if command -v git >/dev/null 2>&1; then
        local c
        c="$(git -C "$_INSTALL_ORCH_ROOT" -c safe.directory='*' rev-parse HEAD 2>/dev/null || true)"
        if [[ -n "$c" ]]; then
            echo "$c"
            return
        fi
    fi
    # Git-free fallback: parse .git/HEAD -> ref -> ref file, or detached SHA.
    # Do NOT strip whitespace from HEAD content — the space in "ref: refs/..."
    # is significant and ${head_content#ref: } depends on it.
    local head_file="${_INSTALL_ORCH_ROOT}/.git/HEAD"
    if [[ -r "$head_file" ]]; then
        local head_content
        # $(...) strips trailing newlines only; spaces are preserved.
        head_content="$(cat "$head_file" 2>/dev/null)"
        if [[ "$head_content" =~ ^[0-9a-f]{40}$ ]]; then
            # Detached HEAD at a SHA
            echo "$head_content"
            return
        elif [[ "$head_content" == ref:* ]]; then
            local ref="${head_content#ref: }"
            local ref_file="${_INSTALL_ORCH_ROOT}/.git/${ref}"
            if [[ -r "$ref_file" ]]; then
                local sha
                sha="$(cat "$ref_file" 2>/dev/null)"
                if [[ "$sha" =~ ^[0-9a-f]{40}$ ]]; then
                    echo "$sha"
                    return
                fi
            fi
            # Packed refs fallback
            local packed="${_INSTALL_ORCH_ROOT}/.git/packed-refs"
            if [[ -r "$packed" ]]; then
                local sha
                sha="$(grep -E "^[0-9a-f]{40} ${ref}$" "$packed" 2>/dev/null | cut -d' ' -f1 || true)"
                if [[ -n "$sha" ]]; then
                    echo "$sha"
                    return
                fi
            fi
        fi
    fi
    # Could not determine commit
    echo ""
}

# --- Mode helpers ---

_set_baked() {
    export BSPP_ORCHESTRATION_SOURCE="baked"
    _commit="$(_read_baked_commit)"
    if [[ -n "$_commit" ]]; then
        export BSPP_ORCHESTRATION_PROVENANCE_COMMIT="$_commit"
    else
        unset BSPP_ORCHESTRATION_PROVENANCE_COMMIT
    fi
}

_mount_present() {
    [[ -e "$_INSTALL_ORCH_ROOT" ]]
}

_mount_valid() {
    [[ -f "$_ORCH_CONTRACT_TOML" && -f "$_ORCH_CONTROL_TOML" && -f "$_ORCH_RUNTIME_TOML" ]]
}

# --- Mode selection ---

# Opt-out: when BSPP_INSTALL_MODE_SKIP=1, skip mount detection entirely.
# Used by the benchmark submit path to preserve baked-wheel determinism.
# The guard must be set in the container environment BEFORE the entrypoint
# sources this helper (e.g., via `env BSPP_INSTALL_MODE_SKIP=1 entrypoint.sh`
# in the srun command, NOT in a heredoc that runs after the entrypoint).
if [[ "${BSPP_INSTALL_MODE_SKIP:-0}" == "1" ]]; then
    _set_baked
    return 0 2>/dev/null || exit 0  # sourced: return; executed: exit cleanly
fi

if [[ "${BSPP_ORCHESTRATION_DEV_MOUNT:-0}" != "1" ]]; then
    # Baked-first default. A present mount without the explicit dev flag fails
    # closed rather than silently overriding the baked source.
    if _mount_present; then
        echo "[entrypoint] orchestration source mount present at ${_INSTALL_ORCH_ROOT} but dev-mount is not enabled;" >&2
        echo "[entrypoint] set BSPP_ORCHESTRATION_DEV_MOUNT=1 to use it, or remove the mount to run the baked source" >&2
        exit 1
    fi
    _set_baked
    return 0 2>/dev/null || exit 0
fi

# Dev-mount explicitly requested.
if ! _mount_present; then
    echo "[entrypoint] BSPP_ORCHESTRATION_DEV_MOUNT=1 but no orchestration source mount at ${_INSTALL_ORCH_ROOT}" >&2
    exit 1
fi
if ! _mount_valid; then
    # Mount path exists but is not a valid orchestration source — fail closed.
    echo "[entrypoint] orchestration source mount present but incomplete: \
missing pyproject.toml under packages/" >&2
    exit 1
fi

# OVERRIDE mode
export BSPP_ORCHESTRATION_SOURCE="override"
_commit="$(_read_override_commit)"
if [[ -n "$_commit" ]]; then
    export BSPP_ORCHESTRATION_PROVENANCE_COMMIT="$_commit"
else
    unset BSPP_ORCHESTRATION_PROVENANCE_COMMIT
fi

# Commit agreement check: the mounted source must match the baked commit
# unless the explicit dev-only mismatch override is set. An undeterminable
# mounted commit OR an undeterminable baked commit fails closed rather than
# silently substituting unverifiable source for the baked wheels.
if [[ "${BSPP_ORCHESTRATION_DEV_MOUNT_ALLOW_MISMATCH:-0}" != "1" ]]; then
    _baked="$(_read_baked_commit)"
    if [[ -z "$_baked" ]]; then
        echo "[entrypoint] baked orchestration source commit is undeterminable; refusing to substitute a mounted source without a verifiable baked commit" >&2
        echo "[entrypoint] set BSPP_ORCHESTRATION_DEV_MOUNT_ALLOW_MISMATCH=1 to override" >&2
        exit 1
    fi
    if [[ -z "$_commit" ]]; then
        echo "[entrypoint] mounted orchestration source commit is undeterminable; refusing to substitute unknown source for the baked wheels" >&2
        echo "[entrypoint] set BSPP_ORCHESTRATION_DEV_MOUNT_ALLOW_MISMATCH=1 to override" >&2
        exit 1
    fi
    if [[ "$_commit" != "$_baked" ]]; then
        echo "[entrypoint] mounted orchestration source commit ${_commit} disagrees with baked commit ${_baked}" >&2
        echo "[entrypoint] set BSPP_ORCHESTRATION_DEV_MOUNT_ALLOW_MISMATCH=1 to override" >&2
        exit 1
    fi
fi

# --no-build-isolation: use the hatchling already baked into the image
# instead of trying to download it from the index (network-constrained
# Slurm/Pyxis environments have no outbound PyPI access).
python -m pip install --no-deps --no-build-isolation --root-user-action=ignore \
    --config-settings editable_mode=strict \
    -e "${_INSTALL_ORCH_ROOT}/packages/orchestration-contract" >/dev/null
python -m pip install --no-deps --no-build-isolation --root-user-action=ignore \
    --config-settings editable_mode=strict \
    -e "${_INSTALL_ORCH_ROOT}/packages/orchestration-control" >/dev/null
python -m pip install --no-deps --no-build-isolation --root-user-action=ignore \
    --config-settings editable_mode=strict \
    -e "${_INSTALL_ORCH_ROOT}/packages/orchestration-runtime" >/dev/null
