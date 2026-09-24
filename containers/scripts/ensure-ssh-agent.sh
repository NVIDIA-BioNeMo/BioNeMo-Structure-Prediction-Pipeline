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
# Ensure an SSH agent socket with a loaded key exists for Docker BuildKit
# --ssh mounts. Idempotent: reuses an existing agent; starts one only when
# SSH_AUTH_SOCK is unset. Key selection: BSPP_SSH_KEY (an explicit private
# key path) wins; otherwise default ~/.ssh/id_* keys are added only when the
# agent is empty.
#
# Source this from build/push scripts that run `docker build --ssh default`.
# =============================================================================
set -euo pipefail

if [[ -z "${SSH_AUTH_SOCK:-}" ]]; then
    echo "SSH_AUTH_SOCK unset; starting a transient ssh-agent for this build..."
    eval "$(ssh-agent -s)" >/dev/null
    trap 'if [[ -n "${SSH_AGENT_PID:-}" ]]; then kill "${SSH_AGENT_PID}" >/dev/null 2>&1 || true; fi' EXIT
fi

if [[ -n "${BSPP_SSH_KEY:-}" ]]; then
    echo "Adding explicit SSH key from BSPP_SSH_KEY: ${BSPP_SSH_KEY}"
    ssh-add "${BSPP_SSH_KEY}" || {
        echo "ERROR: ssh-add ${BSPP_SSH_KEY} failed. Verify the key path and that the key is authorized for the remote." >&2
        exit 1
    }
elif ! ssh-add -l >/dev/null 2>&1; then
    echo "SSH agent has no keys; adding default keys (~/.ssh/id_*)..."
    ssh-add || {
        echo "ERROR: ssh-add failed. Set BSPP_SSH_KEY=/path/to/key and retry." >&2
        exit 1
    }
fi
