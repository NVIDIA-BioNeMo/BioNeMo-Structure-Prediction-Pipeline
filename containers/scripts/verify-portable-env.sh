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
# verify-portable-env.sh — verify (and, with --repair, fix) that the selected
# venv resolves under the repo-local .uv-python/ so it works identically on the
# sandbox and the host (shared mount, same absolute path).
#
# Background: a `uv` command run WITHOUT UV_PYTHON_INSTALL_DIR=$PWD/.uv-python
# silently re-links .venv/bin/python to the invoking user's uv cache
# (a user-local uv cache such as <user>/.local/share/uv/python/...), which does
# not exist on the host. The README environment-setup section is the
# canonical source.
# This script is the established practice: run it before a live test (the
# live-test driver's preflight also checks this as a backstop).
#
# Usage:
#   containers/scripts/verify-portable-env.sh           # check only (exit 1 if broken)
#   containers/scripts/verify-portable-env.sh --repair  # check, and repair in place
#   BSPP_VENV_DIR=.venv/shared containers/scripts/verify-portable-env.sh
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${BSPP_VENV_DIR:-${REPO_ROOT}/.venv}"
if [[ "$VENV_DIR" != /* ]]; then
    VENV_DIR="${REPO_ROOT}/${VENV_DIR}"
fi
VENV_PY="${VENV_DIR}/bin/python"
PYVENV_CFG="${VENV_DIR}/pyvenv.cfg"
UV_PY_DIR="${REPO_ROOT}/.uv-python"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[[ -e "$VENV_PY" ]] || die "no venv at ${VENV_DIR} — create it per the README environment-setup section"

# Find the repo-local managed interpreter (the portable base).
uv_py="$(find "$UV_PY_DIR" -maxdepth 3 -type f -name 'python3.1*' 2>/dev/null | sort | head -1 || true)"
[[ -n "$uv_py" ]] || die "no managed CPython under ${UV_PY_DIR} — run 'UV_PYTHON_INSTALL_DIR=\$PWD/.uv-python uv python install 3.12' first"

resolved="$(readlink -f "$VENV_PY" 2>/dev/null || true)"

if [[ "$resolved" == "$UV_PY_DIR"/* ]]; then
    echo "OK (portable): ${VENV_PY} -> $resolved"
    exit 0
fi

echo "NOT PORTABLE: ${VENV_PY} -> ${resolved:-<unresolved>}"

if [[ "${1:-}" != "--repair" ]]; then
    echo "Run with --repair to re-point the venv to the repo-local base."
    exit 1
fi

# Repair: re-point the interpreter symlink and the pyvenv.cfg home to the
# repo-local base. This preserves the already-installed packages (site-packages
# is untouched) and makes the venv resolve identically on both machines.
ln -sfn "$uv_py" "$VENV_PY"
sed -i "s|^home = .*|home = ${UV_PY_DIR}/$(basename "$(dirname "$(dirname "$uv_py")")")/bin|" "$PYVENV_CFG"

new_resolved="$(readlink -f "$VENV_PY" 2>/dev/null || true)"
[[ "$new_resolved" == "$UV_PY_DIR"/* ]] || die "repair failed: still resolves to $new_resolved"
"$VENV_PY" --version >/dev/null

echo "REPAIRED (portable): ${VENV_PY} -> $new_resolved"
echo "Interpreter: $("$VENV_PY" --version 2>&1)"
