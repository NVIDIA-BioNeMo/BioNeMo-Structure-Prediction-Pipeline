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

set -euo pipefail

ORCHESTRATION_ROOT="${ORCHESTRATION_ROOT:-/workspace/bspp-orchestration}"
TOOLKIT_ROOT="${TOOLKIT_ROOT:-/workspace/AFDB-Integration-Kit}"
PYTHON_BIN="${BSPP_ORCHESTRATION_PYTHON:-${PYTHON_BIN:-/opt/bspp-orchestration-env/.pixi/envs/default/bin/python}}"

fail() {
  echo "BSPP semantic acceptance wrapper failed: $*" >&2
  exit 127
}

[[ -x "$PYTHON_BIN" ]] || fail "missing Python runtime: $PYTHON_BIN"
export PYTHONPATH="${ORCHESTRATION_ROOT}/packages/orchestration-contract/src:${ORCHESTRATION_ROOT}/packages/orchestration-runtime/src${PYTHONPATH:+:${PYTHONPATH}}"

exec "$PYTHON_BIN" -c "from bspp.orchestration.runtime.cli import cli; cli()" \
  validate semantic-acceptance "$@"
