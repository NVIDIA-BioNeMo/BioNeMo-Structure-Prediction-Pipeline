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

# Source the shared orchestration-source install-mode helper.
# Must run before the CUDA compat check so the early-return exec path
# (compat dir absent) also gets override/baked mode + provenance.
source /usr/local/bin/install-orchestration-source.sh

CUDA_COMPAT_DIR="${BSPP_CUDA_COMPAT_DIR:?BSPP_CUDA_COMPAT_DIR must be set}"
CUDA_COMPAT_LIBRARY="${CUDA_COMPAT_DIR}/libcuda.so.1"

# Graceful degradation: if the compat dir is absent, warn but do not fail.
# The colabfold base image may carry its own CUDA runtime.
if [[ ! -d "$CUDA_COMPAT_DIR" ]]; then
  echo "WARNING: CUDA compatibility directory is absent: ${CUDA_COMPAT_DIR} — continuing without compat" >&2
  exec "$@"
fi

[[ -r "$CUDA_COMPAT_DIR" && -x "$CUDA_COMPAT_DIR" ]] || {
  echo "CUDA compatibility directory is not accessible: ${CUDA_COMPAT_DIR}" >&2
  exit 126
}
[[ -L "$CUDA_COMPAT_LIBRARY" ]] || {
  echo "CUDA compatibility library symlink is missing: ${CUDA_COMPAT_LIBRARY}" >&2
  exit 126
}
CUDA_COMPAT_DIR_RESOLVED="$(readlink -f -- "$CUDA_COMPAT_DIR")" || {
  echo "cannot resolve CUDA compatibility directory: ${CUDA_COMPAT_DIR}" >&2
  exit 126
}
CUDA_COMPAT_LIBRARY_RESOLVED="$(readlink -f -- "$CUDA_COMPAT_LIBRARY")" || {
  echo "cannot resolve CUDA compatibility library: ${CUDA_COMPAT_LIBRARY}" >&2
  exit 126
}
[[ -f "$CUDA_COMPAT_LIBRARY_RESOLVED" && -r "$CUDA_COMPAT_LIBRARY_RESOLVED" ]] || {
  echo "CUDA compatibility target is not a readable regular file: ${CUDA_COMPAT_LIBRARY_RESOLVED}" >&2
  exit 126
}
case "$CUDA_COMPAT_LIBRARY_RESOLVED" in
  "$CUDA_COMPAT_DIR_RESOLVED"/*) ;;
  *)
    echo "CUDA compatibility target escapes its directory: ${CUDA_COMPAT_LIBRARY_RESOLVED}" >&2
    exit 126
    ;;
esac

export LD_LIBRARY_PATH="${CUDA_COMPAT_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec "$@"
