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

usage() {
  cat >&2 <<'EOF'
Usage: submit-acceptance-checks.sh --parity-script PATH --semantic-script PATH --verify-script PATH [--dependency afterok:JOB[:JOB...]]
EOF
}

PARITY_SCRIPT=""
SEMANTIC_SCRIPT=""
VERIFY_SCRIPT=""
DEPENDENCY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --parity-script)
      PARITY_SCRIPT="${2:?missing --parity-script value}"
      shift 2
      ;;
    --semantic-script)
      SEMANTIC_SCRIPT="${2:?missing --semantic-script value}"
      shift 2
      ;;
    --verify-script)
      VERIFY_SCRIPT="${2:?missing --verify-script value}"
      shift 2
      ;;
    --dependency)
      DEPENDENCY="${2:?missing --dependency value}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

[[ -n "$PARITY_SCRIPT" ]] || { usage; exit 2; }
[[ -n "$SEMANTIC_SCRIPT" ]] || { usage; exit 2; }
[[ -n "$VERIFY_SCRIPT" ]] || { usage; exit 2; }

submit_args=(--parsable)
if [[ -n "$DEPENDENCY" ]]; then
  submit_args+=(--dependency "$DEPENDENCY")
fi

parity_job="$(sbatch "${submit_args[@]}" "$PARITY_SCRIPT")"
semantic_job="$(sbatch "${submit_args[@]}" "$SEMANTIC_SCRIPT")"
verify_job="$(sbatch --parsable --dependency "afterok:${parity_job}:${semantic_job}" "$VERIFY_SCRIPT")"

printf 'acceptance_jobs:\n'
printf '  parity: %s\n' "$parity_job"
printf '  semantic: %s\n' "$semantic_job"
printf '  verify: %s\n' "$verify_job"
