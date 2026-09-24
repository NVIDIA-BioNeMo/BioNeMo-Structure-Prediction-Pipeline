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

# Sourceable, phase-agnostic Slurm job-monitoring primitives.
#
# These run on a Slurm login node and call squeue/sacct directly. This file
# performs no work on source; source it and call the functions.
#
# Usage (on a login node):
#   source containers/scripts/slurm-job-monitor.sh
#   bspp_monitor_slurm_job 12345 30
#
# Usage (from a workstation, over SSH):
#   ssh <target> "set -e; source '<checkout>/containers/scripts/slurm-job-monitor.sh'; \
#                 bspp_monitor_slurm_job '12345' '30'"
#
# `bspp_monitor_slurm_job` returns 0 only when Slurm accounting conclusively
# reports COMPLETED for the exact job id; any other terminal state, a non-zero
# exit code, or inconclusive accounting returns non-zero.

if [[ -n "${BSPP_SLURM_JOB_MONITOR_SOURCED:-}" ]]; then
  return 0 2>/dev/null || exit 0
fi
readonly BSPP_SLURM_JOB_MONITOR_SOURCED=1

bspp_slurm_terminal_state() {
  case "${1%% *}" in
    COMPLETED|CANCELLED|FAILED|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|BOOT_FAIL|TIMEOUT)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

bspp_job_id_is_valid() {
  [[ "${1:-}" =~ ^[1-9][0-9]*$ ]]
}

bspp_monitor_slurm_job() {
  local job_id="${1:?job id is required}"
  local poll_seconds="${2:-15}"
  local state="" exit_code="" accounting="" top_row="" last_state="" queue_rows=""
  local attempt

  bspp_job_id_is_valid "$job_id" || {
    printf 'slurm-monitor: invalid job id: %s\n' "$job_id" >&2
    return 1
  }
  [[ "$poll_seconds" =~ ^[1-9][0-9]*$ ]] || {
    printf 'slurm-monitor: invalid poll seconds: %s\n' "$poll_seconds" >&2
    return 1
  }
  command -v squeue >/dev/null 2>&1 || {
    printf 'slurm-monitor: squeue is required\n' >&2
    return 1
  }
  command -v sacct >/dev/null 2>&1 || {
    printf 'slurm-monitor: sacct is required\n' >&2
    return 1
  }

  # Poll the live queue until the job leaves it. A job is conclusively terminal
  # only once it is absent from the live queue AND Slurm accounting publishes a
  # terminal state with an exact exit code.
  while :; do
    queue_rows="$(squeue --noheader --jobs "$job_id" --format='%i' 2>/dev/null || true)"
    [[ "$queue_rows" =~ [^[:space:]] ]] || break
    squeue --noheader --jobs "$job_id" --format='job=%i state=%T elapsed=%M node=%N' 2>/dev/null || true
    sleep "$poll_seconds"
  done

  # The job has left the live queue; require conclusive terminal accounting.
  # Accounting can lag the queue by a few seconds, so retry a bounded number of
  # times before declaring the observation inconclusive.
  for ((attempt = 1; attempt <= 12; attempt++)); do
    accounting="$(sacct -X -j "$job_id" --noheader --parsable2 \
      --format=JobIDRaw,State,ExitCode,Elapsed,NodeList 2>/dev/null || true)"
    top_row="$(printf '%s\n' "$accounting" | awk -F'|' -v id="$job_id" '$1 == id {print; exit}')"
    state=""
    exit_code=""
    if [[ -n "$top_row" ]]; then
      IFS='|' read -r _ state exit_code _ _ <<<"$top_row"
    fi
    [[ -z "$state" ]] || last_state="$state"
    if [[ -n "$exit_code" && ! "$exit_code" =~ ^[0-9]+:[0-9]+$ ]]; then
      printf 'slurm-monitor: Slurm accounting returned an invalid ExitCode: %s\n' "$exit_code" >&2
      return 1
    fi
    if [[ -n "$state" && -n "$exit_code" ]] && bspp_slurm_terminal_state "$state"; then
      printf '%s\n' "$top_row"
      if [[ "$state" == "COMPLETED" ]]; then
        return 0
      fi
      printf 'slurm-monitor: job %s reached terminal state %s (exit %s)\n' \
        "$job_id" "$state" "$exit_code" >&2
      return 1
    fi
    sleep 5
  done

  if [[ -n "$last_state" ]]; then
    printf 'slurm-monitor: Slurm accounting did not publish conclusive terminal evidence for job %s after 12 attempts; last state: %s\n' \
      "$job_id" "$last_state" >&2
  else
    printf 'slurm-monitor: Slurm accounting did not publish conclusive terminal evidence for job %s after 12 attempts\n' \
      "$job_id" >&2
  fi
  return 1
}
