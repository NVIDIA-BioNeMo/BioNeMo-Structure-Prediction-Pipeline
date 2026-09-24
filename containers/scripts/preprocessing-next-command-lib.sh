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

# Sourceable Phase-A handoff primitives.  This file performs no work on source.

if [[ -n "${BSPP_PREPROCESSING_NEXT_COMMAND_LIB_SOURCED:-}" ]]; then
  return 0 2>/dev/null || exit 0
fi
readonly BSPP_PREPROCESSING_NEXT_COMMAND_LIB_SOURCED=1

bspp_handoff_error() {
  printf 'preprocessing-handoff: %s\n' "$*" >&2
  return 1
}

bspp_require_command() {
  command -v "$1" >/dev/null 2>&1 || bspp_handoff_error "required command is unavailable: $1"
}

bspp_require_regular_file() {
  local path="${1:?path is required}"
  [[ -f "$path" && ! -L "$path" ]] || bspp_handoff_error "not a regular non-symlink file: $path"
}

bspp_sha256() {
  local path="${1:?path is required}"
  bspp_require_regular_file "$path" || return 1
  bspp_require_command sha256sum || return 1
  sha256sum -- "$path" | awk '{print $1}'
}

bspp_verify_sha256() {
  local path="${1:?path is required}"
  local expected="${2:?expected SHA-256 is required}"
  local observed
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || return 1
  observed="$(bspp_sha256 "$path")" || return 1
  [[ "$observed" == "$expected" ]] || bspp_handoff_error "SHA-256 mismatch for $path"
}

bspp_verify_mode() {
  local path="${1:?path is required}"
  local expected="${2:?mode is required}"
  local normalized observed
  [[ "$expected" =~ ^0?[0-7]{3}$ ]] || return 1
  normalized="${expected#0}"
  observed="$(stat -c '%a' -- "$path")" || return 1
  [[ "$observed" == "$normalized" ]] || bspp_handoff_error "mode mismatch for $path: expected $normalized, observed $observed"
}

bspp_verify_library_self() {
  local expected="${1:?expected library SHA-256 is required}"
  local library_path="${2:-${BASH_SOURCE[0]}}"
  bspp_verify_sha256 "$library_path" "$expected"
}

bspp_write_once() {
  local destination="${1:?destination is required}"
  local mode="${2:?mode is required}"
  local content="${3-}"
  local parent temporary
  parent="$(dirname -- "$destination")"
  mkdir -p -- "$parent" || return 1
  if [[ -e "$destination" || -L "$destination" ]]; then
    [[ -f "$destination" && ! -L "$destination" ]] || return 1
    [[ "$(<"$destination")" == "$content" ]] || {
      bspp_handoff_error "immutable record differs: $destination"
      return 1
    }
    bspp_verify_mode "$destination" "$mode" || return 1
    return
  fi
  temporary="$(mktemp "$parent/.bspp-write-once.XXXXXX")" || return 1
  printf '%s\n' "$content" >"$temporary" || { rm -f -- "$temporary"; return 1; }
  chmod "$mode" "$temporary" || { rm -f -- "$temporary"; return 1; }
  if ! ln -- "$temporary" "$destination" 2>/dev/null; then
    rm -f -- "$temporary"
    [[ -f "$destination" && ! -L "$destination" && "$(<"$destination")" == "$content" ]] || {
      bspp_handoff_error "could not publish immutable record: $destination"
      return 1
    }
    bspp_verify_mode "$destination" "$mode"
    return
  fi
  rm -f -- "$temporary"
  bspp_verify_mode "$destination" "$mode"
}

bspp_stage_verified() {
  local source="${1:?source is required}"
  local expected="${2:?expected SHA-256 is required}"
  local destination="${3:?destination is required}"
  local mode="${4:-0400}"
  local parent temporary
  bspp_verify_sha256 "$source" "$expected" || return 1
  parent="$(dirname -- "$destination")"
  mkdir -p -- "$parent" || return 1
  if [[ -e "$destination" || -L "$destination" ]]; then
    bspp_verify_sha256 "$destination" "$expected" || return 1
    bspp_verify_mode "$destination" "$mode" || return 1
    return
  fi
  temporary="$(mktemp "$parent/.bspp-stage.XXXXXX")" || return 1
  if ! cp -- "$source" "$temporary" || ! chmod "$mode" "$temporary" || \
    ! bspp_verify_sha256 "$temporary" "$expected"; then
    rm -f -- "$temporary"
    return 1
  fi
  if ! ln -- "$temporary" "$destination" 2>/dev/null; then
    rm -f -- "$temporary"
    bspp_verify_sha256 "$destination" "$expected" || return 1
    bspp_verify_mode "$destination" "$mode"
    return
  fi
  rm -f -- "$temporary"
  bspp_verify_sha256 "$destination" "$expected" || return 1
  bspp_verify_mode "$destination" "$mode"
}

bspp_bounded_copy() {
  local source="${1:?source is required}"
  local destination="${2:?destination is required}"
  local maximum_bytes="${3:?maximum byte count is required}"
  local expected="${4:?expected SHA-256 is required}"
  local observed_size
  [[ "$maximum_bytes" =~ ^[1-9][0-9]*$ ]] || return 1
  bspp_require_regular_file "$source" || return 1
  observed_size="$(stat -c '%s' -- "$source")" || return 1
  (( observed_size <= maximum_bytes )) || {
    bspp_handoff_error "bounded copy source exceeds $maximum_bytes bytes: $source"
    return 1
  }
  bspp_stage_verified "$source" "$expected" "$destination" 0400
}

bspp_compare_captured_command() {
  local rendered="${1:?rendered command record is required}"
  local captured="${2:?captured command record is required}"
  bspp_require_regular_file "$rendered" || return 1
  bspp_require_regular_file "$captured" || return 1
  cmp -s -- "$rendered" "$captured" || bspp_handoff_error "captured command differs from rendered intent"
}

bspp_job_id_is_valid() {
  [[ "${1:-}" =~ ^[1-9][0-9]*$ ]]
}

bspp_discover_job() {
  local state_root="${1:?state root is required}"
  local job_name="${2:?job name is required}"
  local job_id_file="$state_root/slurm-job-id"
  local submission_intent="$state_root/submission.intent"
  local job_id matches accounting historical_start intent_epoch
  if [[ -f "$job_id_file" && ! -L "$job_id_file" ]]; then
    bspp_verify_mode "$job_id_file" 0400 || return 2
    job_id="$(<"$job_id_file")"
    bspp_job_id_is_valid "$job_id" || {
      bspp_handoff_error "invalid persisted Slurm job id in $job_id_file"
      return 2
    }
    printf '%s\n' "$job_id"
    return
  fi
  bspp_require_command squeue || return 2
  matches="$(squeue --noheader --name "$job_name" --format='%i' | awk 'NF {print $1}')" || return 2
  if [[ -n "$matches" ]]; then
    [[ "$(printf '%s\n' "$matches" | awk 'NF {count++} END {print count+0}')" == 1 ]] || {
      bspp_handoff_error "multiple live jobs match $job_name"
      return 2
    }
    job_id="$(printf '%s\n' "$matches" | awk 'NF {print; exit}')"
    bspp_job_id_is_valid "$job_id" || return 2
    bspp_write_once "$job_id_file" 0400 "$job_id" || return 2
    printf '%s\n' "$job_id"
    return
  fi
  [[ -e "$submission_intent" || -L "$submission_intent" ]] || return 1
  [[ -f "$submission_intent" && ! -L "$submission_intent" ]] || return 2
  bspp_verify_mode "$submission_intent" 0400 || return 2
  [[ "$(<"$submission_intent")" == *" job_name=$job_name" ]] || {
    bspp_handoff_error "submission intent job name differs from requested discovery name"
    return 2
  }
  bspp_require_command sacct || return 2
  bspp_require_command date || return 2
  intent_epoch="$(stat -c '%Y' -- "$submission_intent")" || return 2
  historical_start="$(date -d "@$intent_epoch" '+%Y-%m-%dT%H:%M:%S')" || return 2
  accounting="$(sacct -X --noheader --parsable2 --name "$job_name" \
    --starttime="$historical_start" --format=JobIDRaw,JobName,State)" || return 2
  matches="$(printf '%s\n' "$accounting" | awk -F'|' -v name="$job_name" \
    '$1 ~ /^[1-9][0-9]*$/ && $2 == name {seen[$1]=1} END {for (id in seen) print id}')"
  [[ -n "$matches" ]] || {
    bspp_handoff_error "submission intent exists but no historical job matches $job_name"
    return 2
  }
  [[ "$(printf '%s\n' "$matches" | awk 'NF {count++} END {print count+0}')" == 1 ]] || {
    bspp_handoff_error "multiple historical jobs match $job_name"
    return 2
  }
  job_id="$(printf '%s\n' "$matches" | awk 'NF {print; exit}')"
  bspp_job_id_is_valid "$job_id" || return 2
  bspp_write_once "$job_id_file" 0400 "$job_id" || return 2
  printf '%s\n' "$job_id"
}

bspp_submit_rendered_sbatch() (
  local script="${1:?rendered sbatch script is required}"
  local expected="${2:?rendered script SHA-256 is required}"
  local state_root="${3:?state root is required}"
  local job_name="${4:?job name is required}"
  local lock_file="$state_root/submission.lock"
  local job_id_file="$state_root/slurm-job-id"
  local submission_intent="$state_root/submission.intent"
  local submission_result="$state_root/submission.result"
  local response job_id rendered_names rendered_job_name
  bspp_verify_sha256 "$script" "$expected" || return 1
  rendered_names="$(sed -n 's/^#SBATCH[[:space:]]\+--job-name=//p' -- "$script")" || return 1
  [[ "$(printf '%s\n' "$rendered_names" | awk 'NF {count++} END {print count+0}')" == 1 ]] || {
    bspp_handoff_error "rendered sbatch must contain exactly one job-name directive"
    return 1
  }
  rendered_job_name="$(printf '%s\n' "$rendered_names" | awk 'NF {print; exit}')"
  [[ "$rendered_job_name" == "$job_name" ]] || {
    bspp_handoff_error "passed job name differs from rendered sbatch directive"
    return 1
  }
  bspp_require_command flock || return 1
  bspp_require_command sbatch || return 1
  mkdir -p -- "$state_root" || return 1
  exec {bspp_submission_lock_fd}>"$lock_file" || return 1
  flock -x "$bspp_submission_lock_fd" || return 1
  if [[ -e "$submission_intent" || -L "$submission_intent" ]]; then
    bspp_write_once "$submission_intent" 0400 "script_sha256=$expected job_name=$job_name" || return 2
  fi
  local discovery_status
  if job_id="$(bspp_discover_job "$state_root" "$job_name")"; then
    printf '%s\n' "$job_id"
    flock -u "$bspp_submission_lock_fd"
    exec {bspp_submission_lock_fd}>&-
    return
  else
    discovery_status=$?
  fi
  [[ "$discovery_status" -eq 1 ]] || return "$discovery_status"
  bspp_write_once "$submission_intent" 0400 "script_sha256=$expected job_name=$job_name" || return 1
  response="$(sbatch --parsable -- "$script")" || {
    bspp_write_once "$state_root/submission.failure" 0400 "sbatch failed" || true
    return 1
  }
  job_id="${response%%;*}"
  bspp_job_id_is_valid "$job_id" || return 1
  bspp_write_once "$job_id_file" 0400 "$job_id" || return 1
  bspp_write_once "$submission_result" 0400 "submitted job_id=$job_id script_sha256=$expected" || return 1
  printf '%s\n' "$job_id"
  flock -u "$bspp_submission_lock_fd"
  exec {bspp_submission_lock_fd}>&-
)

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

bspp_monitor_job() {
  local job_id="${1:?job id is required}"
  local state_root="${2:?state root is required}"
  local poll_seconds="${3:-15}"
  local state="" exit_code="" elapsed="" node="" accounting="" top_row="" last_state=""
  local queue_rows="" attempt record
  bspp_job_id_is_valid "$job_id" || return 1
  [[ "$poll_seconds" =~ ^[1-9][0-9]*$ ]] || return 1
  bspp_require_command squeue || return 1
  bspp_require_command sacct || return 1
  while :; do
    queue_rows="$(squeue --noheader --jobs "$job_id" --format='%i')" || {
      bspp_handoff_error "queue query failed for job $job_id; presence unknown, deferring to accounting" || true
      break
    }
    [[ "$queue_rows" =~ [^[:space:]] ]] || break
    squeue --noheader --jobs "$job_id" --format='job=%i state=%T elapsed=%M node=%N' || return 1
    sleep "$poll_seconds"
  done
  for ((attempt = 1; attempt <= 12; attempt++)); do
    state=""
    exit_code=""
    elapsed=""
    node=""
    top_row=""
    accounting="$(sacct -X -j "$job_id" --noheader --parsable2 \
      --format=JobIDRaw,State,ExitCode,Elapsed,NodeList)" || return 1
    top_row="$(printf '%s\n' "$accounting" | awk -F'|' -v id="$job_id" '$1 == id {print; exit}')"
    if [[ -n "$top_row" ]]; then
      IFS='|' read -r _ state exit_code elapsed node _ <<<"$top_row"
    fi
    [[ -z "$state" ]] || last_state="$state"
    if [[ -n "$exit_code" && ! "$exit_code" =~ ^[0-9]+:[0-9]+$ ]]; then
      bspp_handoff_error "Slurm accounting returned an invalid ExitCode: $exit_code"
      return 1
    fi
    if [[ -n "$state" && -n "$exit_code" && -n "$elapsed" && -n "$node" ]] \
      && bspp_slurm_terminal_state "$state"; then
      mkdir -p -- "$state_root" || return 1
      record="$(printf 'state=%s\nexit_code=%s\nelapsed=%s\nnode=%s' "$state" "$exit_code" "$elapsed" "$node")"
      bspp_write_once "$state_root/monitor.result" 0400 "$record" || return 1
      printf '%s\n' "$accounting"
      if [[ "$state" == "COMPLETED" ]]; then
        return 0
      fi
      return 1
    fi
    sleep 5
  done
  if [[ -n "$last_state" ]]; then
    bspp_handoff_error \
      "Slurm accounting did not publish conclusive terminal evidence for job $job_id after 12 attempts; last state: $last_state"
  else
    bspp_handoff_error \
      "Slurm accounting did not publish conclusive terminal evidence for job $job_id after 12 attempts"
  fi
  return 1
}

bspp_publish_phase_record() {
  local state_root="${1:?state root is required}"
  local phase="${2:?phase is required}"
  local kind="${3:?record kind is required}"
  local value="${4:?record value is required}"
  [[ "$phase" =~ ^[a-z0-9][a-z0-9-]*$ ]] || return 1
  [[ "$kind" == intent || "$kind" == result || "$kind" == job-id ]] || return 1
  bspp_write_once "$state_root/$phase.$kind" 0400 "$value"
}
