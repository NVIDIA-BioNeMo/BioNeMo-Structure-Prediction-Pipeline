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

PYTHON="/opt/bspp/environment/bin/python"
TIMEOUT="/opt/bspp/environment/bin/timeout"
SLEEP="/opt/bspp/environment/bin/sleep"
TAIL="/opt/bspp/environment/bin/tail"
MMSEQS="/usr/local/bin/mmseqs"
COLABFOLD_SEARCH="/usr/local/bin/colabfold_search"
CUDA_DRIVER_PROBE="/opt/bspp/bin/bspp-preprocessing-cuda-driver-probe"

fixture="${1:?usage: bspp-preprocessing-carry-characterization FIXTURE}"
server=""

cleanup() {
  if [[ -n "$server" ]]; then
    kill "$server" 2>/dev/null || true
    wait "$server" 2>/dev/null || true
  fi
}
trap cleanup EXIT

server_failure() {
  local status=0
  wait "$server" || status=$?
  "$TAIL" -n 120 "$fixture/gpuserver.log" >&2 || true
  if [[ "$status" -eq 0 ]]; then
    status=1
  fi
  return "$status"
}

"$PYTHON" "$CUDA_DRIVER_PROBE" "$fixture/cuda-driver-evidence.json"

"$MMSEQS" gpuserver "${BSPP_CARRY_DATABASE_ROOT}/uniref30_2302_db" \
  --db-load-mode 0 --prefilter-mode 1 --max-seqs 10000 \
  >"$fixture/gpuserver.log" 2>&1 &
server=$!

for ((warmup_second = 0; warmup_second < 60; warmup_second++)); do
  if ! kill -0 "$server" 2>/dev/null; then
    echo "gpuserver exited during the 60-second warmup" >&2
    server_status=0
    server_failure || server_status=$?
    exit "$server_status"
  fi
  "$SLEEP" 1
done

"$TIMEOUT" --signal=TERM --kill-after=10s 1800s \
  "$COLABFOLD_SEARCH" --mmseqs "$MMSEQS" \
  "$fixture/remaining.fa" \
  "$BSPP_CARRY_DATABASE_ROOT" \
  "$fixture/search-output" \
  --use-env 0 --pairing_strategy 1 --pair-mode paired --filter 2 \
  --db-load-mode 2 --gpu 1 --threads 64 \
  --gpu-server 1 \
  --db1 uniref30_2302_db --db3 colabfold_envdb_202108_db &
search_supervisor=$!

while kill -0 "$search_supervisor" 2>/dev/null; do
  if ! kill -0 "$server" 2>/dev/null; then
    echo "gpuserver exited while the bounded search was running" >&2
    server_status=0
    wait "$server" || server_status=$?
    kill -TERM "$search_supervisor" 2>/dev/null || true
    wait "$search_supervisor" 2>/dev/null || true
    "$TAIL" -n 120 "$fixture/gpuserver.log" >&2 || true
    if [[ "$server_status" -eq 0 ]]; then
      server_status=1
    fi
    exit "$server_status"
  fi
  "$SLEEP" 1
done

search_status=0
wait "$search_supervisor" || search_status=$?
if [[ "$search_status" -ne 0 ]]; then
  echo "bounded ColabFold search failed with status ${search_status}" >&2
  "$TAIL" -n 120 "$fixture/gpuserver.log" >&2 || true
  exit "$search_status"
fi
if ! kill -0 "$server" 2>/dev/null; then
  echo "gpuserver exited before adapter-owned shutdown" >&2
  server_status=0
  server_failure || server_status=$?
  exit "$server_status"
fi

if ! kill -TERM "$server" 2>/dev/null; then
  echo "gpuserver exited during the adapter-owned shutdown handoff" >&2
  server_status=0
  server_failure || server_status=$?
  exit "$server_status"
fi
wait "$server" 2>/dev/null || true
server=""
trap - EXIT
