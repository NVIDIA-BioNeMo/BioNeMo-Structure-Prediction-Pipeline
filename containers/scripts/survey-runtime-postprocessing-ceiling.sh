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
# Survey the bump ceiling for the runtime-postprocessing `postprocessing` variant.
#
# SCOPE: the `runtime-postprocessing` image family only (what containers/
# builds today). The PyG torch_cluster prebuilt-wheel ceiling exists because
# postprocessing Stage 13 needs the compiled torch_cluster.radius_graph CUDA
# path. The `runtime-preprocessing` and `runtime-folding-*` families (see
# devdocs/project-definition/PROJECT_MANIFEST.md, "Container and registry
# model") will have their own dependency ceilings and their own surveys when
# they land — never read this script's verdict as applying to them.
#
# What it does:
#   1. Reads the current pins from containers/variants/postprocessing.env.
#   2. Walks the PyG wheel index (data.pyg.org/whl) from newest page down and
#      finds the newest torch+cu page with a torch_cluster cp312 linux-x86_64
#      prebuilt wheel — the bump ceiling under the torch-cluster-first rule.
#   3. Control check: the CURRENT pin's page must contain the CURRENT pinned
#      wheel. If it does not, the parser (or upstream naming) changed and the
#      miss results cannot be trusted — the survey reports INVALID instead of
#      a wrong "at ceiling".
#   4. Cross-checks official PyTorch cuXXX wheels, nvidia/cuda devel base
#      image tags, and the available cuda-compat-13-* packages.
#   5. Prints a verdict: AT CEILING / BUMP AVAILABLE / BLOCKED-BY-PYTHON.
#
# Usage:
#   bash containers/scripts/survey-runtime-postprocessing-ceiling.sh
#
# Requirements: curl + outbound internet. Run on the host workstation (the
# sandbox DNS blocks data.pyg.org). Read-only; prints to stdout.
#
# Exit codes:
#   0 = a verdict was reached (see the final section of the output)
#   1 = survey invalid (PyG index unreachable, or control check failed)
# =============================================================================

set -uo pipefail

LOCAL_SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=resolve-repo-root.sh
source "${LOCAL_SCRIPT_DIR}/resolve-repo-root.sh"
REPO_ROOT="$(resolve_orchestration_repo_root)"
VARIANT_ENV="${REPO_ROOT}/containers/variants/postprocessing.env"

fetch() { curl -fsSL --max-time 60 -A "bspp-ceiling-survey/1.0" "$1" 2>/dev/null || true; }

pin() { grep -E "^$1=" "$VARIANT_ENV" | cut -d= -f2-; }

CUR_TORCH="$(pin TORCH_VERSION)"
CUR_TC="$(pin TORCH_CLUSTER_VERSION)"          # e.g. ==1.6.3+pt211cu130
CUR_TC_VER="${CUR_TC#==}"
CUR_PYG="$(pin PYG_WHEEL_INDEX)"
CUR_CUDA="$(pin CUDA_VERSION)"

echo "# runtime-postprocessing postprocessing-variant ceiling survey — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo
echo "== 0. Current pins ($(realpath --relative-to="$PWD" "$VARIANT_ENV" 2>/dev/null || echo "$VARIANT_ENV")) =="
grep -E '^(BASE_IMAGE|CUDA_VERSION|CUDA_COMPAT_PACKAGE|TORCH_VERSION|TORCH_CUDA_FLAVOR|PYG_WHEEL_INDEX|TORCH_CLUSTER_VERSION|TARGET_SMS)=' "$VARIANT_ENV"
echo

echo "== 1. PyG prebuilt wheel index — bump ceiling (torch-cluster-first rule) =="
INDEX="$(fetch https://data.pyg.org/whl/)"
if [[ -z "$INDEX" ]]; then
  echo "(data.pyg.org unreachable — survey INVALID; run from a host with outbound internet)"
  exit 1
fi
mapfile -t PAGES < <(printf '%s\n' "$INDEX" | grep -oE 'torch-[0-9]+\.[0-9]+\.[0-9]+%2Bcu[0-9]+\.html' | sort -uV)
TOTAL=${#PAGES[@]}
echo "total torch+cu pages: $TOTAL"
echo "newest 10 pages:"
for ((i=TOTAL-1; i>=0 && i>=TOTAL-10; i--)); do echo "  ${PAGES[$i]//%2B/+}"; done
echo
echo "-- probing newest pages for a torch_cluster cp312 linux-x86_64 wheel --"
CEILING_PAGE=""
CEILING_WHEEL=""
ANY_PAGE=""
ANY_WHEEL=""
CHECKED=0
for ((i=TOTAL-1; i>=0; i--)); do
  PAGE="${PAGES[$i]}"
  CHECKED=$((CHECKED+1))
  BODY="$(fetch "https://data.pyg.org/whl/$PAGE")"
  if [[ -z "$BODY" ]]; then
    echo "  [ERROR] ${PAGE//%2B/+} unreachable or empty — survey INVALID"
    exit 1
  fi
  CP312_WHEEL="$(printf '%s\n' "$BODY" | grep -oE 'torch_cluster-[0-9][^"<>]*cp312-cp312[^"<>]*x86_64\.whl' | sort -uV | tail -1)"
  if [[ -n "$CP312_WHEEL" ]]; then
    echo "  [HIT]  ${PAGE//%2B/+}  →  $CP312_WHEEL"
    if [[ -z "$CEILING_PAGE" ]]; then
      CEILING_PAGE="$PAGE"
      CEILING_WHEEL="$CP312_WHEEL"
    fi
    break
  else
    MISS_ANY="$(printf '%s\n' "$BODY" | grep -oE 'torch_cluster-[^"<>]*x86_64\.whl' | sort -uV | tail -1)"
    if [[ -n "$MISS_ANY" ]]; then
      echo "  [miss] ${PAGE//%2B/+}  (torch_cluster exists but NOT for cp312; newest any-tag: $MISS_ANY)"
      [[ -z "$ANY_PAGE" ]] && { ANY_PAGE="$PAGE"; ANY_WHEEL="$MISS_ANY"; }
    else
      CLUSTER_MENTIONS="$(printf '%s\n' "$BODY" | grep -c 'torch_cluster' || true)"
      echo "  [miss] ${PAGE//%2B/+}  (page ${#BODY} bytes, torch_cluster mentions: $CLUSTER_MENTIONS)"
    fi
  fi
done
echo

echo "== 2. Control check — current pin page must contain the pinned wheel =="
CONTROL="$(fetch "$CUR_PYG")"
NORMALIZED_CONTROL="${CONTROL//%2B/+}"
if printf '%s\n' "$NORMALIZED_CONTROL" \
  | grep -F "torch_cluster-${CUR_TC_VER}-cp312-cp312" \
  | grep -qF "x86_64.whl"; then
  echo "  OK: $CUR_PYG"
  echo "      contains torch_cluster-${CUR_TC_VER}-cp312-cp312-...x86_64.whl — parser trustworthy"
else
  echo "  FAIL: pinned wheel not found on its own index page: $CUR_PYG"
  echo "  The survey parser or the upstream wheel naming changed; do NOT trust the"
  echo "  miss results above. Survey INVALID."
  exit 1
fi
echo

echo "== 3. Official PyTorch wheels — newest cp312 linux-x86_64 per CUDA flavor =="
for CU in 121 124 126 128 129 130 131 132; do
  BODY="$(fetch "https://download.pytorch.org/whl/cu${CU}/torch/")"
  if [[ -z "$BODY" ]]; then
    echo "  cu$CU: (no index / unreachable)"
  else
    NEWEST="$(printf '%s\n' "$BODY" | grep -oE "torch-2\.[0-9]+\.[0-9]+%2Bcu${CU}-cp312-cp312-manylinux[^\"<>]*x86_64\.whl" | sort -uV | tail -1)"
    echo "  cu$CU: ${NEWEST:-(no cp312 x86_64 wheel found)}"
  fi
done
echo

echo "== 4. Docker Hub nvidia/cuda — newest *-devel-ubuntu24.04 tags =="
TAGS=""
for PAGE_NUMBER in 1 2 3; do
  BODY="$(fetch "https://hub.docker.com/v2/repositories/nvidia/cuda/tags/?name=devel-ubuntu24.04&page_size=100&page=$PAGE_NUMBER")"
  [[ -z "$BODY" ]] && break
  PAGE_TAGS="$(printf '%s\n' "$BODY" | grep -oE '"name"[ ]*:[ ]*"13\.[0-9]+\.[0-9]+-devel-ubuntu24\.04"' | grep -oE '13\.[0-9]+\.[0-9]+-devel-ubuntu24\.04')"
  TAGS="$(printf '%s\n%s\n' "$TAGS" "$PAGE_TAGS")"
  printf '%s\n' "$BODY" | grep -q '"next"[ ]*:[ ]*null' && break
done
printf '%s\n' "$TAGS" | grep -v '^$' | sort -uV | tail -8 | sed 's/^/  /'
echo

echo "== 5. NVIDIA CUDA repo (ubuntu24.04) — cuda-compat-13-* packages =="
REPO_BODY="$(fetch https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/)"
echo "  (listing fetched: ${#REPO_BODY} bytes)"
printf '%s\n' "$REPO_BODY" | grep -oE 'cuda-compat-13[^"<>]*\.deb' | sort -uV | tail -8 | sed 's/^/  /'
echo

echo "== 6. Verdict =="
if [[ -n "$CEILING_PAGE" ]]; then
  CEILING_TORCH="${CEILING_PAGE#torch-}"
  CEILING_TORCH="${CEILING_TORCH%.html}"
  CEILING_TORCH="${CEILING_TORCH//%2B/+}"
  CEILING_CLUSTER_VERSION="${CEILING_WHEEL#torch_cluster-}"
  CEILING_CLUSTER_VERSION="${CEILING_CLUSTER_VERSION%%-cp312*}"
  CEILING_CLUSTER_VERSION="${CEILING_CLUSTER_VERSION//%2B/+}"
  if [[ "$CEILING_TORCH" == "$CUR_TORCH" && "==$CEILING_CLUSTER_VERSION" == "$CUR_TC" ]]; then
    echo "AT CEILING: the newest PyG torch_cluster prebuilt (${CEILING_CLUSTER_VERSION} for torch ${CEILING_TORCH})"
    echo "is exactly what 'postprocessing' pins today. No bump is available under the"
    echo "torch-cluster-first rule; re-run this survey periodically (PyG typically"
    echo "lags new torch releases by weeks)."
    exit 0
  fi
  echo "BUMP AVAILABLE — candidate pins for containers/variants/postprocessing.env:"
  echo "  TORCH_VERSION=$CEILING_TORCH"
  echo "  TORCH_CUDA_FLAVOR=${CEILING_TORCH##*+}"
  echo "  PYG_WHEEL_INDEX=https://data.pyg.org/whl/${CEILING_PAGE}"
  echo "  TORCH_CLUSTER_VERSION===${CEILING_CLUSTER_VERSION}"
  echo
  echo "Change surface: containers/variants/postprocessing.env, containers/pyprojects/latest.toml,"
  echo "tests/test_containers.py (hardcoded asserts), containers/README.md,"
  echo "containers/SM_ACCEPTANCE.md. Keep the structure-parser stack frozen"
  echo "(biotite==1.6.0, gemmi==0.7.5, mdtraj==1.11.1.post1). Pick the matching"
  echo "cuda-compat-13-x from section 5 — all are compatible with driver 535+ on"
  echo "Data Center GPUs per NVIDIA's Application Compatibility Support Matrix."
  echo "Then run the full acceptance loop: build+push → pull-sqsh → GPU smoke →"
  echo "Stage 13 timing → task853 byte parity."
  exit 0
fi
if [[ -n "$ANY_PAGE" ]]; then
  echo "BLOCKED-BY-PYTHON: newer torch pages ship torch_cluster only for other"
  echo "python tags (newest any-tag wheel: $ANY_WHEEL on ${ANY_PAGE//%2B/+})."
  echo "A bump would require moving the variant past Python 3.12, which conflicts"
  echo "with the mounted-source requires-python floor. Escalate before proceeding."
  exit 0
fi
echo "AT CEILING: no newer torch_cluster cp312 linux-x86_64 wheel in the newest"
echo "$CHECKED probed pages (control check passed, so the misses are trustworthy)."
echo "Current pins (${CUR_TORCH} / ${CUR_TC_VER}) remain the bump ceiling."
exit 0
