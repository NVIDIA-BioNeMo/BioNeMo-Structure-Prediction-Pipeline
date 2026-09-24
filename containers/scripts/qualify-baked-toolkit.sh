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
# Build-time baked toolkit qualification — fail-closed on contract drift.
#
# Invoked inside the Dockerfile after iPSAE is compiled from baked source.
# Exits 0 when the iPSAE binary passes a deterministic paired-pdb-pae check
# identical to the runtime qualification contract. Any deviation exits 1
# and aborts the image build.
#
# Sandbox testing: set BSPP_BAKED_TOOLKIT to a simulated toolkit root.
# =============================================================================
set -euo pipefail

TOOLKIT="${BSPP_BAKED_TOOLKIT:-/opt/afdb-toolkit}"
IPSAE_SRC="${TOOLKIT}/afdb_integration_kit/ipsae"
MAKEFILE="${IPSAE_SRC}/Makefile"
FIXTURE_DIR="$(mktemp -d)"
SUMMARY_CSV="${FIXTURE_DIR}/summary.csv"
IPSAE_BIN="${TOOLKIT}/ipsae_cpp"
RESULT_JSON="${TOOLKIT}/qualification-result.json"
QUAL_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ---------------------------------------------------------------------------
# Contract constants — identical to RUNTIME_IPSAE_FIXTURE_* in
# bspp.orchestration.contract.runtime_qualification
# ---------------------------------------------------------------------------
FIXTURE_MODEL_ID="BSPP-RQ-PAIR"
EXPECTED_SCORE_AB="1.000000"
EXPECTED_SCORE_BA="1.000000"

# Contrived 4-atom ALA dimer PDB (A 1–2, B 1–2)
FIXTURE_PDB='ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 90.00           C
ATOM      2  CA  ALA A   2       0.000   2.000   0.000  1.00 90.00           C
ATOM      3  CA  ALA B   1       0.000   0.000   4.000  1.00 90.00           C
ATOM      4  CA  ALA B   2       0.000   2.000   4.000  1.00 90.00           C
END'

# Zero-matrix PAE matching the contract fixture
FIXTURE_PAE='{"max_pae":10.0,"pae":[[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]]}'

cleanup() {
    rm -rf "${FIXTURE_DIR}"
}
trap cleanup EXIT

# =============================================================================
# Step 1 — Validate source presence
# =============================================================================
if [ ! -f "${MAKEFILE}" ]; then
    echo "QUALIFICATION FAILED: Makefile missing at ${MAKEFILE}" >&2
    exit 1
fi

# =============================================================================
# Step 2 — Build iPSAE from baked source (idempotent; Dockerfile already
#          built it, but the qualification script owns its own build for
#          auditability)
# =============================================================================
echo "=== Qualification: building iPSAE from ${IPSAE_SRC}"
make -B -C "${IPSAE_SRC}" CXX=g++

if [ ! -x "${IPSAE_SRC}/ipsae_cpp" ]; then
    echo "QUALIFICATION FAILED: ipsae_cpp not executable after make" >&2
    exit 1
fi

# Prefer the baked binary at the known path if it exists and is executable
if [ -x "${IPSAE_BIN}" ]; then
    IPSAE_CMD="${IPSAE_BIN}"
else
    IPSAE_CMD="${IPSAE_SRC}/ipsae_cpp"
fi

IPSAE_SHA256="$(sha256sum "${IPSAE_CMD}" | awk '{print $1}')"

# =============================================================================
# Step 3 — Create deterministic fixture
# =============================================================================
PDB_FILE="${FIXTURE_DIR}/${FIXTURE_MODEL_ID}-model_v1.pdb"
PAE_FILE="${FIXTURE_DIR}/${FIXTURE_MODEL_ID}-meta_v1.json"

printf '%s\n' "${FIXTURE_PDB}" > "${PDB_FILE}"
printf '%s'   "${FIXTURE_PAE}" > "${PAE_FILE}"

# =============================================================================
# Step 4 — Run baked iPSAE against fixture
# =============================================================================
echo "=== Qualification: running iPSAE paired-pdb-pae check"
"${IPSAE_CMD}" --batch "${FIXTURE_DIR}" 10.0 8.0 --summary "${SUMMARY_CSV}" --workers 1 --quiet 2>&1 || true

# =============================================================================
# Step 5 — Parse CSV and validate
# =============================================================================
if [ ! -f "${SUMMARY_CSV}" ] || [ ! -s "${SUMMARY_CSV}" ]; then
    echo "QUALIFICATION FAILED: iPSAE produced no summary CSV" >&2
    exit 1
fi

# The real iPSAE summary CSV has many columns:
#   pdb_path,pae_cutoff,dist_cutoff,iptm_af,ipsae_AB,ipsae_BA,ipsae_d0chn_AB,...
# Resolve columns by header name, never by fixed position.
CSV_HEADER="$(head -1 "${SUMMARY_CSV}")"
DATA_LINE="$(sed -n '2p' "${SUMMARY_CSV}")"

col_index() {
    local name="$1" header="$2" idx=1 field
    local -a fields
    IFS=',' read -r -a fields <<< "${header}"
    for field in "${fields[@]}"; do
        if [ "${field}" = "${name}" ]; then
            echo "${idx}"
            return 0
        fi
        idx=$((idx + 1))
    done
    return 1
}

AB_COL="$(col_index 'ipsae_AB' "${CSV_HEADER}")" || {
    echo "QUALIFICATION FAILED: missing directional 'ipsae_AB' column in CSV header: ${CSV_HEADER}" >&2
    exit 1
}
BA_COL="$(col_index 'ipsae_BA' "${CSV_HEADER}")" || {
    echo "QUALIFICATION FAILED: missing directional 'ipsae_BA' column in CSV header: ${CSV_HEADER}" >&2
    exit 1
}

MODEL_COL="$(echo "${DATA_LINE}" | cut -d, -f1)"

# The pdb_path column may be absolute (e.g. /tmp/.../BSPP-RQ-PAIR-model_v1.pdb)
# or bare. Strip any directory prefix, then the "-model_v1.pdb" suffix.
MODEL_ID="${MODEL_COL##*/}"
MODEL_ID="${MODEL_ID%-model_v1.pdb}"
if [ "${MODEL_ID}" != "${FIXTURE_MODEL_ID}" ]; then
    echo "QUALIFICATION FAILED: model ID mismatch — got '${MODEL_ID}', expected '${FIXTURE_MODEL_ID}'" >&2
    exit 1
fi

IPSAE_AB="$(echo "${DATA_LINE}" | cut -d, -f"${AB_COL}")"
IPSAE_BA="$(echo "${DATA_LINE}" | cut -d, -f"${BA_COL}")"

# Format to 6 decimal places for exact comparison
IPSAE_AB_FMT="$(printf '%.6f' "${IPSAE_AB}" 2>/dev/null || echo "${IPSAE_AB}")"
IPSAE_BA_FMT="$(printf '%.6f' "${IPSAE_BA}" 2>/dev/null || echo "${IPSAE_BA}")"

if [ "${IPSAE_AB_FMT}" != "${EXPECTED_SCORE_AB}" ]; then
    echo "QUALIFICATION FAILED: ipsae_AB mismatch — got '${IPSAE_AB_FMT}', expected '${EXPECTED_SCORE_AB}'" >&2
    exit 1
fi

if [ "${IPSAE_BA_FMT}" != "${EXPECTED_SCORE_BA}" ]; then
    echo "QUALIFICATION FAILED: ipsae_BA mismatch — got '${IPSAE_BA_FMT}', expected '${EXPECTED_SCORE_BA}'" >&2
    exit 1
fi

# =============================================================================
# Step 6 — Write qualification result
# =============================================================================
cat > "${RESULT_JSON}" <<EOF
{"status":"passed","ipsae_sha256":"${IPSAE_SHA256}","checked_at":"${QUAL_DATE}"}
EOF

echo "=== Qualification PASSED: ipsae_AB=${IPSAE_AB_FMT} ipsae_BA=${IPSAE_BA_FMT}"
echo "    ipsae_sha256: ${IPSAE_SHA256}"
echo "    result: ${RESULT_JSON}"