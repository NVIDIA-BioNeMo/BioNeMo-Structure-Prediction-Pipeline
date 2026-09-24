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
# BSPP orchestration — postprocessing container image
# =============================================================================
#
# The image bakes the scientific/runtime dependency environment with pixi, the
# pinned toolkit + iPSAE, and the orchestration source (Contract + Control +
# Runtime wheels at the pinned commit). A bind-mount of the orchestration
# source remains available as an explicit dev override (see
# containers/scripts/install-orchestration-source.sh).
# =============================================================================

ARG BASE_DIGEST=""
ARG BASE_IMAGE=nvidia/cuda:13.0.3-devel-ubuntu24.04
FROM ${BASE_IMAGE}${BASE_DIGEST:+@${BASE_DIGEST}}

ENV DEBIAN_FRONTEND=noninteractive

ARG TARGETARCH
ARG PIXI_MANIFEST=containers/pyprojects/latest.toml
ARG CUDA_VERSION=13.0
ARG CUDA_COMPAT_PACKAGE=
ARG CUDA_COMPAT_DIR=
ARG TOOLKIT_REPO=https://github.com/PDBeurope/AFDB-Integration-Kit.git
# TOOLKIT_REPO defaults to the public upstream toolkit (release-clean). There
# is no private default; the internal child image inherits this same toolkit.
ARG TOOLKIT_REF=e2fa757aa0cb2cec8e4a8382627fcbbca7599556
ARG TOOLKIT_BRANCH=nvidia-postproc

# ---------------------------------------------------------------------------
# System packages + CLI tools
# ---------------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl ca-certificates git openssh-client build-essential make lz4 zstd \
        apt-transport-https gnupg && \
    rm -rf /var/lib/apt/lists/*

# CUDA forward-compatibility libraries. This is intentionally variant-driven:
# current may leave it empty, latest may install cuda-compat-13-0.
RUN if [ -n "${CUDA_COMPAT_PACKAGE}" ]; then \
        apt-get update && \
        apt-get install -y --no-install-recommends "${CUDA_COMPAT_PACKAGE}" && \
        rm -rf /var/lib/apt/lists/*; \
    fi

# gcloud CLI
RUN curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
        gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
        > /etc/apt/sources.list.d/google-cloud-sdk.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends google-cloud-cli && \
    rm -rf /var/lib/apt/lists/*

# s5cmd
ARG S5CMD_VERSION=2.3.0
ARG S5CMD_SHA256
RUN curl -fsSL -o /tmp/s5cmd.deb \
        "https://github.com/peak/s5cmd/releases/download/v${S5CMD_VERSION}/s5cmd_${S5CMD_VERSION}_linux_${TARGETARCH}.deb" && \
    echo "${S5CMD_SHA256}  /tmp/s5cmd.deb" | sha256sum --check --status && \
    dpkg -i /tmp/s5cmd.deb && rm /tmp/s5cmd.deb

# yq
RUN curl -fsSL -o /usr/local/bin/yq \
        "https://github.com/mikefarah/yq/releases/latest/download/yq_linux_${TARGETARCH}" && \
    chmod +x /usr/local/bin/yq

# ---------------------------------------------------------------------------
# pixi + Python environment
# ---------------------------------------------------------------------------
# The pixi binary is pinned and sha256-verified (not the mutable
# 'curl pixi.sh/install.sh | bash' installer). The conda environment itself is
# still resolved non-locked by 'pixi install' below.
ARG PIXI_VERSION=0.80.0
ARG PIXI_SHA256=387a2d3052e656f61ccf735e6750255451366f45635a2da09116b1f8394b2837
RUN curl -fsSL -H 'Accept: application/octet-stream' \
        "https://github.com/prefix-dev/pixi/releases/download/v${PIXI_VERSION}/pixi-x86_64-unknown-linux-musl" \
        -o /usr/local/bin/pixi \
    && echo "${PIXI_SHA256}  /usr/local/bin/pixi" | sha256sum --check --status \
    && chmod 0755 /usr/local/bin/pixi

WORKDIR /opt/bspp-orchestration-env
COPY ${PIXI_MANIFEST} pixi.toml

# pixi solves without a GPU during image build; this declares the intended
# CUDA user-space version for packages that consult virtual CUDA features.
ENV CONDA_OVERRIDE_CUDA=${CUDA_VERSION}
RUN pixi install

RUN pixi shell-hook > /opt/pixi-activate.sh

# ---------------------------------------------------------------------------
# Baked toolkit — clone, pin, provenance
# KEEP THIS AFTER the dependency layers (pixi, gcloud, s5cmd, yq): the
# toolkit is the most frequently changed Python code, so it must live in the
# last independent layers — a pin bump then rebuilds only from here down and
# reuses the cached dependency layers above.
# The source is an explicit public Git URL (TOOLKIT_REPO build ARG, no private
# default). No SSH agent, no private host, no credential enters any layer.
# ---------------------------------------------------------------------------
RUN git clone --branch "${TOOLKIT_BRANCH}" --single-branch \
        "${TOOLKIT_REPO}" \
        /opt/afdb-toolkit \
    && cd /opt/afdb-toolkit \
    && git checkout "${TOOLKIT_REF}" \
    && git log -1 --format='%H %aI' > /tmp/toolkit_head.txt \
    && git remote remove origin 2>/dev/null || true \
    && rm -rf .git \
    && TOOLKIT_COMMIT=$(awk '{print $1}' /tmp/toolkit_head.txt) \
    && TOOLKIT_DATE=$(awk '{print $2}' /tmp/toolkit_head.txt) \
    && if [ "${TOOLKIT_COMMIT}" != "${TOOLKIT_REF}" ]; then \
        echo "ERROR: checked out ${TOOLKIT_COMMIT}, expected ${TOOLKIT_REF}"; exit 1; \
    fi \
    && cat > /opt/afdb-toolkit/provenance.json <<PROVEOF
{
  "schema_version": 1,
  "repository": "${TOOLKIT_REPO}",
  "branch": "${TOOLKIT_BRANCH}",
  "commit": "${TOOLKIT_REF}",
  "checked_out_at": "${TOOLKIT_DATE}",
  "toolkit_path": "/opt/afdb-toolkit"
}
PROVEOF

# ---------------------------------------------------------------------------
# Build iPSAE from baked toolkit source
# ---------------------------------------------------------------------------
RUN cd /opt/afdb-toolkit/afdb_integration_kit/ipsae \
    && make -B -C . CXX=g++ \
    && chmod 500 ipsae_cpp \
    && cp ipsae_cpp /opt/afdb-toolkit/ipsae_cpp

# ---------------------------------------------------------------------------
# Baked orchestration source — Contract + Control + Runtime wheels at the
# pinned commit. KEEP THIS AFTER the toolkit/iPSAE layers: the orchestration
# source changes every commit, so it must live in the last independent layers
# so a source bump rebuilds only from here down and reuses the cached
# dependency and toolkit layers above. The bind-mount of the orchestration
# source remains a dev-only override (install-orchestration-source.sh).
# ---------------------------------------------------------------------------
ARG SOURCE_DATE_EPOCH
ENV SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH}
COPY containers/postprocessing/context/wheels/ /opt/bspp/wheels/
COPY containers/postprocessing/context/postprocessing-runtime-image.json \
    /opt/bspp/postprocessing-runtime-image.json
RUN . /opt/pixi-activate.sh \
    && python -m pip install --no-deps /opt/bspp/wheels/*.whl \
    && python -c 'import importlib.metadata; importlib.metadata.version("bspp-orchestration-control")' \
    && rm -rf /opt/bspp/wheels

# ---------------------------------------------------------------------------
# Entrypoint + smoke test + qualification
# ---------------------------------------------------------------------------
COPY containers/scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY containers/scripts/install-orchestration-source.sh /usr/local/bin/install-orchestration-source.sh
COPY containers/scripts/smoke-gpu.sh /usr/local/bin/bspp-container-smoke-gpu
COPY containers/scripts/qualify-baked-toolkit.sh /usr/local/bin/qualify-baked-toolkit.sh
COPY containers/postprocessing/image-smoke.py /opt/bspp/bin/bspp-postprocessing-image-smoke
COPY packages/orchestration-contract/src /opt/bspp/lib
COPY packages/orchestration-control/src/bspp/orchestration/control/execution_bootstrap.py /opt/bspp/execution_bootstrap.py
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/install-orchestration-source.sh \
        /usr/local/bin/bspp-container-smoke-gpu /usr/local/bin/qualify-baked-toolkit.sh \
        /opt/bspp/bin/bspp-postprocessing-image-smoke \
    && /usr/local/bin/qualify-baked-toolkit.sh

ENV BSPP_CUDA_COMPAT=off
ENV BSPP_CUDA_COMPAT_DIR=${CUDA_COMPAT_DIR}
# The orchestration-source install-mode helper reads the baked source commit
# from this image JSON (see containers/scripts/install-orchestration-source.sh).
ENV BSPP_IMAGE_JSON=/opt/bspp/postprocessing-runtime-image.json

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]

# The entrypoint validates the baked toolkit against this commit. Wire it to the
# ref actually built, otherwise a non-default TOOLKIT_REF produces an image that
# refuses to start (entrypoint.sh falls back to a hard-coded default).
ENV BSPP_EXPECTED_TOOLKIT_COMMIT=${TOOLKIT_REF}
