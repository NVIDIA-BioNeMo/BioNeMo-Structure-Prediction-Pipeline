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

ARG BASE_DIGEST=""
FROM nvidia/cuda:13.0.3-devel-ubuntu24.04${BASE_DIGEST:+@${BASE_DIGEST}}

ARG SOURCE_DATE_EPOCH
ARG S5CMD_VERSION=2.3.0
ARG S5CMD_SHA256
ARG BIOIR_VERSION=0.1.0
ARG TORCH_SPEC=torch==2.11.0+cu130
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130

ENV SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH} \
    BSPP_CUDA_COMPAT_DIR=/usr/local/cuda-13.0/compat \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONUNBUFFERED=1

# Provision Python 3.12 + pip. The base image nvidia/cuda:13.0.3-devel-ubuntu24.04
# is the Ubuntu 24.04 rootfs plus CUDA 13.0 devel packages and ships neither
# python3 nor pip. Ubuntu 24.04's default python3 is 3.12, satisfying the
# Contract/Control/Runtime wheel requirement (>=3.12,<3.13).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Install torch (public PyTorch index, mirrors containers/variants/postprocessing.env).
RUN pip install --no-cache-dir "${TORCH_SPEC}" --index-url "${TORCH_INDEX_URL}"

# Install BioIR from public PyPI.
RUN pip install --no-cache-dir "bionemo-ir==${BIOIR_VERSION}"

# Runtime dependency install (before BSPP wheels).
RUN pip install --no-cache-dir \
      "click>=8.1,<9" \
      "pydantic>=2,<3" \
      "pyyaml>=6,<7" \
      "tabulate>=0.9,<1" \
      "hatchling>=1.25" \
 && python -c "import click, pydantic, yaml, tabulate, hatchling"

# Triton compiles Python extensions during model construction. Keep the
# development headers at exactly the already-installed interpreter version.
RUN bspp_python_package_version="$(dpkg-query -W -f='${Version}' python3.12)" \
 && apt-get update \
 && apt-get install -y --no-install-recommends "python3.12-dev=${bspp_python_package_version}" \
 && rm -rf /var/lib/apt/lists/*

# BSPP Contract+Control+Runtime wheels (no-deps).
COPY context/wheels/*.whl /tmp/wheels/
RUN pip install --no-deps /tmp/wheels/*.whl \
 && python -c 'import importlib.metadata; importlib.metadata.version("bspp-orchestration-control")' \
 && rm -rf /tmp/wheels

COPY context/folding-bioir-image.json /opt/bspp/folding-runtime-image.json
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
COPY install-orchestration-source.sh /usr/local/bin/install-orchestration-source.sh
COPY image-smoke.py /opt/bspp/bin/bspp-folding-image-smoke
RUN chmod 0755 /usr/local/bin/entrypoint.sh /usr/local/bin/install-orchestration-source.sh \
    /opt/bspp/bin/bspp-folding-image-smoke \
 && apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && curl -fsSL -o /tmp/s5cmd.deb \
        "https://github.com/peak/s5cmd/releases/download/v${S5CMD_VERSION}/s5cmd_${S5CMD_VERSION}_linux_amd64.deb" \
 && echo "${S5CMD_SHA256}  /tmp/s5cmd.deb" | sha256sum --check --status \
 && dpkg -i /tmp/s5cmd.deb \
 && rm /tmp/s5cmd.deb \
 && apt-get purge -y --auto-remove curl \
 && rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bspp-orchestration-runtime", "--help"]
