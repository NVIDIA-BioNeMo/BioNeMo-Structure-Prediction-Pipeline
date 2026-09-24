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
FROM ghcr.io/sokrypton/colabfold:1.6.2-cuda12${BASE_DIGEST:+@${BASE_DIGEST}}

ARG SOURCE_DATE_EPOCH
ARG S5CMD_VERSION=2.3.0
ARG S5CMD_SHA256
ENV SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH} \
    BSPP_CUDA_COMPAT_DIR=/usr/local/cuda-12/compat \
    PYTHONUNBUFFERED=1

# Runtime dependency install (before BSPP wheels). These are the minimal
# transitive deps the executor and --help smoke require. numpy is already
# present in the colabfold base image.
RUN pip install --no-cache-dir \
      "click>=8.1,<9" \
      "pydantic>=2,<3" \
      "pyyaml>=6,<7" \
      "tabulate>=0.9,<1" \
      "hatchling>=1.25" \
 && python -c "import click, pydantic, yaml, tabulate, hatchling"

# BSPP Contract+Control+Runtime wheels (no-deps — the runtime wheel's declared deps
# are not installed in this kernel image).
COPY context/wheels/*.whl /tmp/wheels/
RUN pip install --no-deps /tmp/wheels/*.whl \
 && python -c 'import importlib.metadata; importlib.metadata.version("bspp-orchestration-control")' \
 && rm -rf /tmp/wheels

COPY context/folding-colabfold-image.json /opt/bspp/folding-runtime-image.json
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
COPY install-orchestration-source.sh /usr/local/bin/install-orchestration-source.sh
COPY image-smoke.py /opt/bspp/bin/bspp-folding-image-smoke
RUN chmod 0755 /usr/local/bin/entrypoint.sh /usr/local/bin/install-orchestration-source.sh \
    /opt/bspp/bin/bspp-folding-image-smoke \
 && if command -v colabfold_batch >/dev/null 2>&1; then \
      [ -e /usr/local/bin/colabfold_batch ] \
        || ln -sf "$(command -v colabfold_batch)" /usr/local/bin/colabfold_batch; \
    fi \
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
