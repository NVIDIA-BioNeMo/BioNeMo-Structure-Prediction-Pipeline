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
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04${BASE_DIGEST:+@${BASE_DIGEST}}

ARG SOURCE_DATE_EPOCH
ARG S5CMD_VERSION=2.3.0
ARG S5CMD_SHA256
ARG OPENFOLD_SOURCE_COMMIT=be2ec1841f16c966c65ae0e7599ebbadc725757d
ARG PDBFIXER_REF=94cfa4c0ca551cdc5f13320f9a658efd59f2b881
ARG TORCH_SPEC=torch==2.5.1+cu121
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121

ENV SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH} \
    BSPP_CUDA_COMPAT_DIR=/usr/local/cuda-12.1/compat \
    DEBIAN_FRONTEND=noninteractive \
    MAX_JOBS=4 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# Provision Python 3.12 via deadsnakes PPA. The base image is Ubuntu 22.04
# (Python 3.10), but the Contract/Control/Runtime wheels require >=3.12,<3.13.
# After installation, python3 and python both resolve to python3.12.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates git hhsuite hmmer kalign wget \
        software-properties-common gnupg \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m ensurepip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Scientific dependencies: torch + OpenFold deps (installed under Python 3.12).
RUN python3.12 -m pip install --no-cache-dir "${TORCH_SPEC}" --index-url "${TORCH_INDEX_URL}"

RUN python3.12 -m pip install --no-cache-dir \
      "absl-py==2.1.0" \
      "biopython==1.83" \
      "ml-collections==0.1.1" \
      "modelcif==0.7" \
      "openmm==8.1.1" \
      "pytorch-lightning==2.2.1" \
 && python3.12 -m pip install --no-cache-dir "git+https://github.com/openmm/pdbfixer.git@${PDBFIXER_REF}"

RUN git clone https://github.com/aqlaboratory/openfold.git /opt/openfold \
    && cd /opt/openfold \
    && git checkout --detach ${OPENFOLD_SOURCE_COMMIT}

COPY patch_setup.py /tmp/patch_openfold_setup.py
RUN python3.12 /tmp/patch_openfold_setup.py && rm /tmp/patch_openfold_setup.py

# Install setuptools before OpenFold's setup.py install (ensurepip bundles
# only pip; OpenFold's setup.py imports setuptools at build time).
RUN python3.12 -m pip install --no-cache-dir setuptools

RUN wget -q -P /opt/openfold/openfold/resources \
        https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt \
    && cd /opt/openfold \
    && python3.12 setup.py install

# Runtime dependency install (before BSPP wheels). numpy is pinned <2 because
# OpenFold requires numpy 1.x. The runtime wheel declares numpy>=2.0, but
# pip install --no-deps means that declared dependency is not enforced.
RUN python3.12 -m pip install --no-cache-dir \
      "click>=8.1,<9" \
      "pydantic>=2,<3" \
      "pyyaml>=6,<7" \
      "tabulate>=0.9,<1" \
      "numpy>=1.26,<2" \
      "hatchling>=1.25" \
 && python3.12 -c "import click, pydantic, yaml, tabulate, numpy, hatchling"

# BSPP Contract+Control+Runtime wheels (no-deps).
COPY context/wheels/*.whl /tmp/wheels/
RUN python3.12 -m pip install --no-deps /tmp/wheels/*.whl \
 && python3.12 -c 'import importlib.metadata; importlib.metadata.version("bspp-orchestration-control")' \
 && rm -rf /tmp/wheels

# The upstream run_pretrained_openfold.py has no shebang and is mode 100644.
# The executor invokes it directly (subprocess.run(["run_pretrained_openfold.py", ...])),
# so prepend a shebang and set the executable bit before symlinking it onto PATH.
RUN sed -i '1i #!/usr/bin/env python3' /opt/openfold/run_pretrained_openfold.py \
    && chmod +x /opt/openfold/run_pretrained_openfold.py \
    && ln -sf /opt/openfold/run_pretrained_openfold.py /usr/local/bin/run_pretrained_openfold.py

COPY context/folding-openfold-cli-image.json /opt/bspp/folding-runtime-image.json
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
