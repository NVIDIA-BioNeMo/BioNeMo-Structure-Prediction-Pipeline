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

FROM docker.io/nvidia/cuda@sha256:6b3201183858bad08441837f5a5efc2c75290135cc25fcc87d9ff763190cfd09

ARG SOURCE_DATE_EPOCH
ARG S5CMD_VERSION=2.3.0
ARG S5CMD_SHA256
ARG PIXI_URL
ARG PIXI_SHA256
ARG MMSEQS_URL
ARG MMSEQS_SHA256
ENV SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH} \
    PIXI_NO_PROGRESS=true \
    PIXI_CACHE_DIR=/opt/bspp/pixi-cache \
    PATH=/opt/bspp/environment/bin:${PATH}

# Keep construction in one layer so constrained rootless/VFS builders do not
# duplicate the scientific environment at every step. The build tree is
# deleted before the layer is committed and is not visible in the final image.
COPY . /tmp/bspp-image-build
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && curl -fsSL -H 'Accept: application/octet-stream' "$PIXI_URL" -o /usr/local/bin/pixi \
    && echo "$PIXI_SHA256  /usr/local/bin/pixi" | sha256sum --check --status \
    && chmod 0755 /usr/local/bin/pixi \
    && curl -fsSL -H 'Accept: application/octet-stream' "$MMSEQS_URL" -o /tmp/mmseqs-linux-gpu.tar.gz \
    && echo "$MMSEQS_SHA256  /tmp/mmseqs-linux-gpu.tar.gz" | sha256sum --check --status \
    && chmod 0755 /tmp/bspp-image-build/entrypoint.sh \
        /tmp/bspp-image-build/image-smoke.py \
        /tmp/bspp-image-build/carry-characterization.sh \
        /tmp/bspp-image-build/cuda-driver-probe.py \
    && mkdir -p /opt/bspp/locked /opt/bspp/mmseqs /opt/bspp/bin \
    && cp /tmp/bspp-image-build/pixi.toml /tmp/bspp-image-build/pixi.lock /opt/bspp/locked/ \
    && pixi install --locked --manifest-path /opt/bspp/locked/pixi.toml \
    && ln -s /opt/bspp/locked/.pixi/envs/default /opt/bspp/environment \
    && tar -xzf /tmp/mmseqs-linux-gpu.tar.gz \
        -C /opt/bspp/mmseqs --strip-components=1 \
    && rm -f /tmp/mmseqs-linux-gpu.tar.gz \
    && /opt/bspp/environment/bin/python -m pip install --no-deps \
        /tmp/bspp-image-build/context/wheels/*.whl \
    && /opt/bspp/environment/bin/python -c 'import importlib.metadata; importlib.metadata.version("bspp-orchestration-control")' \
    && install -m 0644 /tmp/bspp-image-build/context/preprocessing-runtime-image.json \
        /opt/bspp/preprocessing-runtime-image.json \
    && install -m 0755 /tmp/bspp-image-build/entrypoint.sh /usr/local/bin/entrypoint.sh \
    && install -m 0755 /tmp/bspp-image-build/image-smoke.py /opt/bspp/bin/bspp-preprocessing-image-smoke \
    && install -m 0755 /tmp/bspp-image-build/carry-characterization.sh \
        /opt/bspp/bin/bspp-preprocessing-carry-characterization \
    && install -m 0755 /tmp/bspp-image-build/cuda-driver-probe.py \
        /opt/bspp/bin/bspp-preprocessing-cuda-driver-probe \
    && ln -sf /opt/bspp/mmseqs/bin/mmseqs /usr/local/bin/mmseqs \
    && ln -sf /opt/bspp/environment/bin/colabfold_search /usr/local/bin/colabfold_search \
    && ln -sf /opt/bspp/environment/bin/tar /usr/bin/tar \
    && ln -sf /opt/bspp/environment/bin/lz4 /usr/bin/lz4 \
    && ln -sf /opt/bspp/environment/bin/flock /usr/bin/flock \
    && ln -sf /opt/bspp/environment/bin/rsync /usr/bin/rsync \
    && ln -sf /opt/bspp/environment/bin/bash /bin/bash \
    && ln -sf /opt/bspp/environment/bin/ls /usr/bin/ls \
    && curl -fsSL -o /tmp/s5cmd.deb \
        "https://github.com/peak/s5cmd/releases/download/v${S5CMD_VERSION}/s5cmd_${S5CMD_VERSION}_linux_amd64.deb" \
    && echo "${S5CMD_SHA256}  /tmp/s5cmd.deb" | sha256sum --check --status \
    && dpkg -i /tmp/s5cmd.deb \
    && rm /tmp/s5cmd.deb \
    && apt-get purge -y --auto-remove curl \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/bspp-image-build /opt/bspp/pixi-cache

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bspp-orchestration-runtime", "--help"]
