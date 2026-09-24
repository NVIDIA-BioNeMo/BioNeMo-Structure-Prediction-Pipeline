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
FROM docker.io/nvidia/cuda:12.6.3-base-ubuntu24.04${BASE_DIGEST:+@${BASE_DIGEST}}

ARG SOURCE_DATE_EPOCH
ARG S5CMD_VERSION=2.3.0
ARG S5CMD_SHA256
ARG PIXI_URL
ARG PIXI_SHA256
ENV SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH} \
    PIXI_NO_PROGRESS=true \
    PIXI_CACHE_DIR=/opt/bspp/pixi-cache \
    PATH=/opt/bspp/environment/bin:${PATH}

# Keep construction in one layer so constrained rootless/VFS builders do not
# duplicate the process environment at every step. The build tree is deleted
# before the layer is committed and is not visible in the final image.
#
# The runtime image carries only the non-fold toolchain (lz4, rsync, tar, flock,
# s5cmd) and the Contract+Control+Runtime wheels via --no-deps. It is a
# folding-executor-only image, not a general bspp-orchestration-runtime image.
COPY . /tmp/bspp-image-build
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && curl -fsSL -H 'Accept: application/octet-stream' "$PIXI_URL" -o /usr/local/bin/pixi \
    && echo "$PIXI_SHA256  /usr/local/bin/pixi" | sha256sum --check --status \
    && chmod 0755 /usr/local/bin/pixi \
    && chmod 0755 /tmp/bspp-image-build/entrypoint.sh \
        /tmp/bspp-image-build/image-smoke.py \
    && mkdir -p /opt/bspp/locked /opt/bspp/bin \
    && cp /tmp/bspp-image-build/pixi.toml /opt/bspp/locked/ \
    && pixi install --manifest-path /opt/bspp/locked/pixi.toml \
    && ln -s /opt/bspp/locked/.pixi/envs/default /opt/bspp/environment \
    && /opt/bspp/environment/bin/python -m pip install --no-deps \
        /tmp/bspp-image-build/context/wheels/*.whl \
    && /opt/bspp/environment/bin/python -c 'import importlib.metadata; importlib.metadata.version("bspp-orchestration-control")' \
    && install -m 0644 /tmp/bspp-image-build/context/folding-runtime-image.json \
        /opt/bspp/folding-runtime-image.json \
    && install -m 0755 /tmp/bspp-image-build/entrypoint.sh /usr/local/bin/entrypoint.sh \
    && install -m 0755 /tmp/bspp-image-build/install-orchestration-source.sh \
        /usr/local/bin/install-orchestration-source.sh \
    && install -m 0755 /tmp/bspp-image-build/image-smoke.py /opt/bspp/bin/bspp-folding-image-smoke \
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
