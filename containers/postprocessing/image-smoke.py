#!/usr/bin/env python
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

"""Self-contained composition smoke for the postprocessing image.

This proves the baked orchestration source (Contract + Control + Runtime
distributions), the baked-mode source selection performed by the shared
install-mode helper, and the embedded image manifest. It deliberately performs
no CUDA init, no GPU work, no nested container, and no scientific inference —
the GPU smoke (`bspp-container-smoke-gpu`) owns that gate.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

_MANIFEST = Path("/opt/bspp/postprocessing-runtime-image.json")
_DISTRIBUTIONS = (
    "bspp-orchestration-contract",
    "bspp-orchestration-control",
    "bspp-orchestration-runtime",
)


def main() -> None:
    # The entrypoint sources install-orchestration-source.sh before exec'ing
    # this script; with no dev mount it must resolve baked mode.
    source = os.environ.get("BSPP_ORCHESTRATION_SOURCE")
    if source != "baked":
        raise RuntimeError(f"orchestration source is not baked: {source!r}")

    for distribution in _DISTRIBUTIONS:
        importlib.metadata.version(distribution)

    manifest_bytes = _MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != 1:
        raise RuntimeError("postprocessing image manifest is not schema version 1")
    source_commit = manifest.get("source_commit")
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise RuntimeError("postprocessing image manifest has a malformed source_commit")
    int(source_commit, 16)
    for key in ("contract_wheel_sha256", "control_wheel_sha256", "runtime_wheel_sha256"):
        digest = manifest.get(key)
        if not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError(f"postprocessing image manifest has a malformed {key}")
        int(digest, 16)

    report = {
        "schema_version": 1,
        "image_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "orchestration_source": source,
        "toolkit_source": os.environ.get("BSPP_TOOLKIT_SOURCE"),
        "source_commit": source_commit,
        "contract_version": importlib.metadata.version("bspp-orchestration-contract"),
        "control_version": importlib.metadata.version("bspp-orchestration-control"),
        "runtime_version": importlib.metadata.version("bspp-orchestration-runtime"),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
