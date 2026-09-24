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

"""Cross-image pixi pin consistency tests.

The pixi binary is pinned (version + sha256) in three places — the folding
runtime image-lock, the postprocessing Dockerfile ARG defaults, and the
preprocessing image-lock. These must stay identical so a future pixi upgrade
cannot silently drift one image onto a different binary.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_PIXI_VERSION = "0.80.0"
EXPECTED_PIXI_SHA256 = "387a2d3052e656f61ccf735e6750255451366f45635a2da09116b1f8394b2837"
EXPECTED_PIXI_URL = (
    f"https://github.com/prefix-dev/pixi/releases/download/v{EXPECTED_PIXI_VERSION}/pixi-x86_64-unknown-linux-musl"
)


def _pixi_pins() -> dict[str, tuple[str, str, str]]:
    folding = json.loads((ROOT / "containers" / "folding" / "runtime" / "image-lock.json").read_text())["pixi"]
    preprocessing = json.loads((ROOT / "containers" / "preprocessing" / "image-lock.json").read_text())["pixi"]
    dockerfile = (ROOT / "containers" / "Dockerfile").read_text()
    postprocessing_version = dockerfile.split("ARG PIXI_VERSION=", maxsplit=1)[1].splitlines()[0]
    postprocessing_sha = dockerfile.split("ARG PIXI_SHA256=", maxsplit=1)[1].splitlines()[0]
    return {
        "folding-runtime": (folding["version"], folding["artifact_sha256"], folding["artifact_url"]),
        "postprocessing": (
            postprocessing_version,
            postprocessing_sha,
            f"https://github.com/prefix-dev/pixi/releases/download/v{postprocessing_version}/pixi-x86_64-unknown-linux-musl",
        ),
        "preprocessing": (preprocessing["version"], preprocessing["artifact_sha256"], preprocessing["artifact_url"]),
    }


def test_pixi_pin_is_identical_across_all_images() -> None:
    pins = _pixi_pins()
    for image, (version, sha256, url) in pins.items():
        assert version == EXPECTED_PIXI_VERSION, f"{image} pixi version drifted: {version}"
        assert sha256 == EXPECTED_PIXI_SHA256, f"{image} pixi sha256 drifted"
        assert url == EXPECTED_PIXI_URL, f"{image} pixi url drifted: {url}"
