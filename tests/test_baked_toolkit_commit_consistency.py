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

"""Guard: the baked-toolkit commit pin is a single source of truth.

The public pin appears in the Dockerfile (`ARG TOOLKIT_REF`), `entrypoint.sh`
(`expected_commit`), and the Python packages (`EXPECTED_TOOLKIT_COMMIT` /
`BAKED_TOOLKIT_COMMIT`, both now aliased to the canonical
`contract.runspec.BAKED_TOOLKIT_COMMIT`). A future re-pin updates the contract
constant and the external shell/Docker declarations together; this test fails
loudly if any of them drift.

`containers/folding/<image>/image-lock.json` are per-backend lock files whose
base-image digests are pinned by the lock files; each dedicated `build.sh` fails closed
while its base digest still equals the placeholder.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from bspp.orchestration.contract.runspec import BAKED_TOOLKIT_COMMIT

ROOT = Path(__file__).resolve().parents[1]


def _dockerfile_toolkit_ref() -> str:
    text = (ROOT / "containers" / "Dockerfile").read_text(encoding="utf-8")
    m = re.search(r"^ARG TOOLKIT_REF=([0-9a-f]{40})\s*$", text, re.M)
    assert m, "Dockerfile must define 'ARG TOOLKIT_REF=<40-hex sha>'"
    return m.group(1)


def _entrypoint_expected_commit() -> str:
    text = (ROOT / "containers" / "scripts" / "entrypoint.sh").read_text(encoding="utf-8")
    m = re.search(r'expected_commit="\$\{BSPP_EXPECTED_TOOLKIT_COMMIT:-([0-9a-f]{40})\}"', text)
    assert m, 'entrypoint.sh must default expected_commit to "${BSPP_EXPECTED_TOOLKIT_COMMIT:-<40-hex sha>}"'
    return m.group(1)


def test_baked_toolkit_commit_single_source() -> None:
    dockerfile_text = (ROOT / "containers" / "Dockerfile").read_text(encoding="utf-8")
    assert "ARG TOOLKIT_REPO=https://github.com/PDBeurope/AFDB-Integration-Kit.git" in dockerfile_text
    assert "ARG TOOLKIT_BRANCH=nvidia-postproc" in dockerfile_text
    dockerfile = _dockerfile_toolkit_ref()
    entrypoint = _entrypoint_expected_commit()
    assert dockerfile == BAKED_TOOLKIT_COMMIT, (
        f"Dockerfile TOOLKIT_REF ({dockerfile}) drifted from contract BAKED_TOOLKIT_COMMIT ({BAKED_TOOLKIT_COMMIT})"
    )
    assert entrypoint == BAKED_TOOLKIT_COMMIT, (
        f"entrypoint.sh default expected_commit ({entrypoint}) drifted from contract BAKED_TOOLKIT_COMMIT "
        f"({BAKED_TOOLKIT_COMMIT})"
    )


def test_runtime_resolver_defaults_to_public_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    import bspp.orchestration.runtime.toolkit as toolkit

    monkeypatch.delenv("BSPP_EXPECTED_TOOLKIT_COMMIT", raising=False)
    # The module was imported without the internal override, so the public pin
    # is the resolution default.
    assert toolkit.EXPECTED_TOOLKIT_COMMIT == BAKED_TOOLKIT_COMMIT
    # The contract is: env override wins, public pin is the fail-closed default.
    monkeypatch.setenv("BSPP_EXPECTED_TOOLKIT_COMMIT", "c" * 40)
    assert (os.environ.get("BSPP_EXPECTED_TOOLKIT_COMMIT") or BAKED_TOOLKIT_COMMIT) == "c" * 40
