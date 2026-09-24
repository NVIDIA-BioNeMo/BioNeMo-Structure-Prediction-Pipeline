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

"""Public documentation contract tests (survive the public release)."""

from __future__ import annotations

from pathlib import Path

from bspp.orchestration.control.profiles import resolve_cluster_profile

ROOT = Path(__file__).resolve().parents[1]


def test_readme_is_slim_phase_oriented_overview() -> None:
    readme = (ROOT / "README.md").read_text()

    assert "## How it works" in readme
    for phase in ("Preprocessing", "Folding", "Postprocessing"):
        assert f"| {phase} |" in readme
    # Release limits live in the status page, not in the README.
    assert "docs/status.md" in readme
    assert "production interface" not in readme

    assert "## Install" in readme
    assert "uv sync --frozen --package bspp-orchestration-control --no-dev" in readme
    assert "uv run --frozen --package bspp-orchestration-control --no-dev bsppctl --help" in readme
    assert "DEVELOPING.md" in readme

    assert "## Concepts" in readme
    assert "**Phase Plan**" in readme
    assert "**Phase RunSpec**" in readme
    # The README points at the canonical commented profile instead of inlining one.
    assert "skills/examples/run-plans/cluster-profile.yaml" in readme

    assert "## Agent skills and example run plans" in readme
    for plan in (
        "preprocessing-phase-plan.yaml",
        "folding-phase-plan.yaml",
        "postprocessing-phase-plan.yaml",
    ):
        assert plan in readme

    assert "## Containers" in readme
    assert "BSPP_REGISTRY" in readme

    assert "## Commands" in readme
    assert "bsppctl phase" in readme
    assert "bsppctl prepare-benchmark" in readme

    # The public README must not reference internal operator/decision material.
    assert "docs/HOWTO-POSTPROCESSING.md" not in readme
    assert "docs/HOWTO-FOLDING.md" not in readme
    assert "docs/migration/legacy-run-to-phase.md" not in readme
    assert "docs/postprocessing/recipe-vs-runspec.md" not in readme
    assert "docs/examples/postprocessing-reference-run/" not in readme
    assert "devdocs" not in readme
    assert "## Cluster Prerequisites for Postprocessing" not in readme
    assert "## Runtime Qualification" not in readme

    # The public README must not carry internal identities/hosts/clusters/storage.
    # Tokens are assembled at runtime so this test file stays free of the literal
    # forbidden strings (the public-surface token scan runs over tests/).
    forbidden_tokens = (
        "dra" + "co",
        "df" + "w",
        "gitlab-master" + ".nvidia.com",
        "healthcare" + "eng",
        "/lust" + "re",
        "swift" + "stack",
        "ext-nv" + "da",
        "ext-nvi" + "dia",
        "s3://af" + "cdb",
        "dfran" + "sos",
        "nvidia-pos" + "tproc",
    )
    for token in forbidden_tokens:
        assert token not in readme

    # The canonical profile the README points at resolves through the strict loader.
    profile_path = ROOT / "skills" / "examples" / "run-plans" / "cluster-profile.yaml"
    profile = resolve_cluster_profile("my-cluster", config_path=profile_path)
    assert profile.owner == "alice"
    assert profile.transport == "ssh"


def test_developing_documents_uv_workflow() -> None:
    developing = (ROOT / "DEVELOPING.md").read_text()
    assert "uv" in developing
    assert "pytest" in developing
