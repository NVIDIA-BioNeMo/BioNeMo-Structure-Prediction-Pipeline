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

"""Cross-artifact integration tests for the public user-skills run-plan examples.

These tests exercise contracts that span story-owned artifacts: the preprocessing
Phase Plan resolves against the shared Cluster Profile, and its chunk execution
intent matches the image-fixed runtime semantics enforced at materialization.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from bspp.orchestration.contract.phase import PhasePlan, phase_plan_from_mapping
from bspp.orchestration.control.phase_attempt_materialization import (
    _verify_preprocessing_runtime_semantics,
)
from bspp.orchestration.control.profiles import (
    resolve_cluster_profile,
    resolve_database_manifest_path,
)

ROOT = Path(__file__).resolve().parents[1]
RUN_PLANS = ROOT / "skills" / "examples" / "run-plans"
PREPROCESSING_PLAN = RUN_PLANS / "preprocessing-phase-plan.yaml"
SHARED_PROFILE = RUN_PLANS / "cluster-profile.yaml"

_FIXED_EXECUTABLES = {
    "mmseqs_executable": "/usr/local/bin/mmseqs",
    "colabfold_search_executable": "/usr/local/bin/colabfold_search",
    "tar_executable": "/usr/bin/tar",
    "lz4_executable": "/usr/bin/lz4",
}


def _load_preprocessing_plan() -> PhasePlan:
    payload = yaml.safe_load(PREPROCESSING_PLAN.read_text())
    assert isinstance(payload, dict)
    return phase_plan_from_mapping(payload)


def test_preprocessing_plan_resolves_database_selection_against_shared_profile(
    tmp_path: Path,
) -> None:
    plan = _load_preprocessing_plan()

    # A minimal repo template supplies the illustrative `my-cluster` key without
    # adding it to the governed production template.
    template = tmp_path / "cluster_profile_templates.yaml"
    template.write_text("clusters:\n  my-cluster:\n    account: my-account\n")

    profile = resolve_cluster_profile("my-cluster", config_path=SHARED_PROFILE, template_path=template)

    manifest_path = resolve_database_manifest_path(profile, plan.payload.database)

    assert plan.payload.database.database_set.identifier == "my-database-set"
    assert plan.payload.database.database_set.version == "1"
    assert plan.payload.database.requested_policy.value == "direct"
    assert str(manifest_path) == "/data/bspp/database/manifest.json"


def test_preprocessing_example_matches_fixed_runtime_semantics() -> None:
    plan = _load_preprocessing_plan()
    site = plan.payload.chunk_execution_intent.site

    _verify_preprocessing_runtime_semantics(plan.payload.chunk_execution_intent)

    for field_name, expected_path in _FIXED_EXECUTABLES.items():
        assert getattr(site, field_name) == expected_path
    assert site.container_image == "/data/bspp/containers/bspp-orchestration-preprocessing.sqsh"
