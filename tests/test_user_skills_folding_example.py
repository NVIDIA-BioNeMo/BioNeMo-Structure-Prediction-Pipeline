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

"""Loader and comment-coverage tests for the public folding Phase Plan example.

The committed example in ``skills/examples/run-plans/folding-phase-plan.yaml`` is a
non-runnable schema-complete folding Phase Plan. These tests prove that its active
YAML loads through the strict ``phase_plan_family_from_mapping`` dispatcher into a
``FoldingPhasePlan`` and that its comments enumerate every accepted authored field
and every enum alternative that would be misleading if left active.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from bspp.orchestration.contract.phase import (
    FoldingPhasePlan,
    PhaseSlurmResources,
    phase_plan_family_from_mapping,
)
from bspp.orchestration.contract.preprocessing_handoff import (
    msa_artifact_set_id,
    verified_local_bundled_artifact_location_id,
)
from bspp.orchestration.control.folding_shard import (
    derive_fold_shard_projection,
    fold_shard_targets_from_plan,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = ROOT / "skills" / "examples" / "run-plans" / "folding-phase-plan.yaml"

FIELD_TOKENS = (
    "schema_version",
    "phase_kind",
    "target_cluster",
    "input_location",
    "payload",
    "artifact_location_id",
    "kind",
    "artifact_set_id",
    "tar_path",
    "bundle_path",
    "bundle_uri",
    "tar_size_bytes",
    "tar_sha256",
    "lz4_size_bytes",
    "lz4_sha256",
    "raw_tar_members",
    "members",
    "verified_at",
    "msa_set",
    "backend",
    "transport",
    "s3_prediction_prefix",
    "msa_set_manifest",
    "artifact_type",
    "expected_chunk_count",
    "member_a3m_paths",
    "requires_paired_query_header",
    "chunks",
    "member_count",
    "logical_bytes",
    "member_lengths",
    "chunk_name",
    "logical_path",
    "sha256",
    "member_name",
    "raw_member_name",
    "size_bytes",
)

ENUM_TOKENS = (
    "verified-local-bundled",
    "verified-remote-bundled",
    "openfold-cli",
    "bioir",
    "colabfold",
    "openfold-trt",
    "publish-to-s3",
    "local",
)


def _load_example() -> FoldingPhasePlan:
    raw = yaml.safe_load(EXAMPLE_PATH.read_text())
    assert isinstance(raw, dict)
    plan = phase_plan_family_from_mapping(raw)
    assert isinstance(plan, FoldingPhasePlan)
    return plan


def _comment_text() -> str:
    text = EXAMPLE_PATH.read_text()
    comment_lines = [line.split("#", 1)[1] for line in text.splitlines() if "#" in line]
    return "\n".join(comment_lines)


def test_folding_example_loads_as_folding_phase_plan() -> None:
    plan = _load_example()

    assert plan.phase_kind == "folding"
    assert plan.target_cluster == "my-cluster"
    assert plan.payload.backend == "openfold-cli"
    assert plan.payload.transport == "local"


def test_folding_example_cross_artifact_identity_is_consistent() -> None:
    plan = _load_example()

    assert plan.payload.msa_set_manifest is not None
    assert plan.payload.msa_set_manifest.artifact_set_id == plan.payload.msa_set.artifact_set_id
    assert plan.payload.msa_set.artifact_set_id == plan.input_location.artifact_set_id


def test_folding_example_comments_cover_every_field() -> None:
    comment_text = _comment_text()

    missing = [token for token in FIELD_TOKENS if token not in comment_text]
    assert missing == []


def test_folding_example_comments_cover_every_enum_alternative() -> None:
    comment_text = _comment_text()

    missing = [token for token in ENUM_TOKENS if token not in comment_text]
    assert missing == []


def test_folding_example_active_transport_selection_is_local() -> None:
    text = EXAMPLE_PATH.read_text()

    # "local" is also a substring of "verified-local-bundled"; the exact active
    # transport selection must be authored on its own line, not only in comments.
    assert "transport: local" in text


def test_folding_example_has_ordered_member_lengths_aligned_to_members() -> None:
    plan = _load_example()
    manifest = plan.payload.msa_set_manifest
    assert manifest is not None
    assert manifest.member_lengths == (1234, 2345)
    assert len(manifest.member_lengths) == len(plan.payload.msa_set.member_a3m_paths)
    assert manifest.member_lengths == tuple(member.size_bytes for member in plan.input_location.members)


def test_folding_example_artifact_set_identity_is_canonical() -> None:
    plan = _load_example()
    manifest = plan.payload.msa_set_manifest
    assert manifest is not None
    expected_set_id = msa_artifact_set_id(
        manifest.chunks,
        manifest.member_count,
        manifest.logical_bytes,
        member_lengths=manifest.member_lengths,
    )
    assert manifest.artifact_set_id == expected_set_id
    assert plan.payload.msa_set.artifact_set_id == expected_set_id
    assert plan.input_location.artifact_set_id == expected_set_id


def test_folding_example_location_identity_is_canonical() -> None:
    plan = _load_example()
    location = plan.input_location
    expected_location_id = verified_local_bundled_artifact_location_id(
        artifact_set_id=location.artifact_set_id,
        tar_path=location.tar_path,
        bundle_path=location.bundle_path,
        bundle_uri=location.bundle_uri,
        tar_size_bytes=location.tar_size_bytes,
        tar_sha256=location.tar_sha256,
        lz4_size_bytes=location.lz4_size_bytes,
        lz4_sha256=location.lz4_sha256,
        raw_tar_members=location.raw_tar_members,
        members=location.members,
    )
    assert location.artifact_location_id == expected_location_id


def test_folding_example_is_eligible_for_shard_derivation() -> None:
    plan = _load_example()

    targets = fold_shard_targets_from_plan(plan)
    assert targets == (("AF-0000000000000001", 1234), ("AF-0000000000000002", 2345))

    scalar_resources = PhaseSlurmResources(
        partition="gpu",
        cpus_per_task=4,
        memory="16G",
        time="01:00:00",
    )
    projection, binding = derive_fold_shard_projection(plan, scalar_resources, "attempt-0001")
    assert binding.worker_count == 1
    assert len(projection.ranks) == 1
    assert [target.target_id for target in projection.ranks[0].targets] == [
        "AF-0000000000000002",
        "AF-0000000000000001",
    ]
