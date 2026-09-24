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

"""Strict bounds and identity tests for postprocessing finalization bundles."""

from __future__ import annotations

from dataclasses import replace

import pytest

from bspp.orchestration.contract.phase import canonical_mapping_digest
from bspp.orchestration.contract.postprocessing_finalization_bundle import (
    POSTPROCESSING_EVIDENCE_TRANSFER_LIMITS_V1,
    POSTPROCESSING_FINALIZATION_FIXED_PATHS,
    PostprocessingAction09AssemblyWitness,
    PostprocessingBundleMemberIdentity,
    PostprocessingEvidenceTransferLimitsV1,
    PostprocessingFinalizationHandoffIndex,
    PostprocessingTarManifest,
    PostprocessingTarMemberIdentity,
    postprocessing_handoff_index_from_mapping,
    postprocessing_tar_manifest_from_mapping,
    validate_postprocessing_bundle_relative_path,
)

RUN_ID = "phase-run-0123456789abcdef0123456789abcdef"
ATTEMPT_ID = "attempt-0001"
SHA = "1" * 64


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("max_indexed_files", 97),
        ("max_relative_path_depth", 9),
        ("max_relative_path_utf8_bytes", 513),
        ("max_path_component_utf8_bytes", 129),
        ("max_file_bytes", 16_777_217),
        ("max_aggregate_bytes", 134_217_729),
        ("max_tar_manifests", 65),
        ("max_members_per_tar_manifest", 16_385),
        ("max_total_tar_members", 131_073),
        ("permitted_suffixes", (".json", ".txt")),
    ),
)
def test_transfer_limits_are_versioned_immutable_constants(field: str, value: object) -> None:
    with pytest.raises(ValueError, match="immutable"):
        replace(POSTPROCESSING_EVIDENCE_TRANSFER_LIMITS_V1, **{field: value})


@pytest.mark.parametrize(
    "value",
    (
        "",
        "/absolute.json",
        "a/../b.json",
        "a/./b.json",
        "a//b.json",
        "a\\b.json",
        "a/b.txt",
        "a/\x00b.json",
        "a/\x1fb.json",
        "a/" + "b" * 129 + ".json",
        "/".join(("a",) * 8) + "/b.json",
        "e\u0301.json",
    ),
)
def test_bundle_relative_paths_reject_unsafe_or_over_limit_values(value: str) -> None:
    with pytest.raises(ValueError):
        validate_postprocessing_bundle_relative_path(value)


def test_bundle_relative_paths_accept_v1_byte_and_depth_boundaries() -> None:
    component = "a" * 123 + ".json"
    path = "/".join(("a",) * 7 + (component,))
    assert len(path.split("/")) == PostprocessingEvidenceTransferLimitsV1().max_relative_path_depth
    assert len(component.encode()) == PostprocessingEvidenceTransferLimitsV1().max_path_component_utf8_bytes
    assert validate_postprocessing_bundle_relative_path(path) == path


def test_tar_manifest_round_trip_binds_members_but_not_physical_stat_identity() -> None:
    members = (
        PostprocessingTarMemberIdentity(path="a/member.json.zst", sha256="2" * 64, size_bytes=3),
        PostprocessingTarMemberIdentity(path="b/member.cif.zst", sha256="3" * 64, size_bytes=4),
    )
    identity = {
        "schema_version": 1,
        "manifest_kind": "postprocessing-tar-manifest-v1",
        "tar_path": "local_tars/shard_1/batch_0.tar",
        "members": [item.to_mapping() for item in members],
    }
    manifest = PostprocessingTarManifest(
        tar_path="local_tars/shard_1/batch_0.tar",
        tar_size_bytes=100,
        stat_device=1,
        stat_inode=2,
        stat_mtime_ns=3,
        members=members,
        manifest_id=canonical_mapping_digest(identity),
    )

    assert postprocessing_tar_manifest_from_mapping(manifest.to_mapping()) == manifest
    assert (
        replace(manifest, tar_size_bytes=999, stat_device=99, stat_inode=100, stat_mtime_ns=101).manifest_id
        == manifest.manifest_id
    )
    with pytest.raises(ValueError, match="manifest id"):
        replace(manifest, manifest_id="4" * 64)
    with pytest.raises(ValueError, match="unsafe"):
        PostprocessingTarMemberIdentity(path="../escape", sha256=SHA, size_bytes=1)


def test_action09_witness_excludes_index_and_aggregate_to_avoid_a_hash_cycle() -> None:
    witness = PostprocessingAction09AssemblyWitness(
        phase_run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        phase_runspec_digest=SHA,
        action_graph_digest=SHA,
        action_id="postprocessing-09-acceptance-adjudication",
        runtime_action_digest=SHA,
        command_digest=SHA,
        assembled_at="2026-09-03T12:00:00Z",
        intended_members=(_member("acceptance/adjudication.json"),),
    )
    assert witness.publication_claim == "none"
    for circular in ("handoff-index.json", "aggregate-action-evidence.json"):
        with pytest.raises(ValueError, match="circular"):
            replace(witness, intended_members=(_member(circular),))


def test_handoff_index_round_trip_requires_exact_layout_and_declared_counts() -> None:
    dynamic = "outputs/tar-manifests/" + "a" * 64 + ".json"
    paths = tuple(sorted((*POSTPROCESSING_FINALIZATION_FIXED_PATHS, dynamic)))
    members = tuple(_member(path) for path in paths)
    index = PostprocessingFinalizationHandoffIndex(
        phase_run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        phase_runspec_digest=SHA,
        action_graph_digest=SHA,
        execution_projection_sha256=SHA,
        acceptance_policy_sha256=SHA,
        members=members,
        tar_manifest_member_counts=((dynamic, 6_977),),
        declared_file_count=len(members),
        declared_aggregate_bytes=len(members),
        declared_tar_manifest_count=1,
        declared_total_tar_members=6_977,
    )

    assert postprocessing_handoff_index_from_mapping(index.to_mapping()) == index
    without_fixed = tuple(item for item in members if item.path != "acceptance/adjudication.json")
    with pytest.raises(ValueError, match="layout"):
        replace(index, members=without_fixed, declared_file_count=len(members) - 1)
    with pytest.raises(ValueError, match="counts or sizes"):
        replace(index, declared_aggregate_bytes=len(members) + 1)
    with pytest.raises(ValueError, match="member count"):
        replace(index, tar_manifest_member_counts=((dynamic, 16_385),), declared_total_tar_members=16_385)


def test_handoff_index_accepts_maximum_tar_manifest_count_and_rejects_plus_one() -> None:
    dynamic = tuple(f"outputs/tar-manifests/{value:064x}.json" for value in range(64))
    paths = tuple(sorted((*POSTPROCESSING_FINALIZATION_FIXED_PATHS, *dynamic)))
    members = tuple(_member(path) for path in paths)
    index = PostprocessingFinalizationHandoffIndex(
        phase_run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        phase_runspec_digest=SHA,
        action_graph_digest=SHA,
        execution_projection_sha256=SHA,
        acceptance_policy_sha256=SHA,
        members=members,
        tar_manifest_member_counts=tuple((path, 1) for path in dynamic),
        declared_file_count=len(members),
        declared_aggregate_bytes=len(members),
        declared_tar_manifest_count=64,
        declared_total_tar_members=64,
    )
    assert index.declared_tar_manifest_count == 64

    extra = f"outputs/tar-manifests/{64:064x}.json"
    with pytest.raises(ValueError, match="counts or sizes"):
        replace(
            index,
            members=tuple(sorted((*members, _member(extra)), key=lambda item: item.path)),
            tar_manifest_member_counts=tuple((path, 1) for path in (*dynamic, extra)),
            declared_file_count=len(members) + 1,
            declared_aggregate_bytes=len(members) + 1,
            declared_tar_manifest_count=65,
            declared_total_tar_members=65,
        )


def _member(path: str) -> PostprocessingBundleMemberIdentity:
    return PostprocessingBundleMemberIdentity(path=path, sha256=SHA, size_bytes=1)
