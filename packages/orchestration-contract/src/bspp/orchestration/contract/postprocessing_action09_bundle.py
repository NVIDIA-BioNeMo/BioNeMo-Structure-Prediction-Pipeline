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

"""Focused postprocessing contracts extracted from postprocessing_finalization_bundle.py."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from bspp.orchestration.contract.phase import validate_phase_attempt_id, validate_phase_run_id
from bspp.orchestration.contract.postprocessing_phase_ids import POSTPROCESSING_ACTION_IDS
from bspp.orchestration.contract.postprocessing_transfer_limits import (
    POSTPROCESSING_EVIDENCE_TRANSFER_LIMITS_V1,
    POSTPROCESSING_FINALIZATION_FIXED_PATHS,
    PostprocessingEvidenceTransferLimitsV1,
    _limits_from_mapping,
    validate_postprocessing_bundle_relative_path,
)
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

_SHA256 = re.compile(r"[0-9a-f]{64}")


_TAR_MANIFEST_PATH = re.compile(r"outputs/tar-manifests/([0-9a-f]{64})\.json")


@dataclass(frozen=True)
class PostprocessingBundleMemberIdentity:
    path: str
    sha256: str
    size_bytes: int
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        validate_postprocessing_bundle_relative_path(self.path)
        _sha(self.sha256, "postprocessing finalization member SHA-256")
        if (
            not isinstance(self.size_bytes, int)
            or isinstance(self.size_bytes, bool)
            or self.size_bytes < 0
            or self.size_bytes > POSTPROCESSING_EVIDENCE_TRANSFER_LIMITS_V1.max_file_bytes
        ):
            raise ValueError("postprocessing finalization member size exceeds the v1 limit")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class PostprocessingAction09AssemblyWitness:
    phase_run_id: str
    attempt_id: str
    phase_runspec_digest: str
    action_graph_digest: str
    action_id: str
    runtime_action_digest: str
    command_digest: str
    assembled_at: str
    intended_members: tuple[PostprocessingBundleMemberIdentity, ...]
    publication_claim: Literal["none"] = "none"
    witness_kind: Literal["postprocessing-action09-prepublication-assembly-v1"] = (
        "postprocessing-action09-prepublication-assembly-v1"
    )
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        validate_phase_run_id(self.phase_run_id)
        validate_phase_attempt_id(self.attempt_id)
        for value in (
            self.phase_runspec_digest,
            self.action_graph_digest,
            self.runtime_action_digest,
            self.command_digest,
        ):
            _sha(value, "postprocessing action 09 witness digest")
        if self.action_id != POSTPROCESSING_ACTION_IDS["acceptance-adjudication"]:
            raise ValueError("postprocessing assembly witness must belong to action 09")
        if not self.assembled_at or self.publication_claim != "none":
            raise ValueError("postprocessing action 09 witness cannot claim publication")
        paths = tuple(item.path for item in self.intended_members)
        forbidden = {"handoff-index.json", "aggregate-action-evidence.json"}
        if not paths or paths != tuple(sorted(paths)) or len(set(paths)) != len(paths) or forbidden & set(paths):
            raise ValueError("postprocessing action 09 witness members are unordered, duplicated, or circular")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "witness_kind": self.witness_kind,
            "publication_claim": self.publication_claim,
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "phase_runspec_digest": self.phase_runspec_digest,
            "action_graph_digest": self.action_graph_digest,
            "action_id": self.action_id,
            "runtime_action_digest": self.runtime_action_digest,
            "command_digest": self.command_digest,
            "assembled_at": self.assembled_at,
            "intended_members": [item.to_mapping() for item in self.intended_members],
        }


@dataclass(frozen=True)
class PostprocessingFinalizationHandoffIndex:
    phase_run_id: str
    attempt_id: str
    phase_runspec_digest: str
    action_graph_digest: str
    execution_projection_sha256: str
    acceptance_policy_sha256: str
    members: tuple[PostprocessingBundleMemberIdentity, ...]
    tar_manifest_member_counts: tuple[tuple[str, int], ...]
    declared_file_count: int
    declared_aggregate_bytes: int
    declared_tar_manifest_count: int
    declared_total_tar_members: int
    limits: PostprocessingEvidenceTransferLimitsV1 = POSTPROCESSING_EVIDENCE_TRANSFER_LIMITS_V1
    layout_kind: Literal["postprocessing-finalization-layout-v1"] = "postprocessing-finalization-layout-v1"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        validate_phase_run_id(self.phase_run_id)
        validate_phase_attempt_id(self.attempt_id)
        for value in (
            self.phase_runspec_digest,
            self.action_graph_digest,
            self.execution_projection_sha256,
            self.acceptance_policy_sha256,
        ):
            _sha(value, "postprocessing handoff index digest")
        if self.layout_kind != self.limits.layout_kind or self.limits != POSTPROCESSING_EVIDENCE_TRANSFER_LIMITS_V1:
            raise ValueError("postprocessing handoff index must use immutable v1 transfer limits")
        paths = tuple(item.path for item in self.members)
        dynamic = tuple(path for path in paths if _TAR_MANIFEST_PATH.fullmatch(path) is not None)
        if (
            not paths
            or paths != tuple(sorted(paths))
            or len(set(paths)) != len(paths)
            or set(paths) != set(POSTPROCESSING_FINALIZATION_FIXED_PATHS) | set(dynamic)
            or len(paths) > self.limits.max_indexed_files
        ):
            raise ValueError("postprocessing handoff index layout is incomplete, extra, unordered, or oversized")
        counts = self.tar_manifest_member_counts
        if tuple(path for path, _ in counts) != dynamic or tuple(sorted(counts)) != counts:
            raise ValueError("postprocessing tar manifest counts differ from indexed manifests")
        if any(
            not isinstance(count, int)
            or isinstance(count, bool)
            or count <= 0
            or count > self.limits.max_members_per_tar_manifest
            for _, count in counts
        ):
            raise ValueError("postprocessing tar manifest member count exceeds the v1 limit")
        aggregate = sum(item.size_bytes for item in self.members)
        total_tar_members = sum(count for _, count in counts)
        if (
            self.declared_file_count != len(paths)
            or self.declared_aggregate_bytes != aggregate
            or aggregate > self.limits.max_aggregate_bytes
            or self.declared_tar_manifest_count != len(dynamic)
            or len(dynamic) > self.limits.max_tar_manifests
            or self.declared_total_tar_members != total_tar_members
            or total_tar_members > self.limits.max_total_tar_members
        ):
            raise ValueError("postprocessing handoff index declared counts or sizes are invalid")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "layout_kind": self.layout_kind,
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "phase_runspec_digest": self.phase_runspec_digest,
            "action_graph_digest": self.action_graph_digest,
            "execution_projection_sha256": self.execution_projection_sha256,
            "acceptance_policy_sha256": self.acceptance_policy_sha256,
            "limits": self.limits.to_mapping(),
            "members": [item.to_mapping() for item in self.members],
            "tar_manifest_member_counts": [
                {"path": path, "member_count": count} for path, count in self.tar_manifest_member_counts
            ],
            "declared_file_count": self.declared_file_count,
            "declared_aggregate_bytes": self.declared_aggregate_bytes,
            "declared_tar_manifest_count": self.declared_tar_manifest_count,
            "declared_total_tar_members": self.declared_total_tar_members,
        }


def postprocessing_handoff_index_from_mapping(
    payload: Mapping[str, object],
) -> PostprocessingFinalizationHandoffIndex:
    _fields(
        payload,
        {
            "schema_version",
            "layout_kind",
            "phase_run_id",
            "attempt_id",
            "phase_runspec_digest",
            "action_graph_digest",
            "execution_projection_sha256",
            "acceptance_policy_sha256",
            "limits",
            "members",
            "tar_manifest_member_counts",
            "declared_file_count",
            "declared_aggregate_bytes",
            "declared_tar_manifest_count",
            "declared_total_tar_members",
        },
        "PostprocessingFinalizationHandoffIndex",
    )
    members = _mapping_list(payload, "members")
    raw_counts = _mapping_list(payload, "tar_manifest_member_counts")
    counts: list[tuple[str, int]] = []
    for item in raw_counts:
        _fields(item, {"path", "member_count"}, "PostprocessingTarManifestMemberCount")
        counts.append((_string(item, "path"), _integer(item, "member_count")))
    limits_payload = _mapping(payload, "limits")
    limits = _limits_from_mapping(limits_payload)
    return PostprocessingFinalizationHandoffIndex(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="PostprocessingFinalizationHandoffIndex"
        ),
        layout_kind=cast("Literal['postprocessing-finalization-layout-v1']", _string(payload, "layout_kind")),
        phase_run_id=_string(payload, "phase_run_id"),
        attempt_id=_string(payload, "attempt_id"),
        phase_runspec_digest=_string(payload, "phase_runspec_digest"),
        action_graph_digest=_string(payload, "action_graph_digest"),
        execution_projection_sha256=_string(payload, "execution_projection_sha256"),
        acceptance_policy_sha256=_string(payload, "acceptance_policy_sha256"),
        limits=limits,
        members=tuple(_bundle_member_from_mapping(item) for item in members),
        tar_manifest_member_counts=tuple(counts),
        declared_file_count=_integer(payload, "declared_file_count"),
        declared_aggregate_bytes=_integer(payload, "declared_aggregate_bytes"),
        declared_tar_manifest_count=_integer(payload, "declared_tar_manifest_count"),
        declared_total_tar_members=_integer(payload, "declared_total_tar_members"),
    )


def postprocessing_action09_assembly_witness_from_mapping(
    payload: Mapping[str, object],
) -> PostprocessingAction09AssemblyWitness:
    _fields(
        payload,
        {
            "schema_version",
            "witness_kind",
            "publication_claim",
            "phase_run_id",
            "attempt_id",
            "phase_runspec_digest",
            "action_graph_digest",
            "action_id",
            "runtime_action_digest",
            "command_digest",
            "assembled_at",
            "intended_members",
        },
        "PostprocessingAction09AssemblyWitness",
    )
    if (
        _string(payload, "witness_kind") != "postprocessing-action09-prepublication-assembly-v1"
        or _string(payload, "publication_claim") != "none"
    ):
        raise ValueError("unsupported postprocessing Action 09 witness discriminator")
    return PostprocessingAction09AssemblyWitness(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="PostprocessingAction09AssemblyWitness"
        ),
        witness_kind="postprocessing-action09-prepublication-assembly-v1",
        publication_claim="none",
        phase_run_id=_string(payload, "phase_run_id"),
        attempt_id=_string(payload, "attempt_id"),
        phase_runspec_digest=_string(payload, "phase_runspec_digest"),
        action_graph_digest=_string(payload, "action_graph_digest"),
        action_id=_string(payload, "action_id"),
        runtime_action_digest=_string(payload, "runtime_action_digest"),
        command_digest=_string(payload, "command_digest"),
        assembled_at=_string(payload, "assembled_at"),
        intended_members=tuple(
            _bundle_member_from_mapping(item) for item in _mapping_list(payload, "intended_members")
        ),
    )


def _bundle_member_from_mapping(payload: Mapping[str, object]) -> PostprocessingBundleMemberIdentity:
    _fields(payload, {"schema_version", "path", "sha256", "size_bytes"}, "PostprocessingBundleMemberIdentity")
    return PostprocessingBundleMemberIdentity(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="PostprocessingBundleMemberIdentity"
        ),
        path=_string(payload, "path"),
        sha256=_string(payload, "sha256"),
        size_bytes=_integer(payload, "size_bytes"),
    )


def _sha(value: str, label: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase SHA-256")


def _schema(value: int, record: str) -> None:
    validate_schema_version(value, record_name=record)


def _fields(payload: Mapping[str, object], allowed: set[str], record: str) -> None:
    unknown = sorted(set(payload) - allowed)
    missing = sorted(allowed - set(payload))
    if unknown or missing:
        raise ValueError(f"{record} fields differ; missing={missing!r}, unknown={unknown!r}")


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _mapping_list(payload: Mapping[str, object], key: str) -> tuple[Mapping[str, object], ...]:
    value = payload.get(key)
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"{key} must be a list of mappings")
    return tuple(cast("Mapping[str, object]", item) for item in value)


def _string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _integer(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


__all__ = [
    "PostprocessingAction09AssemblyWitness",
    "PostprocessingBundleMemberIdentity",
    "PostprocessingFinalizationHandoffIndex",
    "postprocessing_action09_assembly_witness_from_mapping",
    "postprocessing_handoff_index_from_mapping",
]
