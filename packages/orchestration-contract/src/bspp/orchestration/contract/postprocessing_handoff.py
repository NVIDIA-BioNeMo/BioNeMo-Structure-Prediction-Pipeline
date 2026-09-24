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

"""Focused postprocessing contracts extracted from postprocessing_receipt.py."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from bspp.orchestration.contract.phase import (
    canonical_mapping_digest,
    validate_phase_attempt_id,
    validate_phase_run_id,
)
from bspp.orchestration.contract.postprocessing_artifacts import (
    PostprocessingEvidenceArtifact,
    PostprocessingLogicalArtifactSet,
    _artifact_set,
)
from bspp.orchestration.contract.postprocessing_phase_ids import validate_postprocessing_action_id
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

_SHA256 = re.compile(r"[0-9a-f]{64}")


_JOB_ID = re.compile(r"[0-9]+")


_HANDOFF_ID = re.compile(r"postprocessing-handoff-[0-9a-f]{64}")


_ARTIFACT_SET_ID = re.compile(r"postprocessing-artifact-set-[0-9a-f]{64}")


_LOCATION_ID = re.compile(r"postprocessing-artifact-location-[0-9a-f]{64}")


@dataclass(frozen=True)
class PostprocessingArtifactPlacement:
    logical_path: str
    physical_path: str
    sha256: str
    size_bytes: int
    verification_kind: Literal["content-sha256-v1", "inventory-metadata-v1"] = "content-sha256-v1"
    verification_source_path: str | None = None
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        for value in (self.logical_path, self.physical_path):
            if not value or value.startswith("/") or ".." in value.split("/"):
                raise ValueError("postprocessing Artifact placement paths must be root-relative")
        _sha(self.sha256)
        if not isinstance(self.size_bytes, int) or isinstance(self.size_bytes, bool) or self.size_bytes < 0:
            raise ValueError("postprocessing Artifact placement size must be non-negative")
        PostprocessingEvidenceArtifact(
            path=self.logical_path,
            sha256=self.sha256,
            size_bytes=self.size_bytes,
            verification_kind=self.verification_kind,
            verification_source_path=self.verification_source_path,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "logical_path": self.logical_path,
            "physical_path": self.physical_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "verification_kind": self.verification_kind,
            "verification_source_path": self.verification_source_path,
        }


@dataclass(frozen=True)
class PostprocessingVerifiedArtifactLocation:
    artifact_location_id: str
    artifact_set_id: str
    root: str
    verified_members_digest: str
    verified_at: str
    placements: tuple[PostprocessingArtifactPlacement, ...]
    location_kind: Literal["verified-local-output-tree-v1", "verified-object-output-tree-v1"]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        if (
            self.location_kind not in {"verified-local-output-tree-v1", "verified-object-output-tree-v1"}
            or _ARTIFACT_SET_ID.fullmatch(self.artifact_set_id) is None
            or not self.verified_at
        ):
            raise ValueError("postprocessing physical Artifact Location fields are invalid")
        if self.location_kind == "verified-local-output-tree-v1" and not self.root.startswith("/"):
            raise ValueError("verified local output root must be absolute")
        if self.location_kind == "verified-object-output-tree-v1" and "://" not in self.root:
            raise ValueError("verified object output root must be a URI")
        _sha(self.verified_members_digest)
        logical_paths = tuple(item.logical_path for item in self.placements)
        physical_paths = tuple(item.physical_path for item in self.placements)
        if (
            not self.placements
            or logical_paths != tuple(sorted(logical_paths))
            or len(set(logical_paths)) != len(logical_paths)
            or len(set(physical_paths)) != len(physical_paths)
            or self.verified_members_digest
            != canonical_mapping_digest(
                {
                    "schema_version": CURRENT_CONTRACT_SCHEMA_VERSION,
                    "placements": [item.to_mapping() for item in self.placements],
                }
            )
        ):
            raise ValueError("postprocessing physical Artifact placements are invalid")
        if _LOCATION_ID.fullmatch(self.artifact_location_id) is None or self.artifact_location_id != (
            "postprocessing-artifact-location-" + canonical_mapping_digest(self.identity_mapping())
        ):
            raise ValueError("postprocessing physical Artifact Location id differs from its identity")

    def identity_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "location_kind": self.location_kind,
            "artifact_set_id": self.artifact_set_id,
            "root": self.root,
            "verified_members_digest": self.verified_members_digest,
            "verified_at": self.verified_at,
            "placements": [item.to_mapping() for item in self.placements],
        }

    def to_mapping(self) -> dict[str, object]:
        return {"artifact_location_id": self.artifact_location_id, **self.identity_mapping()}


@dataclass(frozen=True)
class PostprocessingTaskReceiptEvidence:
    scheduler_job_id: str
    state: Literal["COMPLETED"]
    exit_code: Literal["0", "0:0"]
    source: Literal["sacct"]
    restarts: int | None = None
    task_index: int | None = None
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        if re.fullmatch(r"[0-9]+(?:_[0-9]+)?", self.scheduler_job_id) is None:
            raise ValueError("postprocessing receipt task scheduler id is invalid")
        if self.state != "COMPLETED" or self.exit_code not in {"0", "0:0"} or self.source != "sacct":
            raise ValueError("postprocessing receipt task must be successful sacct evidence")
        if self.restarts is not None and (
            not isinstance(self.restarts, int) or isinstance(self.restarts, bool) or self.restarts < 0
        ):
            raise ValueError("postprocessing receipt task restarts must be a non-negative integer")
        if self.task_index is not None and (
            not isinstance(self.task_index, int) or isinstance(self.task_index, bool) or self.task_index < 0
        ):
            raise ValueError("postprocessing receipt task index must be non-negative")

    def to_mapping(self) -> dict[str, object]:
        mapping: dict[str, object] = {
            "schema_version": self.schema_version,
            "task_index": self.task_index,
            "scheduler_job_id": self.scheduler_job_id,
            "state": self.state,
            "exit_code": self.exit_code,
            "source": self.source,
        }
        if self.restarts is not None:
            mapping["restarts"] = self.restarts
        return mapping


@dataclass(frozen=True)
class PostprocessingActionReceiptEvidence:
    action_id: str
    runtime_action_digest: str
    parent_job_id: str
    expected_task_indexes: tuple[int, ...]
    tasks: tuple[PostprocessingTaskReceiptEvidence, ...]
    outcome: Literal["succeeded"] = "succeeded"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        validate_postprocessing_action_id(self.action_id)
        _sha(self.runtime_action_digest)
        if _JOB_ID.fullmatch(self.parent_job_id) is None or not self.tasks or self.outcome != "succeeded":
            raise ValueError("postprocessing receipt action requires successful parent/task evidence")
        if tuple(sorted(set(self.expected_task_indexes))) != self.expected_task_indexes:
            raise ValueError("postprocessing receipt expected task indexes must be sorted and unique")
        if self.expected_task_indexes:
            if tuple(item.task_index for item in self.tasks) != self.expected_task_indexes or any(
                item.scheduler_job_id != f"{self.parent_job_id}_{item.task_index}" for item in self.tasks
            ):
                raise ValueError("postprocessing receipt array tasks differ from the exact expected set")
        elif (
            len(self.tasks) != 1
            or self.tasks[0].task_index is not None
            or self.tasks[0].scheduler_job_id != self.parent_job_id
        ):
            raise ValueError("postprocessing receipt non-array action requires its parent task")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "action_id": self.action_id,
            "runtime_action_digest": self.runtime_action_digest,
            "parent_job_id": self.parent_job_id,
            "expected_task_indexes": list(self.expected_task_indexes),
            "tasks": [item.to_mapping() for item in self.tasks],
            "outcome": self.outcome,
        }


@dataclass(frozen=True)
class PostprocessingOutputHandoff:
    handoff_id: str
    phase_run_id: str
    attempt_id: str
    phase_runspec_digest: str
    legacy_run_id: str
    output_namespace: str
    output_dir: str
    object_prefix: str
    artifact_set: PostprocessingLogicalArtifactSet
    physical_locations: tuple[PostprocessingVerifiedArtifactLocation, ...]
    handoff_kind: str = "postprocessing-output-handoff-v1"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        validate_phase_run_id(self.phase_run_id)
        validate_phase_attempt_id(self.attempt_id)
        _sha(self.phase_runspec_digest)
        if self.handoff_kind != "postprocessing-output-handoff-v1":
            raise ValueError("unsupported postprocessing output handoff kind")
        if not all((self.legacy_run_id, self.output_namespace, self.output_dir, self.object_prefix)):
            raise ValueError("postprocessing output handoff paths and names must be non-empty")
        location_ids = tuple(item.artifact_location_id for item in self.physical_locations)
        logical_members = {item.path: item for item in self.artifact_set.flattened_members}
        local_locations = tuple(
            item for item in self.physical_locations if item.location_kind == "verified-local-output-tree-v1"
        )
        if (
            not self.physical_locations
            or location_ids != tuple(sorted(location_ids))
            or len(set(location_ids)) != len(location_ids)
            or len(local_locations) != 1
            or any(item.artifact_set_id != self.artifact_set.artifact_set_id for item in self.physical_locations)
        ):
            raise ValueError("postprocessing output handoff physical locations must verify the exact Artifact Set")
        for location in self.physical_locations:
            placements = {item.logical_path: item for item in location.placements}
            if not set(placements) <= set(logical_members) or any(
                (placement.sha256, placement.size_bytes)
                != (logical_members[path].sha256, logical_members[path].size_bytes)
                or placement.verification_kind != logical_members[path].verification_kind
                or placement.verification_source_path != logical_members[path].verification_source_path
                for path, placement in placements.items()
            ):
                raise ValueError("postprocessing physical placements differ from the logical Artifact Set")
            if location.location_kind == "verified-object-output-tree-v1" and any(
                not path.startswith("postprocessing-output/scientific-output/") for path in placements
            ):
                raise ValueError("object location may only place scientific output members")
        if {item.logical_path for item in local_locations[0].placements} != set(logical_members):
            raise ValueError("verified local location must place every logical output member")
        if _HANDOFF_ID.fullmatch(self.handoff_id) is None or self.handoff_id != postprocessing_handoff_id(
            self.identity_mapping()
        ):
            raise ValueError("postprocessing output handoff id differs from its identity")

    def identity_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "handoff_kind": self.handoff_kind,
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "phase_runspec_digest": self.phase_runspec_digest,
            "legacy_run_id": self.legacy_run_id,
            "output_namespace": self.output_namespace,
            "output_dir": self.output_dir,
            "object_prefix": self.object_prefix,
            "artifact_set": self.artifact_set.to_mapping(),
            "physical_locations": [item.to_mapping() for item in self.physical_locations],
        }

    def to_mapping(self) -> dict[str, object]:
        return {"handoff_id": self.handoff_id, **self.identity_mapping()}


def postprocessing_handoff_id(payload: Mapping[str, object]) -> str:
    return "postprocessing-handoff-" + canonical_mapping_digest(payload)


def _action(payload: Mapping[str, object]) -> PostprocessingActionReceiptEvidence:
    _fields(
        payload,
        {
            "schema_version",
            "action_id",
            "runtime_action_digest",
            "parent_job_id",
            "expected_task_indexes",
            "tasks",
            "outcome",
        },
        "PostprocessingActionReceiptEvidence",
    )
    expected = payload.get("expected_task_indexes")
    if not isinstance(expected, list) or any(not isinstance(item, int) or isinstance(item, bool) for item in expected):
        raise ValueError("postprocessing receipt task indexes must be integers")
    if payload.get("outcome") != "succeeded":
        raise ValueError("postprocessing receipt action outcome must be succeeded")
    return PostprocessingActionReceiptEvidence(
        schema_version=_schema(payload, "PostprocessingActionReceiptEvidence"),
        action_id=_string(payload, "action_id"),
        runtime_action_digest=_string(payload, "runtime_action_digest"),
        parent_job_id=_string(payload, "parent_job_id"),
        expected_task_indexes=tuple(expected),
        tasks=tuple(_task(item) for item in _mapping_list(payload, "tasks")),
        outcome="succeeded",
    )


def _task(payload: Mapping[str, object]) -> PostprocessingTaskReceiptEvidence:
    allowed = {"schema_version", "task_index", "scheduler_job_id", "state", "exit_code", "source", "restarts"}
    if set(payload) != allowed and set(payload) != allowed - {"restarts"}:
        raise ValueError("PostprocessingTaskReceiptEvidence has missing or extra fields")
    index = payload.get("task_index")
    if index is not None and (not isinstance(index, int) or isinstance(index, bool)):
        raise ValueError("postprocessing receipt task index must be integer or null")
    restarts = payload.get("restarts")
    if restarts is not None and (not isinstance(restarts, int) or isinstance(restarts, bool) or restarts < 0):
        raise ValueError("postprocessing receipt task restarts must be a non-negative integer")
    return PostprocessingTaskReceiptEvidence(
        schema_version=_schema(payload, "PostprocessingTaskReceiptEvidence"),
        task_index=index,
        scheduler_job_id=_string(payload, "scheduler_job_id"),
        state=cast("Literal['COMPLETED']", _string(payload, "state")),
        exit_code=cast("Literal['0', '0:0']", _string(payload, "exit_code")),
        source=cast("Literal['sacct']", _string(payload, "source")),
        restarts=restarts,
    )


def _handoff(payload: Mapping[str, object]) -> PostprocessingOutputHandoff:
    _fields(
        payload,
        {
            "schema_version",
            "handoff_kind",
            "handoff_id",
            "phase_run_id",
            "attempt_id",
            "phase_runspec_digest",
            "legacy_run_id",
            "output_namespace",
            "output_dir",
            "object_prefix",
            "artifact_set",
            "physical_locations",
        },
        "PostprocessingOutputHandoff",
    )
    return PostprocessingOutputHandoff(
        schema_version=_schema(payload, "PostprocessingOutputHandoff"),
        handoff_kind=_string(payload, "handoff_kind"),
        handoff_id=_string(payload, "handoff_id"),
        phase_run_id=_string(payload, "phase_run_id"),
        attempt_id=_string(payload, "attempt_id"),
        phase_runspec_digest=_string(payload, "phase_runspec_digest"),
        legacy_run_id=_string(payload, "legacy_run_id"),
        output_namespace=_string(payload, "output_namespace"),
        output_dir=_string(payload, "output_dir"),
        object_prefix=_string(payload, "object_prefix"),
        artifact_set=_artifact_set(_mapping(payload, "artifact_set")),
        physical_locations=tuple(_location(item) for item in _mapping_list(payload, "physical_locations")),
    )


def postprocessing_output_handoff_from_mapping(
    payload: Mapping[str, object],
) -> PostprocessingOutputHandoff:
    """Parse and validate one published postprocessing output handoff."""
    return _handoff(payload)


def _location(payload: Mapping[str, object]) -> PostprocessingVerifiedArtifactLocation:
    _fields(
        payload,
        {
            "schema_version",
            "location_kind",
            "artifact_location_id",
            "artifact_set_id",
            "root",
            "verified_members_digest",
            "verified_at",
            "placements",
        },
        "PostprocessingVerifiedArtifactLocation",
    )
    return PostprocessingVerifiedArtifactLocation(
        schema_version=_schema(payload, "PostprocessingVerifiedArtifactLocation"),
        location_kind=cast(
            "Literal['verified-local-output-tree-v1', 'verified-object-output-tree-v1']",
            _string(payload, "location_kind"),
        ),
        artifact_location_id=_string(payload, "artifact_location_id"),
        artifact_set_id=_string(payload, "artifact_set_id"),
        root=_string(payload, "root"),
        verified_members_digest=_string(payload, "verified_members_digest"),
        verified_at=_string(payload, "verified_at"),
        placements=tuple(_placement(item) for item in _mapping_list(payload, "placements")),
    )


def postprocessing_verified_artifact_location_from_mapping(
    payload: Mapping[str, object],
) -> PostprocessingVerifiedArtifactLocation:
    """Parse and validate one verified postprocessing Artifact Location."""
    return _location(payload)


def _placement(payload: Mapping[str, object]) -> PostprocessingArtifactPlacement:
    _fields(
        payload,
        {
            "schema_version",
            "logical_path",
            "physical_path",
            "sha256",
            "size_bytes",
            "verification_kind",
            "verification_source_path",
        },
        "PostprocessingArtifactPlacement",
    )
    return PostprocessingArtifactPlacement(
        schema_version=_schema(payload, "PostprocessingArtifactPlacement"),
        logical_path=_string(payload, "logical_path"),
        physical_path=_string(payload, "physical_path"),
        sha256=_string(payload, "sha256"),
        size_bytes=_integer(payload, "size_bytes"),
        verification_kind=cast(
            "Literal['content-sha256-v1', 'inventory-metadata-v1']",
            _string(payload, "verification_kind"),
        ),
        verification_source_path=_optional_string(payload, "verification_source_path"),
    )


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


def _fields(payload: Mapping[str, object], expected: set[str], name: str) -> None:
    if set(payload) != expected:
        raise ValueError(f"{name} has missing or extra fields")


def _schema(payload_or_version: Mapping[str, object] | int, name: str) -> int:
    value = payload_or_version.get("schema_version") if isinstance(payload_or_version, Mapping) else payload_or_version
    return validate_schema_version(value, record_name=name)


def _string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_string(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError(f"{key} must be a non-empty string or null")
    return value


def _integer(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _sha(value: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise ValueError("postprocessing receipt digest must be lowercase SHA-256")


__all__ = [
    "PostprocessingActionReceiptEvidence",
    "PostprocessingArtifactPlacement",
    "PostprocessingOutputHandoff",
    "PostprocessingTaskReceiptEvidence",
    "PostprocessingVerifiedArtifactLocation",
    "postprocessing_handoff_id",
    "postprocessing_output_handoff_from_mapping",
    "postprocessing_verified_artifact_location_from_mapping",
]
