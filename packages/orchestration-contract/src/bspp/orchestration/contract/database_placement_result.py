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

"""Strict direct-requested Database Placement Result evidence contract."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal, cast

from bspp.orchestration.contract._database_validation import (
    require_fields as _require_fields,
)
from bspp.orchestration.contract._database_validation import (
    required_int as _required_int,
)
from bspp.orchestration.contract._database_validation import (
    required_mapping as _required_mapping,
)
from bspp.orchestration.contract._database_validation import (
    required_nonempty_str as _required_str,
)
from bspp.orchestration.contract._database_validation import (
    required_str_tuple as _required_str_sequence,
)
from bspp.orchestration.contract._database_validation import (
    validate_database_source_mount_options as _validate_mount_options,
)
from bspp.orchestration.contract._database_validation import (
    validate_placement_absolute_path as _validate_absolute_path,
)
from bspp.orchestration.contract._database_validation import (
    validate_schema as _validate_schema,
)
from bspp.orchestration.contract.database_placement import (
    DATABASE_SOURCE_ROOT,
    LEGACY_DATABASE_SOURCE_ROOT,
    LEGACY_SELECTED_DATABASE_ROOT,
    SELECTED_DATABASE_ROOT,
    DatabaseAccessPolicy,
    DatabasePlacementOutcomeKind,
)
from bspp.orchestration.contract.database_set_provisioning import (
    DatabaseSetIdentity,
    DatabaseSourceMember,
    database_source_member_from_mapping,
    validate_database_source_member_topology,
)
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

DatabasePlacementFailureClassification = Literal[
    "authority-invalid",
    "source-manifest-invalid",
    "source-mount-invalid",
    "source-inventory-invalid",
    "result-publication-failed",
]

_SHA256 = re.compile(r"[0-9a-f]{64}")
_ATTEMPT_ID = re.compile(r"attempt-[0-9]{4}")
_PHASE_RUN_ID = re.compile(r"phase-run-[0-9a-f]{32}")
_ACTION_ID = re.compile(r"preprocessing-chunk-[0-9]{6}")
_FAILURE_CLASSIFICATIONS = frozenset(
    {
        "authority-invalid",
        "source-manifest-invalid",
        "source-mount-invalid",
        "source-inventory-invalid",
        "result-publication-failed",
    }
)
_MAX_FAILURE_ERROR_CHARACTERS = 2048


@dataclass(frozen=True)
class DatabaseSourceMountFacts:
    """Decoded Linux mount authority for the protected source root."""

    mount_id: int
    parent_mount_id: int
    device_major: int
    device_minor: int
    mount_root: str
    mount_point: str
    filesystem_type: str
    mount_source: str
    mount_options: tuple[str, ...]
    super_options: tuple[str, ...]
    read_only: Literal[True]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "DatabaseSourceMountFacts")
        for name, value in (
            ("mount_id", self.mount_id),
            ("parent_mount_id", self.parent_mount_id),
            ("device_major", self.device_major),
            ("device_minor", self.device_minor),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"database source {name} must be a non-negative integer")
        _validate_absolute_path(self.mount_root, "database source mount_root")
        _validate_absolute_path(self.mount_point, "database source mount_point")
        if not self.filesystem_type or "/" in self.filesystem_type or "\x00" in self.filesystem_type:
            raise ValueError("database source filesystem_type must be a non-empty filesystem token")
        if not self.mount_source or "\x00" in self.mount_source:
            raise ValueError("database source mount_source must be non-empty")
        _validate_mount_options(self.mount_options, "mount_options")
        _validate_mount_options(self.super_options, "super_options")
        if self.read_only is not True or "ro" not in self.mount_options or "rw" in self.mount_options:
            raise ValueError("database source mount facts must prove a read-only mount")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "mount_id": self.mount_id,
            "parent_mount_id": self.parent_mount_id,
            "device_major": self.device_major,
            "device_minor": self.device_minor,
            "mount_root": self.mount_root,
            "mount_point": self.mount_point,
            "filesystem_type": self.filesystem_type,
            "mount_source": self.mount_source,
            "mount_options": list(self.mount_options),
            "super_options": list(self.super_options),
            "read_only": self.read_only,
        }


@dataclass(frozen=True)
class DatabaseSourceObservation:
    """One complete metadata-only observation immediately before science."""

    source_container_root: str
    source_manifest_sha256: str
    members: tuple[DatabaseSourceMember, ...]
    verification: Literal["metadata-verified"] = "metadata-verified"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "DatabaseSourceObservation")
        if self.source_container_root not in (DATABASE_SOURCE_ROOT, LEGACY_DATABASE_SOURCE_ROOT):
            raise ValueError(f"database source observation root must be {DATABASE_SOURCE_ROOT!r}")
        if _SHA256.fullmatch(self.source_manifest_sha256) is None:
            raise ValueError("database source observation manifest digest must be lowercase SHA-256")
        validate_database_source_member_topology(self.members, record_name="Database Source Observation")
        if self.verification != "metadata-verified":
            raise ValueError("database source observation verification must be 'metadata-verified'")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_container_root": self.source_container_root,
            "source_manifest_sha256": self.source_manifest_sha256,
            "members": [member.to_mapping() for member in self.members],
            "verification": self.verification,
        }


@dataclass(frozen=True)
class DatabasePostScienceObservation:
    """Complete direct-source facts retained after science, including drift."""

    source_container_root: str
    source_manifest_sha256: str
    members: tuple[DatabaseSourceMember, ...]
    inventory_paths: tuple[str, ...]
    errors: tuple[str, ...]
    verification: Literal["metadata-observed"] = "metadata-observed"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "DatabasePostScienceObservation")
        if self.source_container_root not in (DATABASE_SOURCE_ROOT, LEGACY_DATABASE_SOURCE_ROOT):
            raise ValueError(f"post-science database source root must be {DATABASE_SOURCE_ROOT!r}")
        if _SHA256.fullmatch(self.source_manifest_sha256) is None:
            raise ValueError("post-science observation manifest digest must be lowercase SHA-256")
        if not isinstance(self.members, tuple) or any(
            not isinstance(item, DatabaseSourceMember) for item in self.members
        ):
            raise ValueError("post-science observation members must be an immutable tuple")
        if len({item.logical_name for item in self.members}) != len(self.members):
            raise ValueError("post-science observation logical members must be unique")
        if (
            not isinstance(self.inventory_paths, tuple)
            or tuple(sorted(set(self.inventory_paths))) != self.inventory_paths
            or any(
                not item or item.startswith("/") or ".." in PurePosixPath(item).parts for item in self.inventory_paths
            )
        ):
            raise ValueError("post-science inventory paths must be sorted, unique, safe relative paths")
        if (
            not isinstance(self.errors, tuple)
            or len(self.errors) > 128
            or any(not item or len(item) > 512 or "\x00" in item for item in self.errors)
        ):
            raise ValueError("post-science observation errors must be bounded")
        if self.verification != "metadata-observed":
            raise ValueError("post-science observation verification must be 'metadata-observed'")

    @classmethod
    def from_pre_science(cls, observation: DatabaseSourceObservation) -> DatabasePostScienceObservation:
        return cls(
            source_container_root=observation.source_container_root,
            source_manifest_sha256=observation.source_manifest_sha256,
            members=observation.members,
            inventory_paths=_member_inventory_paths(observation.members),
            errors=(),
        )

    def matches(self, observation: DatabaseSourceObservation) -> bool:
        return (
            self.source_manifest_sha256 == observation.source_manifest_sha256
            and self.members == observation.members
            and self.inventory_paths == _member_inventory_paths(observation.members)
            and not self.errors
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_container_root": self.source_container_root,
            "source_manifest_sha256": self.source_manifest_sha256,
            "members": [member.to_mapping() for member in self.members],
            "inventory_paths": list(self.inventory_paths),
            "errors": list(self.errors),
            "verification": self.verification,
        }


@dataclass(frozen=True)
class DatabasePostScienceObservationFailure:
    """Bounded proof that no complete selected-root observation was possible."""

    source_container_root: str
    source_manifest_sha256: str
    error: str
    verification: Literal["observation-failed"] = "observation-failed"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "DatabasePostScienceObservationFailure")
        if self.source_container_root not in (DATABASE_SOURCE_ROOT, LEGACY_DATABASE_SOURCE_ROOT):
            raise ValueError(f"post-science database source root must be {DATABASE_SOURCE_ROOT!r}")
        if _SHA256.fullmatch(self.source_manifest_sha256) is None:
            raise ValueError("post-science observation failure manifest digest must be lowercase SHA-256")
        if not self.error or len(self.error) > 2048 or "\x00" in self.error:
            raise ValueError("post-science observation failure error must be bounded")
        if self.verification != "observation-failed":
            raise ValueError("post-science observation failure verification must be 'observation-failed'")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_container_root": self.source_container_root,
            "source_manifest_sha256": self.source_manifest_sha256,
            "error": self.error,
            "verification": self.verification,
        }


DatabasePostScienceEvidence = DatabasePostScienceObservation | DatabasePostScienceObservationFailure


@dataclass(frozen=True)
class DatabasePlacementResult:
    """Exclusive pre-science authority for one direct-requested placement."""

    phase_run_id: str
    attempt_id: str
    phase_runspec_digest: str
    action_id: str
    database_set: DatabaseSetIdentity
    requested_policy: DatabaseAccessPolicy
    source_manifest_sha256: str
    branch_kind: Literal["direct-requested"]
    outcome: DatabasePlacementOutcomeKind
    selected_container_root: str
    verification: Literal["metadata-verified"]
    source_mount: DatabaseSourceMountFacts
    pre_science_observation: DatabaseSourceObservation
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "DatabasePlacementResult")
        if _PHASE_RUN_ID.fullmatch(self.phase_run_id) is None:
            raise ValueError("database placement phase_run_id must be an opaque phase-run id")
        if _ATTEMPT_ID.fullmatch(self.attempt_id) is None:
            raise ValueError("database placement attempt_id must use attempt-NNNN")
        if _SHA256.fullmatch(self.phase_runspec_digest) is None:
            raise ValueError("database placement phase_runspec_digest must be lowercase SHA-256")
        if _ACTION_ID.fullmatch(self.action_id) is None:
            raise ValueError("database placement action_id must identify a preprocessing chunk")
        if not isinstance(self.database_set, DatabaseSetIdentity):
            raise ValueError("database placement requires an exact Database Set identity")
        if (
            not isinstance(self.requested_policy, DatabaseAccessPolicy)
            or self.requested_policy != DatabaseAccessPolicy.DIRECT
        ):
            raise ValueError("Database Placement Result supports direct-requested policy only")
        if _SHA256.fullmatch(self.source_manifest_sha256) is None:
            raise ValueError("database placement source_manifest_sha256 must be lowercase SHA-256")
        if self.branch_kind != "direct-requested":
            raise ValueError("Database Placement Result branch must be direct-requested")
        if (
            not isinstance(self.outcome, DatabasePlacementOutcomeKind)
            or self.outcome != DatabasePlacementOutcomeKind.DIRECT_REQUESTED
        ):
            raise ValueError("Database Placement Result outcome must be direct-requested")
        if self.selected_container_root not in (SELECTED_DATABASE_ROOT, LEGACY_SELECTED_DATABASE_ROOT):
            raise ValueError(f"database placement selected root must be {SELECTED_DATABASE_ROOT!r}")
        if self.verification != "metadata-verified":
            raise ValueError("database placement verification must be 'metadata-verified'")
        if not isinstance(self.source_mount, DatabaseSourceMountFacts):
            raise ValueError("database placement requires exact source mount facts")
        source_root = PurePosixPath(DATABASE_SOURCE_ROOT)
        mount_point = PurePosixPath(self.source_mount.mount_point)
        if mount_point != source_root and mount_point not in source_root.parents:
            raise ValueError("database placement source mount must contain the protected source root")
        if not isinstance(self.pre_science_observation, DatabaseSourceObservation):
            raise ValueError("database placement requires one exact pre-science observation")
        if self.pre_science_observation.source_manifest_sha256 != self.source_manifest_sha256:
            raise ValueError("database placement observation must bind the same source manifest digest")

    def to_mapping(self) -> dict[str, object]:
        return {
            "database_placement_result": {
                "schema_version": self.schema_version,
                "phase_run_id": self.phase_run_id,
                "attempt_id": self.attempt_id,
                "phase_runspec_digest": self.phase_runspec_digest,
                "action_id": self.action_id,
                "database_set": self.database_set.to_mapping(),
                "requested_policy": self.requested_policy.value,
                "source_manifest_sha256": self.source_manifest_sha256,
                "branch_kind": self.branch_kind,
                "outcome": self.outcome.value,
                "selected_container_root": self.selected_container_root,
                "verification": self.verification,
                "source_mount": self.source_mount.to_mapping(),
                "pre_science_observation": self.pre_science_observation.to_mapping(),
            }
        }


@dataclass(frozen=True)
class DatabasePlacementFailureEvidence:
    """Bounded, exclusive evidence that direct placement failed before science."""

    phase_run_id: str
    attempt_id: str
    phase_runspec_digest: str
    action_id: str
    database_set: DatabaseSetIdentity
    requested_policy: DatabaseAccessPolicy
    source_manifest_sha256: str
    source_mount: DatabaseSourceMountFacts | None
    science_started: Literal[False]
    classification: DatabasePlacementFailureClassification
    error: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "DatabasePlacementFailureEvidence")
        if _PHASE_RUN_ID.fullmatch(self.phase_run_id) is None:
            raise ValueError("database placement failure phase_run_id must be an opaque phase-run id")
        if _ATTEMPT_ID.fullmatch(self.attempt_id) is None:
            raise ValueError("database placement failure attempt_id must use attempt-NNNN")
        if _SHA256.fullmatch(self.phase_runspec_digest) is None:
            raise ValueError("database placement failure phase_runspec_digest must be lowercase SHA-256")
        if _ACTION_ID.fullmatch(self.action_id) is None:
            raise ValueError("database placement failure action_id must identify a preprocessing chunk")
        if not isinstance(self.database_set, DatabaseSetIdentity):
            raise ValueError("database placement failure requires an exact Database Set identity")
        if not isinstance(self.requested_policy, DatabaseAccessPolicy):
            raise ValueError("database placement failure requested_policy must be canonical")
        if _SHA256.fullmatch(self.source_manifest_sha256) is None:
            raise ValueError("database placement failure source_manifest_sha256 must be lowercase SHA-256")
        if self.source_mount is not None and not isinstance(self.source_mount, DatabaseSourceMountFacts):
            raise ValueError("database placement failure source_mount must be exact mount facts when available")
        if self.science_started is not False:
            raise ValueError("database placement failure science_started must be false")
        if self.classification not in _FAILURE_CLASSIFICATIONS:
            raise ValueError(f"unsupported database placement failure classification: {self.classification!r}")
        if not self.error or len(self.error) > _MAX_FAILURE_ERROR_CHARACTERS or "\x00" in self.error:
            raise ValueError("database placement failure error must be non-empty and bounded to 2048 characters")

    def to_mapping(self) -> dict[str, object]:
        return {
            "database_placement_failure": {
                "schema_version": self.schema_version,
                "phase_run_id": self.phase_run_id,
                "attempt_id": self.attempt_id,
                "phase_runspec_digest": self.phase_runspec_digest,
                "action_id": self.action_id,
                "database_set": self.database_set.to_mapping(),
                "requested_policy": self.requested_policy.value,
                "source_manifest_sha256": self.source_manifest_sha256,
                "source_mount": None if self.source_mount is None else self.source_mount.to_mapping(),
                "science_started": self.science_started,
                "classification": self.classification,
                "error": self.error,
            }
        }


def database_placement_result_from_mapping(payload: Mapping[str, object]) -> DatabasePlacementResult:
    """Strict-load the sole direct-requested Database Placement Result shape."""
    _require_fields(payload, {"database_placement_result"}, name="Database Placement Result document")
    inner = _required_mapping(payload, "database_placement_result")
    _require_fields(
        inner,
        {
            "phase_run_id",
            "attempt_id",
            "phase_runspec_digest",
            "action_id",
            "database_set",
            "requested_policy",
            "source_manifest_sha256",
            "branch_kind",
            "outcome",
            "selected_container_root",
            "verification",
            "schema_version",
            "source_mount",
            "pre_science_observation",
        },
        name="Database Placement Result",
    )
    policy = _database_access_policy(inner.get("requested_policy"))
    outcome = _database_outcome(_required_str(inner, "outcome"))
    branch_kind = _required_str(inner, "branch_kind")
    if branch_kind != "direct-requested":
        raise ValueError("Database Placement Result branch must be direct-requested")
    verification = _required_str(inner, "verification")
    if verification != "metadata-verified":
        raise ValueError("database placement verification must be 'metadata-verified'")
    return DatabasePlacementResult(
        phase_run_id=_required_str(inner, "phase_run_id"),
        attempt_id=_required_str(inner, "attempt_id"),
        phase_runspec_digest=_required_str(inner, "phase_runspec_digest"),
        action_id=_required_str(inner, "action_id"),
        database_set=_database_set_identity_from_mapping(_required_mapping(inner, "database_set")),
        requested_policy=policy,
        source_manifest_sha256=_required_str(inner, "source_manifest_sha256"),
        branch_kind=cast("Literal['direct-requested']", branch_kind),
        outcome=outcome,
        selected_container_root=_required_str(inner, "selected_container_root"),
        verification="metadata-verified",
        source_mount=_database_source_mount_facts_from_mapping(_required_mapping(inner, "source_mount")),
        pre_science_observation=_database_source_observation_from_mapping(
            _required_mapping(inner, "pre_science_observation")
        ),
        schema_version=validate_schema_version(inner.get("schema_version"), record_name="DatabasePlacementResult"),
    )


def canonical_database_placement_result_bytes(result: DatabasePlacementResult) -> bytes:
    """Return exact immutable bytes for the placement-result authority."""
    return (json.dumps(result.to_mapping(), indent=2, sort_keys=True) + "\n").encode()


def database_placement_result_digest(result: DatabasePlacementResult) -> str:
    """Return the #90 linkage digest of canonical Result bytes."""
    return hashlib.sha256(canonical_database_placement_result_bytes(result)).hexdigest()


def database_placement_failure_evidence_from_mapping(
    payload: Mapping[str, object],
) -> DatabasePlacementFailureEvidence:
    """Strict-load the sole bounded direct-placement failure document."""
    _require_fields(payload, {"database_placement_failure"}, name="Database Placement failure document")
    inner = _required_mapping(payload, "database_placement_failure")
    _require_fields(
        inner,
        {
            "schema_version",
            "phase_run_id",
            "attempt_id",
            "phase_runspec_digest",
            "action_id",
            "database_set",
            "requested_policy",
            "source_manifest_sha256",
            "source_mount",
            "science_started",
            "classification",
            "error",
        },
        name="Database Placement failure",
    )
    classification = _required_str(inner, "classification")
    if inner.get("science_started") is not False:
        raise ValueError("database placement failure science_started must be false")
    source_mount_payload = inner.get("source_mount")
    if source_mount_payload is not None and not isinstance(source_mount_payload, Mapping):
        raise ValueError("database placement failure source_mount must be a mapping or null")
    return DatabasePlacementFailureEvidence(
        phase_run_id=_required_str(inner, "phase_run_id"),
        attempt_id=_required_str(inner, "attempt_id"),
        phase_runspec_digest=_required_str(inner, "phase_runspec_digest"),
        action_id=_required_str(inner, "action_id"),
        database_set=_database_set_identity_from_mapping(_required_mapping(inner, "database_set")),
        requested_policy=_database_access_policy(inner.get("requested_policy")),
        source_manifest_sha256=_required_str(inner, "source_manifest_sha256"),
        source_mount=(
            _database_source_mount_facts_from_mapping(source_mount_payload)
            if isinstance(source_mount_payload, Mapping)
            else None
        ),
        science_started=False,
        classification=cast("DatabasePlacementFailureClassification", classification),
        error=_required_str(inner, "error"),
        schema_version=validate_schema_version(
            inner.get("schema_version"), record_name="DatabasePlacementFailureEvidence"
        ),
    )


def canonical_database_placement_failure_evidence_bytes(failure: DatabasePlacementFailureEvidence) -> bytes:
    """Return exact immutable bytes for bounded placement-failure evidence."""
    return (json.dumps(failure.to_mapping(), indent=2, sort_keys=True) + "\n").encode()


def database_placement_failure_evidence_digest(failure: DatabasePlacementFailureEvidence) -> str:
    """Bind action evidence to exact canonical placement-failure bytes."""
    return hashlib.sha256(canonical_database_placement_failure_evidence_bytes(failure)).hexdigest()


def _database_source_mount_facts_from_mapping(payload: Mapping[str, object]) -> DatabaseSourceMountFacts:
    _require_fields(
        payload,
        {
            "mount_id",
            "parent_mount_id",
            "device_major",
            "device_minor",
            "mount_root",
            "mount_point",
            "filesystem_type",
            "mount_source",
            "mount_options",
            "super_options",
            "read_only",
            "schema_version",
        },
        name="database source mount facts",
    )
    if payload.get("read_only") is not True:
        raise ValueError("database source mount facts read_only must be true")
    return DatabaseSourceMountFacts(
        mount_id=_required_int(payload, "mount_id"),
        parent_mount_id=_required_int(payload, "parent_mount_id"),
        device_major=_required_int(payload, "device_major"),
        device_minor=_required_int(payload, "device_minor"),
        mount_root=_required_str(payload, "mount_root"),
        mount_point=_required_str(payload, "mount_point"),
        filesystem_type=_required_str(payload, "filesystem_type"),
        mount_source=_required_str(payload, "mount_source"),
        mount_options=_required_str_sequence(payload, "mount_options"),
        super_options=_required_str_sequence(payload, "super_options"),
        read_only=True,
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="DatabaseSourceMountFacts"),
    )


def _database_source_observation_from_mapping(payload: Mapping[str, object]) -> DatabaseSourceObservation:
    _require_fields(
        payload,
        {"source_container_root", "source_manifest_sha256", "members", "verification", "schema_version"},
        name="database source observation",
    )
    verification = _required_str(payload, "verification")
    if verification != "metadata-verified":
        raise ValueError("database source observation verification must be 'metadata-verified'")
    members = payload.get("members")
    if not isinstance(members, list | tuple):
        raise ValueError("members must be a list")
    return DatabaseSourceObservation(
        source_container_root=_required_str(payload, "source_container_root"),
        source_manifest_sha256=_required_str(payload, "source_manifest_sha256"),
        members=tuple(database_source_member_from_mapping(item) for item in members),
        verification="metadata-verified",
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="DatabaseSourceObservation"),
    )


def database_post_science_observation_from_mapping(payload: Mapping[str, object]) -> DatabasePostScienceObservation:
    _require_fields(
        payload,
        {
            "source_container_root",
            "source_manifest_sha256",
            "members",
            "inventory_paths",
            "errors",
            "verification",
            "schema_version",
        },
        name="post-science database source observation",
    )
    members = payload.get("members")
    inventory_paths = payload.get("inventory_paths")
    errors = payload.get("errors")
    if not isinstance(members, list | tuple):
        raise ValueError("post-science members must be a list")
    if not isinstance(inventory_paths, list | tuple) or not all(isinstance(item, str) for item in inventory_paths):
        raise ValueError("post-science inventory_paths must be a list of strings")
    if not isinstance(errors, list | tuple) or not all(isinstance(item, str) for item in errors):
        raise ValueError("post-science errors must be a list of strings")
    return DatabasePostScienceObservation(
        source_container_root=_required_str(payload, "source_container_root"),
        source_manifest_sha256=_required_str(payload, "source_manifest_sha256"),
        members=tuple(database_source_member_from_mapping(item) for item in members),
        inventory_paths=tuple(inventory_paths),
        errors=tuple(errors),
        verification=cast("Literal['metadata-observed']", _required_str(payload, "verification")),
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="DatabasePostScienceObservation"
        ),
    )


def database_post_science_observation_failure_from_mapping(
    payload: Mapping[str, object],
) -> DatabasePostScienceObservationFailure:
    _require_fields(
        payload,
        {
            "source_container_root",
            "source_manifest_sha256",
            "error",
            "verification",
            "schema_version",
        },
        name="post-science database source observation failure",
    )
    return DatabasePostScienceObservationFailure(
        source_container_root=_required_str(payload, "source_container_root"),
        source_manifest_sha256=_required_str(payload, "source_manifest_sha256"),
        error=_required_str(payload, "error"),
        verification=cast("Literal['observation-failed']", _required_str(payload, "verification")),
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="DatabasePostScienceObservationFailure"
        ),
    )


def database_post_science_evidence_from_mapping(payload: Mapping[str, object]) -> DatabasePostScienceEvidence:
    verification = _required_str(payload, "verification")
    if verification == "metadata-observed":
        return database_post_science_observation_from_mapping(payload)
    if verification == "observation-failed":
        return database_post_science_observation_failure_from_mapping(payload)
    raise ValueError(f"unsupported post-science observation verification: {verification!r}")


def _member_inventory_paths(members: tuple[DatabaseSourceMember, ...]) -> tuple[str, ...]:
    paths: set[str] = set()
    for member in members:
        for value in (member.source_path, member.resolved_path, *(alias.path for alias in member.alias_topology)):
            path = PurePosixPath(value)
            paths.update(PurePosixPath(*path.parts[:index]).as_posix() for index in range(1, len(path.parts) + 1))
    return tuple(sorted(paths))


def _database_access_policy(value: object) -> DatabaseAccessPolicy:
    if not isinstance(value, str):
        raise ValueError("requested_policy must be a canonical string")
    try:
        return DatabaseAccessPolicy(value)
    except ValueError as exc:
        raise ValueError(f"unsupported database access policy: {value!r}") from exc


def _database_outcome(value: str) -> DatabasePlacementOutcomeKind:
    try:
        return DatabasePlacementOutcomeKind(value)
    except ValueError as exc:
        raise ValueError(f"unsupported database placement outcome: {value!r}") from exc


def _database_set_identity_from_mapping(payload: Mapping[str, object]) -> DatabaseSetIdentity:
    _require_fields(payload, {"identifier", "version"}, name="database_set")
    return DatabaseSetIdentity(
        identifier=_required_str(payload, "identifier"),
        version=_required_str(payload, "version"),
    )


__all__ = [
    "DatabasePlacementFailureClassification",
    "DatabasePlacementFailureEvidence",
    "DatabasePlacementResult",
    "DatabasePostScienceEvidence",
    "DatabasePostScienceObservation",
    "DatabasePostScienceObservationFailure",
    "DatabaseSourceMountFacts",
    "DatabaseSourceObservation",
    "canonical_database_placement_failure_evidence_bytes",
    "canonical_database_placement_result_bytes",
    "database_placement_failure_evidence_digest",
    "database_placement_failure_evidence_from_mapping",
    "database_placement_result_digest",
    "database_placement_result_from_mapping",
    "database_post_science_evidence_from_mapping",
    "database_post_science_observation_failure_from_mapping",
    "database_post_science_observation_from_mapping",
]
