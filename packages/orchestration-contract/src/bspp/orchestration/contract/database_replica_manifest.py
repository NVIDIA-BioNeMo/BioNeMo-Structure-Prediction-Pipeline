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

"""Immutable Database Replica member and manifest authority."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from bspp.orchestration.contract._database_validation import (
    require_fields as _require_fields,
)
from bspp.orchestration.contract._database_validation import required_int as _required_int
from bspp.orchestration.contract._database_validation import (
    required_mapping as _required_mapping,
)
from bspp.orchestration.contract._database_validation import required_nonempty_str as _required_str
from bspp.orchestration.contract._database_validation import (
    required_sequence as _required_sequence,
)
from bspp.orchestration.contract._database_validation import (
    validate_nonnegative_int as _validate_nonnegative_int,
)
from bspp.orchestration.contract._database_validation import (
    validate_relative_path as _validate_relative_path,
)
from bspp.orchestration.contract._database_validation import validate_schema as _validate_schema
from bspp.orchestration.contract.database_placement_result import DatabaseSourceObservation
from bspp.orchestration.contract.database_replica_facts import (
    DatabaseReplicaCopyEvidence,
    _copy_evidence_from_mapping,
    _database_set_from_mapping,
    _source_observation_from_mapping,
    _unwrap_exact,
    _validate_sha256,
)
from bspp.orchestration.contract.database_set_provisioning import DatabaseRole, DatabaseSetIdentity
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version


@dataclass(frozen=True)
class DatabaseReplicaMember:
    """One regular logical member in a self-contained replica."""

    role: DatabaseRole
    database_name: str
    logical_name: str
    resolved_source_path: str
    replica_path: str
    size_bytes: int
    mtime_ns: int
    hardlink_group: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "DatabaseReplicaMember")
        if self.role not in {"primary", "metagenomic"}:
            raise ValueError(f"unsupported database replica role: {self.role!r}")
        for name, value in (
            ("database_name", self.database_name),
            ("logical_name", self.logical_name),
            ("resolved_source_path", self.resolved_source_path),
            ("replica_path", self.replica_path),
            ("hardlink_group", self.hardlink_group),
        ):
            _validate_relative_path(value, f"database replica {name}")
        if self.replica_path != self.logical_name:
            raise ValueError("database replica paths must exactly equal logical MMseqs names")
        if self.hardlink_group != self.resolved_source_path:
            raise ValueError("database replica hardlink groups must bind exact resolved sources")
        _validate_nonnegative_int(self.size_bytes, "database replica size_bytes")
        _validate_nonnegative_int(self.mtime_ns, "database replica mtime_ns")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "role": self.role,
            "database_name": self.database_name,
            "logical_name": self.logical_name,
            "resolved_source_path": self.resolved_source_path,
            "replica_path": self.replica_path,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
            "hardlink_group": self.hardlink_group,
        }


@dataclass(frozen=True)
class DatabaseReplicaManifest:
    """Immutable metadata-verified publication record for one cold replica."""

    database_set: DatabaseSetIdentity
    source_manifest_sha256: str
    members: tuple[DatabaseReplicaMember, ...]
    pre_copy_source_observation: DatabaseSourceObservation
    post_copy_source_observation: DatabaseSourceObservation
    copy_evidence: DatabaseReplicaCopyEvidence
    verification: Literal["metadata-verified"] = "metadata-verified"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "DatabaseReplicaManifest")
        if not isinstance(self.database_set, DatabaseSetIdentity):
            raise ValueError("Database Replica Manifest requires an exact Database Set identity")
        _validate_sha256(self.source_manifest_sha256, "Database Replica Manifest source digest")
        if (
            not isinstance(self.members, tuple)
            or not self.members
            or any(not isinstance(item, DatabaseReplicaMember) for item in self.members)
            or tuple(sorted(self.members, key=lambda item: item.logical_name)) != self.members
            or len({item.logical_name for item in self.members}) != len(self.members)
        ):
            raise ValueError("Database Replica Manifest members must be complete, unique and sorted")
        for observation in (self.pre_copy_source_observation, self.post_copy_source_observation):
            if not isinstance(observation, DatabaseSourceObservation):
                raise ValueError("Database Replica Manifest requires complete source observations")
            if observation.source_manifest_sha256 != self.source_manifest_sha256:
                raise ValueError("Database Replica Manifest observations must bind the source digest")
        if self.pre_copy_source_observation != self.post_copy_source_observation:
            raise ValueError("Database Source must remain stable across replica population")
        expected_by_name = {item.logical_name: item for item in self.pre_copy_source_observation.members}
        if set(expected_by_name) != {item.logical_name for item in self.members}:
            raise ValueError("Database Replica Manifest must contain the complete source logical inventory")
        for member in self.members:
            source = expected_by_name[member.logical_name]
            if (
                member.role != source.role
                or member.database_name != source.database_name
                or member.resolved_source_path != source.resolved_path
                or member.size_bytes != source.size_bytes
                or member.mtime_ns != source.mtime_ns
            ):
                raise ValueError("Database Replica member metadata must exactly match its source observation")
        if not isinstance(self.copy_evidence, DatabaseReplicaCopyEvidence):
            raise ValueError("Database Replica Manifest requires exact copy evidence")
        groups: dict[str, tuple[str, ...]] = {}
        for resolved in sorted({item.resolved_source_path for item in self.members}):
            groups[resolved] = tuple(
                sorted(item.logical_name for item in self.members if item.resolved_source_path == resolved)
            )
        outcomes = {item.resolved_source_path: item for item in self.copy_evidence.outcomes}
        if set(outcomes) != set(groups):
            raise ValueError("Database Replica copy evidence must cover each unique resolved source exactly once")
        for resolved, logical_names in groups.items():
            outcome = outcomes[resolved]
            metadata_identities = {
                (expected_by_name[logical_name].size_bytes, expected_by_name[logical_name].mtime_ns)
                for logical_name in logical_names
            }
            if len(metadata_identities) != 1:
                raise ValueError("Database Replica hard-link groups must have one source metadata identity")
            expected_size, _ = next(iter(metadata_identities))
            if outcome.logical_names != logical_names:
                raise ValueError("Database Replica copy outcomes must bind the exact hard-link alias groups")
            if outcome.size_bytes != expected_size:
                raise ValueError("Database Replica copy outcome sizes must match their unique source groups")
        if self.verification != "metadata-verified":
            raise ValueError("Database Replica Manifest verification must be 'metadata-verified'")

    def to_mapping(self) -> dict[str, object]:
        return {
            "database_replica_manifest": {
                "schema_version": self.schema_version,
                "database_set": self.database_set.to_mapping(),
                "source_manifest_sha256": self.source_manifest_sha256,
                "members": [item.to_mapping() for item in self.members],
                "pre_copy_source_observation": self.pre_copy_source_observation.to_mapping(),
                "post_copy_source_observation": self.post_copy_source_observation.to_mapping(),
                "copy_evidence": self.copy_evidence.to_mapping(),
                "verification": self.verification,
            }
        }


def database_replica_manifest_from_mapping(payload: Mapping[str, object]) -> DatabaseReplicaManifest:
    inner = _unwrap_exact(payload, "database_replica_manifest")
    _require_fields(
        inner,
        {
            "schema_version",
            "database_set",
            "source_manifest_sha256",
            "members",
            "pre_copy_source_observation",
            "post_copy_source_observation",
            "copy_evidence",
            "verification",
        },
        "Database Replica Manifest",
    )
    verification = _required_str(inner, "verification")
    if verification != "metadata-verified":
        raise ValueError("Database Replica Manifest verification must be 'metadata-verified'")
    return DatabaseReplicaManifest(
        database_set=_database_set_from_mapping(_required_mapping(inner, "database_set")),
        source_manifest_sha256=_required_str(inner, "source_manifest_sha256"),
        members=tuple(_replica_member_from_mapping(item) for item in _required_sequence(inner, "members")),
        pre_copy_source_observation=_source_observation_from_mapping(
            _required_mapping(inner, "pre_copy_source_observation")
        ),
        post_copy_source_observation=_source_observation_from_mapping(
            _required_mapping(inner, "post_copy_source_observation")
        ),
        copy_evidence=_copy_evidence_from_mapping(_required_mapping(inner, "copy_evidence")),
        verification="metadata-verified",
        schema_version=validate_schema_version(inner.get("schema_version"), record_name="DatabaseReplicaManifest"),
    )


def canonical_database_replica_manifest_bytes(manifest: DatabaseReplicaManifest) -> bytes:
    return (json.dumps(manifest.to_mapping(), indent=2, sort_keys=True) + "\n").encode()


def database_replica_manifest_digest(manifest: DatabaseReplicaManifest) -> str:
    return hashlib.sha256(canonical_database_replica_manifest_bytes(manifest)).hexdigest()


def _replica_member_from_mapping(value: object) -> DatabaseReplicaMember:
    if not isinstance(value, Mapping):
        raise ValueError("Database Replica members must be mappings")
    _require_fields(
        value,
        {
            "schema_version",
            "role",
            "database_name",
            "logical_name",
            "resolved_source_path",
            "replica_path",
            "size_bytes",
            "mtime_ns",
            "hardlink_group",
        },
        "Database Replica member",
    )
    return DatabaseReplicaMember(
        role=cast("DatabaseRole", _required_str(value, "role")),
        database_name=_required_str(value, "database_name"),
        logical_name=_required_str(value, "logical_name"),
        resolved_source_path=_required_str(value, "resolved_source_path"),
        replica_path=_required_str(value, "replica_path"),
        size_bytes=_required_int(value, "size_bytes"),
        mtime_ns=_required_int(value, "mtime_ns"),
        hardlink_group=_required_str(value, "hardlink_group"),
        schema_version=validate_schema_version(value.get("schema_version"), record_name="DatabaseReplicaMember"),
    )


__all__ = [
    "DatabaseReplicaManifest",
    "DatabaseReplicaMember",
    "canonical_database_replica_manifest_bytes",
    "database_replica_manifest_digest",
    "database_replica_manifest_from_mapping",
]
