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

"""Shared strict facts for Database Replica evidence contracts."""

from __future__ import annotations

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
    required_sequence as _required_sequence,
)
from bspp.orchestration.contract._database_validation import (
    required_str_tuple as _required_str_tuple,
)
from bspp.orchestration.contract._database_validation import (
    validate_mount_options as _validate_mount_options,
)
from bspp.orchestration.contract._database_validation import (
    validate_nonnegative_int as _validate_nonnegative_int,
)
from bspp.orchestration.contract._database_validation import (
    validate_relative_path as _validate_relative_path,
)
from bspp.orchestration.contract._database_validation import (
    validate_replica_absolute_path as _validate_absolute_path,
)
from bspp.orchestration.contract._database_validation import (
    validate_schema as _validate_schema,
)
from bspp.orchestration.contract.database_placement import (
    DATABASE_CACHE_ROOT,
    DATABASE_SOURCE_ROOT,
    LEGACY_DATABASE_CACHE_ROOT,
    LEGACY_DATABASE_SOURCE_ROOT,
)
from bspp.orchestration.contract.database_placement_result import DatabaseSourceMountFacts, DatabaseSourceObservation
from bspp.orchestration.contract.database_set_provisioning import (
    DatabaseSetIdentity,
    database_source_member_from_mapping,
)
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

DatabaseCapacityDecision = Literal["sufficient", "insufficient"]

_SHA256 = re.compile(r"[0-9a-f]{64}")
_RSYNC_OPTIONS = ("--archive", "--no-owner", "--no-group", "--no-perms", "--protect-args")


@dataclass(frozen=True)
class DatabaseCacheMountFacts:
    """Decoded Linux mount and writability authority for the protected cache root."""

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
    read_write: Literal[True]
    writable_verified: Literal[True]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "DatabaseCacheMountFacts")
        for name, value in (
            ("mount_id", self.mount_id),
            ("parent_mount_id", self.parent_mount_id),
            ("device_major", self.device_major),
            ("device_minor", self.device_minor),
        ):
            _validate_nonnegative_int(value, f"database cache {name}")
        _validate_absolute_path(self.mount_root, "database cache mount_root")
        _validate_absolute_path(self.mount_point, "database cache mount_point")
        if not self.filesystem_type or "/" in self.filesystem_type or "\x00" in self.filesystem_type:
            raise ValueError("database cache filesystem_type must be a non-empty filesystem token")
        if not self.mount_source or "\x00" in self.mount_source:
            raise ValueError("database cache mount_source must be non-empty")
        _validate_mount_options(self.mount_options, "database cache mount_options")
        _validate_mount_options(self.super_options, "database cache super_options")
        if self.read_write is not True or "rw" not in self.mount_options or "ro" in self.mount_options:
            raise ValueError("database cache mount facts must prove a read-write mount")
        if self.writable_verified is not True:
            raise ValueError("database cache mount facts must prove descriptor-relative writability")
        protected = PurePosixPath(DATABASE_CACHE_ROOT)
        legacy_protected = PurePosixPath(LEGACY_DATABASE_CACHE_ROOT)
        mount_point = PurePosixPath(self.mount_point)
        if (mount_point != protected and mount_point not in protected.parents) and (
            mount_point != legacy_protected and mount_point not in legacy_protected.parents
        ):
            raise ValueError("database cache mount must contain the protected cache root")

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
            "read_write": self.read_write,
            "writable_verified": self.writable_verified,
        }


@dataclass(frozen=True)
class DatabaseWarmCacheMountFacts:
    """Read-only observation of the protected writable cache mount."""

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
    read_write: Literal[True]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "DatabaseWarmCacheMountFacts")
        for name, value in (
            ("mount_id", self.mount_id),
            ("parent_mount_id", self.parent_mount_id),
            ("device_major", self.device_major),
            ("device_minor", self.device_minor),
        ):
            _validate_nonnegative_int(value, f"warm database cache {name}")
        _validate_absolute_path(self.mount_root, "warm database cache mount_root")
        _validate_absolute_path(self.mount_point, "warm database cache mount_point")
        if not self.filesystem_type or "/" in self.filesystem_type or "\x00" in self.filesystem_type:
            raise ValueError("warm database cache filesystem_type must be a non-empty filesystem token")
        if not self.mount_source or "\x00" in self.mount_source:
            raise ValueError("warm database cache mount_source must be non-empty")
        _validate_mount_options(self.mount_options, "warm database cache mount_options")
        _validate_mount_options(self.super_options, "warm database cache super_options")
        if self.read_write is not True or "rw" not in self.mount_options or "ro" in self.mount_options:
            raise ValueError("warm database cache mount facts must prove a read-write mount")
        protected = PurePosixPath(DATABASE_CACHE_ROOT)
        legacy_protected = PurePosixPath(LEGACY_DATABASE_CACHE_ROOT)
        mount_point = PurePosixPath(self.mount_point)
        if (mount_point != protected and mount_point not in protected.parents) and (
            mount_point != legacy_protected and mount_point not in legacy_protected.parents
        ):
            raise ValueError("warm database cache mount must contain the protected cache root")

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
            "read_write": self.read_write,
        }


@dataclass(frozen=True)
class DatabaseCapacityGate:
    """Exact pre-copy user-available capacity decision for one cold replica."""

    available_user_bytes: int
    allocated_replica_bytes: int
    reserved_bytes: int
    required_bytes: int
    decision: DatabaseCapacityDecision
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "DatabaseCapacityGate")
        for name, value in (
            ("available_user_bytes", self.available_user_bytes),
            ("allocated_replica_bytes", self.allocated_replica_bytes),
            ("reserved_bytes", self.reserved_bytes),
            ("required_bytes", self.required_bytes),
        ):
            _validate_nonnegative_int(value, f"database capacity {name}")
        if self.required_bytes != self.allocated_replica_bytes + self.reserved_bytes:
            raise ValueError("database capacity required_bytes must equal allocated plus reserved bytes")
        expected = "sufficient" if self.available_user_bytes >= self.required_bytes else "insufficient"
        if self.decision != expected:
            raise ValueError("database capacity decision does not match the exact equality-inclusive gate")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "available_user_bytes": self.available_user_bytes,
            "allocated_replica_bytes": self.allocated_replica_bytes,
            "reserved_bytes": self.reserved_bytes,
            "required_bytes": self.required_bytes,
            "decision": self.decision,
        }


@dataclass(frozen=True)
class DatabaseRsyncOutcome:
    """One successful pinned rsync for one unique resolved source."""

    resolved_source_path: str
    destination_logical_name: str
    logical_names: tuple[str, ...]
    size_bytes: int
    argv: tuple[str, ...]
    exit_code: Literal[0]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "DatabaseRsyncOutcome")
        _validate_relative_path(self.resolved_source_path, "rsync resolved_source_path")
        _validate_relative_path(self.destination_logical_name, "rsync destination_logical_name")
        if (
            not isinstance(self.logical_names, tuple)
            or not self.logical_names
            or tuple(sorted(set(self.logical_names))) != self.logical_names
        ):
            raise ValueError("rsync logical_names must be a sorted unique immutable tuple")
        for value in self.logical_names:
            _validate_relative_path(value, "rsync logical name")
        if self.destination_logical_name != self.logical_names[0]:
            raise ValueError("rsync destination must be the deterministic first logical name")
        _validate_nonnegative_int(self.size_bytes, "rsync size_bytes")
        if (
            not isinstance(self.argv, tuple)
            or len(self.argv) < 3
            or any(not isinstance(item, str) or not item or "\x00" in item for item in self.argv)
        ):
            raise ValueError("rsync argv must be an exact immutable non-empty tuple")
        expected_argv = (
            "/usr/bin/rsync",
            *_RSYNC_OPTIONS,
            f"{DATABASE_SOURCE_ROOT}/{self.resolved_source_path}",
            f"{DATABASE_CACHE_ROOT}/replicas/.population-<private>/{self.destination_logical_name}",
        )
        legacy_expected_argv = (
            "/usr/bin/rsync",
            *_RSYNC_OPTIONS,
            f"{LEGACY_DATABASE_SOURCE_ROOT}/{self.resolved_source_path}",
            f"{LEGACY_DATABASE_CACHE_ROOT}/replicas/.population-<private>/{self.destination_logical_name}",
        )
        if self.argv not in (expected_argv, legacy_expected_argv):
            raise ValueError("rsync evidence must bind the exact pinned production argv")
        if self.exit_code != 0:
            raise ValueError("replica copy evidence may retain only successful rsync outcomes")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "resolved_source_path": self.resolved_source_path,
            "destination_logical_name": self.destination_logical_name,
            "logical_names": list(self.logical_names),
            "size_bytes": self.size_bytes,
            "argv": list(self.argv),
            "exit_code": self.exit_code,
        }


@dataclass(frozen=True)
class DatabaseReplicaCopyEvidence:
    """Aggregate deterministic evidence for all unique-source transfers."""

    source_count: int
    selected_workers: int
    total_copied_bytes: int
    elapsed_nanoseconds: int
    aggregate_bytes_per_second: int
    outcomes: tuple[DatabaseRsyncOutcome, ...]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "DatabaseReplicaCopyEvidence")
        if not isinstance(self.source_count, int) or isinstance(self.source_count, bool) or self.source_count <= 0:
            raise ValueError("replica copy source_count must be positive")
        if (
            not isinstance(self.selected_workers, int)
            or isinstance(self.selected_workers, bool)
            or self.selected_workers <= 0
            or self.selected_workers > self.source_count
            or self.selected_workers > 8
        ):
            raise ValueError("replica copy selected_workers must be within the source and implementation bounds")
        for name, value in (
            ("total_copied_bytes", self.total_copied_bytes),
            ("elapsed_nanoseconds", self.elapsed_nanoseconds),
            ("aggregate_bytes_per_second", self.aggregate_bytes_per_second),
        ):
            _validate_nonnegative_int(value, f"replica copy {name}")
        if self.elapsed_nanoseconds <= 0:
            raise ValueError("replica copy elapsed_nanoseconds must be positive")
        if (
            not isinstance(self.outcomes, tuple)
            or len(self.outcomes) != self.source_count
            or any(not isinstance(item, DatabaseRsyncOutcome) for item in self.outcomes)
            or tuple(sorted(self.outcomes, key=lambda item: item.resolved_source_path)) != self.outcomes
            or len({item.resolved_source_path for item in self.outcomes}) != len(self.outcomes)
        ):
            raise ValueError("replica copy outcomes must be complete and sorted by unique resolved source")
        if self.total_copied_bytes != sum(item.size_bytes for item in self.outcomes):
            raise ValueError("replica copy byte total must equal the unique-source outcomes")
        expected_rate = self.total_copied_bytes * 1_000_000_000 // self.elapsed_nanoseconds
        if self.aggregate_bytes_per_second != expected_rate:
            raise ValueError("replica copy aggregate throughput must use deterministic integer division")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_count": self.source_count,
            "selected_workers": self.selected_workers,
            "total_copied_bytes": self.total_copied_bytes,
            "elapsed_nanoseconds": self.elapsed_nanoseconds,
            "aggregate_bytes_per_second": self.aggregate_bytes_per_second,
            "outcomes": [item.to_mapping() for item in self.outcomes],
        }


def _cache_mount_from_mapping(payload: Mapping[str, object]) -> DatabaseCacheMountFacts:
    _require_fields(
        payload,
        {
            "schema_version",
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
            "read_write",
            "writable_verified",
        },
        "database cache mount facts",
    )
    if payload.get("read_write") is not True or payload.get("writable_verified") is not True:
        raise ValueError("database cache mount facts require true read_write and writable_verified")
    return DatabaseCacheMountFacts(
        mount_id=_required_int(payload, "mount_id"),
        parent_mount_id=_required_int(payload, "parent_mount_id"),
        device_major=_required_int(payload, "device_major"),
        device_minor=_required_int(payload, "device_minor"),
        mount_root=_required_str(payload, "mount_root"),
        mount_point=_required_str(payload, "mount_point"),
        filesystem_type=_required_str(payload, "filesystem_type"),
        mount_source=_required_str(payload, "mount_source"),
        mount_options=_required_str_tuple(payload, "mount_options"),
        super_options=_required_str_tuple(payload, "super_options"),
        read_write=True,
        writable_verified=True,
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="DatabaseCacheMountFacts"),
    )


def database_warm_cache_mount_facts_from_mapping(payload: Mapping[str, object]) -> DatabaseWarmCacheMountFacts:
    """Strictly parse a read-only warm cache mount observation."""
    _require_fields(
        payload,
        {
            "schema_version",
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
            "read_write",
        },
        "warm database cache mount facts",
    )
    if payload.get("read_write") is not True:
        raise ValueError("warm database cache mount facts require true read_write")
    return DatabaseWarmCacheMountFacts(
        mount_id=_required_int(payload, "mount_id"),
        parent_mount_id=_required_int(payload, "parent_mount_id"),
        device_major=_required_int(payload, "device_major"),
        device_minor=_required_int(payload, "device_minor"),
        mount_root=_required_str(payload, "mount_root"),
        mount_point=_required_str(payload, "mount_point"),
        filesystem_type=_required_str(payload, "filesystem_type"),
        mount_source=_required_str(payload, "mount_source"),
        mount_options=_required_str_tuple(payload, "mount_options"),
        super_options=_required_str_tuple(payload, "super_options"),
        read_write=True,
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="DatabaseWarmCacheMountFacts"
        ),
    )


def _capacity_from_mapping(payload: Mapping[str, object]) -> DatabaseCapacityGate:
    _require_fields(
        payload,
        {
            "schema_version",
            "available_user_bytes",
            "allocated_replica_bytes",
            "reserved_bytes",
            "required_bytes",
            "decision",
        },
        "Database Capacity Gate",
    )
    return DatabaseCapacityGate(
        available_user_bytes=_required_int(payload, "available_user_bytes"),
        allocated_replica_bytes=_required_int(payload, "allocated_replica_bytes"),
        reserved_bytes=_required_int(payload, "reserved_bytes"),
        required_bytes=_required_int(payload, "required_bytes"),
        decision=cast("DatabaseCapacityDecision", _required_str(payload, "decision")),
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="DatabaseCapacityGate"),
    )


def _copy_evidence_from_mapping(payload: Mapping[str, object]) -> DatabaseReplicaCopyEvidence:
    _require_fields(
        payload,
        {
            "schema_version",
            "source_count",
            "selected_workers",
            "total_copied_bytes",
            "elapsed_nanoseconds",
            "aggregate_bytes_per_second",
            "outcomes",
        },
        "Database Replica copy evidence",
    )
    return DatabaseReplicaCopyEvidence(
        source_count=_required_int(payload, "source_count"),
        selected_workers=_required_int(payload, "selected_workers"),
        total_copied_bytes=_required_int(payload, "total_copied_bytes"),
        elapsed_nanoseconds=_required_int(payload, "elapsed_nanoseconds"),
        aggregate_bytes_per_second=_required_int(payload, "aggregate_bytes_per_second"),
        outcomes=tuple(_rsync_outcome_from_mapping(item) for item in _required_sequence(payload, "outcomes")),
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="DatabaseReplicaCopyEvidence"
        ),
    )


def _rsync_outcome_from_mapping(value: object) -> DatabaseRsyncOutcome:
    if not isinstance(value, Mapping):
        raise ValueError("rsync outcomes must be mappings")
    _require_fields(
        value,
        {
            "schema_version",
            "resolved_source_path",
            "destination_logical_name",
            "logical_names",
            "size_bytes",
            "argv",
            "exit_code",
        },
        "Database rsync outcome",
    )
    exit_code = _required_int(value, "exit_code")
    if exit_code != 0:
        raise ValueError("Database rsync outcome exit_code must be zero")
    return DatabaseRsyncOutcome(
        resolved_source_path=_required_str(value, "resolved_source_path"),
        destination_logical_name=_required_str(value, "destination_logical_name"),
        logical_names=_required_str_tuple(value, "logical_names"),
        size_bytes=_required_int(value, "size_bytes"),
        argv=_required_str_tuple(value, "argv"),
        exit_code=0,
        schema_version=validate_schema_version(value.get("schema_version"), record_name="DatabaseRsyncOutcome"),
    )


def _source_mount_from_mapping(payload: Mapping[str, object]) -> DatabaseSourceMountFacts:
    _require_fields(
        payload,
        {
            "schema_version",
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
        },
        "database source mount facts",
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
        mount_options=_required_str_tuple(payload, "mount_options"),
        super_options=_required_str_tuple(payload, "super_options"),
        read_only=True,
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="DatabaseSourceMountFacts"),
    )


def _source_observation_from_mapping(payload: Mapping[str, object]) -> DatabaseSourceObservation:
    _require_fields(
        payload,
        {"schema_version", "source_container_root", "source_manifest_sha256", "members", "verification"},
        "Database Source observation",
    )
    return DatabaseSourceObservation(
        source_container_root=_required_str(payload, "source_container_root"),
        source_manifest_sha256=_required_str(payload, "source_manifest_sha256"),
        members=tuple(database_source_member_from_mapping(item) for item in _required_sequence(payload, "members")),
        verification=cast("Literal['metadata-verified']", _required_str(payload, "verification")),
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="DatabaseSourceObservation"),
    )


def _database_set_from_mapping(payload: Mapping[str, object]) -> DatabaseSetIdentity:
    _require_fields(payload, {"identifier", "version"}, "Database Set identity")
    return DatabaseSetIdentity(
        identifier=_required_str(payload, "identifier"),
        version=_required_str(payload, "version"),
    )


def _unwrap_exact(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    _require_fields(payload, {key}, f"{key} document")
    return _required_mapping(payload, key)


def _validate_sha256(value: str, name: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")


__all__ = [
    "DatabaseCacheMountFacts",
    "DatabaseCapacityDecision",
    "DatabaseCapacityGate",
    "DatabaseReplicaCopyEvidence",
    "DatabaseRsyncOutcome",
    "DatabaseWarmCacheMountFacts",
    "database_warm_cache_mount_facts_from_mapping",
]
