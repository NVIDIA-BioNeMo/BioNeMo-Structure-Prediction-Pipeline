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

"""Policy-bound preprocessing database selection and materialization contracts."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
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
    required_mapping_sequence as _required_mapping_sequence,
)
from bspp.orchestration.contract._database_validation import (
    required_nonempty_str as _required_str,
)
from bspp.orchestration.contract._database_validation import (
    required_str_tuple as _required_str_sequence,
)
from bspp.orchestration.contract._database_validation import (
    validate_placement_absolute_path as _validate_absolute_path,
)
from bspp.orchestration.contract._database_validation import (
    validate_schema as _validate_schema,
)
from bspp.orchestration.contract.database_set_provisioning import (
    DatabaseSetIdentity,
    DatabaseSourceManifest,
    DatabaseSourceMember,
    database_source_manifest_digest,
    database_source_manifest_from_mapping,
)
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

SELECTED_DATABASE_ROOT = "/run/bspp/database/selected"
LEGACY_SELECTED_DATABASE_ROOT = "/run/afcdb/database/selected"
DATABASE_SOURCE_ROOT = "/run/bspp/database/source"
LEGACY_DATABASE_SOURCE_ROOT = "/run/afcdb/database/source"
DATABASE_CACHE_ROOT = "/run/bspp/database/cache"
# Historical evidence was authored against the pre-rename cache root; loaders
# must keep accepting it byte-identically.
LEGACY_DATABASE_CACHE_ROOT = "/run/afcdb/database/cache"
DATABASE_REPLICA_LEASE_TARGET = "/run/bspp/database/replica-lease.lock"
LEGACY_DATABASE_REPLICA_LEASE_TARGET = "/run/afcdb/database/replica-lease.lock"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ATTEMPT_ID = re.compile(r"attempt-[0-9]{4}")
_PHASE_RUN_ID = re.compile(r"phase-run-[0-9a-f]{32}")
_ACTION_ID = re.compile(r"preprocessing-chunk-[0-9]{6}")
_SAFE_UNIX_USER = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,63}")


class DatabaseAccessPolicy(StrEnum):
    """Canonical operator-requested database access policy."""

    STAGE_REQUIRED = "stage-required"
    STAGE_PREFERRED = "stage-preferred"
    DIRECT = "direct"


class DatabasePlacementOutcomeKind(StrEnum):
    """A successful outcome which Runtime may later select."""

    REPLICA_COLD = "replica-cold"
    REPLICA_WARM = "replica-warm"
    DIRECT_REQUESTED = "direct-requested"
    DIRECT_CAPACITY_FALLBACK = "direct-capacity-fallback"


DatabasePlacementBranchKind = Literal["staged", "direct-requested", "direct-capacity-fallback"]
DatabaseMountPurpose = Literal["source", "cache", "selected-replica", "replica-lease", "selected-source"]


@dataclass(frozen=True)
class DatabaseSetSelection:
    """Database Set identity and policy authored by a Phase Plan."""

    database_set: DatabaseSetIdentity
    requested_policy: DatabaseAccessPolicy = DatabaseAccessPolicy.STAGE_REQUIRED
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "DatabaseSetSelection")
        if not isinstance(self.database_set, DatabaseSetIdentity):
            raise ValueError("Database Set selection requires an exact DatabaseSetIdentity")
        if not isinstance(self.requested_policy, DatabaseAccessPolicy):
            raise ValueError("Database Set selection requires a canonical requested policy")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "database_set": self.database_set.to_mapping(),
            "requested_policy": self.requested_policy.value,
        }


@dataclass(frozen=True)
class DatabaseProfileStagingSnapshot:
    """Site-owned staging configuration frozen into a Phase RunSpec."""

    cache_root: str
    unix_user: str
    expected_filesystem_type: str
    reserve_bytes: int
    lock_wait_seconds: int
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "DatabaseProfileStagingSnapshot")
        _validate_absolute_path(self.cache_root, "database cache_root")
        if _SAFE_UNIX_USER.fullmatch(self.unix_user) is None:
            raise ValueError("database staging unix_user must be a safe Unix user identity")
        if not self.expected_filesystem_type or "/" in self.expected_filesystem_type:
            raise ValueError("database expected_filesystem_type must be a non-empty filesystem token")
        if not isinstance(self.reserve_bytes, int) or isinstance(self.reserve_bytes, bool) or self.reserve_bytes < 0:
            raise ValueError("database reserve_bytes must be a non-negative integer")
        if (
            not isinstance(self.lock_wait_seconds, int)
            or isinstance(self.lock_wait_seconds, bool)
            or self.lock_wait_seconds <= 0
        ):
            raise ValueError("database lock_wait_seconds must be a positive integer")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "cache_root": self.cache_root,
            "unix_user": self.unix_user,
            "expected_filesystem_type": self.expected_filesystem_type,
            "reserve_bytes": self.reserve_bytes,
            "lock_wait_seconds": self.lock_wait_seconds,
        }

    @property
    def user_cache_root(self) -> str:
        """Return the only cache namespace writable by this executing user."""
        return str(PurePosixPath(self.cache_root) / "users" / self.unix_user)


@dataclass(frozen=True)
class DatabaseMountDescriptor:
    """One exact protected host-to-container database mount authority."""

    source: str
    target: str
    purpose: DatabaseMountPurpose
    read_only: bool
    source_kind: Literal["file", "directory"] = "directory"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "DatabaseMountDescriptor")
        _validate_absolute_path(self.source, "database mount source")
        _validate_absolute_path(self.target, "database mount target")
        if self.purpose not in {"source", "cache", "selected-replica", "replica-lease", "selected-source"}:
            raise ValueError(f"unsupported database mount purpose: {self.purpose!r}")
        if not isinstance(self.read_only, bool):
            raise ValueError("database mount read_only must be boolean")
        if self.source_kind not in {"file", "directory"}:
            raise ValueError("database mount source_kind must be 'file' or 'directory'")
        if self.purpose == "cache" and self.read_only:
            raise ValueError("database cache placement mount must be read-write")
        if self.purpose != "cache" and not self.read_only:
            raise ValueError("database source and scientific mounts must be read-only")

    def to_mapping(self) -> dict[str, object]:
        mapping: dict[str, object] = {
            "schema_version": self.schema_version,
            "source": self.source,
            "target": self.target,
            "purpose": self.purpose,
            "read_only": self.read_only,
        }
        if self.source_kind == "file":
            mapping["source_kind"] = "file"
        return mapping


@dataclass(frozen=True)
class DatabasePlacementBranch:
    """One exact policy-authorized placement/science branch."""

    branch_kind: DatabasePlacementBranchKind
    authorized_outcomes: tuple[DatabasePlacementOutcomeKind, ...]
    placement_mounts: tuple[DatabaseMountDescriptor, ...]
    scientific_mounts: tuple[DatabaseMountDescriptor, ...]
    finalization_mounts: tuple[DatabaseMountDescriptor, ...]
    gpuserver_argv: tuple[str, ...]
    search_argv: tuple[str, ...]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "DatabasePlacementBranch")
        if self.branch_kind not in {"staged", "direct-requested", "direct-capacity-fallback"}:
            raise ValueError(f"unsupported database placement branch: {self.branch_kind!r}")
        expected_outcomes = {
            "staged": (
                DatabasePlacementOutcomeKind.REPLICA_COLD,
                DatabasePlacementOutcomeKind.REPLICA_WARM,
            ),
            "direct-requested": (DatabasePlacementOutcomeKind.DIRECT_REQUESTED,),
            "direct-capacity-fallback": (DatabasePlacementOutcomeKind.DIRECT_CAPACITY_FALLBACK,),
        }[self.branch_kind]
        if (
            not isinstance(self.authorized_outcomes, tuple)
            or any(not isinstance(item, DatabasePlacementOutcomeKind) for item in self.authorized_outcomes)
            or self.authorized_outcomes != expected_outcomes
        ):
            raise ValueError("database branch outcomes do not match the declared branch")
        for name, mounts in (
            ("placement_mounts", self.placement_mounts),
            ("scientific_mounts", self.scientific_mounts),
            ("finalization_mounts", self.finalization_mounts),
        ):
            if not isinstance(mounts, tuple) or any(not isinstance(item, DatabaseMountDescriptor) for item in mounts):
                raise ValueError(f"{name} must be an immutable tuple of database mount descriptors")
        if self.finalization_mounts:
            raise ValueError("database finalization must not receive payload mounts")
        if any(
            not isinstance(argv, tuple) or not argv or any(not isinstance(item, str) or not item for item in argv)
            for argv in (self.gpuserver_argv, self.search_argv)
        ):
            raise ValueError("database branches must bind exact non-empty Scientific Kernel commands")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "branch_kind": self.branch_kind,
            "authorized_outcomes": [item.value for item in self.authorized_outcomes],
            "placement_mounts": [item.to_mapping() for item in self.placement_mounts],
            "scientific_mounts": [item.to_mapping() for item in self.scientific_mounts],
            "finalization_mounts": [item.to_mapping() for item in self.finalization_mounts],
            "gpuserver_argv": list(self.gpuserver_argv),
            "search_argv": list(self.search_argv),
        }


@dataclass(frozen=True)
class PreprocessingDatabaseBinding:
    """Immutable manifest, policy, branch, command, and mount authority."""

    database_set: DatabaseSetIdentity
    requested_policy: DatabaseAccessPolicy
    source_manifest: DatabaseSourceManifest
    source_manifest_sha256: str
    primary_database_name: str
    metagenomic_database_name: str
    source_manifest_projection: str
    staging: DatabaseProfileStagingSnapshot | None
    branches: tuple[DatabasePlacementBranch, ...]
    selected_container_root: str = SELECTED_DATABASE_ROOT
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, "PreprocessingDatabaseBinding")
        if not isinstance(self.database_set, DatabaseSetIdentity):
            raise ValueError("database binding requires an exact DatabaseSetIdentity")
        if not isinstance(self.requested_policy, DatabaseAccessPolicy):
            raise ValueError("database binding requires a canonical requested policy")
        if not isinstance(self.source_manifest, DatabaseSourceManifest):
            raise ValueError("database binding requires an exact DatabaseSourceManifest")
        if self.database_set != self.source_manifest.database_set:
            raise ValueError("database binding identity must match its embedded source manifest")
        if self.source_manifest_sha256 != database_source_manifest_digest(self.source_manifest):
            raise ValueError("database binding digest must match exact canonical source-manifest bytes")
        if _SHA256.fullmatch(self.source_manifest_sha256) is None:
            raise ValueError("database binding source_manifest_sha256 must be lowercase SHA-256")
        expected_primary, expected_metagenomic = database_source_manifest_names(self.source_manifest)
        if (self.primary_database_name, self.metagenomic_database_name) != (
            expected_primary,
            expected_metagenomic,
        ):
            raise ValueError("database binding names must come from the embedded source manifest")
        if self.selected_container_root not in (SELECTED_DATABASE_ROOT, LEGACY_SELECTED_DATABASE_ROOT):
            raise ValueError(f"database binding selected_container_root must be {SELECTED_DATABASE_ROOT!r}")
        projection = PurePosixPath(self.source_manifest_projection)
        if (
            projection.is_absolute()
            or len(projection.parts) != 3
            or projection.parts[0] != "attempts"
            or _ATTEMPT_ID.fullmatch(projection.parts[1]) is None
            or projection.parts[2] != "database-source-manifest.json"
        ):
            raise ValueError("database source-manifest projection must be attempt-scoped and relative")
        if not isinstance(self.branches, tuple) or any(
            not isinstance(branch, DatabasePlacementBranch) for branch in self.branches
        ):
            raise ValueError("database binding branches must be an immutable tuple")
        expected_kinds = database_branch_kinds(self.requested_policy)
        if tuple(branch.branch_kind for branch in self.branches) != expected_kinds:
            raise ValueError("database binding branches must exactly close the requested policy")
        if any(branch.branch_kind == "staged" for branch in self.branches) and self.staging is None:
            raise ValueError("staged database branches require a complete profile staging snapshot")
        command_pairs = {(branch.gpuserver_argv, branch.search_argv) for branch in self.branches}
        if len(command_pairs) != 1:
            raise ValueError("all database branches must bind identical Scientific Kernel commands")
        expected_mounts = _expected_branch_mounts(self)
        legacy_expected_mounts = _expected_branch_mounts(self, legacy=True)
        observed_mounts = tuple(
            (branch.placement_mounts, branch.scientific_mounts, branch.finalization_mounts) for branch in self.branches
        )
        if observed_mounts not in (expected_mounts, legacy_expected_mounts):
            raise ValueError("database branch mounts do not match protected placement authority")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "database_set": self.database_set.to_mapping(),
            "requested_policy": self.requested_policy.value,
            "source_manifest": self.source_manifest.to_mapping(),
            "source_manifest_sha256": self.source_manifest_sha256,
            "primary_database_name": self.primary_database_name,
            "metagenomic_database_name": self.metagenomic_database_name,
            "source_manifest_projection": self.source_manifest_projection,
            "staging": self.staging.to_mapping() if self.staging is not None else None,
            "branches": [branch.to_mapping() for branch in self.branches],
            "selected_container_root": self.selected_container_root,
        }

    @property
    def identity_protected_targets(self) -> tuple[str, ...]:
        """Return every database host/container namespace forbidden to generic mounts."""
        values = {
            value
            for branch in self.branches
            for mounts in (branch.placement_mounts, branch.scientific_mounts, branch.finalization_mounts)
            for mount in mounts
            for value in (mount.source, mount.target)
        }
        values.add(self.selected_container_root)
        if self.staging is not None:
            values.add(self.staging.cache_root)
            values.add(self.staging.user_cache_root)
        return tuple(sorted(values))


def database_branch_kinds(policy: DatabaseAccessPolicy) -> tuple[DatabasePlacementBranchKind, ...]:
    """Return the exact branch closure authorized by one policy."""
    if not isinstance(policy, DatabaseAccessPolicy):
        raise ValueError(f"unsupported database access policy: {policy!r}")
    if policy == DatabaseAccessPolicy.STAGE_REQUIRED:
        return ("staged",)
    if policy == DatabaseAccessPolicy.STAGE_PREFERRED:
        return ("staged", "direct-capacity-fallback")
    if policy == DatabaseAccessPolicy.DIRECT:
        return ("direct-requested",)
    raise ValueError(f"unsupported database access policy: {policy!r}")


def database_source_manifest_names(manifest: DatabaseSourceManifest) -> tuple[str, str]:
    """Return the exact primary and metagenomic logical database names."""
    names: dict[str, str] = {}
    for member in manifest.members:
        existing = names.setdefault(member.role, member.database_name)
        if existing != member.database_name:
            raise ValueError(f"Database Source Manifest has conflicting {member.role} database names")
    try:
        return names["primary"], names["metagenomic"]
    except KeyError as exc:
        raise ValueError("Database Source Manifest must bind primary and metagenomic names") from exc


def build_preprocessing_database_binding(
    *,
    selection: DatabaseSetSelection,
    source_manifest: DatabaseSourceManifest,
    source_manifest_projection: str,
    staging: DatabaseProfileStagingSnapshot | None,
    gpuserver_argv: tuple[str, ...],
    search_argv: tuple[str, ...],
) -> PreprocessingDatabaseBinding:
    """Materialize exact policy branches from verified manifest and profile authority."""
    if selection.database_set != source_manifest.database_set:
        raise ValueError("selected Database Set does not match the resolved source manifest")
    digest = database_source_manifest_digest(source_manifest)
    primary, metagenomic = database_source_manifest_names(source_manifest)
    branches = tuple(
        _build_database_branch(
            kind=kind,
            source_root=source_manifest.source_root,
            source_manifest_sha256=digest,
            members=source_manifest.members,
            staging=staging,
            gpuserver_argv=gpuserver_argv,
            search_argv=search_argv,
        )
        for kind in database_branch_kinds(selection.requested_policy)
    )
    return PreprocessingDatabaseBinding(
        database_set=selection.database_set,
        requested_policy=selection.requested_policy,
        source_manifest=source_manifest,
        source_manifest_sha256=digest,
        primary_database_name=primary,
        metagenomic_database_name=metagenomic,
        source_manifest_projection=source_manifest_projection,
        staging=staging,
        branches=branches,
    )


def _build_database_branch(
    *,
    kind: DatabasePlacementBranchKind,
    source_root: str,
    source_manifest_sha256: str,
    members: tuple[DatabaseSourceMember, ...],
    staging: DatabaseProfileStagingSnapshot | None,
    gpuserver_argv: tuple[str, ...],
    search_argv: tuple[str, ...],
) -> DatabasePlacementBranch:
    placement_mounts, scientific_mounts = _branch_mounts(
        source_root=source_root,
        source_manifest_sha256=source_manifest_sha256,
        members=members,
        staging=staging,
        kind=kind,
    )
    return DatabasePlacementBranch(
        branch_kind=kind,
        authorized_outcomes=_authorized_outcomes(kind),
        placement_mounts=placement_mounts,
        scientific_mounts=scientific_mounts,
        finalization_mounts=(),
        gpuserver_argv=gpuserver_argv,
        search_argv=search_argv,
    )


def database_set_selection_from_mapping(payload: Mapping[str, object]) -> DatabaseSetSelection:
    """Strict-load a Plan database selection, defaulting only its policy."""
    _require_fields(
        payload,
        {"database_set"},
        optional={"schema_version", "requested_policy"},
        name="Database Set selection",
    )
    identity = _database_set_identity_from_mapping(_required_mapping(payload, "database_set"))
    policy_value = payload.get("requested_policy", DatabaseAccessPolicy.STAGE_REQUIRED.value)
    return DatabaseSetSelection(
        database_set=identity,
        requested_policy=_database_access_policy(policy_value),
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="DatabaseSetSelection"),
    )


def preprocessing_database_binding_from_mapping(payload: Mapping[str, object]) -> PreprocessingDatabaseBinding:
    """Strict-load and fully revalidate an immutable RunSpec database binding."""
    _require_fields(
        payload,
        {
            "database_set",
            "requested_policy",
            "source_manifest",
            "source_manifest_sha256",
            "primary_database_name",
            "metagenomic_database_name",
            "source_manifest_projection",
            "staging",
            "branches",
            "selected_container_root",
        },
        optional={"schema_version"},
        name="preprocessing database binding",
    )
    staging_value = payload.get("staging")
    if staging_value is not None and not isinstance(staging_value, Mapping):
        raise ValueError("database staging snapshot must be a mapping or null")
    return PreprocessingDatabaseBinding(
        database_set=_database_set_identity_from_mapping(_required_mapping(payload, "database_set")),
        requested_policy=_database_access_policy(payload.get("requested_policy")),
        source_manifest=database_source_manifest_from_mapping(_required_mapping(payload, "source_manifest")),
        source_manifest_sha256=_required_str(payload, "source_manifest_sha256"),
        primary_database_name=_required_str(payload, "primary_database_name"),
        metagenomic_database_name=_required_str(payload, "metagenomic_database_name"),
        source_manifest_projection=_required_str(payload, "source_manifest_projection"),
        staging=(
            database_profile_staging_snapshot_from_mapping(cast("Mapping[str, object]", staging_value))
            if staging_value is not None
            else None
        ),
        branches=tuple(
            database_placement_branch_from_mapping(item) for item in _required_mapping_sequence(payload, "branches")
        ),
        selected_container_root=_required_str(payload, "selected_container_root"),
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="PreprocessingDatabaseBinding"
        ),
    )


def database_profile_staging_snapshot_from_mapping(
    payload: Mapping[str, object],
) -> DatabaseProfileStagingSnapshot:
    _require_fields(
        payload,
        {"cache_root", "unix_user", "expected_filesystem_type", "reserve_bytes", "lock_wait_seconds"},
        optional={"schema_version"},
        name="database profile staging snapshot",
    )
    return DatabaseProfileStagingSnapshot(
        cache_root=_required_str(payload, "cache_root"),
        unix_user=_required_str(payload, "unix_user"),
        expected_filesystem_type=_required_str(payload, "expected_filesystem_type"),
        reserve_bytes=_required_int(payload, "reserve_bytes"),
        lock_wait_seconds=_required_int(payload, "lock_wait_seconds"),
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="DatabaseProfileStagingSnapshot"
        ),
    )


def database_placement_branch_from_mapping(payload: Mapping[str, object]) -> DatabasePlacementBranch:
    _require_fields(
        payload,
        {
            "branch_kind",
            "authorized_outcomes",
            "placement_mounts",
            "scientific_mounts",
            "finalization_mounts",
            "gpuserver_argv",
            "search_argv",
        },
        optional={"schema_version"},
        name="database placement branch",
    )
    kind_value = _required_str(payload, "branch_kind")
    if kind_value not in {"staged", "direct-requested", "direct-capacity-fallback"}:
        raise ValueError(f"unsupported database placement branch: {kind_value!r}")
    return DatabasePlacementBranch(
        branch_kind=cast("DatabasePlacementBranchKind", kind_value),
        authorized_outcomes=tuple(
            _database_outcome(item) for item in _required_str_sequence(payload, "authorized_outcomes")
        ),
        placement_mounts=tuple(
            database_mount_descriptor_from_mapping(item)
            for item in _required_mapping_sequence(payload, "placement_mounts")
        ),
        scientific_mounts=tuple(
            database_mount_descriptor_from_mapping(item)
            for item in _required_mapping_sequence(payload, "scientific_mounts")
        ),
        finalization_mounts=tuple(
            database_mount_descriptor_from_mapping(item)
            for item in _required_mapping_sequence(payload, "finalization_mounts")
        ),
        gpuserver_argv=_required_str_sequence(payload, "gpuserver_argv"),
        search_argv=_required_str_sequence(payload, "search_argv"),
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="DatabasePlacementBranch"),
    )


def database_mount_descriptor_from_mapping(payload: Mapping[str, object]) -> DatabaseMountDescriptor:
    _require_fields(
        payload,
        {"source", "target", "purpose", "read_only"},
        optional={"schema_version", "source_kind"},
        name="database mount descriptor",
    )
    purpose = _required_str(payload, "purpose")
    if purpose not in {"source", "cache", "selected-replica", "replica-lease", "selected-source"}:
        raise ValueError(f"unsupported database mount purpose: {purpose!r}")
    read_only = payload.get("read_only")
    if not isinstance(read_only, bool):
        raise ValueError("database mount read_only must be boolean")
    source_kind_value = payload.get("source_kind", "directory")
    if source_kind_value not in {"file", "directory"}:
        raise ValueError(f"unsupported database mount source_kind: {source_kind_value!r}")
    return DatabaseMountDescriptor(
        source=_required_str(payload, "source"),
        target=_required_str(payload, "target"),
        purpose=cast("DatabaseMountPurpose", purpose),
        read_only=read_only,
        source_kind=cast("Literal['file', 'directory']", source_kind_value),
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="DatabaseMountDescriptor"),
    )


def _expected_branch_mounts(
    binding: PreprocessingDatabaseBinding,
    *,
    legacy: bool = False,
) -> tuple[
    tuple[
        tuple[DatabaseMountDescriptor, ...],
        tuple[DatabaseMountDescriptor, ...],
        tuple[DatabaseMountDescriptor, ...],
    ],
    ...,
]:
    return tuple(
        (
            *_branch_mounts(
                source_root=binding.source_manifest.source_root,
                source_manifest_sha256=binding.source_manifest_sha256,
                members=binding.source_manifest.members,
                staging=binding.staging,
                kind=branch.branch_kind,
                legacy=legacy,
            ),
            (),
        )
        for branch in binding.branches
    )


def _branch_mounts(
    *,
    source_root: str,
    source_manifest_sha256: str,
    members: tuple[DatabaseSourceMember, ...],
    staging: DatabaseProfileStagingSnapshot | None,
    kind: DatabasePlacementBranchKind,
    legacy: bool = False,
) -> tuple[tuple[DatabaseMountDescriptor, ...], tuple[DatabaseMountDescriptor, ...]]:
    source_target = LEGACY_DATABASE_SOURCE_ROOT if legacy else DATABASE_SOURCE_ROOT
    cache_target = LEGACY_DATABASE_CACHE_ROOT if legacy else DATABASE_CACHE_ROOT
    selected_target = LEGACY_SELECTED_DATABASE_ROOT if legacy else SELECTED_DATABASE_ROOT
    lease_target = LEGACY_DATABASE_REPLICA_LEASE_TARGET if legacy else DATABASE_REPLICA_LEASE_TARGET
    source_mount = DatabaseMountDescriptor(
        source=source_root,
        target=source_target,
        purpose="source",
        read_only=True,
    )
    if kind == "staged":
        if staging is None:
            raise ValueError("staged database branch requires profile staging authority")
        cache_mount = DatabaseMountDescriptor(
            source=staging.user_cache_root,
            target=cache_target,
            purpose="cache",
            read_only=False,
        )
        replica_source = str(PurePosixPath(staging.user_cache_root) / "replicas" / source_manifest_sha256)
        science_mount = DatabaseMountDescriptor(
            source=replica_source,
            target=selected_target,
            purpose="selected-replica",
            read_only=True,
        )
        lease_source = str(PurePosixPath(staging.user_cache_root) / ".locks" / f"{source_manifest_sha256}.lock")
        lease_mount = DatabaseMountDescriptor(
            source=lease_source,
            target=lease_target,
            purpose="replica-lease",
            read_only=True,
        )
        return (source_mount, cache_mount), (science_mount, lease_mount)
    direct_mounts = tuple(
        DatabaseMountDescriptor(
            source=str(PurePosixPath(source_root) / member.resolved_path),
            target=str(PurePosixPath(selected_target) / member.logical_name),
            purpose="selected-source",
            read_only=True,
            source_kind="file",
        )
        for member in sorted(members, key=lambda item: item.logical_name)
    )
    return (source_mount,), direct_mounts


def _authorized_outcomes(
    kind: DatabasePlacementBranchKind,
) -> tuple[DatabasePlacementOutcomeKind, ...]:
    return {
        "staged": (
            DatabasePlacementOutcomeKind.REPLICA_COLD,
            DatabasePlacementOutcomeKind.REPLICA_WARM,
        ),
        "direct-requested": (DatabasePlacementOutcomeKind.DIRECT_REQUESTED,),
        "direct-capacity-fallback": (DatabasePlacementOutcomeKind.DIRECT_CAPACITY_FALLBACK,),
    }[kind]


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
    "DATABASE_CACHE_ROOT",
    "DATABASE_SOURCE_ROOT",
    "SELECTED_DATABASE_ROOT",
    "DatabaseAccessPolicy",
    "DatabaseMountDescriptor",
    "DatabasePlacementBranch",
    "DatabasePlacementBranchKind",
    "DatabasePlacementOutcomeKind",
    "DatabaseProfileStagingSnapshot",
    "DatabaseSetSelection",
    "PreprocessingDatabaseBinding",
    "build_preprocessing_database_binding",
    "database_branch_kinds",
    "database_mount_descriptor_from_mapping",
    "database_placement_branch_from_mapping",
    "database_profile_staging_snapshot_from_mapping",
    "database_set_selection_from_mapping",
    "database_source_manifest_names",
    "preprocessing_database_binding_from_mapping",
]
