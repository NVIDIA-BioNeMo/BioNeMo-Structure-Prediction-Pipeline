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

"""Strict contracts for explicit cluster-side Database Set Provisioning."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import yaml

from bspp.orchestration.contract._database_validation import (
    require_fields as _require_keys,
)
from bspp.orchestration.contract._database_validation import (
    required_bool as _required_bool,
)
from bspp.orchestration.contract._database_validation import (
    required_int as _required_int,
)
from bspp.orchestration.contract._database_validation import (
    required_list as _required_list,
)
from bspp.orchestration.contract._database_validation import (
    required_mapping as _required_mapping,
)
from bspp.orchestration.contract._database_validation import (
    required_str as _required_str,
)
from bspp.orchestration.contract._database_validation import (
    validate_nonnegative_int as _validate_nonnegative_int,
)
from bspp.orchestration.contract._database_validation import (
    validate_provisioning_absolute_path as _validate_absolute_path,
)
from bspp.orchestration.contract._database_validation import (
    validate_relative_path as _validate_relative_path,
)
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

DatabaseRole = Literal["primary", "metagenomic"]
DatabaseMemberKind = Literal["regular", "symlink"]
ProvisioningDisposition = Literal["published", "reused"]
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CHECKSUM_ALGORITHM = re.compile(r"[a-z0-9][a-z0-9._+-]{0,31}")
_CHECKSUM_VALUE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/+=-]{0,255}")


@dataclass(frozen=True)
class DatabaseSetIdentity:
    """Stable logical Database Set identifier and version."""

    identifier: str
    version: str

    def __post_init__(self) -> None:
        _validate_identifier(self.identifier, "database_set.identifier")
        _validate_identifier(self.version, "database_set.version")

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON/YAML-ready data."""
        return {"identifier": self.identifier, "version": self.version}


@dataclass(frozen=True)
class PreexistingChecksum:
    """A declaration-supplied checksum retained without reading payload bytes."""

    algorithm: str
    value: str

    def __post_init__(self) -> None:
        if _CHECKSUM_ALGORITHM.fullmatch(self.algorithm) is None:
            raise ValueError("preexisting_checksum algorithm must be a safe lowercase token of at most 32 characters")
        if _CHECKSUM_VALUE.fullmatch(self.value) is None:
            raise ValueError("preexisting_checksum value must be a safe non-empty token of at most 256 characters")

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON/YAML-ready data."""
        return {"algorithm": self.algorithm, "value": self.value}


@dataclass(frozen=True)
class DeclaredDatabaseMember:
    """One required logical member and its source-relative path."""

    logical_name: str
    source_path: str
    preexisting_checksum: PreexistingChecksum | None

    def __post_init__(self) -> None:
        _validate_identifier(self.logical_name, "logical_name")
        _validate_relative_path(self.source_path, "source_path")

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON/YAML-ready data."""
        return {
            "logical_name": self.logical_name,
            "source_path": self.source_path,
            "preexisting_checksum": (
                self.preexisting_checksum.to_mapping() if self.preexisting_checksum is not None else None
            ),
        }


@dataclass(frozen=True)
class DeclaredDatabaseRole:
    """One authoritative primary or metagenomic logical database."""

    role: DatabaseRole
    database_name: str
    members: tuple[DeclaredDatabaseMember, ...]

    def __post_init__(self) -> None:
        if self.role not in {"primary", "metagenomic"}:
            raise ValueError(f"unsupported database role: {self.role!r}")
        _validate_identifier(self.database_name, "database_name")
        if not isinstance(self.members, tuple) or not self.members:
            raise ValueError(f"database role {self.role!r} must declare a complete non-empty member inventory")
        logical_names = [member.logical_name for member in self.members]
        if len(logical_names) != len(set(logical_names)):
            raise ValueError(f"database role {self.role!r} has duplicate logical member authority")
        if self.database_name not in logical_names:
            raise ValueError(f"database role {self.role!r} is missing its database_name target")

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON/YAML-ready data."""
        return {
            "role": self.role,
            "database_name": self.database_name,
            "members": [member.to_mapping() for member in self.members],
        }


@dataclass(frozen=True)
class DatabaseSetDeclaration:
    """Explicit source declaration for one Database Set version."""

    database_set: DatabaseSetIdentity
    source_root: str
    roles: tuple[DeclaredDatabaseRole, ...]
    source_kind: Literal["local", "s3"] = "local"
    source_uri: str | None = None
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, record_name="DatabaseSetDeclaration")
        _validate_absolute_path(self.source_root, "source_root")
        if self.source_kind not in {"local", "s3"}:
            raise ValueError(f"unsupported database set source_kind: {self.source_kind!r}")
        if self.source_kind == "s3":
            if (
                self.source_uri is None
                or not self.source_uri.startswith("s3://")
                or self.source_uri.rstrip("/") == "s3://"
            ):
                raise ValueError("s3 source_kind requires a non-empty s3:// source_uri")
        else:
            if self.source_uri is not None:
                raise ValueError("local source_kind requires source_uri to be None")
        if not isinstance(self.roles, tuple):
            raise ValueError("roles must be an immutable tuple")
        role_names = [role.role for role in self.roles]
        if len(role_names) != 2 or set(role_names) != {"primary", "metagenomic"}:
            raise ValueError("roles must contain primary and metagenomic authority exactly once")
        role_order = {"primary": 0, "metagenomic": 1}
        object.__setattr__(self, "roles", tuple(sorted(self.roles, key=lambda role: role_order[role.role])))
        database_names = [role.database_name for role in self.roles]
        if len(database_names) != len(set(database_names)):
            raise ValueError("database roles must have distinct logical database names")
        logical_names = [member.logical_name for role in self.roles for member in role.members]
        if len(logical_names) != len(set(logical_names)):
            raise ValueError("Database Set has duplicate logical member authority")

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON/YAML-ready data."""
        inner: dict[str, object] = {
            "schema_version": self.schema_version,
            "database_set": self.database_set.to_mapping(),
            "source_root": self.source_root,
            "roles": [role.to_mapping() for role in self.roles],
        }
        if self.source_kind != "local":
            inner["source_kind"] = self.source_kind
        if self.source_uri is not None:
            inner["source_uri"] = self.source_uri
        return {"database_set_declaration": inner}


@dataclass(frozen=True)
class DatabaseAlias:
    """One observed source-relative symlink hop."""

    path: str
    target: str

    def __post_init__(self) -> None:
        _validate_relative_path(self.path, "alias path")
        if not self.target or "\x00" in self.target:
            raise ValueError("alias target must be a non-empty path")

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON/YAML-ready data."""
        return {"path": self.path, "target": self.target}


@dataclass(frozen=True)
class DatabaseSourceMember:
    """One complete logical-to-resolved member inventory entry."""

    role: DatabaseRole
    database_name: str
    logical_name: str
    source_path: str
    source_kind: DatabaseMemberKind
    resolved_path: str
    resolved_kind: Literal["regular"]
    size_bytes: int
    mtime_ns: int
    alias_topology: tuple[DatabaseAlias, ...]
    preexisting_checksum: PreexistingChecksum | None

    def __post_init__(self) -> None:
        if self.role not in {"primary", "metagenomic"}:
            raise ValueError(f"unsupported database role: {self.role!r}")
        _validate_identifier(self.database_name, "database_name")
        _validate_identifier(self.logical_name, "logical_name")
        _validate_relative_path(self.source_path, "source_path")
        _validate_relative_path(self.resolved_path, "resolved_path")
        if self.source_kind not in {"regular", "symlink"} or self.resolved_kind != "regular":
            raise ValueError("manifest members must resolve from regular files or confined symlinks to regular files")
        _validate_nonnegative_int(self.size_bytes, "size_bytes")
        _validate_nonnegative_int(self.mtime_ns, "mtime_ns")
        if not isinstance(self.alias_topology, tuple):
            raise ValueError("alias_topology must be an immutable tuple")
        alias_paths = [alias.path for alias in self.alias_topology]
        if len(alias_paths) != len(set(alias_paths)):
            raise ValueError("alias_topology must not contain a cycle or duplicate hop")
        if self.source_kind == "symlink" and not self.alias_topology:
            raise ValueError("symlink source members must declare alias topology")

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON/YAML-ready data."""
        return {
            "role": self.role,
            "database_name": self.database_name,
            "logical_name": self.logical_name,
            "source_path": self.source_path,
            "source_kind": self.source_kind,
            "resolved_path": self.resolved_path,
            "resolved_kind": self.resolved_kind,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
            "alias_topology": [alias.to_mapping() for alias in self.alias_topology],
            "preexisting_checksum": (
                self.preexisting_checksum.to_mapping() if self.preexisting_checksum is not None else None
            ),
        }


@dataclass(frozen=True)
class DatabaseSourceManifest:
    """Immutable metadata-verified inventory for one Database Set version."""

    database_set: DatabaseSetIdentity
    source_root: str
    members: tuple[DatabaseSourceMember, ...]
    verification: Literal["metadata-verified"] = "metadata-verified"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, record_name="DatabaseSourceManifest")
        if self.verification != "metadata-verified":
            raise ValueError("Database Source Manifest verification must be 'metadata-verified'")
        _validate_absolute_path(self.source_root, "manifest source_root")
        validate_database_source_member_topology(self.members, record_name="Database Source Manifest")

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON/YAML-ready data."""
        return {
            "database_source_manifest": {
                "schema_version": self.schema_version,
                "database_set": self.database_set.to_mapping(),
                "verification": self.verification,
                "source_root": self.source_root,
                "members": [member.to_mapping() for member in self.members],
            }
        }


def validate_database_source_member_topology(
    members: tuple[DatabaseSourceMember, ...],
    *,
    record_name: str,
) -> None:
    """Require complete primary/metagenomic logical-member authority."""
    if (
        not isinstance(members, tuple)
        or not members
        or any(not isinstance(member, DatabaseSourceMember) for member in members)
    ):
        raise ValueError(f"{record_name} must contain a complete member inventory")
    logical_names = [member.logical_name for member in members]
    if len(logical_names) != len(set(logical_names)):
        raise ValueError(f"{record_name} has duplicate logical member authority")
    for role_name in ("primary", "metagenomic"):
        role_members = [member for member in members if member.role == role_name]
        database_names = {member.database_name for member in role_members}
        if len(database_names) != 1 or next(iter(database_names), None) not in {
            member.logical_name for member in role_members
        }:
            raise ValueError(f"{record_name} has an incomplete {role_name} inventory")


@dataclass(frozen=True)
class DatabaseSetProvisioningEvidence:
    """Durable result of publishing or exactly reusing a source manifest."""

    database_set: DatabaseSetIdentity
    disposition: ProvisioningDisposition
    manifest_path: str
    manifest_sha256: str
    member_count: int
    verification: Literal["metadata-verified"] = "metadata-verified"
    payload_bytes_copied: bool = False
    payload_bytes_hashed: bool = False
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version, record_name="DatabaseSetProvisioningEvidence")
        if self.disposition not in {"published", "reused"}:
            raise ValueError(f"unsupported provisioning disposition: {self.disposition!r}")
        if self.verification != "metadata-verified":
            raise ValueError("provisioning verification must be 'metadata-verified'")
        if not Path(self.manifest_path).is_absolute():
            raise ValueError("manifest_path must be absolute")
        if _SHA256.fullmatch(self.manifest_sha256) is None:
            raise ValueError("manifest_sha256 must be a lowercase sha256 value")
        if not isinstance(self.member_count, int) or isinstance(self.member_count, bool) or self.member_count <= 0:
            raise ValueError("member_count must be a positive integer")
        if self.payload_bytes_copied is not False or self.payload_bytes_hashed is not False:
            raise ValueError("Database Set Provisioning cannot copy or hash payload bytes")

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic JSON/YAML-ready data."""
        return {
            "database_set_provisioning": {
                "schema_version": self.schema_version,
                "database_set": self.database_set.to_mapping(),
                "disposition": self.disposition,
                "verification": self.verification,
                "manifest_path": self.manifest_path,
                "manifest_sha256": self.manifest_sha256,
                "member_count": self.member_count,
                "payload_bytes_copied": self.payload_bytes_copied,
                "payload_bytes_hashed": self.payload_bytes_hashed,
            }
        }


def database_set_declaration_from_mapping(payload: Mapping[str, object]) -> DatabaseSetDeclaration:
    """Strict-load a Database Set declaration from outer or inner mapping data."""
    inner = _unwrap(payload, "database_set_declaration")
    _require_keys(
        inner,
        {"database_set", "source_root", "roles"},
        "database_set_declaration",
        optional={"schema_version", "source_kind", "source_uri"},
    )
    schema_version = validate_schema_version(inner.get("schema_version"), record_name="DatabaseSetDeclaration")
    identity_payload = _required_mapping(inner, "database_set")
    _require_keys(identity_payload, {"identifier", "version"}, "database_set")
    roles_payload = _required_list(inner, "roles")
    roles = tuple(_database_role_from_mapping(item) for item in roles_payload)
    source_kind_value = inner.get("source_kind", "local")
    if source_kind_value not in {"local", "s3"}:
        raise ValueError(f"unsupported database set source_kind: {source_kind_value!r}")
    source_kind = cast("Literal['local', 's3']", source_kind_value)
    source_uri_value = inner.get("source_uri")
    if source_uri_value is not None and not isinstance(source_uri_value, str):
        raise ValueError("source_uri must be null or a non-empty string")
    if source_uri_value is not None and not source_uri_value:
        raise ValueError("source_uri must be null or a non-empty string")
    return DatabaseSetDeclaration(
        database_set=DatabaseSetIdentity(
            identifier=_required_str(identity_payload, "identifier"),
            version=_required_str(identity_payload, "version"),
        ),
        source_root=_required_str(inner, "source_root"),
        roles=roles,
        source_kind=source_kind,
        source_uri=source_uri_value,
        schema_version=schema_version,
    )


def load_database_set_declaration(path: Path) -> DatabaseSetDeclaration:
    """Strict-load one JSON or YAML Database Set declaration document."""
    payload = yaml.safe_load(path.read_bytes())
    if not isinstance(payload, Mapping):
        raise TypeError("Database Set declaration document must be a mapping")
    return database_set_declaration_from_mapping(payload)


def database_source_manifest_from_mapping(payload: Mapping[str, object]) -> DatabaseSourceManifest:
    """Strict-load a Database Source Manifest from outer or inner mapping data."""
    inner = _unwrap(payload, "database_source_manifest")
    _require_keys(
        inner,
        {"database_set", "verification", "source_root", "members"},
        "database_source_manifest",
        optional={"schema_version"},
    )
    identity = _identity_from_mapping(_required_mapping(inner, "database_set"))
    verification = _required_str(inner, "verification")
    if verification != "metadata-verified":
        raise ValueError("Database Source Manifest verification must be 'metadata-verified'")
    return DatabaseSourceManifest(
        database_set=identity,
        source_root=_required_str(inner, "source_root"),
        members=tuple(database_source_member_from_mapping(item) for item in _required_list(inner, "members")),
        verification="metadata-verified",
        schema_version=validate_schema_version(inner.get("schema_version"), record_name="DatabaseSourceManifest"),
    )


def database_set_provisioning_evidence_from_mapping(
    payload: Mapping[str, object],
) -> DatabaseSetProvisioningEvidence:
    """Strict-load Database Set Provisioning evidence from mapping data."""
    inner = _unwrap(payload, "database_set_provisioning")
    _require_keys(
        inner,
        {
            "database_set",
            "disposition",
            "verification",
            "manifest_path",
            "manifest_sha256",
            "member_count",
            "payload_bytes_copied",
            "payload_bytes_hashed",
        },
        "database_set_provisioning",
        optional={"schema_version"},
    )
    disposition_value = _required_str(inner, "disposition")
    if disposition_value not in {"published", "reused"}:
        raise ValueError(f"unsupported provisioning disposition: {disposition_value!r}")
    disposition = cast(ProvisioningDisposition, disposition_value)
    verification = _required_str(inner, "verification")
    if verification != "metadata-verified":
        raise ValueError("provisioning verification must be 'metadata-verified'")
    return DatabaseSetProvisioningEvidence(
        database_set=_identity_from_mapping(_required_mapping(inner, "database_set")),
        disposition=disposition,
        verification="metadata-verified",
        manifest_path=_required_str(inner, "manifest_path"),
        manifest_sha256=_required_str(inner, "manifest_sha256"),
        member_count=_required_int(inner, "member_count"),
        payload_bytes_copied=_required_bool(inner, "payload_bytes_copied"),
        payload_bytes_hashed=_required_bool(inner, "payload_bytes_hashed"),
        schema_version=validate_schema_version(
            inner.get("schema_version"), record_name="DatabaseSetProvisioningEvidence"
        ),
    )


def canonical_database_source_manifest_bytes(manifest: DatabaseSourceManifest) -> bytes:
    """Return the exact canonical bytes whose digest identifies a source manifest."""
    return (json.dumps(manifest.to_mapping(), indent=2, sort_keys=True) + "\n").encode()


def database_source_manifest_digest(manifest: DatabaseSourceManifest) -> str:
    """Return the SHA-256 identity of the exact canonical manifest document."""
    return hashlib.sha256(canonical_database_source_manifest_bytes(manifest)).hexdigest()


def database_source_member_from_mapping(value: object) -> DatabaseSourceMember:
    """Strict-load one shared Database Source member mapping."""
    if not isinstance(value, Mapping):
        raise ValueError("manifest members must be mappings")
    _require_keys(
        value,
        {
            "role",
            "database_name",
            "logical_name",
            "source_path",
            "source_kind",
            "resolved_path",
            "resolved_kind",
            "size_bytes",
            "mtime_ns",
            "alias_topology",
            "preexisting_checksum",
        },
        "manifest member",
    )
    role_value = _required_str(value, "role")
    if role_value not in {"primary", "metagenomic"}:
        raise ValueError(f"unsupported database role: {role_value!r}")
    role = cast(DatabaseRole, role_value)
    source_kind_value = _required_str(value, "source_kind")
    if source_kind_value not in {"regular", "symlink"}:
        raise ValueError(f"unsupported source kind: {source_kind_value!r}")
    source_kind = cast(DatabaseMemberKind, source_kind_value)
    resolved_kind = _required_str(value, "resolved_kind")
    if resolved_kind != "regular":
        raise ValueError(f"unsupported resolved kind: {resolved_kind!r}")
    aliases = tuple(database_alias_from_mapping(item) for item in _required_list(value, "alias_topology"))
    return DatabaseSourceMember(
        role=role,
        database_name=_required_str(value, "database_name"),
        logical_name=_required_str(value, "logical_name"),
        source_path=_required_str(value, "source_path"),
        source_kind=source_kind,
        resolved_path=_required_str(value, "resolved_path"),
        resolved_kind="regular",
        size_bytes=_required_int(value, "size_bytes"),
        mtime_ns=_required_int(value, "mtime_ns"),
        alias_topology=aliases,
        preexisting_checksum=_checksum_from_value(value.get("preexisting_checksum")),
    )


def database_alias_from_mapping(value: object) -> DatabaseAlias:
    """Strict-load one shared Database Source alias mapping."""
    if not isinstance(value, Mapping):
        raise ValueError("alias_topology entries must be mappings")
    _require_keys(value, {"path", "target"}, "alias")
    return DatabaseAlias(path=_required_str(value, "path"), target=_required_str(value, "target"))


def _identity_from_mapping(value: Mapping[str, object]) -> DatabaseSetIdentity:
    _require_keys(value, {"identifier", "version"}, "database_set")
    return DatabaseSetIdentity(identifier=_required_str(value, "identifier"), version=_required_str(value, "version"))


def _database_role_from_mapping(value: object) -> DeclaredDatabaseRole:
    if not isinstance(value, Mapping):
        raise ValueError("roles entries must be mappings")
    _require_keys(value, {"role", "database_name", "members"}, "database role")
    role_value = _required_str(value, "role")
    if role_value not in {"primary", "metagenomic"}:
        raise ValueError(f"unsupported database role: {role_value!r}")
    role = cast(DatabaseRole, role_value)
    return DeclaredDatabaseRole(
        role=role,
        database_name=_required_str(value, "database_name"),
        members=tuple(_database_member_from_mapping(item) for item in _required_list(value, "members")),
    )


def _database_member_from_mapping(value: object) -> DeclaredDatabaseMember:
    if not isinstance(value, Mapping):
        raise ValueError("members entries must be mappings")
    _require_keys(value, {"logical_name", "source_path", "preexisting_checksum"}, "database member")
    return DeclaredDatabaseMember(
        logical_name=_required_str(value, "logical_name"),
        source_path=_required_str(value, "source_path"),
        preexisting_checksum=_checksum_from_value(value.get("preexisting_checksum")),
    )


def preexisting_checksum_from_mapping(value: object) -> PreexistingChecksum:
    """Strict-load one declaration-retained checksum mapping."""
    if not isinstance(value, Mapping):
        raise ValueError("preexisting_checksum must be a mapping")
    _require_keys(value, {"algorithm", "value"}, "preexisting_checksum")
    algorithm = _required_str(value, "algorithm")
    return PreexistingChecksum(algorithm=algorithm, value=_required_str(value, "value"))


def _checksum_from_value(value: object) -> PreexistingChecksum | None:
    if value is None:
        return None
    return preexisting_checksum_from_mapping(value)


def _unwrap(payload: Mapping[str, object], wrapper: str) -> Mapping[str, object]:
    if wrapper not in payload:
        return payload
    _require_keys(payload, {wrapper}, f"outer {wrapper}")
    return _required_mapping(payload, wrapper)


def _validate_identifier(value: str, name: str) -> None:
    if _IDENTIFIER.fullmatch(value) is None or value in {".", ".."}:
        raise ValueError(f"{name} must be a safe non-empty identifier")


__all__ = [
    "DatabaseAlias",
    "DatabaseMemberKind",
    "DatabaseRole",
    "DatabaseSetDeclaration",
    "DatabaseSetIdentity",
    "DatabaseSetProvisioningEvidence",
    "DatabaseSourceManifest",
    "DatabaseSourceMember",
    "DeclaredDatabaseMember",
    "DeclaredDatabaseRole",
    "PreexistingChecksum",
    "ProvisioningDisposition",
    "canonical_database_source_manifest_bytes",
    "database_alias_from_mapping",
    "database_set_declaration_from_mapping",
    "database_set_provisioning_evidence_from_mapping",
    "database_source_manifest_digest",
    "database_source_manifest_from_mapping",
    "database_source_member_from_mapping",
    "load_database_set_declaration",
    "preexisting_checksum_from_mapping",
    "validate_database_source_member_topology",
]
