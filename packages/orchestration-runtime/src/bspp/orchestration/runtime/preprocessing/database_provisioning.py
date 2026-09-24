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

"""Explicit cluster-side Database Set Provisioning."""

from __future__ import annotations

import fcntl
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

from bspp.orchestration.contract.database_set_provisioning import (
    DatabaseAlias,
    DatabaseMemberKind,
    DatabaseSetDeclaration,
    DatabaseSetProvisioningEvidence,
    DatabaseSourceManifest,
    DatabaseSourceMember,
    DeclaredDatabaseMember,
    DeclaredDatabaseRole,
    ProvisioningDisposition,
    canonical_database_source_manifest_bytes,
    database_source_manifest_digest,
)

_MANIFEST_NAME = "database-source-manifest.json"


class DatabaseProvisioningError(RuntimeError):
    """Database Set Provisioning failed closed."""


@dataclass(frozen=True)
class _ObservedMember:
    member: DatabaseSourceMember
    stability: tuple[tuple[object, ...], ...]


def provision_database_set(
    declaration: DatabaseSetDeclaration,
    *,
    manifest_root: Path,
    evidence_path: Path,
) -> DatabaseSetProvisioningEvidence:
    """Inventory and atomically publish one immutable Database Source Manifest."""
    try:
        source_root = Path(declaration.source_root).resolve(strict=True)
    except OSError as exc:
        raise DatabaseProvisioningError(f"Database Set source root is unavailable: {declaration.source_root}") from exc
    if not source_root.is_dir():
        raise DatabaseProvisioningError(f"Database Set source root is not a directory: {source_root}")

    resolved_manifest_root = manifest_root.resolve(strict=False)
    version_dir = resolved_manifest_root / declaration.database_set.identifier / declaration.database_set.version
    manifest_path = version_dir / _MANIFEST_NAME
    lock_dir = resolved_manifest_root / ".locks" / declaration.database_set.identifier
    lock_path = lock_dir / f"{declaration.database_set.version}.lock"
    resolved_evidence_path = evidence_path.resolve(strict=False)
    managed_paths = {manifest_path.resolve(strict=False), lock_path.resolve(strict=False)}
    if resolved_evidence_path in managed_paths:
        raise DatabaseProvisioningError(
            f"provisioning evidence path collides with managed Database Set path: {resolved_evidence_path}"
        )

    first_observations = _observe_inventory(declaration, source_root)
    second_observations = _observe_inventory(declaration, source_root)
    for first, second in zip(first_observations, second_observations, strict=True):
        if first.stability != second.stability or first.member != second.member:
            raise DatabaseProvisioningError(
                f"unstable Database Set source observation for logical member {first.member.logical_name!r}"
            )

    role_order = {"primary": 0, "metagenomic": 1}
    members = tuple(
        sorted(
            (observed.member for observed in first_observations),
            key=lambda member: (role_order[member.role], member.logical_name),
        )
    )
    manifest = DatabaseSourceManifest(
        database_set=declaration.database_set,
        source_root=str(source_root),
        members=members,
    )
    manifest_bytes = canonical_database_source_manifest_bytes(manifest)
    manifest_root.mkdir(parents=True, exist_ok=True)
    version_dir.mkdir(parents=True, exist_ok=True)
    lock_dir.mkdir(parents=True, exist_ok=True)

    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        disposition = _publish_or_reuse_manifest(manifest_path, manifest_bytes)

    evidence = DatabaseSetProvisioningEvidence(
        database_set=declaration.database_set,
        disposition=disposition,
        manifest_path=str(manifest_path),
        manifest_sha256=database_source_manifest_digest(manifest),
        member_count=len(manifest.members),
    )
    evidence_bytes = (json.dumps(evidence.to_mapping(), indent=2, sort_keys=True) + "\n").encode()
    _atomic_write(resolved_evidence_path, evidence_bytes, mode=0o444)
    return evidence


def _observe_inventory(
    declaration: DatabaseSetDeclaration,
    source_root: Path,
) -> tuple[_ObservedMember, ...]:
    observations: list[_ObservedMember] = []
    for role in declaration.roles:
        for declared_member in role.members:
            observations.append(_observe_member(source_root, role, declared_member))
    expected_count = sum(len(role.members) for role in declaration.roles)
    if len(observations) != expected_count:
        raise DatabaseProvisioningError("Database Set source inventory is incomplete")
    return tuple(observations)


def _observe_member(
    source_root: Path,
    role: DeclaredDatabaseRole,
    declared_member: DeclaredDatabaseMember,
) -> _ObservedMember:
    declared_path = source_root / declared_member.source_path
    try:
        declared_info = declared_path.lstat()
    except OSError as exc:
        raise DatabaseProvisioningError(
            f"missing source member {declared_member.logical_name!r}: {declared_member.source_path}"
        ) from exc
    if stat.S_ISLNK(declared_info.st_mode):
        source_kind: DatabaseMemberKind = "symlink"
    elif stat.S_ISREG(declared_info.st_mode):
        source_kind = "regular"
    else:
        source_kind = "regular"

    pending = list(Path(declared_member.source_path).parts)
    directory = source_root
    aliases: list[DatabaseAlias] = []
    stability: list[tuple[object, ...]] = []
    visited: set[str] = set()

    while pending:
        current = directory / pending.pop(0)
        relative = _confined_relative(current, source_root, declared_member.logical_name)
        relative_text = relative.as_posix()
        try:
            info = current.lstat()
        except OSError as exc:
            raise DatabaseProvisioningError(
                f"missing source member {declared_member.logical_name!r}: {relative_text}"
            ) from exc
        if stat.S_ISLNK(info.st_mode):
            if relative_text in visited:
                raise DatabaseProvisioningError(
                    f"cyclic source alias for logical member {declared_member.logical_name!r}: {relative_text}"
                )
            visited.add(relative_text)
            try:
                target = os.readlink(current)
            except OSError as exc:
                raise DatabaseProvisioningError(
                    f"cannot read source alias for logical member {declared_member.logical_name!r}: {relative_text}"
                ) from exc
            aliases.append(DatabaseAlias(path=relative_text, target=target))
            stability.append(_stat_identity(relative_text, info, target))
            current = Path(target) if Path(target).is_absolute() else current.parent / target
            current = Path(os.path.normpath(current))
            target_relative = _confined_relative(current, source_root, declared_member.logical_name)
            pending = [*target_relative.parts, *pending]
            directory = source_root
            continue
        if pending:
            if not stat.S_ISDIR(info.st_mode):
                raise DatabaseProvisioningError(
                    f"unsupported source path component for logical member {declared_member.logical_name!r}: "
                    f"{relative_text}"
                )
            stability.append(_stat_identity(relative_text, info, None))
            directory = current
            continue
        if not stat.S_ISREG(info.st_mode):
            raise DatabaseProvisioningError(
                f"unsupported source kind for logical member {declared_member.logical_name!r}: {relative_text}"
            )
        stability.append(_stat_identity(relative_text, info, None))
        member = DatabaseSourceMember(
            role=role.role,
            database_name=role.database_name,
            logical_name=declared_member.logical_name,
            source_path=declared_member.source_path,
            source_kind=source_kind,
            resolved_path=relative_text,
            resolved_kind="regular",
            size_bytes=info.st_size,
            mtime_ns=info.st_mtime_ns,
            alias_topology=tuple(aliases),
            preexisting_checksum=declared_member.preexisting_checksum,
        )
        return _ObservedMember(member=member, stability=tuple(stability))
    raise DatabaseProvisioningError(
        f"source inventory is incomplete for logical member {declared_member.logical_name!r}"
    )


def _confined_relative(path: Path, source_root: Path, logical_name: str) -> Path:
    normalized = Path(os.path.abspath(path))
    try:
        return normalized.relative_to(source_root)
    except ValueError as exc:
        raise DatabaseProvisioningError(
            f"source alias escapes declared root for logical member {logical_name!r}: {normalized}"
        ) from exc


def _stat_identity(relative: str, info: os.stat_result, target: str | None) -> tuple[object, ...]:
    return (
        relative,
        stat.S_IFMT(info.st_mode),
        stat.S_IMODE(info.st_mode),
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        target,
    )


def _publish_or_reuse_manifest(path: Path, content: bytes) -> ProvisioningDisposition:
    try:
        info = path.lstat()
    except FileNotFoundError:
        _atomic_write(path, content, mode=0o444)
        return "published"
    if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o222:
        raise DatabaseProvisioningError(f"existing Database Source Manifest is not immutable: {path}")
    if path.read_bytes() != content:
        raise DatabaseProvisioningError(
            f"existing Database Source Manifest identity does not match the declared Database Set version: {path}"
        )
    return "reused"


def _atomic_write(path: Path, content: bytes, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fchmod(handle.fileno(), mode)
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


__all__ = ["DatabaseProvisioningError", "provision_database_set"]
