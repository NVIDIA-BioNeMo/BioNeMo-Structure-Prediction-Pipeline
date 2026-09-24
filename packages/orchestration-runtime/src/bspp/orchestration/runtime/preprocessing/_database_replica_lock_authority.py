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

"""Descriptor-relative Database Replica lock authority operations."""

from __future__ import annotations

import fcntl
import os
import stat
from pathlib import Path

from ._database_placement_errors import DatabasePlacementError
from ._database_replica_errors import ClassifiedDatabaseReplicaError
from ._database_replica_lock_types import (
    CacheExclusiveOwnership,
    IdentityExclusiveOwnership,
    LockContendedError,
    MaintenanceCacheExclusiveOwnership,
    MaintenanceIdentityExclusiveOwnership,
)
from ._filesystem_authority import FilesystemAuthority, _real_directory_stat


def acquire_or_create_exclusive_population_lock(parent_descriptor: int, name: str) -> int:
    """Acquire a population lock without waiting (legacy low-level seam)."""
    return _acquire_lock(
        parent_descriptor,
        name,
        flags=os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
        operation=fcntl.LOCK_EX | fcntl.LOCK_NB,
        description="Database Replica population lock",
    )


def acquire_existing_shared_identity_lock(parent_descriptor: int, name: str) -> int:
    """Acquire an existing warm identity lock without create or write authority."""
    return _acquire_lock(
        parent_descriptor,
        name,
        flags=os.O_RDONLY | os.O_NOFOLLOW,
        operation=fcntl.LOCK_SH | fcntl.LOCK_NB,
        description="Database Replica shared identity lock",
    )


def verify_identity_ownership(cache_descriptor: int, ownership: IdentityExclusiveOwnership) -> None:
    """Rebind every identity-ownership descriptor to its exact visible basename."""
    if cache_descriptor != ownership.cache_descriptor:
        raise ClassifiedDatabaseReplicaError("lock-unavailable", "Database Replica cache ownership changed")
    _verify_cache_root_authority(
        ownership.cache_path,
        cache_descriptor,
        ownership.cache_identity,
    )
    if (
        _relative_authority(
            cache_descriptor,
            ".locks",
            kind="directory",
            expected_mode=0o700,
        )
        != ownership.locks_identity
    ):
        raise ClassifiedDatabaseReplicaError("lock-unavailable", "Database Replica lock directory changed")
    if (
        _descriptor_authority(
            ownership.locks_descriptor,
            kind="directory",
            expected_mode=0o700,
            description="Database Replica held lock directory",
        )
        != ownership.locks_identity
    ):
        raise ClassifiedDatabaseReplicaError("lock-unavailable", "Database Replica held lock directory changed")
    name = f"{ownership.source_manifest_sha256}.lock"
    if (
        _relative_authority(
            ownership.locks_descriptor,
            name,
            kind="file",
            expected_mode=0o600,
        )
        != ownership.lock_identity
        or _descriptor_authority(
            ownership.descriptor,
            kind="file",
            expected_mode=0o600,
            description="Database Replica held identity lock",
        )
        != ownership.lock_identity
    ):
        raise ClassifiedDatabaseReplicaError("lock-unavailable", "Database Replica identity lock changed")


def verify_cache_ownership(cache_descriptor: int, ownership: CacheExclusiveOwnership) -> None:
    """Rebind cache ownership and its prerequisite identity ownership."""
    verify_identity_ownership(cache_descriptor, ownership.identity)
    if (
        _relative_authority(
            ownership.identity.locks_descriptor,
            "cache.lock",
            kind="file",
            expected_mode=0o600,
        )
        != ownership.lock_identity
        or _descriptor_authority(
            ownership.descriptor,
            kind="file",
            expected_mode=0o600,
            description="Database Replica held cache lock",
        )
        != ownership.lock_identity
    ):
        raise ClassifiedDatabaseReplicaError("lock-unavailable", "Database Replica cache lock changed")


def verify_maintenance_cache_ownership(
    cache_descriptor: int,
    ownership: MaintenanceCacheExclusiveOwnership,
) -> None:
    """Rebind maintenance cache EX without asserting an identity prerequisite."""
    if cache_descriptor != ownership.cache_descriptor:
        raise ClassifiedDatabaseReplicaError("lock-unavailable", "Database Replica maintenance cache changed")
    _verify_cache_root_authority(ownership.cache_path, cache_descriptor, ownership.cache_identity)
    if (
        _relative_authority(cache_descriptor, ".locks", kind="directory", expected_mode=0o700)
        != ownership.locks_identity
        or _descriptor_authority(
            ownership.locks_descriptor,
            kind="directory",
            expected_mode=0o700,
            description="Database Replica maintenance held lock directory",
        )
        != ownership.locks_identity
        or _relative_authority(ownership.locks_descriptor, "cache.lock", kind="file", expected_mode=0o600)
        != ownership.lock_identity
        or _descriptor_authority(
            ownership.descriptor,
            kind="file",
            expected_mode=0o600,
            description="Database Replica maintenance held cache lock",
        )
        != ownership.lock_identity
    ):
        raise ClassifiedDatabaseReplicaError("lock-unavailable", "Database Replica maintenance cache lock changed")


def verify_maintenance_identity_ownership(
    cache_descriptor: int,
    ownership: MaintenanceIdentityExclusiveOwnership,
) -> None:
    """Rebind every nonblocking identity lock held by maintenance."""
    verify_maintenance_cache_ownership(cache_descriptor, ownership.cache)
    if not (len(ownership.identities) == len(ownership.descriptors) == len(ownership.lock_identities)):
        raise ClassifiedDatabaseReplicaError("lock-unavailable", "Database Replica maintenance lock set changed")
    for identity, descriptor, expected in zip(
        ownership.identities,
        ownership.descriptors,
        ownership.lock_identities,
        strict=True,
    ):
        if (
            _relative_authority(
                ownership.cache.locks_descriptor,
                f"{identity}.lock",
                kind="file",
                expected_mode=0o600,
            )
            != expected
            or _descriptor_authority(
                descriptor,
                kind="file",
                expected_mode=0o600,
                description="Database Replica maintenance held identity lock",
            )
            != expected
        ):
            raise ClassifiedDatabaseReplicaError(
                "lock-unavailable", "Database Replica maintenance identity lock changed"
            )


def verify_existing_shared_identity_lock(locks_descriptor: int, name: str, descriptor: int) -> None:
    """Rebind a shared identity descriptor to the unchanged visible protocol file."""
    if _relative_authority(
        locks_descriptor,
        name,
        kind="file",
        expected_mode=0o600,
    ) != _descriptor_authority(
        descriptor,
        kind="file",
        expected_mode=0o600,
        description="Database Replica held shared identity lock",
    ):
        raise ClassifiedDatabaseReplicaError("lock-unavailable", "Database Replica shared identity lock changed")


def verify_shared_identity_ownership(
    cache_path: Path,
    cache_descriptor: int,
    cache_identity: FilesystemAuthority,
    locks_descriptor: int,
    name: str,
    descriptor: int,
) -> None:
    """Bind a shared lease, its parent, and the exact held cache-root descriptor."""
    _verify_cache_root_authority(cache_path, cache_descriptor, cache_identity)
    if _relative_authority(
        cache_descriptor,
        ".locks",
        kind="directory",
        expected_mode=0o700,
    ) != _descriptor_authority(
        locks_descriptor,
        kind="directory",
        expected_mode=0o700,
        description="Database Replica held shared lock directory",
    ):
        raise ClassifiedDatabaseReplicaError("lock-unavailable", "Database Replica shared lock directory changed")
    verify_existing_shared_identity_lock(locks_descriptor, name, descriptor)


def _relative_authority(
    parent_descriptor: int,
    name: str,
    *,
    kind: str,
    expected_mode: int,
) -> FilesystemAuthority:
    try:
        info = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except OSError as exc:
        raise ClassifiedDatabaseReplicaError(
            "lock-unavailable",
            f"Database Replica lock authority changed: {name!r}",
        ) from exc
    return _validated_authority(
        info,
        kind=kind,
        expected_mode=expected_mode,
        description=f"Database Replica lock authority {name!r}",
    )


def _descriptor_authority(
    descriptor: int,
    *,
    kind: str,
    expected_mode: int,
    description: str,
) -> FilesystemAuthority:
    try:
        info = os.fstat(descriptor)
    except OSError as exc:
        raise ClassifiedDatabaseReplicaError("lock-unavailable", "Database Replica lock descriptor changed") from exc
    return _validated_authority(
        info,
        kind=kind,
        expected_mode=expected_mode,
        description=description,
    )


def _validated_authority(
    info: os.stat_result,
    *,
    kind: str,
    expected_mode: int,
    description: str,
) -> FilesystemAuthority:
    expected_type = stat.S_IFDIR if kind == "directory" else stat.S_IFREG
    if (
        stat.S_IFMT(info.st_mode) != expected_type
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) != expected_mode
    ):
        raise ClassifiedDatabaseReplicaError(
            "lock-unavailable",
            f"{description} is invalid",
        )
    return FilesystemAuthority(
        file_type=expected_type,
        device=info.st_dev,
        inode=info.st_ino,
        owner_uid=info.st_uid,
        permissions=stat.S_IMODE(info.st_mode),
    )


def _verify_cache_root_authority(
    path: Path,
    descriptor: int,
    expected: FilesystemAuthority,
) -> None:
    try:
        visible = _real_directory_stat(path, description="protected Database Replica cache root")
        held = os.fstat(descriptor)
    except (DatabasePlacementError, OSError) as exc:
        raise ClassifiedDatabaseReplicaError(
            "lock-unavailable",
            "Database Replica cache-root authority changed",
        ) from exc
    expected_mode = expected.permissions
    if (
        _validated_authority(
            visible,
            kind="directory",
            expected_mode=expected_mode,
            description="Database Replica visible cache-root authority",
        )
        != expected
        or _validated_authority(
            held,
            kind="directory",
            expected_mode=expected_mode,
            description="Database Replica held cache-root authority",
        )
        != expected
    ):
        raise ClassifiedDatabaseReplicaError(
            "lock-unavailable",
            "Database Replica cache-root authority changed",
        )


def _acquire_lock(
    parent_descriptor: int,
    name: str,
    *,
    flags: int,
    operation: int,
    description: str,
) -> int:
    descriptor: int | None = None
    acquired = False
    try:
        descriptor = _open_validated_lock(parent_descriptor, name, flags=flags)
        fcntl.flock(descriptor, operation)
        if _relative_authority(
            parent_descriptor,
            name,
            kind="file",
            expected_mode=0o600,
        ) != _descriptor_authority(
            descriptor,
            kind="file",
            expected_mode=0o600,
            description=f"Database Replica held lock authority {name!r}",
        ):
            raise ClassifiedDatabaseReplicaError("lock-unavailable", f"Database Replica lock changed: {name!r}")
        acquired = True
        return descriptor
    except BlockingIOError as exc:
        raise LockContendedError(
            "lock-unavailable",
            f"{description} is already held",
        ) from exc
    except ClassifiedDatabaseReplicaError:
        raise
    except OSError as exc:
        raise ClassifiedDatabaseReplicaError(
            "lock-unavailable",
            f"cannot acquire {description}: {name!r}",
        ) from exc
    finally:
        if descriptor is not None and not acquired:
            os.close(descriptor)


def _open_validated_lock(parent_descriptor: int, name: str, *, flags: int) -> int:
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=parent_descriptor)
    except OSError as exc:
        raise ClassifiedDatabaseReplicaError(
            "lock-unavailable",
            f"Database Replica lock is unavailable: {name!r}",
        ) from exc
    try:
        info = _lock_descriptor_stat(descriptor)
        parent_device = os.fstat(parent_descriptor).st_dev
    except OSError as exc:
        os.close(descriptor)
        raise ClassifiedDatabaseReplicaError(
            "lock-unavailable",
            f"Database Replica lock authority is unavailable: {name!r}",
        ) from exc
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.geteuid()
        or info.st_dev != parent_device
        or stat.S_IMODE(info.st_mode) != 0o600
    ):
        os.close(descriptor)
        raise ClassifiedDatabaseReplicaError(
            "lock-unavailable",
            f"Database Replica lock authority is invalid: {name!r}",
        )
    return descriptor


def _lock_descriptor_stat(descriptor: int) -> os.stat_result:
    """Isolated OS boundary for validating a newly opened lock file."""
    return os.fstat(descriptor)


def _open_private_locks_directory(cache_descriptor: int) -> int:
    try:
        os.mkdir(".locks", 0o700, dir_fd=cache_descriptor)
    except FileExistsError:
        pass
    except OSError as exc:
        raise ClassifiedDatabaseReplicaError(
            "lock-unavailable", "cannot create Database Replica lock directory"
        ) from exc
    descriptor: int | None = None
    try:
        info = os.stat(".locks", dir_fd=cache_descriptor, follow_symlinks=False)
        descriptor = os.open(".locks", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=cache_descriptor)
        visible_authority = _validated_authority(
            info,
            kind="directory",
            expected_mode=0o700,
            description="Database Replica visible lock directory authority",
        )
        opened_authority = _descriptor_authority(
            descriptor,
            kind="directory",
            expected_mode=0o700,
            description="Database Replica held lock directory authority",
        )
        if visible_authority != opened_authority or info.st_dev != os.fstat(cache_descriptor).st_dev:
            raise ClassifiedDatabaseReplicaError(
                "lock-unavailable",
                "Database Replica lock directory authority is invalid",
            )
    except ClassifiedDatabaseReplicaError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise ClassifiedDatabaseReplicaError(
            "lock-unavailable", "Database Replica lock directory is unavailable"
        ) from exc
    assert descriptor is not None
    return descriptor


def _open_existing_private_locks_directory(cache_descriptor: int) -> int:
    """Open maintenance lock authority without manufacturing protocol state."""
    descriptor: int | None = None
    try:
        info = os.stat(".locks", dir_fd=cache_descriptor, follow_symlinks=False)
        descriptor = os.open(".locks", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=cache_descriptor)
        visible = _validated_authority(
            info,
            kind="directory",
            expected_mode=0o700,
            description="Database Replica maintenance visible lock directory",
        )
        opened = _descriptor_authority(
            descriptor,
            kind="directory",
            expected_mode=0o700,
            description="Database Replica maintenance held lock directory",
        )
        if visible != opened or info.st_dev != os.fstat(cache_descriptor).st_dev:
            raise ClassifiedDatabaseReplicaError(
                "lock-unavailable", "Database Replica maintenance lock directory authority is invalid"
            )
        return descriptor
    except ClassifiedDatabaseReplicaError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise ClassifiedDatabaseReplicaError(
            "lock-unavailable", "Database Replica maintenance lock directory is unavailable"
        ) from exc


__all__ = [
    "acquire_existing_shared_identity_lock",
    "acquire_or_create_exclusive_population_lock",
    "verify_cache_ownership",
    "verify_existing_shared_identity_lock",
    "verify_identity_ownership",
    "verify_maintenance_cache_ownership",
    "verify_maintenance_identity_ownership",
    "verify_shared_identity_ownership",
]
