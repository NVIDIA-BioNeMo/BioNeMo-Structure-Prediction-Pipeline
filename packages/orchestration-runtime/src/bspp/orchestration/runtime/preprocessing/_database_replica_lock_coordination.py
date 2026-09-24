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

"""Bounded ordinary and maintenance Database Replica lock coordination."""

from __future__ import annotations

import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from ._database_replica_errors import ClassifiedDatabaseReplicaError
from ._database_replica_lock_authority import (
    _acquire_lock,
    _descriptor_authority,
    _open_existing_private_locks_directory,
    _open_private_locks_directory,
    _open_validated_lock,
    verify_cache_ownership,
    verify_identity_ownership,
    verify_maintenance_cache_ownership,
    verify_maintenance_identity_ownership,
)
from ._database_replica_lock_types import (
    CacheExclusiveOwnership,
    IdentityExclusiveOwnership,
    LockContendedError,
    LockWait,
    MaintenanceCacheExclusiveOwnership,
    MaintenanceIdentityContendedError,
    MaintenanceIdentityExclusiveOwnership,
)
from ._filesystem_authority import FilesystemAuthority


@contextmanager
def hold_exclusive_identity(
    cache_descriptor: int,
    *,
    cache_path: Path,
    cache_identity: FilesystemAuthority,
    source_manifest_sha256: str,
    wait: LockWait,
) -> Iterator[IdentityExclusiveOwnership]:
    """Wait within the attempt deadline for identity-exclusive ownership."""
    locks_descriptor = _open_private_locks_directory(cache_descriptor)
    descriptor: int | None = None
    try:
        descriptor, contended = _acquire_lock_until(
            locks_descriptor,
            f"{source_manifest_sha256}.lock",
            wait=wait,
            description="Database Replica identity-exclusive lock",
        )
        ownership = IdentityExclusiveOwnership(
            cache_descriptor=cache_descriptor,
            cache_path=cache_path,
            cache_identity=cache_identity,
            locks_descriptor=locks_descriptor,
            locks_identity=_descriptor_authority(
                locks_descriptor,
                kind="directory",
                expected_mode=0o700,
                description="Database Replica held lock directory",
            ),
            descriptor=descriptor,
            lock_identity=_descriptor_authority(
                descriptor,
                kind="file",
                expected_mode=0o600,
                description="Database Replica held identity lock",
            ),
            source_manifest_sha256=source_manifest_sha256,
            contended=contended,
        )
        verify_identity_ownership(cache_descriptor, ownership)
        yield ownership
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(locks_descriptor)


@contextmanager
def hold_exclusive_cache(
    identity: IdentityExclusiveOwnership,
    *,
    wait: LockWait,
) -> Iterator[CacheExclusiveOwnership]:
    """Wait for cache ownership after, and only while, identity ownership is live."""
    try:
        verify_identity_ownership(identity.cache_descriptor, identity)
    except OSError as exc:
        raise ClassifiedDatabaseReplicaError(
            "lock-unavailable",
            "Database Replica identity ownership ended before cache ownership",
        ) from exc
    descriptor, contended = _acquire_lock_until(
        identity.locks_descriptor,
        "cache.lock",
        wait=wait,
        description="Database Replica cache-exclusive lock",
    )
    try:
        ownership = CacheExclusiveOwnership(
            identity=identity,
            descriptor=descriptor,
            lock_identity=_descriptor_authority(
                descriptor,
                kind="file",
                expected_mode=0o600,
                description="Database Replica held cache lock",
            ),
            contended=contended,
        )
        verify_cache_ownership(identity.cache_descriptor, ownership)
        yield ownership
    finally:
        os.close(descriptor)


@contextmanager
def hold_exclusive_cache_for_maintenance(
    cache_descriptor: int,
    *,
    cache_path: Path,
    cache_identity: FilesystemAuthority,
    wait: LockWait,
) -> Iterator[MaintenanceCacheExclusiveOwnership]:
    """Acquire cache EX first for the explicit acceptance maintenance protocol."""
    locks_descriptor = _open_existing_private_locks_directory(cache_descriptor)
    descriptor: int | None = None
    try:
        descriptor, contended = _acquire_lock_until(
            locks_descriptor,
            "cache.lock",
            wait=wait,
            description="Database Replica maintenance cache-exclusive lock",
            flags=os.O_RDWR | os.O_NOFOLLOW,
        )
        ownership = MaintenanceCacheExclusiveOwnership(
            cache_descriptor=cache_descriptor,
            cache_path=cache_path,
            cache_identity=cache_identity,
            locks_descriptor=locks_descriptor,
            locks_identity=_descriptor_authority(
                locks_descriptor,
                kind="directory",
                expected_mode=0o700,
                description="Database Replica maintenance lock directory",
            ),
            descriptor=descriptor,
            lock_identity=_descriptor_authority(
                descriptor,
                kind="file",
                expected_mode=0o600,
                description="Database Replica maintenance cache lock",
            ),
            contended=contended,
        )
        verify_maintenance_cache_ownership(cache_descriptor, ownership)
        yield ownership
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(locks_descriptor)


@contextmanager
def hold_exclusive_identities_for_maintenance(
    cache_descriptor: int,
    ownership: MaintenanceCacheExclusiveOwnership,
    identities: tuple[str, ...],
) -> Iterator[MaintenanceIdentityExclusiveOwnership]:
    """Acquire every sorted relevant identity EX once, without ever waiting."""
    if identities != tuple(sorted(set(identities))):
        raise ValueError("maintenance identity locks must be sorted and unique")
    verify_maintenance_cache_ownership(cache_descriptor, ownership)
    descriptors: list[int] = []
    acquired: list[str] = []
    lock_identities: list[FilesystemAuthority] = []
    try:
        for identity in identities:
            try:
                descriptor = _acquire_lock(
                    ownership.locks_descriptor,
                    f"{identity}.lock",
                    flags=os.O_RDWR | os.O_NOFOLLOW,
                    operation=fcntl.LOCK_EX | fcntl.LOCK_NB,
                    description="Database Replica maintenance identity-exclusive lock",
                )
            except LockContendedError as exc:
                raise MaintenanceIdentityContendedError(identity, tuple(acquired)) from exc
            descriptors.append(descriptor)
            acquired.append(identity)
            lock_identities.append(
                _descriptor_authority(
                    descriptor,
                    kind="file",
                    expected_mode=0o600,
                    description="Database Replica maintenance identity lock",
                )
            )
        held = MaintenanceIdentityExclusiveOwnership(
            cache=ownership,
            identities=tuple(acquired),
            descriptors=tuple(descriptors),
            lock_identities=tuple(lock_identities),
        )
        verify_maintenance_identity_ownership(cache_descriptor, held)
        yield held
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _acquire_lock_until(
    parent_descriptor: int,
    name: str,
    *,
    wait: LockWait,
    description: str,
    flags: int = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
) -> tuple[int, bool]:
    descriptor: int | None = None
    acquired = False
    contended = False
    try:
        descriptor = _open_validated_lock(parent_descriptor, name, flags=flags)
        while True:
            if contended and wait.monotonic() >= wait.deadline:
                raise ClassifiedDatabaseReplicaError(
                    "lock-unavailable",
                    f"timed out waiting for {description}",
                )
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                return descriptor, contended
            except BlockingIOError:
                contended = True
                remaining = wait.deadline - wait.monotonic()
                if remaining <= 0:
                    raise ClassifiedDatabaseReplicaError(
                        "lock-unavailable",
                        f"timed out waiting for {description}",
                    ) from None
                wait.sleeper(min(wait.poll_quantum_seconds, remaining))
            except OSError as exc:
                raise ClassifiedDatabaseReplicaError(
                    "lock-unavailable",
                    f"cannot acquire {description}: {name!r}",
                ) from exc
    finally:
        if descriptor is not None and not acquired:
            os.close(descriptor)


__all__ = [
    "hold_exclusive_cache",
    "hold_exclusive_cache_for_maintenance",
    "hold_exclusive_identities_for_maintenance",
    "hold_exclusive_identity",
]
