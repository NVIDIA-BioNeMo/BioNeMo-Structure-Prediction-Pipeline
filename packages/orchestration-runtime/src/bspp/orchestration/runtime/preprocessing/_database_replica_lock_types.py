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

"""Database Replica lock value types and classified contention errors."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ._database_replica_errors import ClassifiedDatabaseReplicaError
from ._filesystem_authority import FilesystemAuthority

_POLL_QUANTUM_SECONDS = 0.05


class LockContendedError(ClassifiedDatabaseReplicaError):
    """Private signal for an otherwise-valid descriptor whose flock contended."""


class MaintenanceIdentityContendedError(LockContendedError):
    """One maintenance identity lock contended after earlier locks were held."""

    def __init__(self, source_manifest_sha256: str, acquired_identities: tuple[str, ...]) -> None:
        super().__init__("lock-unavailable", f"Database Replica identity lock is active: {source_manifest_sha256}")
        self.source_manifest_sha256 = source_manifest_sha256
        self.acquired_identities = acquired_identities


@dataclass(frozen=True)
class LockWait:
    """One absolute monotonic deadline shared by every lock in an attempt."""

    deadline: float
    monotonic: Callable[[], float]
    sleeper: Callable[[float], None]
    poll_quantum_seconds: float = _POLL_QUANTUM_SECONDS

    @classmethod
    def for_timeout(
        cls,
        timeout_seconds: int,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        poll_quantum_seconds: float = _POLL_QUANTUM_SECONDS,
    ) -> LockWait:
        if timeout_seconds <= 0:
            raise ValueError("Database Replica lock timeout must be positive")
        if poll_quantum_seconds <= 0:
            raise ValueError("Database Replica lock poll quantum must be positive")
        return cls(
            deadline=monotonic() + timeout_seconds,
            monotonic=monotonic,
            sleeper=sleeper,
            poll_quantum_seconds=poll_quantum_seconds,
        )


@dataclass(frozen=True)
class IdentityExclusiveOwnership:
    """Live exclusive ownership of one manifest identity."""

    cache_descriptor: int
    cache_path: Path
    cache_identity: FilesystemAuthority
    locks_descriptor: int
    locks_identity: FilesystemAuthority
    descriptor: int
    lock_identity: FilesystemAuthority
    source_manifest_sha256: str
    contended: bool


@dataclass(frozen=True)
class CacheExclusiveOwnership:
    """Live cache-wide ownership acquired after a specific identity."""

    identity: IdentityExclusiveOwnership
    descriptor: int
    lock_identity: FilesystemAuthority
    contended: bool


@dataclass(frozen=True)
class MaintenanceCacheExclusiveOwnership:
    """Live cache-wide ownership acquired without an identity prerequisite."""

    cache_descriptor: int
    cache_path: Path
    cache_identity: FilesystemAuthority
    locks_descriptor: int
    locks_identity: FilesystemAuthority
    descriptor: int
    lock_identity: FilesystemAuthority
    contended: bool


@dataclass(frozen=True)
class MaintenanceIdentityExclusiveOwnership:
    """All relevant identity locks held nonblocking beneath maintenance cache EX."""

    cache: MaintenanceCacheExclusiveOwnership
    identities: tuple[str, ...]
    descriptors: tuple[int, ...]
    lock_identities: tuple[FilesystemAuthority, ...]


__all__ = [
    "CacheExclusiveOwnership",
    "FilesystemAuthority",
    "IdentityExclusiveOwnership",
    "LockContendedError",
    "LockWait",
    "MaintenanceCacheExclusiveOwnership",
    "MaintenanceIdentityContendedError",
    "MaintenanceIdentityExclusiveOwnership",
]
