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

"""Value types and narrow capabilities for cache maintenance."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from bspp.orchestration.contract.database_cache_maintenance import (
    DatabaseAcceptanceCacheProfile,
    DatabaseCacheMaintenanceEvidence,
    DatabaseCacheRemovedEntry,
)

from ._filesystem_authority import (
    DirectoryAuthority,
    FilesystemAuthority,
    FilesystemObjectIdentity,
)
from ._linux_mount_authority import (
    FrozenDirectoryMountAuthority,
    FrozenRegularFileMountAuthority,
)
from ._owned_tree import OwnedTreeRemover


class DatabaseCacheMaintenanceError(RuntimeError):
    """The acceptance-only cache operation refused or failed closed."""


@dataclass(frozen=True)
class DatabaseCacheMaintenancePaths:
    """Physical site inputs for the private maintenance coordinator."""

    mountinfo: Path


@dataclass(frozen=True)
class _Scope:
    profile: DatabaseAcceptanceCacheProfile
    effective_user: str
    effective_uid: int
    cache_root: Path
    user_namespace: Path
    replicas_path: Path
    cache_root_descriptor: int
    users_descriptor: int
    user_descriptor: int
    cache_root_identity: DirectoryAuthority
    users_identity: DirectoryAuthority
    user_identity: DirectoryAuthority


@dataclass(frozen=True)
class _Entry:
    kind: str
    identity: str
    basename: str
    authority: FilesystemAuthority

    def evidence(self) -> DatabaseCacheRemovedEntry:
        return DatabaseCacheRemovedEntry(
            kind=self.kind,  # type: ignore[arg-type]
            source_manifest_sha256=self.identity,
            basename=self.basename,
            device=self.authority.device,
            inode=self.authority.inode,
        )


@dataclass(frozen=True)
class _DatabaseAcceptanceCacheAuthority:
    profile: DatabaseAcceptanceCacheProfile
    configured_sibling_cache_roots: tuple[Path, ...]


@dataclass(frozen=True)
class _PhysicalCacheRootBinding:
    path: Path
    descriptor: int | None
    identity: DirectoryAuthority | None
    mount_authority: FrozenDirectoryMountAuthority | None
    configured_path_uses_symlink: bool


@dataclass(frozen=True)
class _SelectedDeletionScopeAuthority:
    cache_root: FrozenDirectoryMountAuthority
    users_root: FrozenDirectoryMountAuthority
    user_namespace: FrozenDirectoryMountAuthority
    locks: FrozenDirectoryMountAuthority | None
    replicas: FrozenDirectoryMountAuthority | None

    def core_chain(self) -> tuple[FrozenDirectoryMountAuthority, ...]:
        return (self.cache_root, self.users_root, self.user_namespace)

    def managed(self) -> tuple[FrozenDirectoryMountAuthority, ...]:
        optional = tuple(item for item in (self.locks, self.replicas) if item is not None)
        return (*self.core_chain(), *optional)


@dataclass(frozen=True)
class _DeletionTargetAuthority:
    selected: _SelectedDeletionScopeAuthority
    sibling_roots: tuple[_PhysicalCacheRootBinding, ...]
    mountinfo_path: Path


@dataclass(frozen=True)
class _MaintenanceLockMountAuthority:
    cache_lock: FrozenRegularFileMountAuthority
    identity_locks: tuple[tuple[str, FrozenRegularFileMountAuthority], ...] = ()


@dataclass(frozen=True)
class _EvidencePublicationAuthority:
    parent_identity: DirectoryAuthority
    mount_authority: FrozenDirectoryMountAuthority


@dataclass(frozen=True)
class _CreatedEvidenceDirectory:
    parent_descriptor: int
    basename: str
    descriptor: int
    identity: DirectoryAuthority


@dataclass
class _EvidenceDestination:
    path: Path
    parent_descriptor: int
    parent_identity: DirectoryAuthority
    descriptor: int | None
    destination_identity: FilesystemObjectIdentity | None
    existing: DatabaseCacheMaintenanceEvidence | None
    created_directories: tuple[_CreatedEvidenceDirectory, ...] = ()
    finalization_attempted: bool = False
    published: bool = False
    publication_authority: _EvidencePublicationAuthority | None = None


@dataclass(frozen=True)
class DatabaseCacheEvidenceStore:
    """Cohesive evidence lifecycle used by the private coordinator."""

    reserve: Callable[[Path], _EvidenceDestination]
    authorize: Callable[[_EvidenceDestination, _DeletionTargetAuthority], None]
    verify: Callable[..., None]
    finish: Callable[[_EvidenceDestination | None, DatabaseCacheMaintenanceEvidence], None]
    close: Callable[[_EvidenceDestination], None]


__all__ = [
    "DatabaseCacheEvidenceStore",
    "DatabaseCacheMaintenanceError",
    "DatabaseCacheMaintenancePaths",
    "OwnedTreeRemover",
    "_CreatedEvidenceDirectory",
    "_DatabaseAcceptanceCacheAuthority",
    "_DeletionTargetAuthority",
    "_Entry",
    "_EvidenceDestination",
    "_EvidencePublicationAuthority",
    "_MaintenanceLockMountAuthority",
    "_PhysicalCacheRootBinding",
    "_Scope",
    "_SelectedDeletionScopeAuthority",
]
