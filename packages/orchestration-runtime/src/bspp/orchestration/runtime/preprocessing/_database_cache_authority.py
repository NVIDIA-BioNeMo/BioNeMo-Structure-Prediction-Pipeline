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

"""Cache-maintenance deletion, mount, containment, and lock authority."""

from __future__ import annotations

import os
from pathlib import Path

from ._database_cache_evidence import (
    _verify_evidence_destination,
    _verify_evidence_parent,
)
from ._database_cache_maintenance_types import (
    DatabaseCacheMaintenanceError,
    _DeletionTargetAuthority,
    _EvidenceDestination,
    _EvidencePublicationAuthority,
    _MaintenanceLockMountAuthority,
    _PhysicalCacheRootBinding,
    _Scope,
    _SelectedDeletionScopeAuthority,
)
from ._database_cache_scope import (
    _configured_path_uses_symlink,
    _open_physical_cache_root,
)
from ._filesystem_authority import (
    FilesystemObjectIdentity,
    filesystem_object_identity,
)
from ._linux_mount_authority import (
    FrozenDirectoryMountAuthority,
    FrozenRegularFileMountAuthority,
    LinuxMountAuthorityError,
    LinuxMountContainmentError,
    directory_mount_backings_overlap,
    freeze_directory_mount_authority,
    freeze_evidence_mount_authority,
    freeze_regular_file_mount_authority,
    require_exact_mount_child,
    verify_frozen_directory_mount_authority,
    verify_frozen_regular_file_mount_authority,
)


def _authorize_evidence_publication(
    destination: _EvidenceDestination,
    target: _DeletionTargetAuthority,
) -> None:
    """Enable terminal publication only after complete physical separation proof."""
    selected_managed = target.selected.managed()
    forbidden = {
        FilesystemObjectIdentity(device=item.identity.device, inode=item.identity.inode) for item in selected_managed
    }
    forbidden.update(
        FilesystemObjectIdentity(
            device=binding.mount_authority.identity.device,
            inode=binding.mount_authority.identity.inode,
        )
        for binding in target.sibling_roots
        if binding.mount_authority is not None
    )
    try:
        _verify_evidence_parent(destination.path.parent, destination.parent_descriptor, destination.parent_identity)
        if destination.descriptor is not None:
            _verify_evidence_destination(destination, require_empty=True)
        _verify_deletion_target_bindings(target, require_separation=False)
        if _descriptor_ancestry_contains(destination.parent_descriptor, forbidden):
            raise DatabaseCacheMaintenanceError(
                "maintenance evidence destination is physically within the managed cache"
            )
        managed_directories = tuple((item.role, item.path, item.descriptor) for item in selected_managed)
        frozen = freeze_evidence_mount_authority(
            evidence_parent_path=destination.path.parent,
            evidence_parent_descriptor=destination.parent_descriptor,
            managed_directories=managed_directories,
            mountinfo_path=target.mountinfo_path,
        )
        if any(
            binding.mount_authority is not None
            and directory_mount_backings_overlap(binding.mount_authority, frozen.evidence_parent)
            for binding in target.sibling_roots
        ):
            raise DatabaseCacheMaintenanceError(
                "maintenance evidence destination is physically within a configured sibling cache"
            )
        destination.publication_authority = _EvidencePublicationAuthority(
            parent_identity=destination.parent_identity,
            mount_authority=frozen.evidence_parent,
        )
    except LinuxMountContainmentError as exc:
        raise DatabaseCacheMaintenanceError(
            "maintenance evidence destination is physically within the managed cache"
        ) from exc
    except LinuxMountAuthorityError as exc:
        raise DatabaseCacheMaintenanceError("maintenance evidence mount authority is unavailable") from exc


def _freeze_deletion_target_authority(
    scope: _Scope,
    *,
    sibling_root_bindings: tuple[_PhysicalCacheRootBinding, ...],
    locks_descriptor: int | None,
    replicas_descriptor: int | None,
    mountinfo_path: Path,
) -> _DeletionTargetAuthority:
    try:
        return _freeze_deletion_target_mount_authority(
            scope,
            sibling_root_bindings=sibling_root_bindings,
            locks_descriptor=locks_descriptor,
            replicas_descriptor=replicas_descriptor,
            mountinfo_path=mountinfo_path,
        )
    except LinuxMountAuthorityError as exc:
        raise DatabaseCacheMaintenanceError("maintenance evidence mount authority is unavailable") from exc


def _freeze_deletion_target_mount_authority(
    scope: _Scope,
    *,
    sibling_root_bindings: tuple[_PhysicalCacheRootBinding, ...],
    locks_descriptor: int | None,
    replicas_descriptor: int | None,
    mountinfo_path: Path,
) -> _DeletionTargetAuthority:
    cache_root_authority = freeze_directory_mount_authority(
        role="cache-root",
        path=scope.cache_root,
        descriptor=scope.cache_root_descriptor,
        mountinfo_path=mountinfo_path,
    )
    users_root_authority = freeze_directory_mount_authority(
        role="users-root",
        path=scope.cache_root / "users",
        descriptor=scope.users_descriptor,
        mountinfo_path=mountinfo_path,
    )
    user_namespace_authority = freeze_directory_mount_authority(
        role="user-namespace",
        path=scope.user_namespace,
        descriptor=scope.user_descriptor,
        mountinfo_path=mountinfo_path,
    )
    locks_authority: FrozenDirectoryMountAuthority | None = None
    if locks_descriptor is not None:
        locks_authority = freeze_directory_mount_authority(
            role="locks",
            path=scope.user_namespace / ".locks",
            descriptor=locks_descriptor,
            mountinfo_path=mountinfo_path,
        )
    replicas_authority: FrozenDirectoryMountAuthority | None = None
    if replicas_descriptor is not None:
        replicas_authority = freeze_directory_mount_authority(
            role="replicas",
            path=scope.replicas_path,
            descriptor=replicas_descriptor,
            mountinfo_path=mountinfo_path,
        )
    selected = _SelectedDeletionScopeAuthority(
        cache_root=cache_root_authority,
        users_root=users_root_authority,
        user_namespace=user_namespace_authority,
        locks=locks_authority,
        replicas=replicas_authority,
    )
    return _DeletionTargetAuthority(
        selected=selected,
        sibling_roots=sibling_root_bindings,
        mountinfo_path=mountinfo_path,
    )


def _verify_deletion_target_authority(target: _DeletionTargetAuthority) -> None:
    _verify_deletion_target_bindings(target, require_separation=True)


def _verify_deletion_target_bindings(
    target: _DeletionTargetAuthority,
    *,
    require_separation: bool,
) -> None:
    selected_managed = target.selected.managed()
    try:
        for selected in selected_managed:
            verify_frozen_directory_mount_authority(selected, mountinfo_path=target.mountinfo_path)
    except LinuxMountAuthorityError as exc:
        raise DatabaseCacheMaintenanceError("maintenance deletion target mount authority changed") from exc
    for binding in target.sibling_roots:
        if binding.mount_authority is None:
            current = _open_physical_cache_root(binding.path)
            if current is not None:
                current_descriptor, _ = current
                os.close(current_descriptor)
                raise DatabaseCacheMaintenanceError("configured sibling cache-root authority changed")
            continue
        try:
            verify_frozen_directory_mount_authority(
                binding.mount_authority,
                mountinfo_path=target.mountinfo_path,
            )
        except LinuxMountAuthorityError as exc:
            raise DatabaseCacheMaintenanceError("configured sibling cache-root authority changed") from exc
        current_uses_symlink = _configured_path_uses_symlink(binding.path)
        current = _open_physical_cache_root(binding.path)
        if current is None:
            raise DatabaseCacheMaintenanceError("configured sibling cache-root authority changed")
        current_descriptor, current_identity = current
        try:
            if current_identity != binding.identity or current_uses_symlink != binding.configured_path_uses_symlink:
                raise DatabaseCacheMaintenanceError("configured sibling cache-root authority changed")
        finally:
            os.close(current_descriptor)
    if not require_separation:
        return
    _verify_selected_deletion_scope_hierarchy(target.selected)
    for binding in target.sibling_roots:
        if binding.configured_path_uses_symlink:
            raise DatabaseCacheMaintenanceError(
                f"configured sibling cache root must not traverse symlinks: {binding.path}"
            )
        sibling = binding.mount_authority
        if sibling is None:
            continue
        for selected in selected_managed:
            exact_alias = (selected.identity.device, selected.identity.inode) == (
                sibling.identity.device,
                sibling.identity.inode,
            )
            if exact_alias or directory_mount_backings_overlap(selected, sibling):
                raise DatabaseCacheMaintenanceError(
                    f"acceptance database cache scope physically overlaps configured sibling root: {binding.path}"
                )


def _verify_selected_deletion_scope_hierarchy(target: _SelectedDeletionScopeAuthority) -> None:
    edges = [
        (target.cache_root, target.users_root, "users"),
        (target.users_root, target.user_namespace, target.user_namespace.path.name),
    ]
    if target.locks is not None:
        edges.append((target.user_namespace, target.locks, ".locks"))
    if target.replicas is not None:
        edges.append((target.user_namespace, target.replicas, "replicas"))
    for parent, child, basename in edges:
        try:
            require_exact_mount_child(parent, child, basename=basename)
        except LinuxMountAuthorityError as exc:
            raise DatabaseCacheMaintenanceError(
                f"acceptance database cache hierarchy physically escapes at {child.role}"
            ) from exc
    if target.locks is None:
        _require_relative_directory_absent(target.user_namespace.descriptor, ".locks")
    if target.replicas is None:
        _require_relative_directory_absent(target.user_namespace.descriptor, "replicas")


def _freeze_maintenance_lock_mount_authority(
    target: _DeletionTargetAuthority,
    *,
    cache_lock_descriptor: int,
    identities: tuple[str, ...] = (),
    identity_lock_descriptors: tuple[int, ...] = (),
) -> _MaintenanceLockMountAuthority:
    if identities != tuple(sorted(set(identities))) or len(identities) != len(identity_lock_descriptors):
        raise DatabaseCacheMaintenanceError("maintenance identity lock authority set is invalid")
    locks = target.selected.locks
    if locks is None:
        raise DatabaseCacheMaintenanceError("maintenance lock directory authority is unavailable")
    _verify_deletion_target_authority(target)
    try:
        cache_lock = freeze_regular_file_mount_authority(
            role="cache-lock",
            path=locks.path / "cache.lock",
            descriptor=cache_lock_descriptor,
            mountinfo_path=target.mountinfo_path,
        )
        require_exact_mount_child(locks, cache_lock, basename="cache.lock")
        identity_locks: list[tuple[str, FrozenRegularFileMountAuthority]] = []
        for identity, descriptor in zip(identities, identity_lock_descriptors, strict=True):
            basename = f"{identity}.lock"
            authority = freeze_regular_file_mount_authority(
                role="identity-lock",
                path=locks.path / basename,
                descriptor=descriptor,
                mountinfo_path=target.mountinfo_path,
            )
            require_exact_mount_child(locks, authority, basename=basename)
            identity_locks.append((identity, authority))
    except LinuxMountAuthorityError as exc:
        raise DatabaseCacheMaintenanceError("maintenance lock mount authority is unavailable") from exc
    return _MaintenanceLockMountAuthority(
        cache_lock=cache_lock,
        identity_locks=tuple(identity_locks),
    )


def _verify_maintenance_lock_mount_authority(
    target: _DeletionTargetAuthority,
    authority: _MaintenanceLockMountAuthority,
) -> None:
    locks = target.selected.locks
    if locks is None:
        raise DatabaseCacheMaintenanceError("maintenance lock directory authority is unavailable")
    _verify_deletion_target_authority(target)
    try:
        verify_frozen_regular_file_mount_authority(
            authority.cache_lock,
            mountinfo_path=target.mountinfo_path,
        )
        require_exact_mount_child(locks, authority.cache_lock, basename="cache.lock")
        for identity, identity_lock in authority.identity_locks:
            verify_frozen_regular_file_mount_authority(
                identity_lock,
                mountinfo_path=target.mountinfo_path,
            )
            require_exact_mount_child(locks, identity_lock, basename=f"{identity}.lock")
    except LinuxMountAuthorityError as exc:
        raise DatabaseCacheMaintenanceError("maintenance lock mount authority changed") from exc


def _require_relative_directory_absent(parent_descriptor: int, basename: str) -> None:
    try:
        os.stat(basename, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise DatabaseCacheMaintenanceError(
            f"acceptance database cache absence authority is unavailable: {basename!r}"
        ) from exc
    raise DatabaseCacheMaintenanceError(f"acceptance database cache absent directory appeared: {basename!r}")


def _descriptor_ancestry_contains(descriptor: int, identities: set[FilesystemObjectIdentity]) -> bool:
    current = os.dup(descriptor)
    visited: set[FilesystemObjectIdentity] = set()
    try:
        while True:
            info = os.fstat(current)
            identity = filesystem_object_identity(info)
            if identity in identities:
                return True
            if identity in visited:
                raise DatabaseCacheMaintenanceError("maintenance evidence ancestry is cyclic")
            visited.add(identity)
            parent = os.open("..", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current)
            try:
                parent_info = os.fstat(parent)
            except BaseException:
                os.close(parent)
                raise
            parent_identity = filesystem_object_identity(parent_info)
            if parent_identity == identity:
                os.close(parent)
                return False
            os.close(current)
            current = parent
    except DatabaseCacheMaintenanceError:
        raise
    except OSError as exc:
        raise DatabaseCacheMaintenanceError("maintenance evidence ancestry is unavailable") from exc
    finally:
        os.close(current)


__all__ = [
    "_authorize_evidence_publication",
    "_freeze_deletion_target_authority",
    "_freeze_maintenance_lock_mount_authority",
    "_verify_deletion_target_authority",
    "_verify_maintenance_lock_mount_authority",
]
