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

"""Immutable cache-maintenance evidence reservation and publication."""

from __future__ import annotations

import os
import stat
import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from bspp.orchestration.contract.database_cache_maintenance import (
    DatabaseCacheMaintenanceEvidence,
    canonical_database_cache_maintenance_evidence_bytes,
    load_database_cache_maintenance_evidence,
)

from ._database_cache_maintenance_types import (
    DatabaseCacheMaintenanceError,
    _CreatedEvidenceDirectory,
    _EvidenceDestination,
    _EvidencePublicationAuthority,
)
from ._filesystem_authority import (
    DirectoryAuthority,
    _real_directory_stat,
    directory_authority,
    filesystem_object_identity,
)
from ._linux_mount_authority import (
    freeze_directory_mount_authority,
)

_MOUNTINFO_PATH = Path("/proc/self/mountinfo")


@dataclass(frozen=True)
class DatabaseCacheEvidenceDurability:
    """Filesystem durability operations used during terminal publication."""

    write_all: Callable[[int, bytes], None]
    sync: Callable[[int], None]
    load: Callable[..., DatabaseCacheMaintenanceEvidence]
    verify: Callable[..., None]


def publish_database_cache_maintenance_evidence(
    evidence: DatabaseCacheMaintenanceEvidence,
    destination: Path,
) -> None:
    """Publish 0444 canonical evidence exclusively, accepting only an exact collision."""
    reserved = _reserve_evidence_destination(destination)
    try:
        mount_authority = freeze_directory_mount_authority(
            role="evidence-parent",
            path=reserved.path.parent,
            descriptor=reserved.parent_descriptor,
            mountinfo_path=_MOUNTINFO_PATH,
        )
        reserved.publication_authority = _EvidencePublicationAuthority(
            parent_identity=reserved.parent_identity,
            mount_authority=mount_authority,
        )
        _finish_evidence_destination(reserved, evidence)
    finally:
        _close_evidence_destination(reserved)


def _reserve_evidence_destination(destination: Path) -> _EvidenceDestination:
    """Bind and durability-probe evidence authority before cache mutation."""
    parent_descriptor: int | None = None
    created_directories: tuple[_CreatedEvidenceDirectory, ...] = ()
    probe_descriptor: int | None = None
    destination_descriptor: int | None = None
    destination_created = False
    retained = False
    probe_name = f".{destination.name}.preflight-{uuid.uuid4().hex}"
    try:
        if destination.name in {"", ".", ".."} or ".." in destination.parts:
            raise DatabaseCacheMaintenanceError("maintenance evidence destination path is invalid")
        parent_descriptor, created_directories = _open_or_create_evidence_parent(destination.parent)
        parent_info = _real_directory_stat(destination.parent, description="maintenance evidence parent")
        parent_mode = stat.S_IMODE(parent_info.st_mode)
        if parent_info.st_uid != os.geteuid() or parent_mode & 0o022:
            raise DatabaseCacheMaintenanceError("maintenance evidence parent authority is unsafe")
        parent_identity = directory_authority(parent_info)
        if _evidence_parent_identity(os.fstat(parent_descriptor)) != parent_identity:
            raise DatabaseCacheMaintenanceError("maintenance evidence parent changed while opened")
        probe_descriptor = os.open(
            probe_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_descriptor,
        )
        _write_all(probe_descriptor, b"bspp-maintenance-evidence-preflight\n")
        os.fsync(probe_descriptor)
        os.close(probe_descriptor)
        probe_descriptor = None
        os.unlink(probe_name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
        _verify_evidence_parent(destination.parent, parent_descriptor, parent_identity)
        try:
            destination_descriptor = os.open(
                destination.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o444,
                dir_fd=parent_descriptor,
            )
        except FileExistsError:
            existing = load_database_cache_maintenance_evidence(Path(destination.name), dir_fd=parent_descriptor)
            result = _EvidenceDestination(
                path=destination,
                parent_descriptor=parent_descriptor,
                parent_identity=parent_identity,
                descriptor=None,
                destination_identity=None,
                existing=existing,
                created_directories=created_directories,
            )
            retained = True
            return result
        destination_created = True
        os.fchmod(destination_descriptor, 0o444)
        info = os.fstat(destination_descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o444
            or info.st_size != 0
        ):
            os.close(destination_descriptor)
            destination_descriptor = None
            raise DatabaseCacheMaintenanceError("maintenance evidence reservation authority is invalid")
        os.fsync(parent_descriptor)
        reserved = _EvidenceDestination(
            path=destination,
            parent_descriptor=parent_descriptor,
            parent_identity=parent_identity,
            descriptor=destination_descriptor,
            destination_identity=filesystem_object_identity(info),
            existing=None,
            created_directories=created_directories,
        )
        _verify_evidence_destination(reserved, require_empty=True)
        retained = True
        return reserved
    except DatabaseCacheMaintenanceError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise DatabaseCacheMaintenanceError("maintenance evidence authority preflight failed") from exc
    finally:
        if probe_descriptor is not None:
            os.close(probe_descriptor)
        if parent_descriptor is not None:
            with suppress(FileNotFoundError):
                os.unlink(probe_name, dir_fd=parent_descriptor)
            if not retained:
                if destination_created:
                    with suppress(FileNotFoundError):
                        os.unlink(destination.name, dir_fd=parent_descriptor)
                if destination_descriptor is not None:
                    os.close(destination_descriptor)
                os.close(parent_descriptor)
                _cleanup_created_evidence_directories(created_directories, remove=True)


def _open_or_create_evidence_parent(
    path: Path,
) -> tuple[int, tuple[_CreatedEvidenceDirectory, ...]]:
    """Open/create one parent path component-wise without following links."""
    absolute = Path(os.path.abspath(path))
    current = os.open(absolute.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    created: list[_CreatedEvidenceDirectory] = []
    try:
        for component in absolute.parts[1:]:
            made = False
            try:
                before = os.stat(component, dir_fd=current, follow_symlinks=False)
            except FileNotFoundError:
                os.mkdir(component, 0o700, dir_fd=current)
                made = True
                before = os.stat(component, dir_fd=current, follow_symlinks=False)
            before_identity = _evidence_directory_component_identity(before)
            descriptor = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=current,
            )
            try:
                opened_identity = _evidence_directory_component_identity(os.fstat(descriptor))
                after_identity = _evidence_directory_component_identity(
                    os.stat(component, dir_fd=current, follow_symlinks=False)
                )
                if opened_identity != before_identity or after_identity != before_identity:
                    raise DatabaseCacheMaintenanceError("maintenance evidence parent component changed while opened")
                if made and (opened_identity.owner_uid != os.geteuid() or opened_identity.permissions != 0o700):
                    raise DatabaseCacheMaintenanceError("created maintenance evidence parent authority is invalid")
                if made:
                    created.append(
                        _CreatedEvidenceDirectory(
                            parent_descriptor=os.dup(current),
                            basename=component,
                            descriptor=os.dup(descriptor),
                            identity=opened_identity,
                        )
                    )
            except BaseException:
                os.close(descriptor)
                raise
            os.close(current)
            current = descriptor
        return current, tuple(created)
    except BaseException:
        os.close(current)
        _cleanup_created_evidence_directories(tuple(created), remove=True)
        raise


def _evidence_directory_component_identity(info: os.stat_result) -> DirectoryAuthority:
    if not stat.S_ISDIR(info.st_mode):
        raise DatabaseCacheMaintenanceError("maintenance evidence parent must not traverse non-directories")
    return directory_authority(info)


def _cleanup_created_evidence_directories(
    created: tuple[_CreatedEvidenceDirectory, ...],
    *,
    remove: bool,
) -> None:
    """Close construction authority and remove only unchanged invocation-owned empties."""
    for item in reversed(created):
        try:
            if remove:
                visible = os.stat(item.basename, dir_fd=item.parent_descriptor, follow_symlinks=False)
                if (
                    _evidence_directory_component_identity(os.fstat(item.descriptor)) == item.identity
                    and _evidence_directory_component_identity(visible) == item.identity
                ):
                    os.rmdir(item.basename, dir_fd=item.parent_descriptor)
                    os.fsync(item.parent_descriptor)
        except (DatabaseCacheMaintenanceError, OSError):
            pass
        finally:
            os.close(item.descriptor)
            os.close(item.parent_descriptor)


def _finish_evidence_destination(
    destination: _EvidenceDestination | None,
    evidence: DatabaseCacheMaintenanceEvidence,
) -> None:
    _finish_evidence_destination_with_durability(
        destination,
        evidence,
        durability=DatabaseCacheEvidenceDurability(
            write_all=_write_all,
            sync=os.fsync,
            load=load_database_cache_maintenance_evidence,
            verify=_verify_evidence_destination,
        ),
    )


def _finish_evidence_destination_with_durability(
    destination: _EvidenceDestination | None,
    evidence: DatabaseCacheMaintenanceEvidence,
    *,
    durability: DatabaseCacheEvidenceDurability,
) -> None:
    if destination is None:
        raise DatabaseCacheMaintenanceError("maintenance evidence authority was not reserved")
    _verify_evidence_publication_authority(destination)
    if destination.existing is not None:
        if destination.existing == evidence:
            _verify_retained_evidence_parent(destination)
            return
        raise DatabaseCacheMaintenanceError("maintenance evidence collision is not an exact canonical match")
    if destination.descriptor is None:
        raise DatabaseCacheMaintenanceError("maintenance evidence reservation is unavailable")
    destination.finalization_attempted = True
    content = canonical_database_cache_maintenance_evidence_bytes(evidence)
    _verify_evidence_publication_authority(destination)
    durability.verify(destination, require_empty=True)
    try:
        durability.write_all(destination.descriptor, content)
        durability.sync(destination.descriptor)
        durability.sync(destination.parent_descriptor)
        if (
            durability.load(
                Path(destination.path.name),
                dir_fd=destination.parent_descriptor,
            )
            != evidence
        ):
            raise DatabaseCacheMaintenanceError("published maintenance evidence failed exact reload")
        durability.verify(destination, require_empty=False)
        destination.published = True
    except BaseException as exc:
        try:
            _reset_evidence_destination_after_failed_finalization(destination)
        except (DatabaseCacheMaintenanceError, OSError, TypeError, ValueError) as reset_exc:
            raise DatabaseCacheMaintenanceError("maintenance evidence finalization recovery failed") from reset_exc
        if isinstance(exc, DatabaseCacheMaintenanceError):
            raise
        if isinstance(exc, (OSError, TypeError, ValueError)):
            raise DatabaseCacheMaintenanceError("maintenance evidence finalization failed") from exc
        raise


def _reset_evidence_destination_after_failed_finalization(destination: _EvidenceDestination) -> None:
    """Restore a retained unpublished evidence inode for one failure record retry."""
    if destination.descriptor is None:
        raise DatabaseCacheMaintenanceError("maintenance evidence reservation is unavailable")
    _verify_evidence_destination(destination, require_empty=False)
    os.ftruncate(destination.descriptor, 0)
    os.fsync(destination.descriptor)
    if os.lseek(destination.descriptor, 0, os.SEEK_SET) != 0:
        raise DatabaseCacheMaintenanceError("maintenance evidence descriptor offset reset failed")
    _verify_evidence_destination(destination, require_empty=True)
    destination.finalization_attempted = False


def _close_evidence_destination(destination: _EvidenceDestination) -> None:
    descriptor = destination.descriptor
    remove_created_directories = descriptor is not None and not destination.published
    try:
        if descriptor is not None and not destination.published:
            try:
                visible = os.stat(destination.path.name, dir_fd=destination.parent_descriptor, follow_symlinks=False)
                opened = os.fstat(descriptor)
                if (
                    destination.destination_identity is not None
                    and filesystem_object_identity(visible) == destination.destination_identity
                    and filesystem_object_identity(opened) == destination.destination_identity
                ):
                    os.unlink(destination.path.name, dir_fd=destination.parent_descriptor)
                    os.fsync(destination.parent_descriptor)
            except FileNotFoundError:
                pass
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(destination.parent_descriptor)
        _cleanup_created_evidence_directories(
            destination.created_directories,
            remove=remove_created_directories,
        )


def _verify_evidence_destination(destination: _EvidenceDestination, *, require_empty: bool) -> None:
    if destination.descriptor is None or destination.destination_identity is None:
        raise DatabaseCacheMaintenanceError("maintenance evidence reservation is unavailable")
    _verify_retained_evidence_parent(destination)
    try:
        visible = os.stat(destination.path.name, dir_fd=destination.parent_descriptor, follow_symlinks=False)
        opened = os.fstat(destination.descriptor)
    except OSError as exc:
        raise DatabaseCacheMaintenanceError("maintenance evidence reservation changed") from exc
    if (
        not stat.S_ISREG(visible.st_mode)
        or not stat.S_ISREG(opened.st_mode)
        or visible.st_uid != os.geteuid()
        or opened.st_uid != os.geteuid()
        or stat.S_IMODE(visible.st_mode) != 0o444
        or stat.S_IMODE(opened.st_mode) != 0o444
        or filesystem_object_identity(visible) != destination.destination_identity
        or filesystem_object_identity(opened) != destination.destination_identity
        or (require_empty and (visible.st_size != 0 or opened.st_size != 0))
    ):
        raise DatabaseCacheMaintenanceError("maintenance evidence reservation authority changed")


def _verify_retained_evidence_parent(destination: _EvidenceDestination) -> None:
    if _evidence_parent_identity(os.fstat(destination.parent_descriptor)) != destination.parent_identity:
        raise DatabaseCacheMaintenanceError("retained maintenance evidence parent changed")


def _verify_evidence_publication_authority(destination: _EvidenceDestination) -> None:
    authority = destination.publication_authority
    if authority is None:
        raise DatabaseCacheMaintenanceError("maintenance evidence publication authority is unavailable")
    _verify_retained_evidence_parent(destination)
    if (
        authority.parent_identity != destination.parent_identity
        or authority.mount_authority.descriptor != destination.parent_descriptor
        or DirectoryAuthority(
            device=authority.mount_authority.identity.device,
            inode=authority.mount_authority.identity.inode,
            owner_uid=authority.mount_authority.identity.uid,
            permissions=authority.mount_authority.identity.mode,
        )
        != destination.parent_identity
    ):
        raise DatabaseCacheMaintenanceError("maintenance evidence publication authority changed")


def _verify_evidence_parent(
    path: Path,
    descriptor: int,
    expected: DirectoryAuthority,
) -> None:
    visible = _real_directory_stat(path, description="maintenance evidence parent")
    if _evidence_parent_identity(visible) != expected or _evidence_parent_identity(os.fstat(descriptor)) != expected:
        raise DatabaseCacheMaintenanceError("maintenance evidence parent changed")


def _evidence_parent_identity(info: os.stat_result) -> DirectoryAuthority:
    if not stat.S_ISDIR(info.st_mode):
        raise DatabaseCacheMaintenanceError("maintenance evidence parent must be a directory")
    return directory_authority(info)


def _write_all(descriptor: int, content: bytes) -> None:
    offset = 0
    while offset < len(content):
        written = os.write(descriptor, content[offset:])
        if written <= 0:
            raise OSError("short maintenance evidence write")
        offset += written


__all__ = [
    "_close_evidence_destination",
    "_finish_evidence_destination",
    "_reserve_evidence_destination",
    "_verify_evidence_destination",
    "_verify_evidence_parent",
    "_verify_evidence_publication_authority",
    "publish_database_cache_maintenance_evidence",
]
