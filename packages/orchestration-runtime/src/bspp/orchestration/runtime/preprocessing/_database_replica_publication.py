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

"""Exclusive cache ownership and atomic no-replace Database Replica publication."""

from __future__ import annotations

import ctypes
import errno
import json
import os
import shutil
import stat
import sys
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from bspp.orchestration.contract.database_replica import (
    DatabaseReplicaManifest,
    DatabaseReplicaMember,
    canonical_database_replica_manifest_bytes,
    database_replica_manifest_from_mapping,
)

from ._database_placement_errors import DatabasePlacementError
from ._database_replica_copy import validate_replica_payload
from ._database_replica_errors import ClassifiedDatabaseReplicaError
from ._database_replica_lock_authority import verify_cache_ownership
from ._database_replica_lock_types import CacheExclusiveOwnership
from ._filesystem_authority import (
    FilesystemAuthority,
    _real_directory_stat,
    filesystem_authority,
    filesystem_object_identity,
)

_RENAME_NOREPLACE = 1
_REPLICA_MANIFEST_NAME = "replica-manifest.json"
_RenameAt2 = Callable[[int, bytes, int, bytes, int], int]


@dataclass(frozen=True)
class PopulationWorkspace:
    """Held descriptor authority for one exclusively owned hidden population."""

    replicas_descriptor: int
    temporary_descriptor: int
    temporary_name: str


def open_cache_root(path: Path) -> tuple[int, FilesystemAuthority]:
    """Open a real cache root and bind its descriptor to the path identity."""
    info = _real_directory_stat(path, description="protected Database Replica cache root")
    if info.st_uid != os.geteuid():
        raise DatabasePlacementError("protected Database Replica cache root has the wrong owner")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        raise DatabasePlacementError(f"protected Database Replica cache root is unavailable: {path}") from exc
    identity = _directory_identity(info)
    if _directory_identity(os.fstat(descriptor)) != identity:
        os.close(descriptor)
        raise DatabasePlacementError("protected Database Replica cache root changed while it was opened")
    return descriptor, identity


def verify_cache_root_binding(
    path: Path,
    descriptor: int,
    expected_identity: FilesystemAuthority,
) -> None:
    """Require the path and held descriptor to retain one cache-root authority."""
    if (
        _directory_identity(os.fstat(descriptor)) != expected_identity
        or _directory_identity(_real_directory_stat(path, description="protected Database Replica cache root"))
        != expected_identity
    ):
        raise DatabasePlacementError("protected Database Replica cache root changed during population")


def require_renameat2() -> _RenameAt2:
    """Bind the required Linux/libc descriptor-relative renameat2 symbol."""
    if sys.platform != "linux":
        raise DatabasePlacementError("Database Replica publication requires Linux renameat2")
    try:
        library = ctypes.CDLL(None, use_errno=True)
        function = library.renameat2
    except (AttributeError, OSError) as exc:
        raise DatabasePlacementError("Database Replica publication requires libc renameat2") from exc
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    return cast(_RenameAt2, function)


@contextmanager
def exclusive_population_workspace(
    cache_descriptor: int,
    *,
    ownership: CacheExclusiveOwnership,
    source_manifest_sha256: str,
) -> Iterator[PopulationWorkspace]:
    """Create one private temp while caller-held identity and cache ownership remain live."""
    if ownership.identity.source_manifest_sha256 != source_manifest_sha256:
        raise ClassifiedDatabaseReplicaError(
            "lock-unavailable",
            "Database Replica cache ownership does not match the population identity",
        )
    verify_cache_ownership(cache_descriptor, ownership)
    replicas_descriptor: int | None = None
    temporary_descriptor: int | None = None
    temporary_name = f".population-{source_manifest_sha256}-{uuid.uuid4().hex}"
    try:
        replicas_descriptor = _open_private_directory(cache_descriptor, "replicas")
        try:
            os.stat(source_manifest_sha256, dir_fd=replicas_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ClassifiedDatabaseReplicaError(
                "replica-validation-failed",
                "manifest-addressed Database Replica already exists; warm reuse is deferred",
            )
        try:
            # The hidden temp lives beside its final name. Linux cannot move a
            # 0555 directory across parents because updating `..` requires
            # write authority on that directory; same-parent rename preserves
            # the pre-publication immutable validation without a visible
            # mutable window.
            os.mkdir(temporary_name, 0o700, dir_fd=replicas_descriptor)
            temporary_descriptor = os.open(
                temporary_name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=replicas_descriptor,
            )
        except OSError as exc:
            raise ClassifiedDatabaseReplicaError(
                "publication-failed",
                "cannot create unique private Database Replica population",
            ) from exc
        cache_device = os.fstat(cache_descriptor).st_dev
        if (
            os.fstat(replicas_descriptor).st_dev != cache_device
            or os.fstat(temporary_descriptor).st_dev != cache_device
        ):
            raise ClassifiedDatabaseReplicaError(
                "cache-authority-invalid",
                "Database Replica temp and destination must remain on the cache filesystem",
            )
        yield PopulationWorkspace(
            replicas_descriptor=replicas_descriptor,
            temporary_descriptor=temporary_descriptor,
            temporary_name=temporary_name,
        )
    finally:
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)
        if replicas_descriptor is not None:
            _remove_owned_temporary(replicas_descriptor, temporary_name)
        if replicas_descriptor is not None:
            os.close(replicas_descriptor)


def make_replica_immutable(
    temporary_descriptor: int,
    *,
    manifest: DatabaseReplicaManifest,
    members: tuple[DatabaseReplicaMember, ...],
) -> None:
    """Write the canonical manifest, fsync all files, and freeze files/tree."""
    manifest_bytes = canonical_database_replica_manifest_bytes(manifest)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            _REPLICA_MANIFEST_NAME,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=temporary_descriptor,
        )
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(manifest_bytes)
            handle.flush()
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    except OSError as exc:
        raise DatabasePlacementError("cannot write immutable Database Replica Manifest") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
            descriptor = None
    for member in members:
        descriptor = None
        try:
            descriptor = os.open(member.replica_path, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=temporary_descriptor)
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
        except OSError as exc:
            raise DatabasePlacementError(f"cannot freeze Database Replica member: {member.replica_path}") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
                descriptor = None
    os.fchmod(temporary_descriptor, 0o555)
    os.fsync(temporary_descriptor)
    validate_replica_payload(temporary_descriptor, members, immutable=True)
    _verify_manifest_at(temporary_descriptor, manifest)


def publish_replica_noreplace(
    renameat2: _RenameAt2,
    workspace: PopulationWorkspace,
    *,
    destination_name: str,
) -> None:
    """Atomically expose one complete directory with descriptor-relative RENAME_NOREPLACE."""
    ctypes.set_errno(0)
    result = renameat2(
        workspace.replicas_descriptor,
        os.fsencode(workspace.temporary_name),
        workspace.replicas_descriptor,
        os.fsencode(destination_name),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        label = errno.errorcode.get(error_number, "UNKNOWN")
        raise DatabasePlacementError(
            f"atomic Database Replica publication failed closed: errno={error_number} ({label})"
        )
    try:
        os.fsync(workspace.replicas_descriptor)
    except OSError as exc:
        raise ClassifiedDatabaseReplicaError(
            "publication-failed",
            "cannot durably confirm atomic Database Replica publication",
        ) from exc


def verify_published_replica(
    workspace: PopulationWorkspace,
    *,
    destination_name: str,
    manifest: DatabaseReplicaManifest,
    members: tuple[DatabaseReplicaMember, ...],
) -> None:
    """Descriptor-open and completely revalidate the newly visible immutable winner."""
    try:
        descriptor = os.open(
            destination_name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=workspace.replicas_descriptor,
        )
    except OSError as exc:
        raise DatabasePlacementError("published Database Replica is unavailable") from exc
    try:
        if stat.S_IMODE(os.fstat(descriptor).st_mode) != 0o555:
            raise DatabasePlacementError("published Database Replica root must be immutable 0555")
        validate_replica_payload(descriptor, members, immutable=True)
        _verify_manifest_at(descriptor, manifest)
    finally:
        os.close(descriptor)


def _open_private_directory(parent_descriptor: int, name: str) -> int:
    try:
        os.mkdir(name, 0o700, dir_fd=parent_descriptor)
    except FileExistsError:
        pass
    except OSError as exc:
        raise ClassifiedDatabaseReplicaError(
            "cache-authority-invalid",
            f"cannot create private cache directory {name!r}",
        ) from exc
    try:
        info = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        descriptor = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_descriptor)
    except OSError as exc:
        raise ClassifiedDatabaseReplicaError(
            "cache-authority-invalid",
            f"private cache directory is unavailable: {name!r}",
        ) from exc
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) != 0o700
        or filesystem_object_identity(info) != filesystem_object_identity(os.fstat(descriptor))
        or info.st_dev != os.fstat(parent_descriptor).st_dev
    ):
        os.close(descriptor)
        raise ClassifiedDatabaseReplicaError(
            "cache-authority-invalid",
            f"private cache directory authority is invalid: {name!r}",
        )
    return descriptor


def _verify_manifest_at(root_descriptor: int, expected: DatabaseReplicaManifest) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(_REPLICA_MANIFEST_NAME, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=root_descriptor)
        info = os.fstat(descriptor)
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read()
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o444:
            raise DatabasePlacementError("Database Replica Manifest must be an immutable 0444 regular file")
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise DatabasePlacementError("Database Replica Manifest must be a mapping")
        observed = database_replica_manifest_from_mapping(payload)
        if raw != canonical_database_replica_manifest_bytes(observed) or observed != expected:
            raise DatabasePlacementError("Database Replica Manifest failed exact canonical verification")
    except DatabasePlacementError:
        raise
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise DatabasePlacementError("Database Replica Manifest is unavailable or invalid") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _remove_owned_temporary(staging_descriptor: int, name: str) -> None:
    try:
        info = os.stat(name, dir_fd=staging_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError:
        return
    if not stat.S_ISDIR(info.st_mode):
        return
    try:
        os.chmod(name, 0o700, dir_fd=staging_descriptor, follow_symlinks=False)
        shutil.rmtree(name, dir_fd=staging_descriptor)
    except OSError:
        # A caught failure may retain only its own hidden, ineligible temp if
        # the filesystem itself prevents safe removal. #95 owns stale repair.
        return


def _directory_identity(info: os.stat_result) -> FilesystemAuthority:
    return filesystem_authority(info)


__all__ = [
    "PopulationWorkspace",
    "exclusive_population_workspace",
    "make_replica_immutable",
    "open_cache_root",
    "publish_replica_noreplace",
    "require_renameat2",
    "verify_cache_root_binding",
    "verify_published_replica",
]
