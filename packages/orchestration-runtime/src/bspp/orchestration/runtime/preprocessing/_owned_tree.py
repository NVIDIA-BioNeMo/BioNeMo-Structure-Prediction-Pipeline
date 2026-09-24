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

"""Descriptor-relative claiming and removal of one caller-owned cache tree."""

from __future__ import annotations

import ctypes
import errno
import os
import stat
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace

from ._database_replica_errors import ClassifiedDatabaseReplicaError
from ._database_replica_publication import require_renameat2
from ._filesystem_authority import FilesystemAuthority, filesystem_authority

EntryAuthority = FilesystemAuthority
type IdentityValidator = Callable[..., EntryAuthority]
type RenameAt2Factory = Callable[[], Callable[[int, bytes, int, bytes, int], int]]

_RENAME_NOREPLACE = 1
_NESTED_CLAIM_PREFIX = ".cleanup-claim-"


@dataclass(frozen=True)
class ExpectedDirectoryBinding:
    """Optional pre-opened authority for a root directory claim."""

    authority: EntryAuthority
    descriptor: int | None = None


@dataclass(frozen=True)
class OwnedTreeRemover:
    """Narrow filesystem capability used by private cleanup coordinators."""

    remove: Callable[..., object]


def remove_owned_tree(
    parent_descriptor: int,
    name: str,
    *,
    cache_device: int,
    expected_binding: ExpectedDirectoryBinding | None = None,
    root_claim_prefix: str | None = None,
    identity_validator: IdentityValidator | None = None,
    renameat2_factory: RenameAt2Factory = require_renameat2,
) -> EntryAuthority:
    """Atomically claim and remove one exact same-device, effective-user tree."""
    validate_identity = identity_validator or _owned_entry_identity
    try:
        visible = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if expected_binding is not None and expected_binding.descriptor is not None:
            descriptor = os.dup(expected_binding.descriptor)
        else:
            descriptor = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_descriptor)
    except OSError as exc:
        raise ClassifiedDatabaseReplicaError(
            "replica-validation-failed",
            f"owned cache entry is not a safe directory: {name!r}",
        ) from exc
    try:
        expected = validate_identity(visible, cache_device=cache_device, require_directory=True, name=name)
        if expected_binding is not None and expected != expected_binding.authority:
            raise ClassifiedDatabaseReplicaError(
                "replica-validation-failed", f"owned cache entry changed before cleanup: {name!r}"
            )
        if (
            validate_identity(os.fstat(descriptor), cache_device=cache_device, require_directory=True, name=name)
            != expected
        ):
            raise ClassifiedDatabaseReplicaError(
                "replica-validation-failed", f"owned cache entry changed while opened: {name!r}"
            )
        claim_name = _claim_owned_entry(
            parent_descriptor,
            name,
            descriptor,
            cache_device=cache_device,
            require_directory=True,
            expected=expected,
            claim_name=(
                f"{root_claim_prefix}{uuid.uuid4().hex}"
                if root_claim_prefix is not None
                else f"{_NESTED_CLAIM_PREFIX}{uuid.uuid4().hex}"
            ),
            identity_validator=validate_identity,
            renameat2_factory=renameat2_factory,
        )
        os.fchmod(descriptor, 0o700)
        removal_authority = replace(expected, permissions=0o700)
        for child in sorted(os.listdir(descriptor)):
            child_info = os.stat(child, dir_fd=descriptor, follow_symlinks=False)
            if stat.S_ISDIR(child_info.st_mode):
                child_authority = validate_identity(
                    child_info, cache_device=cache_device, require_directory=True, name=child
                )
                remove_owned_tree(
                    descriptor,
                    child,
                    cache_device=cache_device,
                    expected_binding=ExpectedDirectoryBinding(authority=child_authority),
                    identity_validator=validate_identity,
                    renameat2_factory=renameat2_factory,
                )
            else:
                child_authority = validate_identity(
                    child_info, cache_device=cache_device, require_directory=False, name=child
                )
                _unlink_owned_entry(
                    descriptor,
                    child,
                    cache_device=cache_device,
                    expected=child_authority,
                    identity_validator=validate_identity,
                    renameat2_factory=renameat2_factory,
                )
        os.fsync(descriptor)
        rebound = validate_identity(
            os.stat(claim_name, dir_fd=parent_descriptor, follow_symlinks=False),
            cache_device=cache_device,
            require_directory=True,
            name=claim_name,
        )
        opened = validate_identity(
            os.fstat(descriptor), cache_device=cache_device, require_directory=True, name=claim_name
        )
        if rebound != removal_authority or opened != removal_authority:
            raise ClassifiedDatabaseReplicaError(
                "replica-validation-failed", f"owned cache entry changed before removal: {name!r}"
            )
        _require_owned_entry_binding(
            parent_descriptor,
            claim_name,
            descriptor,
            cache_device=cache_device,
            require_directory=True,
            expected=removal_authority,
            identity_validator=validate_identity,
        )
        os.rmdir(claim_name, dir_fd=parent_descriptor)
        try:
            os.fsync(parent_descriptor)
        except OSError as exc:
            raise ClassifiedDatabaseReplicaError(
                "replica-validation-failed", f"cannot durably confirm private cache entry removal: {claim_name!r}"
            ) from exc
        try:
            os.stat(claim_name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return expected
        raise ClassifiedDatabaseReplicaError(
            "replica-validation-failed", f"private owned cache entry claim remained after removal: {claim_name!r}"
        )
    except ClassifiedDatabaseReplicaError:
        raise
    except OSError as exc:
        raise ClassifiedDatabaseReplicaError(
            "replica-validation-failed", f"cannot safely remove owned cache entry: {name!r}"
        ) from exc
    finally:
        os.close(descriptor)


def _unlink_owned_entry(
    parent_descriptor: int,
    name: str,
    *,
    cache_device: int,
    expected: EntryAuthority,
    identity_validator: IdentityValidator,
    renameat2_factory: RenameAt2Factory,
) -> None:
    entry_descriptor: int | None = None
    try:
        entry_descriptor = os.open(name, os.O_PATH | os.O_NOFOLLOW, dir_fd=parent_descriptor)
        _require_owned_entry_binding(
            parent_descriptor,
            name,
            entry_descriptor,
            cache_device=cache_device,
            require_directory=False,
            expected=expected,
            identity_validator=identity_validator,
        )
        claim_name = _claim_owned_entry(
            parent_descriptor,
            name,
            entry_descriptor,
            cache_device=cache_device,
            require_directory=False,
            expected=expected,
            claim_name=f"{_NESTED_CLAIM_PREFIX}{uuid.uuid4().hex}",
            identity_validator=identity_validator,
            renameat2_factory=renameat2_factory,
        )
        os.unlink(claim_name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
        try:
            os.stat(claim_name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return
        raise ClassifiedDatabaseReplicaError(
            "replica-validation-failed", f"private owned cache child claim remained after removal: {claim_name!r}"
        )
    except ClassifiedDatabaseReplicaError:
        raise
    except OSError as exc:
        raise ClassifiedDatabaseReplicaError(
            "replica-validation-failed", f"cannot safely remove owned cache child: {name!r}"
        ) from exc
    finally:
        if entry_descriptor is not None:
            os.close(entry_descriptor)


def _claim_owned_entry(
    parent_descriptor: int,
    source_name: str,
    descriptor: int,
    *,
    cache_device: int,
    require_directory: bool,
    expected: EntryAuthority,
    claim_name: str,
    identity_validator: IdentityValidator,
    renameat2_factory: RenameAt2Factory,
) -> str:
    renameat2 = renameat2_factory()
    ctypes.set_errno(0)
    result = renameat2(
        parent_descriptor,
        os.fsencode(source_name),
        parent_descriptor,
        os.fsencode(claim_name),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        label = errno.errorcode.get(error_number, "UNKNOWN")
        raise ClassifiedDatabaseReplicaError(
            "replica-validation-failed",
            f"cannot atomically claim owned cache entry: errno={error_number} ({label})",
        )
    try:
        os.fsync(parent_descriptor)
    except OSError as exc:
        raise ClassifiedDatabaseReplicaError(
            "replica-validation-failed", f"cannot durably confirm private cache claim: {claim_name!r}"
        ) from exc
    try:
        _require_owned_entry_binding(
            parent_descriptor,
            claim_name,
            descriptor,
            cache_device=cache_device,
            require_directory=require_directory,
            expected=expected,
            identity_validator=identity_validator,
        )
    except ClassifiedDatabaseReplicaError as exc:
        _restore_mismatched_claim(
            parent_descriptor,
            source_name=source_name,
            claim_name=claim_name,
            renameat2_factory=renameat2_factory,
        )
        raise ClassifiedDatabaseReplicaError(
            "replica-validation-failed", f"owned cache entry changed at its atomic claim boundary: {source_name!r}"
        ) from exc
    return claim_name


def _restore_mismatched_claim(
    parent_descriptor: int,
    *,
    source_name: str,
    claim_name: str,
    renameat2_factory: RenameAt2Factory,
) -> None:
    renameat2 = renameat2_factory()
    ctypes.set_errno(0)
    result = renameat2(
        parent_descriptor,
        os.fsencode(claim_name),
        parent_descriptor,
        os.fsencode(source_name),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            return
        label = errno.errorcode.get(error_number, "UNKNOWN")
        raise ClassifiedDatabaseReplicaError(
            "replica-validation-failed",
            f"cannot restore mismatched private cache claim: errno={error_number} ({label})",
        )
    try:
        os.fsync(parent_descriptor)
    except OSError as exc:
        raise ClassifiedDatabaseReplicaError(
            "replica-validation-failed", "cannot durably confirm mismatched private cache claim restoration"
        ) from exc


def _require_owned_entry_binding(
    parent_descriptor: int,
    name: str,
    descriptor: int,
    *,
    cache_device: int,
    require_directory: bool,
    expected: EntryAuthority,
    identity_validator: IdentityValidator,
) -> None:
    try:
        opened = identity_validator(
            os.fstat(descriptor), cache_device=cache_device, require_directory=require_directory, name=name
        )
        visible = identity_validator(
            os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False),
            cache_device=cache_device,
            require_directory=require_directory,
            name=name,
        )
    except ClassifiedDatabaseReplicaError:
        raise
    except OSError as exc:
        raise ClassifiedDatabaseReplicaError(
            "replica-validation-failed", f"owned cache entry cannot be rebound before removal: {name!r}"
        ) from exc
    if opened != expected or visible != expected:
        raise ClassifiedDatabaseReplicaError(
            "replica-validation-failed", f"owned cache entry changed at the removal boundary: {name!r}"
        )


def _owned_entry_identity(
    info: os.stat_result,
    *,
    cache_device: int,
    require_directory: bool,
    name: str,
) -> EntryAuthority:
    is_expected_type = stat.S_ISDIR(info.st_mode) if require_directory else not stat.S_ISDIR(info.st_mode)
    if not is_expected_type or info.st_uid != os.geteuid() or info.st_dev != cache_device:
        raise ClassifiedDatabaseReplicaError(
            "replica-validation-failed", f"owned cache tree contains unsafe authority: {name!r}"
        )
    return filesystem_authority(info)


__all__ = ["EntryAuthority", "ExpectedDirectoryBinding", "IdentityValidator", "RenameAt2Factory", "remove_owned_tree"]
