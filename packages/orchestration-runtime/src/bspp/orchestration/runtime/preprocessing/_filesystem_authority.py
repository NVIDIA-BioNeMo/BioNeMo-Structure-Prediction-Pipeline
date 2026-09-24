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

"""Shared filesystem authority primitives for direct Database Placement."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path

from ._database_placement_errors import DatabasePlacementError


@dataclass(frozen=True)
class FilesystemAuthority:
    """Owner- and mode-sensitive authority for a filesystem object."""

    file_type: int
    device: int
    inode: int
    owner_uid: int
    permissions: int


@dataclass(frozen=True)
class FilesystemObservation:
    """Stable source/evidence observation without ownership policy."""

    file_type: int
    device: int
    inode: int
    size: int
    modified_ns: int


@dataclass(frozen=True)
class PathObservation:
    """Filesystem observation plus the literal symlink target, when present."""

    identity: FilesystemObservation
    symlink_target: str | None


@dataclass(frozen=True)
class FilesystemObjectIdentity:
    """Device and inode identity for an already type-validated object."""

    device: int
    inode: int


@dataclass(frozen=True)
class FilesystemBinding:
    """Type-sensitive binding for immutable evidence files and directories."""

    file_type: int
    device: int
    inode: int


@dataclass(frozen=True)
class DirectoryAuthority:
    """Owner- and mode-sensitive authority for a prevalidated directory."""

    device: int
    inode: int
    owner_uid: int
    permissions: int


def _real_directory_stat(path: Path, *, description: str) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as exc:
        raise DatabasePlacementError(f"{description} is unavailable: {path}") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise DatabasePlacementError(f"{description} must be a real non-symlink directory")
    current = Path(path.anchor)
    try:
        for component in path.parts[1:]:
            current /= component
            if stat.S_ISLNK(current.lstat().st_mode):
                raise DatabasePlacementError(f"{description} must not traverse symlink components")
    except OSError as exc:
        raise DatabasePlacementError(f"{description} is unavailable: {path}") from exc
    return info


def filesystem_authority(info: os.stat_result) -> FilesystemAuthority:
    return FilesystemAuthority(
        file_type=stat.S_IFMT(info.st_mode),
        device=info.st_dev,
        inode=info.st_ino,
        owner_uid=info.st_uid,
        permissions=stat.S_IMODE(info.st_mode),
    )


def filesystem_observation(info: os.stat_result) -> FilesystemObservation:
    return FilesystemObservation(
        file_type=stat.S_IFMT(info.st_mode),
        device=info.st_dev,
        inode=info.st_ino,
        size=info.st_size,
        modified_ns=info.st_mtime_ns,
    )


def filesystem_object_identity(info: os.stat_result) -> FilesystemObjectIdentity:
    return FilesystemObjectIdentity(device=info.st_dev, inode=info.st_ino)


def filesystem_binding(info: os.stat_result) -> FilesystemBinding:
    return FilesystemBinding(file_type=stat.S_IFMT(info.st_mode), device=info.st_dev, inode=info.st_ino)


def directory_authority(info: os.stat_result) -> DirectoryAuthority:
    return DirectoryAuthority(
        device=info.st_dev,
        inode=info.st_ino,
        owner_uid=info.st_uid,
        permissions=stat.S_IMODE(info.st_mode),
    )


_stat_identity = filesystem_observation


__all__ = [
    "DirectoryAuthority",
    "FilesystemAuthority",
    "FilesystemBinding",
    "FilesystemObjectIdentity",
    "FilesystemObservation",
    "PathObservation",
]
