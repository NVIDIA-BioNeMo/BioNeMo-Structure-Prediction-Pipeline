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

"""Linux mount-facts observation for the protected Database Source root."""

from __future__ import annotations

import os
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from bspp.orchestration.contract.database_placement import DATABASE_CACHE_ROOT, DATABASE_SOURCE_ROOT
from bspp.orchestration.contract.database_placement_result import DatabaseSourceMountFacts
from bspp.orchestration.contract.database_replica import DatabaseCacheMountFacts, DatabaseWarmCacheMountFacts

from ._database_placement_errors import DatabasePlacementError


@dataclass(frozen=True)
class _MountInfoRecord:
    mount_id: int
    parent_mount_id: int
    device_major: int
    device_minor: int
    mount_root: str
    mount_point: str
    filesystem_type: str
    mount_source: str
    mount_options: tuple[str, ...]
    super_options: tuple[str, ...]


def observe_source_mount(root_descriptor: int, mountinfo_path: Path) -> DatabaseSourceMountFacts:
    """Bind an open source descriptor to its longest covering read-only mount."""
    result = _observe_mount(
        root_descriptor,
        protected_root=Path(DATABASE_SOURCE_ROOT),
        mountinfo_path=mountinfo_path,
        require_read_only=True,
        description="protected Database Source",
    )
    return DatabaseSourceMountFacts(
        mount_id=result.mount_id,
        parent_mount_id=result.parent_mount_id,
        device_major=result.device_major,
        device_minor=result.device_minor,
        mount_root=result.mount_root,
        mount_point=result.mount_point,
        filesystem_type=result.filesystem_type,
        mount_source=result.mount_source,
        mount_options=result.mount_options,
        super_options=result.super_options,
        read_only=True,
    )


def observe_cache_mount(
    root_descriptor: int,
    mountinfo_path: Path,
    *,
    expected_filesystem_type: str,
) -> DatabaseCacheMountFacts:
    """Bind an open cache descriptor to its longest covering read-write mount."""
    result = _observe_mount(
        root_descriptor,
        protected_root=Path(DATABASE_CACHE_ROOT),
        mountinfo_path=mountinfo_path,
        require_read_only=False,
        description="protected Database Replica cache",
    )
    if result.filesystem_type != expected_filesystem_type:
        raise DatabasePlacementError(
            "protected Database Replica cache filesystem type does not match RunSpec authority"
        )
    _verify_cache_writable(root_descriptor)
    return DatabaseCacheMountFacts(
        mount_id=result.mount_id,
        parent_mount_id=result.parent_mount_id,
        device_major=result.device_major,
        device_minor=result.device_minor,
        mount_root=result.mount_root,
        mount_point=result.mount_point,
        filesystem_type=result.filesystem_type,
        mount_source=result.mount_source,
        mount_options=result.mount_options,
        super_options=result.super_options,
        read_write=True,
        writable_verified=True,
    )


def observe_warm_cache_mount(
    root_descriptor: int,
    mountinfo_path: Path,
    *,
    expected_filesystem_type: str,
) -> DatabaseWarmCacheMountFacts:
    """Observe cache mount authority without writing anywhere beneath it."""
    result = _observe_mount(
        root_descriptor,
        protected_root=Path(DATABASE_CACHE_ROOT),
        mountinfo_path=mountinfo_path,
        require_read_only=False,
        description="protected Database Replica cache",
    )
    if result.filesystem_type != expected_filesystem_type:
        raise DatabasePlacementError(
            "protected Database Replica cache filesystem type does not match RunSpec authority"
        )
    return DatabaseWarmCacheMountFacts(
        mount_id=result.mount_id,
        parent_mount_id=result.parent_mount_id,
        device_major=result.device_major,
        device_minor=result.device_minor,
        mount_root=result.mount_root,
        mount_point=result.mount_point,
        filesystem_type=result.filesystem_type,
        mount_source=result.mount_source,
        mount_options=result.mount_options,
        super_options=result.super_options,
        read_write=True,
    )


def _observe_mount(
    root_descriptor: int,
    *,
    protected_root: Path,
    mountinfo_path: Path,
    require_read_only: bool,
    description: str,
) -> _MountInfoRecord:
    root_info = os.fstat(root_descriptor)
    try:
        lines = mountinfo_path.read_text().splitlines()
    except OSError as exc:
        raise DatabasePlacementError("Linux /proc/self/mountinfo is unavailable") from exc
    candidates: list[_MountInfoRecord] = []
    for line in lines:
        record = _parse_mountinfo_line(line)
        mount_point = Path(record.mount_point)
        if protected_root == mount_point or mount_point in protected_root.parents:
            candidates.append(record)
    if not candidates:
        raise DatabasePlacementError(f"{description} root is not covered by a mountinfo record")
    longest = max(len(Path(item.mount_point).parts) for item in candidates)
    selected = [item for item in candidates if len(Path(item.mount_point).parts) == longest]
    if len(selected) != 1:
        raise DatabasePlacementError(f"{description} mount authority is ambiguous")
    result = selected[0]
    if require_read_only:
        if "ro" not in result.mount_options or "rw" in result.mount_options:
            raise DatabasePlacementError(f"{description} mount must be read-only")
    elif "rw" not in result.mount_options or "ro" in result.mount_options:
        raise DatabasePlacementError(f"{description} mount must be read-write")
    if (os.major(root_info.st_dev), os.minor(root_info.st_dev)) != (
        result.device_major,
        result.device_minor,
    ):
        raise DatabasePlacementError(f"{description} device does not match mountinfo")
    return result


def _verify_cache_writable(root_descriptor: int) -> None:
    probe_name = f".bspp-write-probe-{os.getpid()}"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            probe_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=root_descriptor,
        )
        os.fsync(descriptor)
        os.unlink(probe_name, dir_fd=root_descriptor)
        os.fsync(root_descriptor)
    except OSError as exc:
        raise DatabasePlacementError("protected Database Replica cache is not descriptor-writable") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        with suppress(FileNotFoundError):
            os.unlink(probe_name, dir_fd=root_descriptor)


def _parse_mountinfo_line(line: str) -> _MountInfoRecord:
    fields = line.split()
    if fields.count("-") != 1:
        raise DatabasePlacementError("malformed Linux mountinfo separator")
    separator = fields.index("-")
    before = fields[:separator]
    after = fields[separator + 1 :]
    if len(before) < 6 or len(after) < 3:
        raise DatabasePlacementError("malformed Linux mountinfo record")
    try:
        mount_id = int(before[0])
        parent_mount_id = int(before[1])
        major_text, minor_text = before[2].split(":", 1)
        device_major = int(major_text)
        device_minor = int(minor_text)
    except (TypeError, ValueError) as exc:
        raise DatabasePlacementError("malformed Linux mountinfo identity") from exc
    return _MountInfoRecord(
        mount_id=mount_id,
        parent_mount_id=parent_mount_id,
        device_major=device_major,
        device_minor=device_minor,
        mount_root=_decode_mountinfo_path(before[3]),
        mount_point=_decode_mountinfo_path(before[4]),
        filesystem_type=after[0],
        mount_source=_decode_mountinfo_path(after[1]),
        mount_options=tuple(sorted(before[5].split(","))),
        super_options=tuple(sorted(after[2].split(","))),
    )


def _decode_mountinfo_path(value: str) -> str:
    decoded: list[str] = []
    index = 0
    replacements = {"040": " ", "011": "\t", "012": "\n", "134": "\\"}
    while index < len(value):
        if value[index] != "\\":
            decoded.append(value[index])
            index += 1
            continue
        code = value[index + 1 : index + 4]
        if len(code) != 3 or code not in replacements:
            raise DatabasePlacementError(f"unsupported or malformed Linux mountinfo escape in {value!r}")
        decoded.append(replacements[code])
        index += 4
    result = "".join(decoded)
    if not result:
        raise DatabasePlacementError("decoded Linux mountinfo path must be non-empty")
    return result


__all__ = ["observe_cache_mount", "observe_source_mount", "observe_warm_cache_mount"]
