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

"""Linux mount-aware authority for retained directory descriptors."""

from __future__ import annotations

import ctypes
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

_AT_NO_AUTOMOUNT = 0x800
_AT_EMPTY_PATH = 0x1000
_STATX_TYPE = 0x0001
_STATX_MODE = 0x0002
_STATX_UID = 0x0008
_STATX_INO = 0x0100
_STATX_BASIC_STATS = 0x07FF
_STATX_MNT_ID = 0x1000
_REQUIRED_STATX_MASK = _STATX_TYPE | _STATX_MODE | _STATX_UID | _STATX_INO | _STATX_MNT_ID


class LinuxMountAuthorityError(RuntimeError):
    """A retained Linux mount binding could not be proved stable."""


class LinuxMountContainmentError(LinuxMountAuthorityError):
    """The evidence parent is backed by a managed filesystem subtree."""


class _StatxTimestamp(ctypes.Structure):
    _fields_ = [
        ("tv_sec", ctypes.c_int64),
        ("tv_nsec", ctypes.c_uint32),
        ("reserved", ctypes.c_int32),
    ]


class _Statx(ctypes.Structure):
    _fields_ = [
        ("mask", ctypes.c_uint32),
        ("blksize", ctypes.c_uint32),
        ("attributes", ctypes.c_uint64),
        ("nlink", ctypes.c_uint32),
        ("uid", ctypes.c_uint32),
        ("gid", ctypes.c_uint32),
        ("mode", ctypes.c_uint16),
        ("spare0", ctypes.c_uint16),
        ("ino", ctypes.c_uint64),
        ("size", ctypes.c_uint64),
        ("blocks", ctypes.c_uint64),
        ("attributes_mask", ctypes.c_uint64),
        ("atime", _StatxTimestamp),
        ("btime", _StatxTimestamp),
        ("ctime", _StatxTimestamp),
        ("mtime", _StatxTimestamp),
        ("rdev_major", ctypes.c_uint32),
        ("rdev_minor", ctypes.c_uint32),
        ("dev_major", ctypes.c_uint32),
        ("dev_minor", ctypes.c_uint32),
        ("mount_id", ctypes.c_uint64),
        ("dio_mem_align", ctypes.c_uint32),
        ("dio_offset_align", ctypes.c_uint32),
        ("subvolume", ctypes.c_uint64),
        ("atomic_write_unit_min", ctypes.c_uint32),
        ("atomic_write_unit_max", ctypes.c_uint32),
        ("atomic_write_segments_max", ctypes.c_uint32),
        ("dio_read_offset_align", ctypes.c_uint32),
        ("atomic_write_unit_max_opt", ctypes.c_uint32),
        ("spare2", ctypes.c_uint32),
        ("spare3", ctypes.c_uint64 * 8),
    ]


_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC_STATX = getattr(_LIBC, "statx", None)
if _LIBC_STATX is not None:
    _LIBC_STATX.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.POINTER(_Statx),
    )
    _LIBC_STATX.restype = ctypes.c_int


_NodeKind = Literal["directory", "regular-file"]


@dataclass(frozen=True)
class _NodeIdentity:
    device: int
    inode: int
    uid: int
    mode: int


@dataclass(frozen=True)
class _StatxIdentity:
    mount_id: int
    device_major: int
    device_minor: int
    inode: int
    uid: int
    mode: int


@dataclass(frozen=True)
class _MountInfoRecord:
    raw: str
    mount_id: int
    parent_mount_id: int
    device_major: int
    device_minor: int
    mount_root: PurePosixPath
    mount_point: PurePosixPath


@dataclass(frozen=True)
class _NodeMountAuthority:
    role: str
    path: Path
    descriptor: int
    identity: _NodeIdentity
    statx: _StatxIdentity
    mountinfo: _MountInfoRecord
    backing_coordinate: PurePosixPath
    node_kind: _NodeKind
    follow_visible: bool = False


@dataclass(frozen=True)
class _NodeSnapshotRequest:
    role: str
    path: Path
    descriptor: int
    node_kind: _NodeKind
    follow_visible: bool = False


@dataclass(frozen=True)
class _NodeObservation:
    identity: _NodeIdentity
    statx: _StatxIdentity


FrozenDirectoryMountAuthority = _NodeMountAuthority
FrozenRegularFileMountAuthority = _NodeMountAuthority


@dataclass(frozen=True)
class FrozenEvidenceMountAuthority:
    """Complete retained mount facts that must remain exact before mutation."""

    evidence_parent: _NodeMountAuthority
    managed: tuple[_NodeMountAuthority, ...]


def freeze_directory_mount_authority(
    *,
    role: str,
    path: Path,
    descriptor: int,
    mountinfo_path: Path,
    follow_visible: bool = False,
) -> FrozenDirectoryMountAuthority:
    """Freeze one retained and visible directory's complete mount authority."""
    return _snapshot_mount_authorities(
        (
            _NodeSnapshotRequest(
                role=role,
                path=path,
                descriptor=descriptor,
                node_kind="directory",
                follow_visible=follow_visible,
            ),
        ),
        mountinfo_path=mountinfo_path,
    )[0]


def verify_frozen_directory_mount_authority(
    frozen: FrozenDirectoryMountAuthority,
    *,
    mountinfo_path: Path,
) -> None:
    """Require one retained directory and its visible path to match its freeze."""
    observed = freeze_directory_mount_authority(
        role=frozen.role,
        path=frozen.path,
        descriptor=frozen.descriptor,
        mountinfo_path=mountinfo_path,
        follow_visible=frozen.follow_visible,
    )
    if observed != frozen:
        raise LinuxMountAuthorityError("retained directory mount authority changed")


def freeze_regular_file_mount_authority(
    *,
    role: str,
    path: Path,
    descriptor: int,
    mountinfo_path: Path,
) -> FrozenRegularFileMountAuthority:
    """Freeze one retained regular file and its exact visible mount authority."""
    return _snapshot_mount_authorities(
        (
            _NodeSnapshotRequest(
                role=role,
                path=path,
                descriptor=descriptor,
                node_kind="regular-file",
            ),
        ),
        mountinfo_path=mountinfo_path,
    )[0]


def verify_frozen_regular_file_mount_authority(
    frozen: FrozenRegularFileMountAuthority,
    *,
    mountinfo_path: Path,
) -> None:
    """Require one retained regular file and its visible path to match its freeze."""
    observed = freeze_regular_file_mount_authority(
        role=frozen.role,
        path=frozen.path,
        descriptor=frozen.descriptor,
        mountinfo_path=mountinfo_path,
    )
    if observed != frozen:
        raise LinuxMountAuthorityError("retained regular-file mount authority changed")


def require_exact_mount_child(
    parent: FrozenDirectoryMountAuthority,
    child: FrozenDirectoryMountAuthority | FrozenRegularFileMountAuthority,
    *,
    basename: str,
) -> None:
    """Require an exact direct child on the parent's unchanged mount backing."""
    if basename in {"", ".", ".."} or Path(basename).name != basename:
        raise LinuxMountAuthorityError("retained mount child basename is invalid")
    expected_path = Path(os.path.abspath(parent.path / basename))
    expected_backing = parent.backing_coordinate / basename
    if (
        parent.node_kind != "directory"
        or child.path != expected_path
        or child.identity.device != parent.identity.device
        or child.statx.mount_id != parent.statx.mount_id
        or child.mountinfo != parent.mountinfo
        or child.backing_coordinate != expected_backing
    ):
        raise LinuxMountAuthorityError("retained mount authority is not an exact direct child")


def directory_mount_backings_overlap(
    left: FrozenDirectoryMountAuthority,
    right: FrozenDirectoryMountAuthority,
) -> bool:
    """Return whether either retained directory contains the other in backing storage."""
    if left.identity.device != right.identity.device:
        return False
    return (
        left.backing_coordinate == right.backing_coordinate
        or left.backing_coordinate in right.backing_coordinate.parents
        or right.backing_coordinate in left.backing_coordinate.parents
    )


def freeze_evidence_mount_authority(
    *,
    evidence_parent_path: Path,
    evidence_parent_descriptor: int,
    managed_directories: tuple[tuple[str, Path, int], ...],
    mountinfo_path: Path,
) -> FrozenEvidenceMountAuthority:
    """Freeze mount-aware backing coordinates and reject managed containment."""
    frozen = _observe_evidence_mount_authority(
        evidence_parent_path=evidence_parent_path,
        evidence_parent_descriptor=evidence_parent_descriptor,
        managed_directories=managed_directories,
        mountinfo_path=mountinfo_path,
    )
    for managed in frozen.managed:
        if directory_mount_backings_overlap(managed, frozen.evidence_parent):
            raise LinuxMountContainmentError("evidence parent backing is within a managed directory")
    return frozen


def verify_frozen_evidence_mount_authority(
    frozen: FrozenEvidenceMountAuthority,
    *,
    mountinfo_path: Path,
) -> None:
    """Re-observe every retained binding and require the complete freeze."""
    observed = _observe_evidence_mount_authority(
        evidence_parent_path=frozen.evidence_parent.path,
        evidence_parent_descriptor=frozen.evidence_parent.descriptor,
        managed_directories=tuple((item.role, item.path, item.descriptor) for item in frozen.managed),
        mountinfo_path=mountinfo_path,
    )
    if observed != frozen:
        raise LinuxMountAuthorityError("retained directory mount authority changed")


def _observe_evidence_mount_authority(
    *,
    evidence_parent_path: Path,
    evidence_parent_descriptor: int,
    managed_directories: tuple[tuple[str, Path, int], ...],
    mountinfo_path: Path,
) -> FrozenEvidenceMountAuthority:
    requests = (
        _NodeSnapshotRequest(
            role="evidence-parent",
            path=evidence_parent_path,
            descriptor=evidence_parent_descriptor,
            node_kind="directory",
        ),
        *(
            _NodeSnapshotRequest(role=role, path=path, descriptor=descriptor, node_kind="directory")
            for role, path, descriptor in managed_directories
        ),
    )
    observed = _snapshot_mount_authorities(requests, mountinfo_path=mountinfo_path)
    return FrozenEvidenceMountAuthority(evidence_parent=observed[0], managed=observed[1:])


def _snapshot_mount_authorities(
    requests: tuple[_NodeSnapshotRequest, ...],
    *,
    mountinfo_path: Path,
) -> tuple[_NodeMountAuthority, ...]:
    """Bind retained descriptors and visible paths around one required mountinfo read."""
    if not requests:
        return ()
    before = tuple(_observe_descriptor(item.descriptor, node_kind=item.node_kind) for item in requests)
    required_mount_ids = frozenset(item.statx.mount_id for item in before)
    records = _read_mountinfo(mountinfo_path, required_mount_ids=required_mount_ids)
    after = tuple(_observe_descriptor(item.descriptor, node_kind=item.node_kind) for item in requests)
    visible = tuple(_observe_visible(item) for item in requests)
    authorities: list[_NodeMountAuthority] = []
    for request, retained_before, retained_after, visible_after in zip(requests, before, after, visible, strict=True):
        if retained_after != retained_before:
            raise LinuxMountAuthorityError("retained mount authority changed during Linux mountinfo observation")
        if visible_after != retained_before:
            raise LinuxMountAuthorityError("visible directory mount authority changed")
        authorities.append(_build_mount_authority(request, retained_before, records))
    return tuple(authorities)


def _observe_descriptor(descriptor: int, *, node_kind: _NodeKind) -> _NodeObservation:
    identity = _node_identity(descriptor, node_kind=node_kind)
    descriptor_statx = _statx_identity(descriptor)
    _require_statx_matches_fstat(descriptor_statx, identity, node_kind=node_kind)
    return _NodeObservation(identity=identity, statx=descriptor_statx)


def _observe_visible(request: _NodeSnapshotRequest) -> _NodeObservation:
    visible_descriptor: int | None = None
    try:
        flags = os.O_RDONLY
        if request.node_kind == "directory":
            flags |= os.O_DIRECTORY
        if not request.follow_visible:
            flags |= os.O_NOFOLLOW
        visible_descriptor = os.open(request.path, flags)
        return _observe_descriptor(visible_descriptor, node_kind=request.node_kind)
    except OSError as exc:
        raise LinuxMountAuthorityError("visible directory mount authority is unavailable") from exc
    finally:
        if visible_descriptor is not None:
            os.close(visible_descriptor)


def _build_mount_authority(
    request: _NodeSnapshotRequest,
    observation: _NodeObservation,
    records: dict[int, _MountInfoRecord],
) -> _NodeMountAuthority:
    try:
        mountinfo = records[observation.statx.mount_id]
    except KeyError as exc:
        raise LinuxMountAuthorityError("retained mount ID is absent from Linux mountinfo") from exc
    if (mountinfo.device_major, mountinfo.device_minor) != (
        observation.statx.device_major,
        observation.statx.device_minor,
    ):
        raise LinuxMountAuthorityError("retained mount device disagrees with Linux mountinfo")
    normalized_path = Path(os.path.abspath(request.path))
    try:
        relative = PurePosixPath(normalized_path.as_posix()).relative_to(mountinfo.mount_point)
    except ValueError as exc:
        raise LinuxMountAuthorityError("visible directory is outside its retained mountpoint") from exc
    coordinate = mountinfo.mount_root.joinpath(relative)
    if ".." in coordinate.parts:
        raise LinuxMountAuthorityError("retained mount coordinate is malformed")
    return _NodeMountAuthority(
        role=request.role,
        path=normalized_path,
        descriptor=request.descriptor,
        identity=observation.identity,
        statx=observation.statx,
        mountinfo=mountinfo,
        backing_coordinate=coordinate,
        node_kind=request.node_kind,
        follow_visible=request.follow_visible,
    )


def _node_identity(descriptor: int, *, node_kind: _NodeKind) -> _NodeIdentity:
    try:
        info = os.fstat(descriptor)
    except OSError as exc:
        raise LinuxMountAuthorityError("retained directory identity is unavailable") from exc
    expected_type = stat.S_IFDIR if node_kind == "directory" else stat.S_IFREG
    if stat.S_IFMT(info.st_mode) != expected_type:
        raise LinuxMountAuthorityError(f"retained mount authority is not a {node_kind}")
    return _NodeIdentity(
        device=info.st_dev,
        inode=info.st_ino,
        uid=info.st_uid,
        mode=stat.S_IMODE(info.st_mode),
    )


def _statx_identity(descriptor: int) -> _StatxIdentity:
    if _LIBC_STATX is None or ctypes.sizeof(_Statx) != 256:
        raise LinuxMountAuthorityError("Linux statx mount authority is unavailable")
    result = _Statx()
    ctypes.set_errno(0)
    status = _LIBC_STATX(
        descriptor,
        b"",
        _AT_EMPTY_PATH | _AT_NO_AUTOMOUNT,
        _STATX_BASIC_STATS | _STATX_MNT_ID,
        ctypes.byref(result),
    )
    if status != 0:
        error = ctypes.get_errno()
        raise LinuxMountAuthorityError("Linux statx mount authority is unavailable") from OSError(
            error,
            os.strerror(error),
        )
    if result.mask & _REQUIRED_STATX_MASK != _REQUIRED_STATX_MASK:
        raise LinuxMountAuthorityError("Linux statx mount authority is incomplete")
    return _StatxIdentity(
        mount_id=result.mount_id,
        device_major=result.dev_major,
        device_minor=result.dev_minor,
        inode=result.ino,
        uid=result.uid,
        mode=result.mode,
    )


def _require_statx_matches_fstat(
    statx_identity: _StatxIdentity,
    identity: _NodeIdentity,
    *,
    node_kind: _NodeKind,
) -> None:
    expected_type = stat.S_IFDIR if node_kind == "directory" else stat.S_IFREG
    if (
        os.makedev(statx_identity.device_major, statx_identity.device_minor) != identity.device
        or statx_identity.inode != identity.inode
        or statx_identity.uid != identity.uid
        or stat.S_IFMT(statx_identity.mode) != expected_type
        or stat.S_IMODE(statx_identity.mode) != identity.mode
    ):
        raise LinuxMountAuthorityError("Linux statx identity disagrees with retained directory")


def _read_mountinfo(path: Path, *, required_mount_ids: frozenset[int]) -> dict[int, _MountInfoRecord]:
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise LinuxMountAuthorityError("Linux mountinfo is unavailable") from exc
    records: dict[int, _MountInfoRecord] = {}
    for line in lines:
        fields = line.split()
        if not fields:
            raise LinuxMountAuthorityError("Linux mountinfo is malformed")
        try:
            mount_id = int(fields[0])
        except (TypeError, ValueError) as exc:
            raise LinuxMountAuthorityError("Linux mountinfo is malformed") from exc
        if mount_id <= 0:
            raise LinuxMountAuthorityError("Linux mountinfo is malformed")
        if mount_id not in required_mount_ids:
            continue
        if fields.count("-") != 1:
            raise LinuxMountAuthorityError("Linux mountinfo is malformed")
        separator = fields.index("-")
        before = fields[:separator]
        after = fields[separator + 1 :]
        if len(before) < 6 or len(after) < 3:
            raise LinuxMountAuthorityError("Linux mountinfo is malformed")
        try:
            parent_mount_id = int(before[1])
            major_text, minor_text = before[2].split(":", 1)
            device_major = int(major_text)
            device_minor = int(minor_text)
            mount_root = _decode_mountinfo_path(before[3])
            mount_point = _decode_mountinfo_path(before[4])
        except (IndexError, TypeError, ValueError) as exc:
            raise LinuxMountAuthorityError("Linux mountinfo is malformed") from exc
        if mount_id in records:
            raise LinuxMountAuthorityError("Linux mountinfo mount ID is ambiguous")
        records[mount_id] = _MountInfoRecord(
            raw=line,
            mount_id=mount_id,
            parent_mount_id=parent_mount_id,
            device_major=device_major,
            device_minor=device_minor,
            mount_root=mount_root,
            mount_point=mount_point,
        )
    if records.keys() != required_mount_ids:
        raise LinuxMountAuthorityError("required retained mount ID is absent from Linux mountinfo")
    return records


def _decode_mountinfo_path(value: str) -> PurePosixPath:
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
            raise LinuxMountAuthorityError("Linux mountinfo is malformed")
        decoded.append(replacements[code])
        index += 4
    result = PurePosixPath("".join(decoded))
    if not result.is_absolute() or ".." in result.parts:
        raise LinuxMountAuthorityError("Linux mountinfo is malformed")
    return result


def _coordinate_contains(
    managed: _NodeMountAuthority,
    evidence_parent: _NodeMountAuthority,
) -> bool:
    if managed.identity.device != evidence_parent.identity.device:
        return False
    return (
        evidence_parent.backing_coordinate == managed.backing_coordinate
        or managed.backing_coordinate in evidence_parent.backing_coordinate.parents
    )


__all__ = [
    "FrozenDirectoryMountAuthority",
    "FrozenEvidenceMountAuthority",
    "FrozenRegularFileMountAuthority",
    "LinuxMountAuthorityError",
    "LinuxMountContainmentError",
    "directory_mount_backings_overlap",
    "freeze_directory_mount_authority",
    "freeze_evidence_mount_authority",
    "freeze_regular_file_mount_authority",
    "require_exact_mount_child",
    "verify_frozen_directory_mount_authority",
    "verify_frozen_evidence_mount_authority",
    "verify_frozen_regular_file_mount_authority",
]
