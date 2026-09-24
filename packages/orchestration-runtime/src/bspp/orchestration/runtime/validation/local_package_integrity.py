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

"""Independent local tar member and digest validation for release acceptance."""

from __future__ import annotations

import hashlib
import os
import stat
import tarfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Final

from bspp.orchestration.contract.release_acceptance import LocalIntegrityReport

MAX_MEMBERS: Final = 100_000
MAX_MEMBER_BYTES: Final = 1024 * 1024 * 1024
MAX_TOTAL_BYTES: Final = 8 * 1024 * 1024 * 1024


def validate_local_tar_integrity(
    archive_path: Path,
    expected_member_sha256: Mapping[str, str],
) -> LocalIntegrityReport:
    """Recompute one tar inventory and its regular-member digests."""
    expected = {_member_name(name): _sha256(digest) for name, digest in expected_member_sha256.items()}
    observed: dict[str, str] = {}
    total_bytes = 0
    before = archive_path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError("local archive must be an exclusive regular file")
    descriptor = os.open(archive_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as handle:
        opened = os.fstat(handle.fileno())
        if _stable_metadata(opened) != _stable_metadata(before):
            raise ValueError("local archive changed during open")
        with tarfile.open(fileobj=handle, mode="r:*") as archive:
            for count, member in enumerate(archive, start=1):
                if count > MAX_MEMBERS:
                    raise ValueError("local archive member count exceeds bound")
                name = _member_name(member.name)
                if name in observed:
                    raise ValueError(f"duplicate local archive member: {name}")
                if not member.isfile() or member.issparse():
                    raise ValueError(f"local archive member must be a non-sparse regular file: {name}")
                if member.size < 0 or member.size > MAX_MEMBER_BYTES:
                    raise ValueError(f"local archive member size exceeds bound: {name}")
                total_bytes += member.size
                if total_bytes > MAX_TOTAL_BYTES:
                    raise ValueError("local archive total payload exceeds bound")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"local archive member cannot be read: {name}")
                digest = hashlib.sha256()
                read_bytes = 0
                while chunk := stream.read(1024 * 1024):
                    read_bytes += len(chunk)
                    if read_bytes > member.size:
                        raise ValueError(f"local archive member exceeds declared size: {name}")
                    digest.update(chunk)
                if read_bytes != member.size:
                    raise ValueError(f"local archive member is truncated: {name}")
                observed[name] = digest.hexdigest()
        finished = os.fstat(handle.fileno())
    current = archive_path.lstat()
    if _stable_metadata(finished) != _stable_metadata(opened) or _stable_metadata(current) != _stable_metadata(opened):
        raise ValueError("local archive changed during validation")
    mismatches = tuple(
        sorted(
            (name for name in set(expected).intersection(observed) if expected[name] != observed[name]),
            key=str.encode,
        )
    )
    return LocalIntegrityReport(
        1,
        tuple(sorted(expected, key=str.encode)),
        tuple(sorted(observed, key=str.encode)),
        mismatches,
    )


def _member_name(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("local archive member name must be non-empty")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe local archive member name: {value!r}")
    return value


def _sha256(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("expected local archive digest must be lowercase SHA-256")
    return value


def _stable_metadata(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_nlink, value.st_size, value.st_mtime_ns, value.st_ctime_ns


__all__ = ["validate_local_tar_integrity"]
