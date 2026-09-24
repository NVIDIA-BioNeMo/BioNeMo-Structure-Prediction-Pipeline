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

"""Preprocessing reference-database intake from S3.

Downloads each declared member of a ``DatabaseSetDeclaration`` with
``source_kind == "s3"`` from ``{source_uri.rstrip('/')}/{source_path}``
to ``{source_root}/{source_path}``, verifying the downloaded SHA-256 against
the member's ``preexisting_checksum`` (which must be a 64-hex SHA-256 value).
No-op for ``source_kind == "local"``.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Protocol

from bspp.orchestration.contract.database_set_provisioning import DatabaseSetDeclaration
from bspp.orchestration.runtime.data_movement.common import PlannedTransfer, TransferResult

_SHA256 = re.compile(r"[0-9a-f]{64}")

_STREAM_BLOCK_SIZE = 1024 * 1024


class DatabaseIntakeError(Exception):
    """Raised when remote database set intake fails closed."""


class _TransferCallable(Protocol):
    def __call__(self, src: str, dst: str) -> TransferResult | PlannedTransfer: ...


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_STREAM_BLOCK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def download_database_set_from_s3(
    declaration: DatabaseSetDeclaration,
    *,
    transfer: _TransferCallable | None = None,
) -> None:
    """Download an S3-sourced Database Set to its ``source_root``.

    For ``source_kind == "s3"``, each member is downloaded from
    ``{source_uri.rstrip('/')}/{source_path}`` to
    ``{source_root}/{source_path}`` and its SHA-256 is verified against the
    member's ``preexisting_checksum`` (algorithm must be ``sha256`` and the
    value must be 64-hex).  Missing, unsupported, or non-64-hex checksums fail
    closed.  No-op for ``source_kind == "local"``.
    """
    if declaration.source_kind == "local":
        return
    if declaration.source_kind != "s3":
        raise DatabaseIntakeError(f"unsupported database set source_kind: {declaration.source_kind!r}")
    if declaration.source_uri is None:
        raise DatabaseIntakeError("s3 source_kind requires a non-None source_uri")

    if transfer is None:
        from bspp.orchestration.runtime.data_movement.s3 import transfer as s3_transfer

        transfer = s3_transfer.cp

    stripped_uri = declaration.source_uri.rstrip("/")
    source_root = Path(declaration.source_root)

    for role in declaration.roles:
        for member in role.members:
            checksum = member.preexisting_checksum
            if checksum is None:
                raise DatabaseIntakeError(f"s3 database member {member.logical_name!r} has no preexisting_checksum")
            if checksum.algorithm != "sha256":
                raise DatabaseIntakeError(
                    f"s3 database member {member.logical_name!r} checksum algorithm must be 'sha256', "
                    f"got {checksum.algorithm!r}"
                )
            if _SHA256.fullmatch(checksum.value) is None:
                raise DatabaseIntakeError(
                    f"s3 database member {member.logical_name!r} checksum value must be 64-hex sha256"
                )
            src_uri = f"{stripped_uri}/{member.source_path}"
            dst_path = source_root / member.source_path
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            result = transfer(src_uri, str(dst_path))
            if isinstance(result, PlannedTransfer):
                raise DatabaseIntakeError(
                    f"database member {member.logical_name!r} intake requires an executed transfer, not a dry-run plan"
                )
            if not result.ok:
                raise DatabaseIntakeError(
                    f"database member {member.logical_name!r} download failed with returncode {result.returncode}: "
                    f"{result.stderr_tail.strip()}"
                )
            if dst_path.is_symlink() or not dst_path.is_file():
                raise DatabaseIntakeError(f"downloaded database member must be a regular non-symlink file: {dst_path}")
            actual = _hash_file(dst_path)
            if actual != checksum.value:
                raise DatabaseIntakeError(
                    f"database member {member.logical_name!r} sha256 {actual} does not match declared "
                    f"checksum {checksum.value}"
                )


__all__ = ["DatabaseIntakeError", "download_database_set_from_s3"]
