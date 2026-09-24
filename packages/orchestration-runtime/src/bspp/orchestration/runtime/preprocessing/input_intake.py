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

"""Preprocessing remote FASTA input intake.

Downloads a ``VerifiedRemoteInputLocation`` from S3 (``s3://``),
verifies the downloaded bytes against the declared ``size_bytes`` and
``sha256``, and returns the verified local path.  The transfer boundary is
injectable so tests can supply a fake ``s3_transfer.cp`` without touching
production code.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Protocol

from bspp.orchestration.contract.phase import VerifiedRemoteInputLocation
from bspp.orchestration.runtime.data_movement.common import PlannedTransfer, TransferResult


class InputIntakeError(Exception):
    """Raised when remote FASTA intake fails closed."""


class _TransferCallable(Protocol):
    def __call__(self, src: str, dst: str) -> TransferResult | PlannedTransfer: ...


_STREAM_BLOCK_SIZE = 1024 * 1024


def download_remote_fasta(
    location: VerifiedRemoteInputLocation,
    destination: Path,
    *,
    transfer: _TransferCallable | None = None,
) -> Path:
    """Download ``location.source_uri`` to ``destination`` and verify the bytes.

    Rejects a ``PlannedTransfer`` (dry-run) before ``.ok`` is checked.
    Fails closed on any non-OK ``TransferResult``, non-positive ``size_bytes``,
    non-regular-file destination, symlink destination, or size/SHA-256 drift
    versus the declared values.

    Returns the verified ``destination`` path.
    """
    if transfer is None:
        from bspp.orchestration.runtime.data_movement.s3 import transfer as s3_transfer

        transfer = s3_transfer.cp

    destination.parent.mkdir(parents=True, exist_ok=True)
    result = transfer(location.source_uri, str(destination))
    if isinstance(result, PlannedTransfer):
        raise InputIntakeError("remote FASTA intake requires an executed transfer, not a dry-run plan")
    if not result.ok:
        raise InputIntakeError(
            f"remote FASTA download failed with returncode {result.returncode}: {result.stderr_tail.strip()}"
        )
    if destination.is_symlink() or not destination.is_file():
        raise InputIntakeError(f"downloaded FASTA must be a regular non-symlink file: {destination}")
    if location.size_bytes <= 0:
        raise InputIntakeError(f"declared size_bytes must be positive, got {location.size_bytes}")
    digest = hashlib.sha256()
    size_bytes = 0
    with destination.open("rb") as handle:
        while chunk := handle.read(_STREAM_BLOCK_SIZE):
            digest.update(chunk)
            size_bytes += len(chunk)
    if size_bytes != location.size_bytes:
        raise InputIntakeError(
            f"downloaded FASTA size {size_bytes} does not match declared size_bytes {location.size_bytes}"
        )
    observed = digest.hexdigest()
    if observed != location.sha256:
        raise InputIntakeError(f"downloaded FASTA sha256 {observed} does not match declared sha256 {location.sha256}")
    return destination


def normalize_fasta_to_identity_headers(path: Path) -> Path:
    """Strip descriptions from FASTA headers, keeping only ``>{identity}``.

    Reads the FASTA, strips descriptions from each header (keeps only the
    first whitespace-delimited token after ``>``), writes atomically (temp file
    + ``os.replace``).  Returns ``path``.

    Assumes strict-two-line FASTA format (one header line, one sequence line
    per record).  Multiline sequence lines are preserved verbatim but are not
    tested; the benchmark corpus is strict-two-line.

    Extends mmsa's ``reshape.py`` behavior (first token only, ledger
    ``P46-INPUT-002``) into the Phase's intake flow.
    """
    temp_path = path.with_name(f".{path.name}.normalize.tmp")
    try:
        with (
            path.open("r", encoding="utf-8") as source,
            temp_path.open("w", encoding="utf-8") as dest,
        ):
            for line in source:
                if line.startswith(">"):
                    tokens = line[1:].split(maxsplit=1)
                    if not tokens:
                        msg = f"FASTA header line has no identity token: {line!r}"
                        raise InputIntakeError(msg)
                    identity = tokens[0]
                    dest.write(f">{identity}\n")
                else:
                    dest.write(line)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    os.replace(temp_path, path)
    return path


__all__ = ["InputIntakeError", "download_remote_fasta", "normalize_fasta_to_identity_headers"]
