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

"""File transfer utilities with retry logic for Lustre I/O errors.

Provides retry-wrapped copy and extraction operations to handle transient
Lustre filesystem errors (ESTALE, EAGAIN, EIO, EBUSY).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bspp.orchestration.runtime.constants import IO_BASE_DELAY, IO_MAX_RETRIES, RETRYABLE_ERRNOS
from bspp.orchestration.runtime.postprocessing.failure_adapter import (
    TransportFailure,
    transport_failure_from_oserror,
)

logger = logging.getLogger(__name__)

__all__ = [
    "copy_files_to_scratch",
    "copy_outputs_to_destination",
    "extract_archive_to_scratch",
    "io_with_retry",
]


def io_with_retry(
    func: Callable[..., Any],
    *args: Any,
    max_retries: int = IO_MAX_RETRIES,
    base_delay: float = IO_BASE_DELAY,
    retryable_errnos: set[int] | None = None,
) -> Any:
    """Execute *func(*args)* with exponential-backoff retry on transient I/O errors.

    Args:
        func: Callable to invoke.
        *args: Positional arguments forwarded to *func*.
        max_retries: Maximum number of retries (default from constants).
        base_delay: Base delay in seconds for exponential backoff.
        retryable_errnos: Set of errno values that trigger a retry.
            Defaults to :data:`~bspp.orchestration.runtime.constants.RETRYABLE_ERRNOS`.

    Returns:
        The return value of *func*.

    Raises:
        OSError: If all retries are exhausted or the error is not retryable.
        TransportFailure: If a retryable ``OSError`` maps to an audited group-A
            transport failure (connection reset, timeout, or temporary DNS).
    """
    if retryable_errnos is None:
        retryable_errnos = RETRYABLE_ERRNOS

    last_exc: OSError | None = None
    for attempt in range(max_retries + 1):
        try:
            return func(*args)
        except OSError as exc:
            last_exc = exc
            if attempt < max_retries and exc.errno in retryable_errnos:
                delay = base_delay * (2**attempt)
                logger.warning(
                    "Transient I/O error (attempt %d/%d, retrying in %.0fs): %s",
                    attempt + 1,
                    max_retries,
                    delay,
                    exc,
                )
                time.sleep(delay)
                continue
            audited = transport_failure_from_oserror(exc)
            if audited is not None:
                raise audited from exc
            raise

    # Should not reach here, but satisfy type checker
    assert last_exc is not None
    raise last_exc


def copy_files_to_scratch(
    src_dir: Path,
    dst_dir: Path,
    filenames: list[str],
) -> int:
    """Copy files from *src_dir* to *dst_dir* with retry on I/O errors.

    Args:
        src_dir: Source directory.
        dst_dir: Destination directory (created if needed).
        filenames: List of filenames to copy.

    Returns:
        Number of files successfully copied.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0

    for fname in filenames:
        src = src_dir / fname
        dst = dst_dir / fname
        if dst.exists():
            copied += 1
            continue
        try:
            io_with_retry(shutil.copy2, src, dst)
            copied += 1
        except (OSError, TransportFailure):
            logger.warning("Failed to copy %s -> %s", src, dst)

    return copied


def extract_archive_to_scratch(
    archive: Path,
    scratch_dir: Path,
) -> int:
    """Extract a ``.tar.lz4`` archive to *scratch_dir*.

    Uses ``lz4 -dc | tar xf -`` pipeline for extraction.

    Args:
        archive: Path to the ``.tar.lz4`` file.
        scratch_dir: Destination directory (created if needed).

    Returns:
        Number of files extracted.

    Raises:
        FileNotFoundError: If *archive* does not exist.
        RuntimeError: If extraction fails.
    """
    if not archive.exists():
        msg = f"Archive not found: {archive}"
        raise FileNotFoundError(msg)

    scratch_dir.mkdir(parents=True, exist_ok=True)

    # Count files first
    lz4_list = subprocess.Popen(
        ["lz4", "-dc", str(archive)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    tar_list = subprocess.Popen(
        ["tar", "tf", "-"],
        stdin=lz4_list.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert lz4_list.stdout is not None
    lz4_list.stdout.close()
    tar_out, _ = tar_list.communicate()
    lz4_list.wait()
    file_count = sum(
        1 for line in tar_out.decode().strip().splitlines() if line.strip() and not line.strip().endswith("/")
    )

    # Extract
    lz4_proc = subprocess.Popen(
        ["lz4", "-dc", str(archive)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    tar_proc = subprocess.Popen(
        ["tar", "xf", "-", "-C", str(scratch_dir)],
        stdin=lz4_proc.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert lz4_proc.stdout is not None
    lz4_proc.stdout.close()
    tar_proc.communicate()
    lz4_proc.wait()

    if lz4_proc.returncode != 0 or tar_proc.returncode != 0:
        msg = f"Extraction failed for {archive.name}: lz4 rc={lz4_proc.returncode}, tar rc={tar_proc.returncode}"
        raise RuntimeError(msg)

    logger.info("Extracted %d files from %s to %s", file_count, archive.name, scratch_dir)
    return file_count


def copy_outputs_to_destination(
    src_dir: Path,
    dst_dir: Path,
    *,
    flat: bool = False,
) -> int:
    """Copy output files from *src_dir* to *dst_dir* with retry.

    Args:
        src_dir: Source directory containing output files.
        dst_dir: Destination directory.
        flat: If ``True``, copy all files into *dst_dir* without preserving
            subdirectory structure.  If ``False`` (default), preserve the
            relative directory structure.

    Returns:
        Number of files successfully copied.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0

    for item in src_dir.rglob("*"):
        if not item.is_file():
            continue

        if flat:
            dest = dst_dir / item.name
        else:
            rel = item.relative_to(src_dir)
            dest = dst_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)

        try:
            io_with_retry(shutil.copy2, item, dest)
            copied += 1
        except (OSError, TransportFailure):
            logger.warning("Failed to copy %s -> %s", item, dest)

    return copied
