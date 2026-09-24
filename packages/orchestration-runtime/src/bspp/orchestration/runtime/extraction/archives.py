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

"""Extract .tar.lz4 archives from a staging directory.

Core extraction logic: find archives, decompress via lz4+tar, track
completion with marker files.  No CLI, no cluster-specific paths.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import multiprocessing.context
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logger = logging.getLogger(__name__)

EXTRACTED_MARKER_DIR = ".extracted"
"""Name of the subdirectory used to record which archives have been extracted."""


# ---------------------------------------------------------------------------
# Marker helpers
# ---------------------------------------------------------------------------


def _marker_path(marker_dir: Path, archive: Path) -> Path:
    """Return the path to the marker file for *archive* inside *marker_dir*."""
    return marker_dir / archive.name


def _is_extracted(marker_dir: Path, archive: Path) -> bool:
    """Return ``True`` if a marker exists for *archive*."""
    return _marker_path(marker_dir, archive).exists()


def _create_marker(marker_dir: Path, archive: Path) -> None:
    """Create a zero-byte marker indicating *archive* was fully extracted."""
    marker_dir.mkdir(parents=True, exist_ok=True)
    _marker_path(marker_dir, archive).touch()


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def find_archives(staging_dir: Path) -> list[Path]:
    """Find all ``.tar.lz4`` files in *staging_dir* (recursively).

    dm downloads preserve S3 key structure (e.g. ``staging/structures/*.tar.lz4``)
    while s5cmd flattens to ``staging/*.tar.lz4``.  Searching recursively handles
    both layouts.

    Returns a sorted list of paths (empty list if directory does not exist).
    """
    if not staging_dir.exists():
        return []
    return sorted(staging_dir.rglob("*.tar.lz4"))


# ---------------------------------------------------------------------------
# Single-archive extraction
# ---------------------------------------------------------------------------


def extract_archive(
    archive: Path,
    output_dir: Path,
    *,
    keep_archive: bool = False,
) -> int:
    """Extract a single ``.tar.lz4`` archive via ``lz4 -dc | tar xf -``.

    Args:
        archive: Path to the ``.tar.lz4`` file.
        output_dir: Destination directory for extracted files.
        keep_archive: If ``False`` (default), delete the archive after
            successful extraction.

    Returns:
        Number of files extracted from the archive.

    Raises:
        RuntimeError: If ``lz4`` or ``tar`` exits with a non-zero code.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # List contents first to count files
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
    tar_list_out, _ = tar_list.communicate()
    lz4_list.wait()
    file_count = sum(
        1 for line in tar_list_out.decode().strip().splitlines() if line.strip() and not line.strip().endswith("/")
    )

    # Actual extraction
    lz4_proc = subprocess.Popen(
        ["lz4", "-dc", str(archive)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    tar_proc = subprocess.Popen(
        ["tar", "xf", "-", "-C", str(output_dir)],
        stdin=lz4_proc.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert lz4_proc.stdout is not None
    lz4_proc.stdout.close()
    _, _tar_stderr = tar_proc.communicate()
    lz4_proc.wait()

    if lz4_proc.returncode != 0 or tar_proc.returncode != 0:
        raise RuntimeError(
            f"Extraction failed for {archive.name}: lz4 rc={lz4_proc.returncode}, tar rc={tar_proc.returncode}"
        )

    if not keep_archive:
        archive.unlink()

    logger.debug("Extracted %d files from %s", file_count, archive.name)
    return file_count


# ---------------------------------------------------------------------------
# Batch extraction (sequential / parallel)
# ---------------------------------------------------------------------------


def _extract_one(
    archive: Path,
    output_dir: Path,
    *,
    keep_archive: bool,
    marker_dir: Path | None,
) -> int:
    """Extract a single archive, optionally creating a marker on success."""
    count = extract_archive(archive, output_dir, keep_archive=keep_archive)
    if marker_dir is not None:
        _create_marker(marker_dir, archive)
    return count


def extract_sequential(
    archives: list[Path],
    output_dir: Path,
    *,
    keep_archives: bool = False,
    marker_dir: Path | None = None,
) -> int:
    """Extract *archives* one at a time.

    Args:
        archives: List of ``.tar.lz4`` paths.
        output_dir: Destination directory.
        keep_archives: If ``False``, delete each archive after extraction.
        marker_dir: If given, skip archives that already have a marker and
            create a marker for each newly extracted archive.

    Returns:
        Total number of files extracted across all archives.
    """
    total = 0
    for archive in archives:
        if marker_dir is not None and _is_extracted(marker_dir, archive):
            logger.info("Skipping already-extracted %s", archive.name)
            continue
        count = extract_archive(archive, output_dir, keep_archive=keep_archives)
        total += count
        if marker_dir is not None:
            _create_marker(marker_dir, archive)
        logger.info("Extracted %s (%d files)", archive.name, count)
    return total


def extract_parallel(
    archives: list[Path],
    output_dir: Path,
    *,
    keep_archives: bool = False,
    marker_dir: Path | None = None,
    workers: int = 4,
) -> int:
    """Extract *archives* in parallel using a :class:`~concurrent.futures.ProcessPoolExecutor`.

    Args:
        archives: List of ``.tar.lz4`` paths.
        output_dir: Destination directory.
        keep_archives: If ``False``, delete each archive after extraction.
        marker_dir: If given, skip already-extracted archives and create
            markers for newly extracted ones.
        workers: Number of parallel extraction workers.

    Returns:
        Total number of files extracted across all archives.
    """
    pending = [a for a in archives if marker_dir is None or not _is_extracted(marker_dir, a)]
    if not pending:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    total = 0

    with ProcessPoolExecutor(max_workers=workers, mp_context=_parallel_start_context()) as executor:
        futures = {
            executor.submit(
                _extract_one,
                a,
                output_dir,
                keep_archive=keep_archives,
                marker_dir=marker_dir,
            ): a
            for a in pending
        }
        for future in as_completed(futures):
            archive = futures[future]
            count = future.result()
            total += count
            logger.info("Extracted %s (%d files)", archive.name, count)

    return total


def _parallel_start_context() -> multiprocessing.context.BaseContext:
    """Return a non-fork context to avoid Python 3.12 fork-in-thread warnings."""

    available = mp.get_all_start_methods()
    if "spawn" in available:
        return mp.get_context("spawn")
    if "forkserver" in available:
        return mp.get_context("forkserver")
    return mp.get_context()
