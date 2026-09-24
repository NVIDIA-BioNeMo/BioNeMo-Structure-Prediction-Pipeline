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

"""Executing archive producer for validated ``ArchiveBatchPlan`` records.

This module is the only reconciliation point between the frozen non-executing
planner (``runtime.folding.archive``) and the frozen transport contract
(``contract.prediction_bundle``):

* The planner renders the pinned ``tar | lz4`` pipeline and its on-disk
  ``archive_name`` as ``{run_tag}_{archive_index:05d}.tar.lz4`` (with an
  underscore between the attempt letter and the batch index).
* ``PredictionArchiveBundle.bundle_name`` requires the canonical
  ``bspp_<YYMMDD>_<HHMM>_<letter><batch:05d>.tar.lz4`` pattern (no underscore
  between the letter and the batch index).

Both artifacts are frozen and out of bounds for this producer, so the emitted
``bundle_name`` is derived from the batch's own identity in the contract's
canonical form while the archive bytes remain written at the plan's pinned
``stdout_path``/``archive_path``. After the planner output is verified, the
producer materializes ``archive_path.with_name(bundle_name)`` as a same-directory
hard link to the verified bytes. ``bundle_name`` is therefore a
transport-resolvable filename, not a detached label; the
frozen planner filename is retained as the execution artifact and no archive
bytes are copied.

If the canonical alias cannot be created, or a pre-existing canonical alias
names foreign evidence, the producer fails closed rather than relabeling
unverified evidence.

Same-attempt restart is ownership-aware and restart-stable:

* A canonical alias that is hard-linked to this attempt's planner output (one
  shared inode) is the verified bundle produced by the first invocation. The
  producer reloads and reuses it unchanged (positive size + SHA-256) and never
  re-runs the pipeline into that evidence.
* A canonical alias without that hard-link ownership is foreign evidence, so
  the producer fails closed without touching it.
* When no canonical alias exists (first publication, or a crash that left an
  unlinked planner output), the pipeline writes new bytes to a fresh
  same-directory temporary path which is published by atomic rename — the
  ``atomic_write_bytes`` pattern from ``emitter_support``. A path hard-linked
  to verified evidence is therefore never truncated in place: a rename
  replaces the directory entry while the verified inode survives under its
  canonical name.

The ``CommandRunner`` protocol is a local pipeline seam, deliberately distinct
from ``slurm.monitor.CommandRunner``: folding execution must not depend on the
slurm module, and the archive pipeline needs both pinned argv halves plus the
stdout target. Unit tests inject a fake runner (never a real subprocess).
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from string import ascii_lowercase
from typing import Protocol

from bspp.orchestration.contract.folding_archive import (
    ArchiveBatchPlan,
    ArchivePlanOptions,
    FoldingResultInventory,
)
from bspp.orchestration.contract.prediction_bundle import PredictionArchiveBundle
from bspp.orchestration.runtime.folding.archive import plan_folding_archives

__all__ = [
    "ArchiveCommandResult",
    "CommandRunner",
    "compute_sha256",
    "derive_archive_run_tag",
    "execute_archive_batch",
    "produce_archives",
]


@dataclass(frozen=True)
class ArchiveCommandResult:
    """Result of one injected ``tar | lz4 > archive`` pipeline invocation."""

    returncode: int
    stdout: str = ""
    stderr: str = ""


class CommandRunner(Protocol):
    """Local pipeline runner: pipe ``tar_argv`` into ``lz4_argv`` to ``stdout_path``.

    A production runner executes the pinned argv halves and redirects the
    compressed bytes to ``stdout_path``. Unit tests inject a fake that records
    the pinned argv and materializes bytes at the archive path.
    """

    def __call__(
        self,
        tar_argv: Sequence[str],
        lz4_argv: Sequence[str],
        *,
        stdout_path: Path,
    ) -> ArchiveCommandResult: ...


def derive_archive_run_tag(attempt_timestamp: datetime, letter: str) -> str:
    """Derive the immutable attempt-owned run tag from a UTC timestamp and letter.

    The run tag is derived once from the immutable UTC
    attempt timestamp and recorded for reuse on restart — never re-derived from
    the wall clock. This function is a pure function of its inputs.
    """
    if not isinstance(letter, str) or len(letter) != 1 or letter not in ascii_lowercase:
        msg = "letter must be exactly one lowercase ASCII letter"
        raise ValueError(msg)
    if not isinstance(attempt_timestamp, datetime):
        msg = "attempt_timestamp must be a datetime"
        raise ValueError(msg)
    offset = attempt_timestamp.utcoffset()
    if offset is not None and offset != timedelta(0):
        msg = "attempt_timestamp must be UTC (or naive, interpreted as UTC)"
        raise ValueError(msg)
    return f"bspp_{attempt_timestamp:%y%m%d_%H%M}_{letter}"


def compute_sha256(path: Path) -> str:
    """Return the lowercase 64-hex SHA-256 of a file's bytes, streamed."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def execute_archive_batch(batch: ArchiveBatchPlan, *, runner: CommandRunner) -> PredictionArchiveBundle:
    """Execute one validated batch and emit its transport-bundle evidence.

    Stages each member into ``batch.stage_dir`` under its collision-safe
    ``stage_name``, then resolves the bundle ownership-aware:

    * Same-attempt restart: an existing ``bundle_name`` alias hard-linked to
      the planner output is the verified bundle from the first invocation. It
      is reloaded, re-verified (regular file, positive size, SHA-256), and
      reused unchanged; the archive pipeline is not run again.
    * An existing ``bundle_name`` alias without that hard-link ownership is
      foreign evidence and fails closed, its bytes untouched.
    * Otherwise the plan's pinned argv runs through the injectable runner into
      a fresh temporary path; after verification (positive size + SHA-256) the
      bytes are published to the planner output by atomic rename (never an
      in-place truncation) and the contract-valid ``bundle_name`` is
      materialized as a same-directory hard link to those verified bytes.

    The returned ``PredictionArchiveBundle`` membership always matches the plan.
    """
    stage_dir = Path(batch.stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    for member in batch.members:
        shutil.copyfile(Path(member.source_path), stage_dir / member.stage_name)

    archive_path = Path(batch.stdout_path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    member_ids = tuple(member.stage_name for member in batch.members)
    bundle_name = f"{batch.manifest_record.run_tag}{batch.archive_index:05d}.tar.lz4"
    canonical_path = archive_path.with_name(bundle_name)

    if canonical_path.exists():
        if archive_path.exists() and os.path.samefile(archive_path, canonical_path):
            # Same-attempt restart: the verified bundle pair is already
            # materialized. Reload and reuse it unchanged; never re-run the
            # pipeline into verified evidence.
            if not canonical_path.is_file():
                msg = f"canonical bundle alias is not a file: {canonical_path}"
                raise ValueError(msg)
            size = canonical_path.stat().st_size
            if size <= 0:
                msg = f"canonical bundle alias is empty: {canonical_path}"
                raise ValueError(msg)
            return PredictionArchiveBundle(
                bundle_name=bundle_name,
                member_ids=member_ids,
                member_count=len(member_ids),
                sha256=compute_sha256(canonical_path),
                size_bytes=size,
                created_at=None,
            )
        msg = (
            "canonical bundle alias already exists but is not hard-linked to "
            f"this attempt's verified planner output: {canonical_path}"
        )
        raise ValueError(msg)

    # First publication (or recovery from a crash that left an unlinked planner
    # output): produce new bytes at a fresh same-directory temporary path and
    # publish by atomic rename, mirroring emitter_support.atomic_write_bytes,
    # so a path hard-linked to verified evidence is never truncated in place.
    fd, tmp_name = tempfile.mkstemp(dir=str(archive_path.parent))
    tmp_path = Path(tmp_name)
    try:
        os.fchmod(fd, 0o644)
        os.close(fd)
        result = runner(batch.tar_argv, batch.lz4_argv, stdout_path=tmp_path)
        if result.returncode != 0:
            msg = f"archive pipeline failed with returncode {result.returncode}: {result.stderr[-500:]}"
            raise RuntimeError(msg)

        size = tmp_path.stat().st_size
        if size <= 0:
            msg = f"archive pipeline produced an empty archive at {archive_path}"
            raise ValueError(msg)

        archive_sha = compute_sha256(tmp_path)
        os.replace(tmp_path, archive_path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise

    try:
        os.link(archive_path, canonical_path)
    except OSError as exc:
        msg = f"failed to create canonical bundle alias {canonical_path}: {exc}"
        raise ValueError(msg) from exc

    canonical_sha = compute_sha256(canonical_path)
    if canonical_sha != archive_sha or canonical_path.stat().st_size != size:
        msg = f"canonical bundle alias does not match planner output: {canonical_path}"
        raise ValueError(msg)

    return PredictionArchiveBundle(
        bundle_name=bundle_name,
        member_ids=member_ids,
        member_count=len(member_ids),
        sha256=canonical_sha,
        size_bytes=size,
        created_at=None,
    )


def produce_archives(
    inventory: FoldingResultInventory,
    options: ArchivePlanOptions,
    *,
    runner: CommandRunner,
    attempt_timestamp: datetime,
    letter: str,
) -> tuple[PredictionArchiveBundle, ...]:
    """Plan and execute every archive batch for one attempt.

    Derives the attempt-owned run tag from ``attempt_timestamp`` and ``letter``,
    replaces the caller-provided run tag on the frozen options, consumes the
    frozen planner, and executes every batch in plan order.
    """
    run_tag = derive_archive_run_tag(attempt_timestamp, letter)
    effective_options = replace(options, run_tag=run_tag)
    plan = plan_folding_archives(inventory, effective_options)
    return tuple(execute_archive_batch(batch, runner=runner) for batch in plan.batches)
