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

"""Archive-task and logical-shard planning for the native worker.

The legacy WP8a archive worker maps one SLURM array task to one archive. Each
archive is then split into a fixed number of logical sub-shards using contiguous
balanced slices, so eight models with ``shards_per_archive=2`` become
``(0:4, 4:8)``.
"""

from __future__ import annotations

import fcntl
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from bspp.orchestration.runtime.worker.types import ArchiveTaskPlan, BatchPlan, SubShardPlan


def _require_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"{name} must be an integer, got {value!r}"
        raise ValueError(msg)


def _require_positive_int(name: str, value: int) -> None:
    _require_int(name, value)
    if value <= 0:
        msg = f"{name} must be positive, got {value}"
        raise ValueError(msg)


def _require_non_negative_int(name: str, value: int) -> None:
    _require_int(name, value)
    if value < 0:
        msg = f"{name} must be non-negative, got {value}"
        raise ValueError(msg)


def task_id_to_archive_idx(task_id: int, archive_count: int) -> int:
    """Return the archive index selected by a zero-based array task ID."""
    _require_int("task_id", task_id)
    _require_non_negative_int("archive_count", archive_count)
    if not 0 <= task_id < archive_count:
        msg = f"task_id {task_id} is outside archive range 0..{archive_count - 1}"
        raise IndexError(msg)
    return task_id


def archive_active_marker_name(task_id: int) -> str:
    """Return the coordination marker filename for an in-flight archive task.

    The native worker writes this marker for the duration of one archive task
    (extraction through final upload) and removes it in its ``finally`` block.
    The archive cleanup tool refuses to delete a staged archive while its
    marker is present, so a concurrent worker rerun cannot lose its input.
    """
    _require_int("task_id", task_id)
    if task_id < 0:
        msg = f"task_id must be non-negative, got {task_id}"
        raise ValueError(msg)
    return f".archive_active_{task_id}"


def archive_active_lock_path(output_dir: Path, task_id: int) -> Path:
    """Return the per-archive worker-active coordination lock path."""
    return output_dir / f"{archive_active_marker_name(task_id)}.lock"


@contextmanager
def archive_active_lock(output_dir: Path, task_id: int) -> Iterator[None]:
    """Hold the per-archive worker-active coordination lock.

    The native worker holds this lock for the full duration of one archive task
    (while its ``.archive_active_<task_id>`` marker exists). The archive cleanup
    tool acquires the same lock across its marker check and deletion, so a
    worker cannot write the marker and begin extraction between the cleanup
    tool's check and its ``unlink``.
    """
    lock_path = archive_active_lock_path(output_dir, task_id)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def logical_shard_id_for_sub_shard(
    archive_idx: int,
    sub_shard_idx: int,
    shards_per_archive: int,
) -> int:
    """Return the global logical shard ID for an archive-local sub-shard."""
    _require_int("archive_idx", archive_idx)
    _require_int("sub_shard_idx", sub_shard_idx)
    _require_positive_int("shards_per_archive", shards_per_archive)
    if archive_idx < 0:
        msg = f"archive_idx must be non-negative, got {archive_idx}"
        raise IndexError(msg)
    if not 0 <= sub_shard_idx < shards_per_archive:
        msg = f"sub_shard_idx {sub_shard_idx} is outside sub-shard range 0..{shards_per_archive - 1}"
        raise IndexError(msg)
    return archive_idx * shards_per_archive + sub_shard_idx


def sub_shard_model_range(
    total_model_count: int,
    sub_shard_idx: int,
    shards_per_archive: int,
) -> tuple[int, int]:
    """Return the half-open archive-local model range for one sub-shard."""
    _require_non_negative_int("total_model_count", total_model_count)
    _require_int("sub_shard_idx", sub_shard_idx)
    _require_positive_int("shards_per_archive", shards_per_archive)
    if not 0 <= sub_shard_idx < shards_per_archive:
        msg = f"sub_shard_idx {sub_shard_idx} is outside sub-shard range 0..{shards_per_archive - 1}"
        raise IndexError(msg)

    base_size = total_model_count // shards_per_archive
    remainder = total_model_count % shards_per_archive
    extra_before = min(sub_shard_idx, remainder)
    model_start = sub_shard_idx * base_size + extra_before
    model_end = model_start + base_size + (1 if sub_shard_idx < remainder else 0)
    return model_start, model_end


def split_model_ids_for_archive(
    model_ids: Sequence[str],
    shards_per_archive: int,
) -> tuple[tuple[str, ...], ...]:
    """Split archive model IDs into contiguous logical sub-shards."""
    _require_positive_int("shards_per_archive", shards_per_archive)
    model_id_tuple = tuple(model_ids)
    return tuple(
        model_id_tuple[model_start:model_end]
        for model_start, model_end in (
            sub_shard_model_range(len(model_id_tuple), sub_shard_idx, shards_per_archive)
            for sub_shard_idx in range(shards_per_archive)
        )
    )


def plan_batches(
    model_ids: Sequence[str],
    shard_dir: Path,
    batch_size: int,
) -> tuple[BatchPlan, ...]:
    """Split sub-shard model IDs into contiguous processing batches."""
    _require_positive_int("batch_size", batch_size)
    model_id_tuple = tuple(model_ids)
    return tuple(
        BatchPlan(
            batch_index=batch_index,
            model_ids=model_id_tuple[batch_start : batch_start + batch_size],
            done_marker=shard_dir / f".batch_{batch_index}_done",
        )
        for batch_index, batch_start in enumerate(range(0, len(model_id_tuple), batch_size))
    )


def plan_sub_shards(
    archive_idx: int,
    model_ids: Sequence[str],
    shards_per_archive: int,
    shard_root: Path,
) -> tuple[SubShardPlan, ...]:
    """Build immutable plans for every logical sub-shard within an archive."""
    _require_int("archive_idx", archive_idx)
    if archive_idx < 0:
        msg = f"archive_idx must be non-negative, got {archive_idx}"
        raise IndexError(msg)
    _require_positive_int("shards_per_archive", shards_per_archive)

    total_model_count = len(model_ids)
    plans: list[SubShardPlan] = []
    for sub_shard_idx in range(shards_per_archive):
        model_start, model_end = sub_shard_model_range(total_model_count, sub_shard_idx, shards_per_archive)
        logical_shard_id = logical_shard_id_for_sub_shard(archive_idx, sub_shard_idx, shards_per_archive)
        plans.append(
            SubShardPlan(
                logical_shard_id=logical_shard_id,
                sub_shard_index=sub_shard_idx,
                model_start=model_start,
                model_end=model_end,
                shard_dir=shard_root / f"shard_{logical_shard_id}",
            ),
        )
    return tuple(plans)


def plan_archive_task(
    task_id: int,
    archive_names: Sequence[str],
    archive_model_ids: Sequence[str],
    shards_per_archive: int,
    shard_root: Path,
) -> ArchiveTaskPlan:
    """Build the archive task plan for a zero-based array task ID."""
    archive_idx = task_id_to_archive_idx(task_id, len(archive_names))
    return ArchiveTaskPlan(
        archive_index=archive_idx,
        archive_name=archive_names[archive_idx],
        shards_per_archive=shards_per_archive,
        logical_shards=plan_sub_shards(archive_idx, archive_model_ids, shards_per_archive, shard_root),
    )


__all__ = [
    "archive_active_lock",
    "archive_active_lock_path",
    "archive_active_marker_name",
    "logical_shard_id_for_sub_shard",
    "plan_archive_task",
    "plan_batches",
    "plan_sub_shards",
    "split_model_ids_for_archive",
    "sub_shard_model_range",
    "task_id_to_archive_idx",
]
