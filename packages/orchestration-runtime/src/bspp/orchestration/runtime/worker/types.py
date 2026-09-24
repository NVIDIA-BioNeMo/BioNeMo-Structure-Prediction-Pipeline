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

"""Shared native-worker data types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

CleanupPolicy = Literal["always", "on_success", "never"]


@dataclass(frozen=True, slots=True)
class TaskContext:
    """Runtime task identity resolved by CLI or SLURM glue."""

    job_id: str
    array_task_id: int
    array_task_count: int | None = None
    node_name: str | None = None
    dry_run: bool = False


@dataclass(frozen=True, slots=True)
class SubShardPlan:
    """One logical shard within an archive task."""

    logical_shard_id: int
    sub_shard_index: int
    model_start: int
    model_end: int
    shard_dir: Path


@dataclass(frozen=True, slots=True)
class ArchiveTaskPlan:
    """Archive selected by a task and the logical shards it owns."""

    archive_index: int
    archive_name: str
    shards_per_archive: int
    logical_shards: tuple[SubShardPlan, ...]


@dataclass(frozen=True, slots=True)
class BatchPlan:
    """A batch of model IDs within a logical shard."""

    batch_index: int
    model_ids: tuple[str, ...]
    done_marker: Path


@dataclass(frozen=True, slots=True)
class ScratchWorkspace:
    """Scratch directories used by one archive task."""

    root: Path
    input_dir: Path
    work_dir: Path
    cleanup_policy: CleanupPolicy = "always"


@dataclass(frozen=True, slots=True)
class PipelineCommand:
    """External production-pipeline invocation."""

    argv: tuple[str, ...]
    env: tuple[tuple[str, str], ...] = ()
    working_dir: Path | None = None


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Result of a production-pipeline subprocess run."""

    exit_code: int
    elapsed_seconds: float
    stdout_path: Path | None = None
    stderr_path: Path | None = None


@dataclass(frozen=True, slots=True)
class UploadResult:
    """Upload or local-fallback accounting for one transfer step."""

    success: bool
    uploaded_files: tuple[Path, ...]
    destination_prefix: str | None = None
    fallback_used: bool = False


@dataclass(frozen=True, slots=True)
class UploadedMarker:
    """Data written to a shard `.uploaded` marker."""

    s3_prefix: str | None
    total_files: int
    total_batches: int
    metadata_files: int
    timestamp: str
    shard_id: int
    model_count: int


@dataclass(frozen=True, slots=True)
class WorkerResult:
    """Top-level native archive task result."""

    archive_name: str
    logical_shards: tuple[int, ...]
    processed_models: int
    failed_models: tuple[str, ...]
    uploaded_files: int
    exit_code: int


__all__ = [
    "ArchiveTaskPlan",
    "BatchPlan",
    "CleanupPolicy",
    "PipelineCommand",
    "PipelineResult",
    "ScratchWorkspace",
    "SubShardPlan",
    "TaskContext",
    "UploadResult",
    "UploadedMarker",
    "WorkerResult",
]
