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

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from bspp.orchestration.runtime.worker import (
    ArchiveTaskPlan,
    BatchPlan,
    PipelineCommand,
    PipelineResult,
    ScratchWorkspace,
    SubShardPlan,
    TaskContext,
    UploadedMarker,
    UploadResult,
    WorkerResult,
)


def test_worker_types_are_small_immutable_records(tmp_path: Path) -> None:
    sub_shard = SubShardPlan(
        logical_shard_id=557,
        sub_shard_index=1,
        model_start=5000,
        model_end=10000,
        shard_dir=tmp_path / "shard_557",
    )
    task_plan = ArchiveTaskPlan(
        archive_index=278,
        archive_name="bspp_260224_1859_00050.tar.lz4",
        shards_per_archive=2,
        logical_shards=(sub_shard,),
    )

    assert task_plan.logical_shards[0].logical_shard_id == 557
    with pytest.raises(FrozenInstanceError):
        sub_shard.logical_shard_id = 558  # type: ignore[misc]


def test_worker_types_cover_runtime_contract(tmp_path: Path) -> None:
    context = TaskContext(job_id="9614363", array_task_id=278, array_task_count=1043, node_name="batch-node")
    batch = BatchPlan(batch_index=4, model_ids=("AF-0000000210229946",), done_marker=tmp_path / ".batch_4_done")
    scratch = ScratchWorkspace(root=tmp_path, input_dir=tmp_path / "input", work_dir=tmp_path / "work")
    command = PipelineCommand(
        argv=("python", "production_pipeline.py", "--input", str(scratch.input_dir)),
        env=(("DUCKDB_MEMORY_LIMIT", "1GB"),),
        working_dir=scratch.work_dir,
    )
    pipeline = PipelineResult(exit_code=0, elapsed_seconds=51.28)
    upload = UploadResult(success=True, uploaded_files=(tmp_path / "shard_557_batch_4.tar",))
    marker = UploadedMarker(
        s3_prefix=None,
        total_files=6,
        total_batches=5,
        metadata_files=1,
        timestamp="2026-05-07T01:24:34Z",
        shard_id=557,
        model_count=5000,
    )
    worker = WorkerResult(
        archive_name="bspp_260224_1859_00050.tar.lz4",
        logical_shards=(556, 557),
        processed_models=10000,
        failed_models=("AF-0000000210229946",),
        uploaded_files=12,
        exit_code=0,
    )

    assert context.array_task_id == 278
    assert batch.model_ids == ("AF-0000000210229946",)
    assert scratch.cleanup_policy == "always"
    assert command.env == (("DUCKDB_MEMORY_LIMIT", "1GB"),)
    assert pipeline.exit_code == 0
    assert upload.fallback_used is False
    assert marker.total_batches == 5
    assert worker.logical_shards == (556, 557)
