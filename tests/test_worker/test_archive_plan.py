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

from pathlib import Path

import pytest

from bspp.orchestration.runtime.worker.archive_plan import (
    logical_shard_id_for_sub_shard,
    plan_archive_task,
    plan_batches,
    plan_sub_shards,
    split_model_ids_for_archive,
    sub_shard_model_range,
    task_id_to_archive_idx,
)


def _model_ids(count: int) -> list[str]:
    return [f"AF-{index:016d}" for index in range(1, count + 1)]


def test_task_id_to_archive_idx_maps_zero_based_array_tasks() -> None:
    assert task_id_to_archive_idx(0, 3) == 0
    assert task_id_to_archive_idx(2, 3) == 2


@pytest.mark.parametrize("task_id", [-1, 3])
def test_task_id_to_archive_idx_rejects_out_of_range(task_id: int) -> None:
    with pytest.raises(IndexError):
        task_id_to_archive_idx(task_id, 3)


@pytest.mark.parametrize("archive_count", [-1, True])
def test_task_id_to_archive_idx_rejects_invalid_archive_count(archive_count: int) -> None:
    with pytest.raises(ValueError):
        task_id_to_archive_idx(0, archive_count)


def test_logical_shard_id_uses_archive_major_formula() -> None:
    assert logical_shard_id_for_sub_shard(0, 0, 2) == 0
    assert logical_shard_id_for_sub_shard(0, 1, 2) == 1
    assert logical_shard_id_for_sub_shard(4, 0, 2) == 8
    assert logical_shard_id_for_sub_shard(4, 1, 2) == 9


@pytest.mark.parametrize(
    ("archive_idx", "sub_shard_idx", "shards_per_archive"),
    [(-1, 0, 2), (0, -1, 2), (0, 2, 2)],
)
def test_logical_shard_id_rejects_out_of_range_indexes(
    archive_idx: int,
    sub_shard_idx: int,
    shards_per_archive: int,
) -> None:
    with pytest.raises(IndexError):
        logical_shard_id_for_sub_shard(archive_idx, sub_shard_idx, shards_per_archive)


def test_split_model_ids_for_archive_uses_legacy_contiguous_slicing_for_two_shards() -> None:
    model_ids = _model_ids(8)

    assert split_model_ids_for_archive(model_ids, shards_per_archive=2) == (
        tuple(model_ids[:4]),
        tuple(model_ids[4:]),
    )


def test_split_model_ids_for_archive_keeps_final_partial_sub_shard() -> None:
    model_ids = _model_ids(5)

    assert split_model_ids_for_archive(model_ids, shards_per_archive=2) == (
        tuple(model_ids[:3]),
        tuple(model_ids[3:]),
    )
    assert sub_shard_model_range(total_model_count=5, sub_shard_idx=1, shards_per_archive=2) == (3, 5)


def test_split_model_ids_for_archive_returns_empty_trailing_sub_shards() -> None:
    model_ids = _model_ids(2)

    assert split_model_ids_for_archive(model_ids, shards_per_archive=4) == (
        (model_ids[0],),
        (model_ids[1],),
        (),
        (),
    )


@pytest.mark.parametrize(
    ("total_model_count", "shards_per_archive", "expected_ranges"),
    [
        (7, 3, ((0, 3), (3, 5), (5, 7))),
        (10, 3, ((0, 4), (4, 7), (7, 10))),
        (10, 4, ((0, 3), (3, 6), (6, 8), (8, 10))),
        (13, 4, ((0, 4), (4, 7), (7, 10), (10, 13))),
    ],
)
def test_split_model_ids_for_archive_uses_legacy_balanced_remainder_distribution(
    total_model_count: int,
    shards_per_archive: int,
    expected_ranges: tuple[tuple[int, int], ...],
) -> None:
    model_ids = _model_ids(total_model_count)

    assert (
        tuple(
            sub_shard_model_range(total_model_count, sub_shard_idx, shards_per_archive)
            for sub_shard_idx in range(shards_per_archive)
        )
        == expected_ranges
    )
    assert split_model_ids_for_archive(model_ids, shards_per_archive) == tuple(
        tuple(model_ids[model_start:model_end]) for model_start, model_end in expected_ranges
    )


def test_plan_batches_uses_contiguous_batches_and_done_markers(tmp_path: Path) -> None:
    model_ids = _model_ids(5)
    shard_dir = tmp_path / "shard_7"

    batches = plan_batches(model_ids, shard_dir, batch_size=2)

    assert [batch.batch_index for batch in batches] == [0, 1, 2]
    assert [batch.model_ids for batch in batches] == [
        tuple(model_ids[:2]),
        tuple(model_ids[2:4]),
        tuple(model_ids[4:]),
    ]
    assert [batch.done_marker for batch in batches] == [
        shard_dir / ".batch_0_done",
        shard_dir / ".batch_1_done",
        shard_dir / ".batch_2_done",
    ]


def test_plan_batches_rejects_invalid_batch_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        plan_batches(["AF-0000000000000001"], tmp_path / "shard_0", batch_size=0)


def test_plan_sub_shards_populates_ranges_and_shard_dirs(tmp_path: Path) -> None:
    model_ids = _model_ids(5)

    plans = plan_sub_shards(archive_idx=3, model_ids=model_ids, shards_per_archive=2, shard_root=tmp_path)

    assert [(plan.logical_shard_id, plan.sub_shard_index, plan.model_start, plan.model_end) for plan in plans] == [
        (6, 0, 0, 3),
        (7, 1, 3, 5),
    ]
    assert [plan.shard_dir for plan in plans] == [tmp_path / "shard_6", tmp_path / "shard_7"]


def test_plan_archive_task_uses_existing_dataclasses(tmp_path: Path) -> None:
    model_ids = _model_ids(8)

    plan = plan_archive_task(
        task_id=1,
        archive_names=["a.tar.lz4", "b.tar.lz4"],
        archive_model_ids=model_ids,
        shards_per_archive=2,
        shard_root=tmp_path,
    )

    assert plan.archive_index == 1
    assert plan.archive_name == "b.tar.lz4"
    assert plan.shards_per_archive == 2
    assert [(shard.logical_shard_id, shard.model_start, shard.model_end) for shard in plan.logical_shards] == [
        (2, 0, 4),
        (3, 4, 8),
    ]
