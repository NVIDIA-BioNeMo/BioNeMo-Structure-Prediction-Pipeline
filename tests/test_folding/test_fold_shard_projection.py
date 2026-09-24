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

"""Contract and planner tests for the canonical fold shard projection."""

from __future__ import annotations

import hashlib

import pytest

from bspp.orchestration.contract.folding_shard import (
    FoldShardProjection,
    FoldShardProjectionBinding,
    FoldShardRank,
    FoldShardTarget,
    fold_shard_projection_binding_from_mapping,
    fold_shard_projection_document_bytes,
    fold_shard_projection_from_mapping,
)
from bspp.orchestration.control.folding_shard import FOLD_SHARD_LPT_VERSION, plan_fold_shards


def _targets() -> tuple[tuple[str, int], ...]:
    return (("A", 5), ("B", 3), ("C", 3), ("D", 2), ("E", 2), ("F", 1))


def test_lpt_is_deterministic_and_balanced() -> None:
    targets = _targets()
    first = plan_fold_shards(targets, 2)
    second = plan_fold_shards(targets, 2)
    assert first == second

    assert first.worker_count == 2
    assert first.lpt_version == FOLD_SHARD_LPT_VERSION
    rank_targets = {rank.global_rank: tuple(t.target_id for t in rank.targets) for rank in first.ranks}
    # Descending length order with target_id tie-break, then least-loaded rank
    # with rank-index tie-break, yields the exact LPT assignment below.
    assert rank_targets[0] == ("A", "D", "F")
    assert rank_targets[1] == ("B", "C", "E")
    loads = {rank.global_rank: sum(t.member_length for t in rank.targets) for rank in first.ranks}
    assert loads[0] == loads[1] == 8


def test_lpt_rank_index_tie_break_and_target_id_order() -> None:
    # Equal lengths sort by target_id; equal loads assign to the lowest rank.
    projection = plan_fold_shards((("B", 2), ("A", 2)), 2)
    rank_targets = {rank.global_rank: tuple(t.target_id for t in rank.targets) for rank in projection.ranks}
    assert rank_targets[0] == ("A",)
    assert rank_targets[1] == ("B",)


def test_projection_round_trips_through_mapping() -> None:
    projection = plan_fold_shards(_targets(), 2)
    reloaded = fold_shard_projection_from_mapping(projection.to_mapping())
    assert reloaded == projection
    assert reloaded.digest == projection.digest


def test_projection_document_bytes_are_stable() -> None:
    projection = plan_fold_shards(_targets(), 2)
    document = fold_shard_projection_document_bytes(projection)
    assert document == projection.to_json().encode()
    assert fold_shard_projection_document_bytes(plan_fold_shards(_targets(), 2)) == document


def test_binding_round_trips_and_binds_document_bytes() -> None:
    projection = plan_fold_shards(_targets(), 2)
    document = fold_shard_projection_document_bytes(projection)
    binding = FoldShardProjectionBinding(
        location="attempts/attempt-0001/fold-shard-projection.json",
        sha256=hashlib.sha256(document).hexdigest(),
        size_bytes=len(document),
        worker_count=2,
        lpt_version=FOLD_SHARD_LPT_VERSION,
    )
    reloaded = fold_shard_projection_binding_from_mapping(binding.to_mapping())
    assert reloaded == binding
    assert binding.sha256 == hashlib.sha256(document).hexdigest()
    assert binding.size_bytes == len(document)
    assert binding.worker_count == 2
    assert binding.lpt_version == FOLD_SHARD_LPT_VERSION


def test_target_validation_rejects_bad_records() -> None:
    with pytest.raises(ValueError, match="target_id"):
        FoldShardTarget(target_id="", member_length=1)
    with pytest.raises(ValueError, match="member_length"):
        FoldShardTarget(target_id="A", member_length=0)


def test_rank_validation_rejects_negative_rank() -> None:
    with pytest.raises(ValueError, match="global_rank"):
        FoldShardRank(global_rank=-1, targets=(FoldShardTarget(target_id="A", member_length=1),))


def test_projection_validation_rejects_non_covering_ranks() -> None:
    with pytest.raises(ValueError, match=r"cover 0\.\.worker_count-1"):
        FoldShardProjection(
            worker_count=2,
            lpt_version=1,
            ranks=(FoldShardRank(global_rank=0, targets=(FoldShardTarget(target_id="A", member_length=1),)),),
        )


def test_projection_validation_rejects_duplicate_targets() -> None:
    with pytest.raises(ValueError, match="unique across all ranks"):
        FoldShardProjection(
            worker_count=2,
            lpt_version=1,
            ranks=(
                FoldShardRank(global_rank=0, targets=(FoldShardTarget(target_id="A", member_length=1),)),
                FoldShardRank(global_rank=1, targets=(FoldShardTarget(target_id="A", member_length=1),)),
            ),
        )


def test_binding_validation_rejects_bad_sha256_and_size() -> None:
    with pytest.raises(ValueError, match="sha256"):
        FoldShardProjectionBinding(
            location="attempts/attempt-0001/fold-shard-projection.json",
            sha256="not-hex",
            size_bytes=0,
            worker_count=1,
            lpt_version=1,
        )
    with pytest.raises(ValueError, match="size_bytes"):
        FoldShardProjectionBinding(
            location="attempts/attempt-0001/fold-shard-projection.json",
            sha256="a" * 64,
            size_bytes=-1,
            worker_count=1,
            lpt_version=1,
        )


def test_projection_loader_rejects_unknown_fields() -> None:
    mapping = plan_fold_shards(_targets(), 2).to_mapping()
    mapping["bogus"] = True
    with pytest.raises(ValueError, match="Unknown FoldShardProjection field"):
        fold_shard_projection_from_mapping(mapping)
