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

"""Pure folding queue strategy implementations.

Port Baseline:
``3864d0eda67e70979b8e48f00ed6a08f9e71c59e:folding/openfold-pipeline/scripts/preprocess_trt_bionemo.py:401-489``.
Runtime-model defaults come from
``3864d0eda67e70979b8e48f00ed6a08f9e71c59e:folding/openfold-pipeline/python/batch_optimizer.py:65-123``.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable
from dataclasses import dataclass

from bspp.orchestration.contract.folding_index import FoldingIndexRecord
from bspp.orchestration.contract.folding_queue import (
    FoldingQueueAssignment,
    FoldingQueueConfig,
    FoldingQueuePlan,
)


@dataclass(frozen=True)
class _ExactLengthBatch:
    length_batch: str
    batch_id: int
    sequence_length: int
    records: tuple[FoldingIndexRecord, ...]
    estimated_runtime: float


def plan_round_robin_assignments(records: Iterable[FoldingIndexRecord], config: FoldingQueueConfig) -> FoldingQueuePlan:
    """Apply the baseline length-sort and modulo assignment."""
    source_records = _validated_records(records)
    ordered = sorted(source_records, key=_source_sort_key)
    assignments = tuple(
        _assignment(
            record,
            worker_id=index % config.worker_count,
            batch_id=index,
        )
        for index, record in enumerate(ordered)
    )
    return FoldingQueuePlan(config=config, assignments=assignments)


def plan_runtime_balanced_assignments(
    records: Iterable[FoldingIndexRecord], config: FoldingQueueConfig
) -> FoldingQueuePlan:
    """Greedily assign descending pinned runtime estimates to least-loaded workers."""
    source_records = _validated_records(records)
    estimated = tuple(
        (
            max(
                config.runtime_coefficient * record.sequence_length * record.msa_depth + config.base_overhead_seconds,
                60.0,
            ),
            record,
        )
        for record in source_records
    )
    ordered = sorted(estimated, key=lambda item: (-item[0], item[1].sequence_length, item[1].source_ordinal))

    worker_loads = [(0.0, worker_id) for worker_id in range(config.worker_count)]
    heapq.heapify(worker_loads)
    assignments: list[FoldingQueueAssignment] = []
    for batch_id, (estimated_runtime, record) in enumerate(ordered):
        current_load, worker_id = heapq.heappop(worker_loads)
        assignments.append(_assignment(record, worker_id=worker_id, batch_id=batch_id))
        heapq.heappush(worker_loads, (current_load + estimated_runtime, worker_id))
    return FoldingQueuePlan(config=config, assignments=tuple(assignments))


def plan_exact_length_assignments(
    records: Iterable[FoldingIndexRecord], config: FoldingQueueConfig
) -> FoldingQueuePlan:
    """Create count-limited same-length batches and greedily place them."""
    source_records = _validated_records(records)
    ordered = sorted(source_records, key=_source_sort_key)
    batches = _create_exact_length_batches(ordered, max_proteins_per_batch=config.max_proteins_per_batch)

    worker_loads = [(0.0, worker_id) for worker_id in range(config.worker_count)]
    heapq.heapify(worker_loads)
    batch_workers: dict[str, int] = {}
    for batch in sorted(batches, key=lambda item: (-item.estimated_runtime, item.length_batch)):
        current_load, worker_id = heapq.heappop(worker_loads)
        batch_workers[batch.length_batch] = worker_id
        heapq.heappush(worker_loads, (current_load + batch.estimated_runtime, worker_id))

    assignments = [
        _assignment(
            record,
            worker_id=batch_workers[batch.length_batch],
            batch_id=batch.batch_id,
            length_batch=batch.length_batch,
        )
        for batch in batches
        for record in batch.records
    ]
    assignments.sort(key=lambda item: (item.worker_id, item.sequence_length, item.batch_id, item.source_ordinal))
    return FoldingQueuePlan(config=config, assignments=tuple(assignments))


def _create_exact_length_batches(
    ordered: list[FoldingIndexRecord], *, max_proteins_per_batch: int
) -> tuple[_ExactLengthBatch, ...]:
    grouped: dict[int, list[FoldingIndexRecord]] = {}
    for record in ordered:
        grouped.setdefault(record.sequence_length, []).append(record)

    batches: list[_ExactLengthBatch] = []
    batch_id = 0
    for sequence_length in sorted(grouped):
        length_records = grouped[sequence_length]
        for batch_in_length, start in enumerate(range(0, len(length_records), max_proteins_per_batch)):
            batch_records = tuple(length_records[start : start + max_proteins_per_batch])
            mean_msa_depth = sum(record.msa_depth for record in batch_records) / len(batch_records)
            estimated_runtime = sequence_length**1.5 * len(batch_records) * (1 + mean_msa_depth * 0.001)
            batches.append(
                _ExactLengthBatch(
                    length_batch=f"{sequence_length}_{batch_in_length}",
                    batch_id=batch_id,
                    sequence_length=sequence_length,
                    records=batch_records,
                    estimated_runtime=estimated_runtime,
                )
            )
            batch_id += 1
    return tuple(batches)


def _validated_records(records: Iterable[FoldingIndexRecord]) -> tuple[FoldingIndexRecord, ...]:
    source_records = tuple(records)
    if not all(isinstance(record, FoldingIndexRecord) for record in source_records):
        msg = "folding queue input must contain only FoldingIndexRecord values"
        raise ValueError(msg)
    _reject_duplicates((record.source_ordinal for record in source_records), field_name="source_ordinal")
    _reject_duplicates((record.protein_id for record in source_records), field_name="protein_id")
    _reject_duplicates((record.msa_path for record in source_records), field_name="msa_path")
    return source_records


def _reject_duplicates(values: Iterable[object], *, field_name: str) -> None:
    seen: set[object] = set()
    for value in values:
        if value in seen:
            msg = f"duplicate {field_name} {value!r} in folding queue input"
            raise ValueError(msg)
        seen.add(value)


def _source_sort_key(record: FoldingIndexRecord) -> tuple[int, int, str, str]:
    return (record.sequence_length, record.source_ordinal, record.protein_id, record.msa_path)


def _assignment(
    record: FoldingIndexRecord,
    *,
    worker_id: int,
    batch_id: int,
    length_batch: str | None = None,
) -> FoldingQueueAssignment:
    return FoldingQueueAssignment(
        source_ordinal=record.source_ordinal,
        worker_id=worker_id,
        batch_id=batch_id,
        sequence_length=record.sequence_length,
        msa_path=record.msa_path,
        protein_id=record.protein_id,
        length_batch=length_batch,
    )


__all__ = [
    "plan_exact_length_assignments",
    "plan_round_robin_assignments",
    "plan_runtime_balanced_assignments",
]
