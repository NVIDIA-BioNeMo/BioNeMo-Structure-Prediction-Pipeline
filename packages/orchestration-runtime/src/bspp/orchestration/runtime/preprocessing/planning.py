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

"""Deterministic, non-executing preprocessing planning.

Port Baseline:
419813dbb5a3949e5e16f289f974d9f95e94bf01:utils/generate_tranches.sh:77-90,
419813dbb5a3949e5e16f289f974d9f95e94bf01:scripts/split_file.sh:3-10, and
419813dbb5a3949e5e16f289f974d9f95e94bf01:scripts/prepare_input_folder_structure.py:44-66.
"""

from __future__ import annotations

from pathlib import Path

from bspp.orchestration.contract.preprocessing import (
    PreprocessingChunk,
    PreprocessingChunkAssignment,
    PreprocessingFastaRecord,
    PreprocessingInput,
    PreprocessingPlanOptions,
    PreprocessingTranche,
    PreprocessingWorkPlan,
)
from bspp.orchestration.runtime.preprocessing.fasta import parse_preprocessing_fasta


def plan_preprocessing_fasta(path: Path, options: PreprocessingPlanOptions) -> PreprocessingWorkPlan:
    """Return a deterministic work plan for a declared FASTA path."""
    records = parse_preprocessing_fasta(path, normalization_mode=options.normalization_mode)
    preprocessing_input = PreprocessingInput(
        source_path=str(path),
        normalization_mode=options.normalization_mode,
        records=records,
    )
    return _build_work_plan(preprocessing_input, options, source_stem=path.stem)


def plan_preprocessing_records(
    *,
    source_path: str,
    records: tuple[PreprocessingFastaRecord, ...],
    options: PreprocessingPlanOptions,
) -> PreprocessingWorkPlan:
    """Plan an explicitly selected tuple, including deliberately empty work."""
    preprocessing_input = PreprocessingInput(
        source_path=source_path,
        normalization_mode=options.normalization_mode,
        records=records,
    )
    return _build_work_plan(preprocessing_input, options, source_stem=Path(source_path).stem)


def _build_work_plan(
    preprocessing_input: PreprocessingInput,
    options: PreprocessingPlanOptions,
    *,
    source_stem: str,
) -> PreprocessingWorkPlan:
    records = preprocessing_input.records
    if not records:
        return PreprocessingWorkPlan(
            input=preprocessing_input,
            options=options,
            tranches=(),
            chunks=(),
            assignments=(),
        )
    records_per_tranche = (len(records) + options.requested_tranches - 1) // options.requested_tranches
    tranches: list[PreprocessingTranche] = []
    chunks: list[PreprocessingChunk] = []
    for tranche_ordinal, start in enumerate(range(0, len(records), records_per_tranche)):
        tranche_records = records[start : start + records_per_tranche]
        tranche_name = f"tranche{tranche_ordinal:02d}"
        tranches.append(
            PreprocessingTranche(
                name=tranche_name,
                fasta_name=f"{source_stem}_{tranche_name}.fa",
                ordinal=tranche_ordinal,
                record_ordinals=tuple(record.source_ordinal for record in tranche_records),
            )
        )
        chunk_count = (len(tranche_records) + options.records_per_chunk - 1) // options.records_per_chunk
        if chunk_count > 100_000:
            msg = f"{tranche_name} requires {chunk_count} chunks; five-digit suffix capacity is 100000"
            raise ValueError(msg)
        for tranche_chunk_ordinal, chunk_start in enumerate(range(0, len(tranche_records), options.records_per_chunk)):
            chunk_records = tranche_records[chunk_start : chunk_start + options.records_per_chunk]
            chunks.append(
                PreprocessingChunk(
                    name=f"{source_stem}_{tranche_name}_{tranche_chunk_ordinal:05d}.fa",
                    tranche_name=tranche_name,
                    ordinal=len(chunks),
                    tranche_chunk_ordinal=tranche_chunk_ordinal,
                    record_ordinals=tuple(record.source_ordinal for record in chunk_records),
                )
            )
    return PreprocessingWorkPlan(
        input=preprocessing_input,
        options=options,
        tranches=tuple(tranches),
        chunks=tuple(chunks),
        assignments=_assign_chunks(tuple(chunks), options),
    )


def _assign_chunks(
    chunks: tuple[PreprocessingChunk, ...],
    options: PreprocessingPlanOptions,
) -> tuple[PreprocessingChunkAssignment, ...]:
    total_workers = options.nodes * options.gpus_per_node
    assignments: list[PreprocessingChunkAssignment] = []
    tranche_names = dict.fromkeys(chunk.tranche_name for chunk in chunks)
    for tranche_name in tranche_names:
        worker_counts = [0] * total_workers
        tranche_chunks = sorted(
            (chunk for chunk in chunks if chunk.tranche_name == tranche_name),
            key=lambda candidate: candidate.name,
        )
        for ordinal, chunk in enumerate(tranche_chunks):
            global_worker_index = ordinal % total_workers
            node_index = global_worker_index // options.gpus_per_node
            gpu_index = global_worker_index % options.gpus_per_node
            worker_label = f"n{node_index}g{gpu_index}"
            assignments.append(
                PreprocessingChunkAssignment(
                    chunk_name=chunk.name,
                    global_worker_index=global_worker_index,
                    node_index=node_index,
                    gpu_index=gpu_index,
                    worker_label=worker_label,
                    worker_ordinal=worker_counts[global_worker_index],
                    source_path=f"splitted/{chunk.name}",
                    staged_path=f"{worker_label}/{chunk.name}",
                )
            )
            worker_counts[global_worker_index] += 1
    return tuple(assignments)


__all__ = ["plan_preprocessing_fasta", "plan_preprocessing_records"]
