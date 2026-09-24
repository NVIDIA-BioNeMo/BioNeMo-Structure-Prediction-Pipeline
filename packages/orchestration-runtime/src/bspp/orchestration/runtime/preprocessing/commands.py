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

"""Pure preprocessing Scientific Kernel command rendering.

Port Baseline:
419813dbb5a3949e5e16f289f974d9f95e94bf01:scripts/msa_on_GPU.sh:48-61 and
419813dbb5a3949e5e16f289f974d9f95e94bf01:scripts/msa_batch.sh:12-29.
"""

from __future__ import annotations

from bspp.orchestration.contract.preprocessing import PreprocessingChunk, PreprocessingFastaRecord
from bspp.orchestration.contract.preprocessing_execution import (
    ExpectedA3M,
    PreprocessingChunkExecutionIntent,
    PreprocessingChunkExecutionPlan,
    PreprocessingEvidencePlan,
    PreprocessingRuntimeCoordinates,
    PreprocessingScientificConfig,
    PreprocessingScientificIntent,
    PreprocessingSiteConfig,
    PreprocessingSiteIntent,
    expected_member_name,
    materialize_preprocessing_chunk_execution_plan,
    member_name_conforms,
)
from bspp.orchestration.runtime.preprocessing.packaging import plan_preprocessing_package


def plan_preprocessing_chunk_execution(
    *,
    chunk: PreprocessingChunk,
    records: tuple[PreprocessingFastaRecord, ...],
    expected_a3m_members: tuple[str, ...],
    scientific: PreprocessingScientificConfig,
    site: PreprocessingSiteConfig,
    runtime: PreprocessingRuntimeCoordinates,
) -> PreprocessingChunkExecutionPlan:
    """Render exact pinned argv for one planned chunk without executing it."""
    if tuple(record.source_ordinal for record in records) != chunk.record_ordinals:
        msg = "source records must exactly match the chunk record ordinals"
        raise ValueError(msg)
    if len(expected_a3m_members) != len(records):
        msg = "expected A3M members must provide exactly one member per source record"
        raise ValueError(msg)
    if len(set(expected_a3m_members)) != len(expected_a3m_members):
        msg = "expected A3M member names must be unique"
        raise ValueError(msg)
    if scientific.require_afdb_model_id_stem:
        for record, member_name in zip(records, expected_a3m_members, strict=True):
            if not member_name_conforms(member_name):
                msg = (
                    "A3M member stem must carry a discoverable AFDB or PDB assembly model ID "
                    "(AFDB_AF[-_]<16 digits> or AFDB_AF[-_]<16 digits>_AF[-_]<16 digits> "
                    r"or pdb_[a-z0-9]+_assembly_\d+): "
                    f"{member_name}"
                )
                raise ValueError(msg)
            derived = expected_member_name(record.identity)
            if member_name != derived:
                msg = (
                    "A3M member name must equal the adapter-derived record-identity name "
                    f"(declared {member_name!r}, derived {derived!r})"
                )
                raise ValueError(msg)
    expected_a3ms = tuple(
        ExpectedA3M(
            chunk_name=chunk.name,
            record_identity=record.identity,
            source_ordinal=record.source_ordinal,
            source_header=record.header,
            member_name=member_name,
        )
        for record, member_name in zip(records, expected_a3m_members, strict=True)
    )
    input_folder = f"n{runtime.slurm_node_id}g{runtime.gpu_id}"
    scratch_output_directory = _join(
        site.scratch_output_root,
        f"{input_folder}_{runtime.submission_counter}",
    )
    raw_search_output_directory = _join(
        site.scratch_output_root,
        "raw-search",
        f"{input_folder}_{runtime.submission_counter}",
    )
    chunk_stem = chunk.name.removesuffix(".fa")
    scratch_log_directory = _join(
        site.scratch_output_root,
        "logs",
        f"{input_folder}_{runtime.submission_counter}",
    )
    evidence = PreprocessingEvidencePlan(
        chunk_name=chunk.name,
        raw_search_output_directory=raw_search_output_directory,
        scratch_output_directory=scratch_output_directory,
        scratch_log_directory=scratch_log_directory,
        scratch_log_path=_join(scratch_log_directory, f"{chunk_stem}.log"),
        scratch_record_path=_join(scratch_log_directory, f"{chunk_stem}.record"),
        a3m_record_glob=_join(scratch_output_directory, "*.a3m"),
        durable_log_path=_join(site.project_logs_root, f"{chunk_stem}.log"),
        durable_record_path=_join(site.project_logs_root, f"{chunk_stem}.record"),
    )
    package = plan_preprocessing_package(
        chunk_name=chunk.name,
        expected_a3ms=expected_a3ms,
        site=site,
        staging_directory=scratch_output_directory,
    )
    intent = PreprocessingChunkExecutionIntent(
        chunk_name=chunk.name,
        scientific=PreprocessingScientificIntent(
            schema_version=scientific.schema_version,
            max_sequences=scientific.max_sequences,
            use_env=scientific.use_env,
            require_afdb_model_id_stem=scientific.require_afdb_model_id_stem,
        ),
        site=PreprocessingSiteIntent(
            mmseqs_executable=site.mmseqs_executable,
            colabfold_search_executable=site.colabfold_search_executable,
            tar_executable=site.tar_executable,
            lz4_executable=site.lz4_executable,
            input_root=site.input_root,
            scratch_output_root=site.scratch_output_root,
            project_logs_root=site.project_logs_root,
            finished_msa_root=site.finished_msa_root,
            split_input_root=site.split_input_root,
            finished_input_root=site.finished_input_root,
            container_image=site.container_image,
            container_mounts=site.container_mounts,
            max_concurrency=site.max_concurrency,
            gpu_delay_seconds=site.gpu_delay_seconds,
        ),
        runtime=runtime,
        expected_a3ms=expected_a3ms,
        evidence=evidence,
        package=package,
    )
    return materialize_preprocessing_chunk_execution_plan(
        intent,
        selected_database_root=site.database_root,
        primary_database_name=scientific.primary_database_name,
        metagenomic_database_name=scientific.metagenomic_database_name,
    )


def _join(root: str, *parts: str) -> str:
    return "/".join((root.rstrip("/"), *parts))


__all__ = ["plan_preprocessing_chunk_execution"]
