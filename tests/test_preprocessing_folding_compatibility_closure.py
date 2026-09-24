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

"""Local public-seam acceptance for preprocessing-to-folding compatibility closure.

The synthetic files below prove deterministic orchestration semantics only. They
do not execute or validate either phase's Scientific Kernel.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from bspp.orchestration.contract import (
    FoldingArchiveLifecycleRequest,
    FoldingCheckpointInputs,
    FoldingCheckpointState,
    FoldingIndex,
    FoldingIndexLifecycleRequest,
    FoldingLifecycleConfig,
    FoldingManifestEvidence,
    FoldingMergeLifecycleRequest,
    FoldingPreflightLifecycleRequest,
    FoldingQueueConfig,
    FoldingQueueLayout,
    FoldingQueueStrategy,
    FoldingResumeLifecycleRequest,
    FoldingStatusLifecycleRequest,
    FoldingSubmitLifecycleRequest,
    folding_index_from_mapping,
    folding_lifecycle_config_from_mapping,
    folding_queue_plan_from_mapping,
    folding_queue_render_result_from_mapping,
)
from bspp.orchestration.contract.preprocessing import (
    PreprocessingChunk,
    PreprocessingPlanOptions,
    PreprocessingWorkPlan,
)
from bspp.orchestration.contract.preprocessing_execution import (
    PreprocessingChunkExecutionPlan,
    PreprocessingRuntimeCoordinates,
    PreprocessingScientificConfig,
    PreprocessingSiteConfig,
)
from bspp.orchestration.contract.preprocessing_state import (
    PreprocessingArchiveEvidence,
    PreprocessingChunkState,
    PreprocessingPairedEvidence,
    PreprocessingSourceMembership,
)
from bspp.orchestration.runtime import folding as folding_runtime
from bspp.orchestration.runtime.preprocessing.commands import plan_preprocessing_chunk_execution
from bspp.orchestration.runtime.preprocessing.planning import plan_preprocessing_fasta
from bspp.orchestration.runtime.preprocessing.retry import plan_preprocessing_retries
from bspp.orchestration.runtime.preprocessing.state import interpret_preprocessing_chunk_state


def _write_equivalent_fasta_fixtures(tmp_path: Path) -> tuple[Path, Path]:
    records = tuple((f"protein-{ordinal:02d}", "ACD" + "E" * ordinal) for ordinal in range(9))
    strict = tmp_path / "strict" / "proteins.fa"
    multiline = tmp_path / "multiline" / "proteins.fa"
    strict.parent.mkdir(parents=True)
    multiline.parent.mkdir(parents=True)
    strict.write_text("".join(f">{identity}\n{sequence}\n" for identity, sequence in records))
    multiline.write_text("".join(f">{identity}\n{sequence[:2]}\n{sequence[2:]}\n" for identity, sequence in records))
    return strict, multiline


def _preprocessing_execution_plans(
    tmp_path: Path,
) -> tuple[PreprocessingWorkPlan, tuple[PreprocessingChunkExecutionPlan, ...]]:
    strict, _multiline = _write_equivalent_fasta_fixtures(tmp_path)
    work_plan = plan_preprocessing_fasta(
        strict,
        PreprocessingPlanOptions(
            requested_tranches=1,
            records_per_chunk=3,
            nodes=1,
            gpus_per_node=3,
            normalization_mode="strict-two-line",
        ),
    )
    scientific = PreprocessingScientificConfig(
        primary_database_name="uniref30_2302_db",
        metagenomic_database_name="colabfold_envdb_202108_db",
        max_sequences=10_000,
        require_afdb_model_id_stem=False,
    )
    site = PreprocessingSiteConfig(
        mmseqs_executable="/opt/bin/mmseqs",
        colabfold_search_executable="/opt/bin/colabfold_search",
        tar_executable="/usr/bin/tar",
        lz4_executable="/usr/bin/lz4",
        database_root=str(tmp_path / "declared-databases"),
        input_root=str(tmp_path / "declared-input"),
        scratch_output_root=str(tmp_path / "declared-scratch"),
        project_logs_root=str(tmp_path / "declared-logs"),
        finished_msa_root=str(tmp_path / "declared-finished-msa"),
        split_input_root=str(tmp_path / "declared-split"),
        finished_input_root=str(tmp_path / "declared-finished-input"),
        container_image=str(tmp_path / "declared-container.sqsh"),
        container_mounts=("/databases",),
        max_concurrency=3,
        gpu_delay_seconds=0,
    )
    executions: list[PreprocessingChunkExecutionPlan] = []
    for chunk, assignment in zip(work_plan.chunks, work_plan.assignments, strict=True):
        records = tuple(work_plan.input.records[ordinal] for ordinal in chunk.record_ordinals)
        executions.append(
            plan_preprocessing_chunk_execution(
                chunk=chunk,
                records=records,
                expected_a3m_members=tuple(f"AFDB_{record.identity}.a3m" for record in records),
                scientific=scientific,
                site=site,
                runtime=PreprocessingRuntimeCoordinates(
                    slurm_node_id=assignment.node_index,
                    gpu_id=assignment.gpu_index,
                    submission_counter=assignment.worker_ordinal,
                ),
            )
        )
    return work_plan, tuple(executions)


def _source(
    execution: PreprocessingChunkExecutionPlan, *, split: bool, finished: bool
) -> PreprocessingSourceMembership:
    package = execution.package
    chunk = execution.chunk_name
    stem = chunk.removesuffix(".fa")
    return PreprocessingSourceMembership(
        chunk_name=chunk,
        pristine_path=f"/pristine/{chunk}",
        pristine_present=True,
        pristine_record_count=len(execution.expected_a3ms),
        split_path=package.completed_input_source_path,
        split_present=split,
        finished_input_path=package.completed_input_path,
        finished_input_present=finished,
        shared_finished_read_tar_path=f"/shared-read/{stem}.tar",
        shared_finished_read_present=False,
        shared_finished_write_tar_path=f"/shared-write/{stem}.tar",
        shared_finished_write_present=False,
    )


def _paired(
    execution: PreprocessingChunkExecutionPlan,
    *,
    members: tuple[str, ...] | None,
    log_lines: tuple[str, ...] | None,
) -> PreprocessingPairedEvidence:
    record_lines = (
        None
        if members is None
        else tuple(f"-rw-r--r-- 1 user group 10 Aug 14 00:00 /output/{member}" for member in members)
    )
    return PreprocessingPairedEvidence(
        chunk_name=execution.chunk_name,
        durable_record_path=execution.evidence.durable_record_path,
        durable_log_path=execution.evidence.durable_log_path,
        record_lines=record_lines,
        log_lines=log_lines,
    )


def _archive(
    execution: PreprocessingChunkExecutionPlan,
    *,
    members: tuple[str, ...] | None,
    tar_size_bytes: int | None = None,
    lz4_size_bytes: int | None = None,
) -> PreprocessingArchiveEvidence:
    return PreprocessingArchiveEvidence(
        chunk_name=execution.chunk_name,
        durable_tar_path=execution.package.durable_tar_path,
        durable_lz4_path=execution.package.durable_lz4_path,
        tar_size_bytes=None if members is None else (4096 if tar_size_bytes is None else tar_size_bytes),
        lz4_size_bytes=None if members is None else (1024 if lz4_size_bytes is None else lz4_size_bytes),
        tar_members=None if members is None else tuple(f"./{member}" for member in members),
    )


def _interpret(
    chunk: PreprocessingChunk,
    execution: PreprocessingChunkExecutionPlan,
    *,
    source: PreprocessingSourceMembership,
    members: tuple[str, ...] | None,
    log_lines: tuple[str, ...] | None,
    archive_members: tuple[str, ...] | None,
    tar_size_bytes: int | None = None,
    lz4_size_bytes: int | None = None,
) -> PreprocessingChunkState:
    return interpret_preprocessing_chunk_state(
        chunk=chunk,
        expected_a3ms=execution.expected_a3ms,
        evidence_plan=execution.evidence,
        package_plan=execution.package,
        expected_num_records=len(execution.expected_a3ms),
        source=source,
        paired_evidence=_paired(execution, members=members, log_lines=log_lines),
        archive_evidence=_archive(
            execution,
            members=archive_members,
            tar_size_bytes=tar_size_bytes,
            lz4_size_bytes=lz4_size_bytes,
        ),
    )


def test_preprocessing_normalization_commands_and_package_plans_are_deterministic(tmp_path: Path) -> None:
    strict_path, multiline_path = _write_equivalent_fasta_fixtures(tmp_path)
    strict_options = PreprocessingPlanOptions(
        requested_tranches=1,
        records_per_chunk=3,
        nodes=1,
        gpus_per_node=3,
        normalization_mode="strict-two-line",
    )
    multiline_options = replace(strict_options, normalization_mode="normalize-multiline")

    strict = plan_preprocessing_fasta(strict_path, strict_options)
    multiline = plan_preprocessing_fasta(multiline_path, multiline_options)

    assert tuple((item.identity, item.sequence) for item in strict.input.records) == tuple(
        (item.identity, item.sequence) for item in multiline.input.records
    )
    assert tuple(item.identity for item in strict.input.records) == tuple(f"protein-{index:02d}" for index in range(9))
    assert strict.chunks == multiline.chunks
    assert strict.assignments == multiline.assignments
    assert plan_preprocessing_fasta(strict_path, strict_options) == strict

    duplicate = tmp_path / "duplicate.fa"
    duplicate.write_text(">same\nAAA\n>same\nBBB\n")
    with pytest.raises(ValueError, match="duplicate FASTA identity"):
        plan_preprocessing_fasta(duplicate, strict_options)

    before = set(tmp_path.rglob("*"))
    work_plan, executions = _preprocessing_execution_plans(tmp_path / "execution")
    assert len(executions) == 3
    for execution in executions:
        records = tuple(
            work_plan.input.records[ordinal]
            for ordinal in next(
                chunk for chunk in work_plan.chunks if chunk.name == execution.chunk_name
            ).record_ordinals
        )
        replay = plan_preprocessing_chunk_execution(
            chunk=next(chunk for chunk in work_plan.chunks if chunk.name == execution.chunk_name),
            records=records,
            expected_a3m_members=tuple(item.member_name for item in execution.expected_a3ms),
            scientific=execution.scientific,
            site=execution.site,
            runtime=execution.runtime,
        )
        assert replay == execution
        assert execution.gpuserver_argv[:2] == ("/opt/bin/mmseqs", "gpuserver")
        assert execution.search_argv[:3] == ("/opt/bin/colabfold_search", "--mmseqs", "/opt/bin/mmseqs")
        assert execution.package.declared_stage_members == tuple(item.member_name for item in execution.expected_a3ms)
        assert execution.evidence.durable_log_path.rsplit("/", 1)[-1] == execution.chunk_name.replace(".fa", ".log")
        assert execution.evidence.durable_record_path.rsplit("/", 1)[-1] == execution.chunk_name.replace(
            ".fa", ".record"
        )
    created_by_planning = set(tmp_path.rglob("*")) - before
    assert created_by_planning == {
        tmp_path / "execution",
        tmp_path / "execution" / "strict",
        tmp_path / "execution" / "strict" / "proteins.fa",
        tmp_path / "execution" / "multiline",
        tmp_path / "execution" / "multiline" / "proteins.fa",
    }
    assert not any(
        Path(path).exists()
        for execution in executions
        for path in (
            execution.package.scratch_tar_path,
            execution.package.scratch_lz4_path,
            execution.package.durable_tar_path,
            execution.package.durable_lz4_path,
            execution.package.completed_input_path,
            execution.evidence.durable_log_path,
            execution.evidence.durable_record_path,
        )
    )
    assert not any(
        Path(path).exists()
        for execution in executions
        for path in (
            execution.site.database_root,
            execution.site.input_root,
            execution.site.scratch_output_root,
            execution.site.project_logs_root,
            execution.site.finished_msa_root,
            execution.site.split_input_root,
            execution.site.finished_input_root,
            execution.site.container_image,
        )
    )


def test_preprocessing_state_authority_retry_and_completion_close_locally(tmp_path: Path) -> None:
    work_plan, executions = _preprocessing_execution_plans(tmp_path)
    declared = tuple(item.member_name for item in executions[0].expected_a3ms)

    completed = _interpret(
        work_plan.chunks[0],
        executions[0],
        source=_source(executions[0], split=False, finished=True),
        members=declared,
        log_lines=("search complete",),
        archive_members=(*declared, "scratch-directory/"),
    )
    # Under n-arity relaxation, the <=2 a3m_count_not_above_two check is removed.
    # Substitute a still-reachable invalid reason (tar_empty) so the retry plan
    # again contains both an invalid and a retryable chunk (N4).
    invalid_members = tuple(item.member_name for item in executions[1].expected_a3ms)
    invalid_chunk = _interpret(
        work_plan.chunks[1],
        executions[1],
        source=_source(executions[1], split=False, finished=True),
        members=invalid_members,
        log_lines=("search complete",),
        archive_members=invalid_members,
        tar_size_bytes=0,
        lz4_size_bytes=0,
    )
    retry_members = tuple(item.member_name for item in executions[2].expected_a3ms)
    retryable = _interpret(
        work_plan.chunks[2],
        executions[2],
        source=_source(executions[2], split=False, finished=True),
        members=retry_members,
        log_lines=("Skipping query after transient failure",),
        archive_members=retry_members,
    )

    assert (completed.state, completed.reason_codes, completed.eligible_for_retry) == (
        "completed",
        ("complete",),
        False,
    )
    assert completed.extra_tar_members == ("scratch-directory/",)
    assert (invalid_chunk.state, invalid_chunk.reason_codes, invalid_chunk.eligible_for_retry) == (
        "invalid",
        ("tar_empty", "tar_lz4_empty"),
        True,
    )
    assert (retryable.state, retryable.reason_codes, retryable.eligible_for_retry) == (
        "retryable",
        ("search_skipped",),
        True,
    )
    assert tuple(action.operation for action in invalid_chunk.retry_actions) == ("copy", "remove", "remove")
    assert tuple(action.operation for action in retryable.retry_actions) == ("copy", "remove", "remove")

    partial = _interpret(
        work_plan.chunks[2],
        executions[2],
        source=_source(executions[2], split=False, finished=False),
        members=None,
        log_lines=("partial evidence",),
        archive_members=retry_members,
    )
    assert (partial.state, partial.reason_codes) == ("retryable", ("missing_record_evidence",))
    assert tuple(action.operation for action in partial.retry_actions) == ("copy", "remove", "remove")
    assert all(not Path(action.target_path).exists() for action in partial.retry_actions)
    with pytest.raises(FrozenInstanceError):
        partial.retry_actions[0].operation = "remove"  # type: ignore[misc]

    valid_source = _source(executions[0], split=False, finished=True)
    authority_cases = (
        (replace(valid_source, pristine_record_count=2), "pristine_record_count"),
        (replace(valid_source, split_path=f"/other/{executions[0].chunk_name}"), "split membership"),
        (replace(valid_source, finished_input_path=f"/other/{executions[0].chunk_name}"), "finished membership"),
    )
    for source, message in authority_cases:
        with pytest.raises(ValueError, match=message):
            _interpret(
                work_plan.chunks[0],
                executions[0],
                source=source,
                members=declared,
                log_lines=("search complete",),
                archive_members=declared,
            )

    retry_plan = plan_preprocessing_retries(work_plan=work_plan, states=(completed, invalid_chunk, retryable))
    assert retry_plan == plan_preprocessing_retries(work_plan=work_plan, states=(completed, invalid_chunk, retryable))
    assert retry_plan.eligible_chunk_names == (work_plan.chunks[1].name, work_plan.chunks[2].name)
    assert retry_plan.invalid_chunk_names == (work_plan.chunks[1].name,)
    assert tuple(action.operation for action in retry_plan.actions) == (
        "copy",
        "remove",
        "remove",
        "copy",
        "remove",
        "remove",
    )
    assert work_plan.chunks[0].name not in retry_plan.eligible_chunk_names

    all_completed = tuple(
        _interpret(
            chunk,
            execution,
            source=_source(execution, split=False, finished=True),
            members=tuple(item.member_name for item in execution.expected_a3ms),
            log_lines=("search complete",),
            archive_members=tuple(item.member_name for item in execution.expected_a3ms),
        )
        for chunk, execution in zip(work_plan.chunks, executions, strict=True)
    )
    finished = plan_preprocessing_retries(work_plan=work_plan, states=all_completed)
    assert finished.eligible_chunk_names == ()
    assert finished.actions == ()
    assert tuple(item.completion_percentage for item in finished.tranche_progress) == ("100.00",)
    assert tuple(item.validated_completion_percentage for item in finished.tranche_progress) == ("100.00",)


def _write_a3ms(tmp_path: Path) -> tuple[Path, ...]:
    root = tmp_path / "a3m"
    root.mkdir(parents=True, exist_ok=True)
    paths = tuple(root / f"protein-{ordinal:02d}.a3m" for ordinal in range(3))
    for ordinal, path in enumerate(paths):
        sequence = "A" * (ordinal + 3)
        path.write_text(f"#{len(sequence)} 1\n>query\n{sequence}\n>hit\n{sequence}\n")
    return paths


def _folding_config(
    tmp_path: Path,
    paths: tuple[Path, ...],
    **overrides: object,
) -> FoldingLifecycleConfig:
    batch_info = tmp_path / "batch_info.parquet"
    container = tmp_path / "openfold.sqsh"
    trt_root = tmp_path / "trt-bionemo"
    engines = tmp_path / "trt-engines"
    checkpoint = tmp_path / "params.pt"
    tmp_path.mkdir(parents=True, exist_ok=True)
    batch_info.write_bytes(b"fixture")
    container.write_bytes(b"fixture")
    trt_root.mkdir(exist_ok=True)
    engines.mkdir(exist_ok=True)
    checkpoint.write_bytes(b"fixture")
    payload: dict[str, object] = {
        "schema_version": 1,
        "run_name": "closure",
        "declared_a3m_inputs": [str(path) for path in paths],
        "batch_info_path": str(batch_info),
        "output_dir": str(tmp_path / "output"),
        "work_dir": str(tmp_path / "queue-writes"),
        "predictions_root": str(tmp_path / "predictions"),
        "checkpoint_dir": str(tmp_path / "declared-checkpoints"),
        "container_image": str(container),
        "trt_bionemo_dir": str(trt_root),
        "trt_engines_dir": str(engines),
        "trt_checkpoint_path": str(checkpoint),
        "model_preset": "model_1_multimer_v3",
        "max_recycling_iters": 4,
        "trt_fallback_threshold": 1536,
        "preprocessing_enabled": True,
        "sort_by_length": False,
        "index_worker_count": 2,
        "queue_strategy": "runtime_balanced",
        "queue_layout": "per_node",
        "node_count": 2,
        "gpus_per_node": 2,
        "worker_count": 2,
        "cpus_per_task": 4,
        "max_proteins_per_batch": 2,
        "job_name": "closure",
        "account": "local",
        "partition": "local",
        "walltime": "00:10:00",
        "archive_enabled": True,
        "archive_run_tag": "closure",
        "archive_stage_root": str(tmp_path / "declared-stage"),
        "archive_root": str(tmp_path / "declared-archives"),
        "proteins_per_archive": 2,
        "archive_start_index": 0,
        "archive_max_archives": None,
        "archive_shuffle": False,
        "archive_shuffle_seed": 42,
        "lz4_executable": "lz4",
    }
    payload.update(overrides)
    return folding_lifecycle_config_from_mapping(payload)


def _folding_index(tmp_path: Path) -> tuple[tuple[Path, ...], FoldingIndex]:
    paths = _write_a3ms(tmp_path)
    return paths, folding_runtime.index_folding_a3ms(paths, sort_by_length=False)


def test_folding_index_lifecycle_and_historical_batch_normalization_agree(tmp_path: Path) -> None:
    paths, index = _folding_index(tmp_path)
    config = _folding_config(tmp_path / "config", paths)
    lifecycle = folding_runtime.plan_folding_index_lifecycle(FoldingIndexLifecycleRequest(config=config))

    assert tuple(record.protein_id for record in index.records) == ("protein-00", "protein-01", "protein-02")
    assert tuple(record.sequence_length for record in index.records) == (3, 4, 5)
    assert lifecycle.index == index
    assert folding_index_from_mapping(index.to_mapping()) == index
    assert folding_runtime.index_folding_a3ms(paths, sort_by_length=False) == index

    historical = folding_runtime.normalize_batch_info(
        (
            {"path": [str(paths[0]), str(paths[1])], "seq_length": 97},
            {"protein_id": "protein-02", "path": str(paths[2]), "total_length": 5},
        ),
        sort_by_length=False,
    )
    assert tuple(record.protein_id for record in historical.records) == ("protein-00", "protein-01", "protein-02")
    assert tuple(record.source_ordinal for record in historical.records) == (0, 1, 2)
    assert tuple(record.sequence_length for record in historical.records) == (97, 97, 5)
    assert tuple(record.total_length for record in historical.records) == (3, 4, 5)


@pytest.mark.parametrize(
    ("strategy", "layout", "expected_assignments"),
    (
        (
            "round_robin",
            "per_node",
            (("protein-00", 0, 0, 0), ("protein-01", 1, 1, 1), ("protein-02", 0, 2, 2)),
        ),
        (
            "exact_length",
            "legacy_per_gpu",
            (("protein-02", 0, 2, 2), ("protein-00", 1, 0, 0), ("protein-01", 1, 1, 1)),
        ),
        (
            "runtime_balanced",
            "per_node",
            (("protein-00", 0, 0, 0), ("protein-01", 1, 1, 1), ("protein-02", 0, 2, 2)),
        ),
    ),
)
def test_all_folding_queue_strategies_and_layouts_render_without_writes(
    tmp_path: Path,
    strategy: FoldingQueueStrategy,
    layout: FoldingQueueLayout,
    expected_assignments: tuple[tuple[str, int, int, int], ...],
) -> None:
    _paths, index = _folding_index(tmp_path)
    config = FoldingQueueConfig(
        strategy=strategy,
        layout=layout,
        worker_count=2,
        max_proteins_per_batch=2,
    )
    plan = folding_runtime.plan_folding_queue(index.records, config)
    rendered = folding_runtime.render_folding_queues(plan)
    manifest = folding_runtime.render_folding_queue_manifest(
        plan,
        created="2026-08-14T00:00:00Z",
        batch_info_path="/declared/batch_info.parquet",
        output_dir="/declared/output",
        num_gpus=2,
        num_nodes=2 if layout == "per_node" else None,
        target_runtime_hours=1.0,
        max_total_residues=10_000,
        force_rerun=False,
        rebalance=False,
        skip_completed=True,
    )

    assert (
        tuple(
            (assignment.protein_id, assignment.worker_id, assignment.batch_id, assignment.source_ordinal)
            for assignment in plan.assignments
        )
        == expected_assignments
    )
    assert {assignment.protein_id for assignment in plan.assignments} == {
        "protein-00",
        "protein-01",
        "protein-02",
    }
    assert all(
        artifact.content.startswith("batch_id,seq_length,msa_path,protein_id\r\n") for artifact in rendered.artifacts
    )
    assert all("\n" not in artifact.content.replace("\r\n", "") for artifact in rendered.artifacts)
    assert '"total_proteins": 3' in manifest
    assert folding_queue_plan_from_mapping(plan.to_mapping()) == plan
    assert folding_queue_render_result_from_mapping(rendered.to_mapping()) == rendered
    assert folding_runtime.plan_folding_queue(index.records, config) == plan
    assert not any((tmp_path / artifact.file_name).exists() for artifact in rendered.artifacts)


def _checkpoint_inputs(tmp_path: Path) -> FoldingCheckpointInputs:
    tmp_path.mkdir(parents=True)
    node_completed = tmp_path / "completed_node_0.csv"
    node_completed.write_text(
        "protein_id,runtime_seconds,timestamp,node_id,gpu_id\nprotein-00,1,2026-08-14T00:00:00,node-0,0\n"
    )
    current_failed = tmp_path / "failed_gpu_current_0.csv"
    current_failed.write_text(
        "protein_id,error_message,timestamp,node_id,gpu_id\nprotein-01,TIMEOUT,2026-08-14T00:01:00,node-0,1\n"
    )
    historical_failed = tmp_path / "failed_gpu_historical_0.csv"
    historical_failed.write_text(
        "protein_id,error_message,timestamp\nprotein-02,CUDA_OUT_OF_MEMORY,2026-08-14T00:02:00\n"
    )
    merged_completed = tmp_path / "completed_all.csv"
    merged_completed.write_text(
        "protein_id,runtime_seconds,timestamp,node_id,gpu_id\nprotein-00,4,2026-08-14T00:03:00,node-global,0\n"
    )
    return FoldingCheckpointInputs(
        node_completion_shards=(str(node_completed),),
        node_failure_shards=(),
        current_gpu_completion_shards=(),
        current_gpu_failure_shards=(str(current_failed),),
        historical_gpu_completion_shards=(),
        historical_gpu_failure_shards=(str(historical_failed),),
        merged_completion_view=str(merged_completed),
        merged_failure_view=None,
    )


def _write_result_pair(root: Path, protein_id: str, *, json: bool = True) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{protein_id}_unrelaxed_rank_001_model.pdb").write_text("SYNTHETIC PDB\n")
    if json:
        (root / f"{protein_id}_scores_rank_001_model.json").write_text("{}\n")


def test_folding_checkpoint_materialization_result_recovery_and_resume_are_coherent(tmp_path: Path) -> None:
    paths, index = _folding_index(tmp_path / "folding")
    config = _folding_config(tmp_path / "config", paths)
    merge = folding_runtime.plan_folding_merge_lifecycle(
        FoldingMergeLifecycleRequest(config=config, checkpoint_inputs=_checkpoint_inputs(tmp_path / "checkpoints"))
    )

    assert tuple(record.protein_id for record in merge.state.completions) == ("protein-00",)
    assert merge.state.completions[0].runtime_seconds == 4
    assert merge.state.completions[0].source_ordinal == 2
    assert tuple(record.protein_id for record in merge.state.failures) == ("protein-01", "protein-02")
    assert merge.completed_view_path.endswith("/declared-checkpoints/completed_all.csv")
    assert merge.failed_view_path.endswith("/declared-checkpoints/failed_all.csv")
    assert not Path(merge.completed_view_path).exists()
    assert not Path(merge.failed_view_path).exists()

    materialized = tmp_path / "materialized"
    materialized.mkdir()
    completed_path = materialized / "completed_all.csv"
    failed_path = materialized / "failed_all.csv"
    folding_runtime.write_merged_checkpoint_views(
        merge.state,
        completed_path=completed_path,
        failed_path=failed_path,
    )
    completed_bytes = completed_path.read_bytes()
    failed_bytes = failed_path.read_bytes()
    assert completed_bytes == (
        b"protein_id,runtime_seconds,timestamp,node_id,gpu_id\r\nprotein-00,4.0,2026-08-14T00:03:00,node-global,0\r\n"
    )
    assert failed_bytes == (
        b"protein_id,error_message,timestamp,node_id,gpu_id\r\n"
        b"protein-01,TIMEOUT,2026-08-14T00:01:00,node-0,1\r\n"
        b"protein-02,CUDA_OUT_OF_MEMORY,2026-08-14T00:02:00,,\r\n"
    )
    reloaded = folding_runtime.load_checkpoint_state(
        current_node_completion_shards=(completed_path,),
        current_node_failure_shards=(failed_path,),
    )
    folding_runtime.write_merged_checkpoint_views(
        reloaded,
        completed_path=completed_path,
        failed_path=failed_path,
    )
    assert (
        folding_runtime.load_checkpoint_state(
            current_node_completion_shards=(completed_path,),
            current_node_failure_shards=(failed_path,),
        )
        == reloaded
    )

    results = tmp_path / "results"
    _write_result_pair(results, "protein-00")
    _write_result_pair(results, "protein-01", json=False)
    (results / "config.json").write_text("{}\n")
    inventory = folding_runtime.scan_folding_results(index, results)
    assert tuple(item.status for item in inventory.associations) == ("complete", "pdb-only", "missing")
    assert tuple(item.reason for item in inventory.unmatched) == ("unplanned-identity",)
    resume = folding_runtime.plan_folding_resume_lifecycle(
        FoldingResumeLifecycleRequest(
            config=config,
            index=index,
            checkpoint_state=merge.state,
            result_inventory=inventory,
            retry_failed=False,
        )
    )
    assert resume.recovered_result_ids == ("protein-00",)
    assert resume.remaining_work.completed_model_ids == ("protein-00",)
    assert resume.remaining_work.transient_failure_ids == ("protein-01",)
    assert resume.remaining_work.permanent_failure_ids == ("protein-02",)
    assert resume.remaining_work.remaining_model_ids == ("protein-01",)
    retried = folding_runtime.plan_folding_resume_lifecycle(
        replace(
            FoldingResumeLifecycleRequest(
                config=config,
                index=index,
                checkpoint_state=merge.state,
                result_inventory=inventory,
                retry_failed=False,
            ),
            retry_failed=True,
        )
    )
    assert retried.remaining_work.remaining_model_ids == ("protein-01", "protein-02")

    (results / "protein-01_scores_rank_001_model.json").write_text("{}\n")
    _write_result_pair(results, "protein-02")
    complete_inventory = folding_runtime.scan_folding_results(index, results)
    complete = folding_runtime.plan_folding_resume_lifecycle(
        FoldingResumeLifecycleRequest(
            config=config,
            index=index,
            checkpoint_state=merge.state,
            result_inventory=complete_inventory,
            retry_failed=False,
        )
    )
    assert complete.remaining_work.remaining_model_ids == ()
    assert complete.submit_decision.queue_writes == ()
    assert complete.submit_decision.execution_authorized is False


def test_folding_preflight_submit_archive_replay_and_status_are_nonexecuting(tmp_path: Path) -> None:
    paths, index = _folding_index(tmp_path / "folding")
    config = _folding_config(tmp_path / "config", paths)
    results = tmp_path / "results"
    for record in index.records:
        _write_result_pair(results, record.protein_id)
    (results / "config.json").write_text("{}\n")
    inventory = folding_runtime.scan_folding_results(index, results)

    submit_request = FoldingSubmitLifecycleRequest(config=config, index=index)
    submit_preflight_request = FoldingPreflightLifecycleRequest(
        action="submit",
        config=config,
        index=index,
        checkpoint_inputs=None,
        checkpoint_state=None,
        result_inventory=None,
        scan_predictions_root=None,
    )
    submit_preflight = folding_runtime.preflight_folding_lifecycle(submit_preflight_request)
    submit = folding_runtime.plan_folding_submit_lifecycle(submit_request)
    assert submit_preflight == folding_runtime.preflight_folding_lifecycle(submit_preflight_request)
    assert submit == folding_runtime.plan_folding_submit_lifecycle(submit_request)
    assert submit_preflight.can_plan is True
    assert submit.submission_planned is True
    assert submit.execution_authorized is False
    assert not Path(config.work_dir).exists()

    archive_preflight_request = FoldingPreflightLifecycleRequest(
        action="archive",
        config=config,
        index=index,
        checkpoint_inputs=None,
        checkpoint_state=None,
        result_inventory=inventory,
        scan_predictions_root=None,
    )
    archive_preflight = folding_runtime.preflight_folding_lifecycle(archive_preflight_request)
    result_check = next(check for check in archive_preflight.checks if check.name == "result")
    assert (result_check.status, archive_preflight.can_plan) == ("warning", True)

    selected_request = FoldingArchiveLifecycleRequest(
        config=config,
        index=index,
        result_inventory=inventory,
        scan_predictions_root=None,
        selected_protein_ids=("protein-00",),
        prior_manifest=(),
        force=False,
    )
    selected = folding_runtime.plan_folding_archive_lifecycle(selected_request)
    assert tuple(batch.protein_ids for batch in selected.archive_plan.batches) == (("protein-00",),)
    assert selected.execution_authorized is False
    assert selected.manifest_jsonl == folding_runtime.render_archive_manifest_jsonl(selected.archive_plan)
    prior = folding_runtime.parse_archive_manifest_jsonl(selected.manifest_jsonl)
    assert prior == tuple(batch.manifest_record for batch in selected.archive_plan.batches)

    replay_request = replace(selected_request, prior_manifest=prior)
    replay = folding_runtime.plan_folding_archive_lifecycle(replay_request)
    assert replay.archive_plan.batches == ()
    assert replay.archive_plan.skipped_prior_protein_ids == ("protein-00",)

    forced_request = replace(
        selected_request,
        selected_protein_ids=None,
        prior_manifest=prior,
        force=True,
    )
    forced = folding_runtime.plan_folding_archive_lifecycle(forced_request)
    assert forced == folding_runtime.plan_folding_archive_lifecycle(forced_request)
    assert tuple(protein for batch in forced.archive_plan.batches for protein in batch.protein_ids) == (
        "protein-00",
        "protein-01",
        "protein-02",
    )
    assert min(batch.archive_index for batch in forced.archive_plan.batches) > max(
        record.archive_index for record in prior
    )
    assert config.archive_stage_root is not None
    assert config.archive_root is not None
    assert not Path(config.archive_stage_root).exists()
    assert not Path(config.archive_root).exists()

    (results / "config.json").unlink()
    coherent_inventory = folding_runtime.scan_folding_results(index, results)
    assert coherent_inventory.unmatched == ()
    status_request = FoldingStatusLifecycleRequest(
        config=config,
        index=index,
        manifest=FoldingManifestEvidence(
            total_proteins=3,
            queue_artifact_count=len(submit.queue_writes),
            queue_layout=config.queue_layout,
        ),
        checkpoint_state=FoldingCheckpointState(completions=(), failures=()),
        result_inventory=coherent_inventory,
        archive_plan=forced.archive_plan,
    )
    status = folding_runtime.interpret_folding_status_lifecycle(status_request)
    assert status == folding_runtime.interpret_folding_status_lifecycle(status_request)
    assert status.state == "complete"
    assert status.evidence_kinds == ("manifest", "checkpoint", "result", "archive")
    assert status.completed_model_count == 3
    assert status.invalid_evidence == ()
