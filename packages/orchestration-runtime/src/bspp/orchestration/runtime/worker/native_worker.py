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

"""Native archive worker composition entry points."""

from __future__ import annotations

import csv
import fcntl
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from bspp.orchestration.contract.runspec import RunSpec
from bspp.orchestration.runtime.toolkit import resolve_toolkit
from bspp.orchestration.runtime.worker.archive_plan import (
    archive_active_lock_path,
    archive_active_marker_name,
    plan_archive_task,
    plan_batches,
)
from bspp.orchestration.runtime.worker.batch_state import (
    flush_local_failed_models,
    load_failed_model_ids,
    record_batch_failed,
    record_prefiltered_failed,
    remove_shard_failed_file_if_clean,
    should_skip_batch,
    write_batch_marker,
)
from bspp.orchestration.runtime.worker.gpu import GpuKeepalivePolicy, KeepaliveProcess, gpu_keepalive
from bspp.orchestration.runtime.worker.manifest import (
    ManifestMode,
    apply_heterodimer_id_rewrites,
    filter_homodimer_manifest_rows,
    filter_manifest_csv,
    persist_shard_manifest_parquet,
    read_heterodimer_id_rewrite_plan,
    write_shard_manifest_csv,
)
from bspp.orchestration.runtime.worker.metadata import (
    MetadataCommandRunner,
    clean_metadata_json_outputs,
    clean_sub_shard_metadata_state,
    finalize_metadata,
    metadata_destination_dir,
    plan_metadata_finalization,
    prefilter_batch_inputs,
)
from bspp.orchestration.runtime.worker.model_inventory import discover_model_ids, filter_model_ids_by_allowlist
from bspp.orchestration.runtime.worker.output_layout import (
    batch_tar_name,
    join_destination_prefix,
    metadata_tar_name,
    success_outputs_dir,
)
from bspp.orchestration.runtime.worker.packaging import ZstdCompressor, create_outputs_tar
from bspp.orchestration.runtime.worker.pipeline_adapter import (
    PipelineCommandRunner,
    ProductionPipelineOptions,
    ProductionPipelinePaths,
    build_production_pipeline_command,
    run_production_pipeline_command,
)
from bspp.orchestration.runtime.worker.scratch import (
    ArchiveExtractor,
    cleanup_scratch_workspace,
    create_scratch_workspace,
    extract_archive_to_workspace,
    plan_scratch_workspace,
    relink_renamed_inputs,
)
from bspp.orchestration.runtime.worker.types import PipelineCommand, PipelineResult, TaskContext, WorkerResult
from bspp.orchestration.runtime.worker.upload import (
    UploadCommandRunner,
    clean_batch_outputs,
    cleanup_success_outputs,
    copy_batch_flat_to_success,
    upload_files_to_s3,
    upload_single_file_to_s3,
    upload_slot,
    uploaded_marker_is_complete,
    write_uploaded_marker,
)

RETRY_FAILED_STATUSES = frozenset(
    {
        "model_failed",
        "prefilter_failed",
        "pipeline_failed",
        "upload_failed_lustre_fallback",
    },
)
TAR_MANIFEST_FIELDS = (
    "timestamp",
    "run_name",
    "tar_type",
    "s3_uri",
    "tar_name",
    "source_archive",
    "shard_id",
    "task_id",
    "batch_id",
    "member_count",
    "size_bytes",
    "compression",
)
PROCESSING_LOG_FIELDS = (
    "model_id",
    "original_id",
    "upload_status",
    "tar_of_origin",
    "shard_id",
    "task_id",
    "failure_reason",
    "timestamp",
)
ANALYSIS_METADATA_FIELDS = (
    "timestamp",
    "run_name",
    "model_id",
    "original_id",
    "upload_status",
    "passes_quality_threshold",
    "ipsae_max",
    "pdockq2_max",
    "quality_ipsae_threshold",
    "quality_pdockq2_threshold",
    "source_archive",
    "shard_id",
    "task_id",
    "batch_id",
    "batch_started_at",
    "batch_finished_at",
    "failure_reason",
    "expected_output_files_json",
    "scores_json",
)
EXPECTED_OUTPUT_SUFFIXES = (
    "-model_v1.cif",
    "-model_v1.pdb",
    "-model_v1.bcif",
    "-confidence_v1.json",
    "-predicted_aligned_error_v1.json",
    "-model_v1_clashes.json",
    "-model_v1_interface.json",
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class WorkerDependencies:
    """Dependency-injected boundaries for native archive worker composition."""

    archive_names: Sequence[str] | None = None
    archive_extractor: ArchiveExtractor | None = None
    pipeline_runner: PipelineCommandRunner | None = None
    metadata_runner: MetadataCommandRunner | None = None
    upload_runner: UploadCommandRunner | None = None
    zstd_compressor: ZstdCompressor | None = None
    gpu_keepalive_policy: GpuKeepalivePolicy = field(default_factory=GpuKeepalivePolicy)
    keepalive_process_factory: Callable[[], KeepaliveProcess] | None = None
    pipeline_script: Path | None = None
    dataset_config: Path | None = None
    provider_json: Path | None = None
    python_cmd: str | None = None
    now: Callable[[], datetime] = _utc_now
    sleep: Callable[[float], None] | None = None


@dataclass(frozen=True, slots=True)
class BatchSyncResult:
    """Upload/copy accounting for one successful batch."""

    status: str
    uploaded_files: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MetadataTransferResult:
    """Metadata finalization and transfer accounting for one shard."""

    success: bool
    metadata_files: tuple[str, ...]
    exit_code: int = 0


def run_archive_task(spec: RunSpec, task: TaskContext, deps: WorkerDependencies | None = None) -> WorkerResult:
    """Run the smallest native archive-mode worker path for one archive task.

    This composes extraction, manifest filtering, processing, packaging, and
    upload helpers for the native Slurm runtime.
    """

    spec.validate()
    active_deps = deps or WorkerDependencies()
    archive_names = tuple(active_deps.archive_names or _load_archive_names(spec.paths.output_dir / "archive_list.json"))
    archive_name = archive_names[task.array_task_id]
    archive_path = spec.paths.staging_dir / archive_name
    workspace = create_scratch_workspace(plan_scratch_workspace(spec.worker.scratch_dir, task))
    active_marker = spec.paths.output_dir / archive_active_marker_name(task.array_task_id)
    active_lock_fd: int | None = None
    if not task.dry_run:
        active_marker.parent.mkdir(parents=True, exist_ok=True)
        active_lock_fd = os.open(
            str(archive_active_lock_path(spec.paths.output_dir, task.array_task_id)),
            os.O_CREAT | os.O_RDWR,
            0o666,
        )
        fcntl.flock(active_lock_fd, fcntl.LOCK_EX)
        active_marker.touch()
    processed_models = 0
    uploaded_file_count = 0
    failed_model_ids: set[str] = set()
    exit_code = 0

    try:
        if not task.dry_run:
            extract_archive_to_workspace(archive_path, workspace, extractor=active_deps.archive_extractor)

        archive_model_ids = _discover_workspace_model_ids(workspace.input_dir)
        allowlist = _load_allowlist(spec.paths.output_dir / "allowlists" / f"{archive_name}.txt")
        if allowlist is not None:
            archive_model_ids = filter_model_ids_by_allowlist(archive_model_ids, allowlist).kept_model_ids
        if not archive_model_ids:
            return WorkerResult(
                archive_name=archive_name,
                logical_shards=(),
                processed_models=0,
                failed_models=(),
                uploaded_files=0,
                exit_code=1,
            )

        archive_plan = plan_archive_task(
            task.array_task_id,
            archive_names,
            archive_model_ids,
            spec.worker.shards_per_archive,
            spec.paths.output_dir,
        )
        logical_shards: list[int] = []
        total_shards = len(archive_names) * spec.worker.shards_per_archive
        for sub_shard in archive_plan.logical_shards:
            shard_original_ids = archive_model_ids[sub_shard.model_start : sub_shard.model_end]
            if not shard_original_ids:
                continue
            logical_shards.append(sub_shard.logical_shard_id)
            clean_sub_shard_metadata_state(workspace.work_dir)
            sub_result = _run_sub_shard(
                spec,
                task,
                active_deps,
                archive_name=archive_name,
                workspace_input_dir=workspace.input_dir,
                work_dir=workspace.work_dir,
                shard_dir=sub_shard.shard_dir,
                logical_shard_id=sub_shard.logical_shard_id,
                total_shards=total_shards,
                original_model_ids=shard_original_ids,
            )
            processed_models += sub_result.processed_models
            uploaded_file_count += sub_result.uploaded_files
            failed_model_ids.update(sub_result.failed_model_ids)
            if sub_result.exit_code != 0:
                exit_code = sub_result.exit_code

        return WorkerResult(
            archive_name=archive_name,
            logical_shards=tuple(logical_shards),
            processed_models=processed_models,
            failed_models=tuple(sorted(failed_model_ids)),
            uploaded_files=uploaded_file_count,
            exit_code=exit_code,
        )
    finally:
        if not task.dry_run:
            active_marker.unlink(missing_ok=True)
            assert active_lock_fd is not None
            fcntl.flock(active_lock_fd, fcntl.LOCK_UN)
            os.close(active_lock_fd)
        cleanup_scratch_workspace(workspace, success=exit_code == 0)


@dataclass(frozen=True, slots=True)
class _SubShardResult:
    processed_models: int
    failed_model_ids: frozenset[str]
    uploaded_files: int
    exit_code: int


def _run_sub_shard(
    spec: RunSpec,
    task: TaskContext,
    deps: WorkerDependencies,
    *,
    archive_name: str,
    workspace_input_dir: Path,
    work_dir: Path,
    shard_dir: Path,
    logical_shard_id: int,
    total_shards: int,
    original_model_ids: tuple[str, ...],
) -> _SubShardResult:
    shard_dir.mkdir(parents=True, exist_ok=True)
    local_tar_restart = bool(
        spec.storage.local_tar_dir and not spec.worker.self_upload and not spec.worker.retry_failed_only,
    )
    if local_tar_restart and uploaded_marker_is_complete(shard_dir):
        return _SubShardResult(0, frozenset(), 0, 0)
    if local_tar_restart:
        _fail_on_stale_local_tar_ledgers(spec, logical_shard_id)

    mode: ManifestMode = "heterodimer" if spec.worker.heterodimers else "homodimer"
    original_manifest = filter_manifest_csv(spec.references.manifest_csv, original_model_ids, mode=mode)
    if original_manifest.row_count == 0:
        return _SubShardResult(0, frozenset(), 0, 1)

    persisted_manifest = original_manifest
    pipeline_manifest = original_manifest
    rename_map: dict[str, str] = {}
    if mode == "heterodimer" and spec.references.heterodimer_id_manifest is not None:
        rewrite_plan = read_heterodimer_id_rewrite_plan(original_manifest, spec.references.heterodimer_id_manifest)
        rename_map = rewrite_plan.rename_map
        if rename_map:
            persisted_manifest = apply_heterodimer_id_rewrites(original_manifest, rewrite_plan)
            pipeline_manifest = persisted_manifest
            relink_renamed_inputs(workspace_input_dir, rename_map)

    runnable_original_ids = original_model_ids
    if spec.worker.retry_failed_only:
        retry_ids = load_retry_failed_ids_from_state(
            spec.paths.output_dir / "processing_log.csv",
            shard_dir / "failed_models.tsv",
        )
        runnable_original_ids = tuple(
            model_id
            for model_id in original_model_ids
            if model_id in retry_ids or rename_map.get(model_id, model_id) in retry_ids
        )
        if not runnable_original_ids:
            return _SubShardResult(0, frozenset(), 0, 0)
        retry_final_ids = tuple(rename_map.get(model_id, model_id) for model_id in runnable_original_ids)
        persisted_manifest = filter_homodimer_manifest_rows(
            persisted_manifest.rows,
            retry_final_ids,
            fieldnames=persisted_manifest.fieldnames,
        )
        pipeline_manifest = persisted_manifest

    persist_shard_manifest_parquet(
        persisted_manifest,
        shard_id=logical_shard_id,
        dataset_tag=spec.dataset.name,
        output_path=shard_dir / "shard_manifest.parquet",
    )
    work_manifest = work_dir / "shard_manifest.csv"
    write_shard_manifest_csv(pipeline_manifest, work_manifest)

    mapping_file: Path | None = None
    if mode == "homodimer":
        mapping_file = work_dir / "shard_mapping.tsv"
        mapping_file.write_text("\n".join(runnable_original_ids) + "\n", encoding="utf-8")

    batches = plan_batches(runnable_original_ids, shard_dir, spec.worker.batch_size)
    global_failed_path = spec.paths.output_dir / "failed_models.tsv"
    shard_failed_path = shard_dir / "failed_models.tsv"
    processed_models = 0
    uploaded_file_count = 0
    had_failures = False
    exit_code = 0

    for batch in batches:
        if should_skip_batch(
            batch.done_marker,
            retry_failed_only=spec.worker.retry_failed_only,
            local_tar_restart_requires_rerun=local_tar_restart and not (shard_dir / ".uploaded").exists(),
        ):
            continue

        batch_started_at = deps.now().isoformat()
        prefilter = prefilter_batch_inputs(batch.model_ids, workspace_input_dir)
        if prefilter.bad_models:
            record_prefiltered_failed(
                prefilter.bad_models,
                global_failed_path=global_failed_path,
                shard_failed_path=shard_failed_path,
            )
            had_failures = True
            _append_processing_log_rows(
                spec.paths.output_dir / "processing_log.csv",
                _processing_rows(
                    prefilter.bad_models,
                    status="prefilter_failed",
                    archive_name=archive_name,
                    logical_shard_id=logical_shard_id,
                    task_id=task.array_task_id,
                    now=deps.now,
                ),
            )
            batch_finished_at = deps.now().isoformat()
            _append_analysis_metadata_rows(
                spec.analysis_metadata.csv_path if spec.analysis_metadata.enabled else None,
                _analysis_metadata_rows(
                    deps,
                    spec,
                    work_dir=work_dir,
                    original_ids=tuple(model_id for model_id, _reason in prefilter.bad_models),
                    model_ids=tuple(model_id for model_id, _reason in prefilter.bad_models),
                    status_by_id={model_id: "prefilter_failed" for model_id, _reason in prefilter.bad_models},
                    failure_reasons={model_id: reason for model_id, reason in prefilter.bad_models},
                    archive_name=archive_name,
                    logical_shard_id=logical_shard_id,
                    task_id=task.array_task_id,
                    batch_id=batch.batch_index,
                    batch_started_at=batch_started_at,
                    batch_finished_at=batch_finished_at,
                    include_scores=False,
                ),
            )

        if not prefilter.good_model_ids:
            continue

        pipeline_ids = _pipeline_model_ids(
            prefilter.good_model_ids,
            rename_map,
            retry_failed_only=spec.worker.retry_failed_only,
        )
        if mapping_file is not None:
            mapping_file.write_text("\n".join(pipeline_ids) + "\n", encoding="utf-8")
        (work_dir / "model_ids.txt").write_text("\n".join(pipeline_ids) + "\n", encoding="utf-8")
        write_shard_manifest_csv(
            filter_homodimer_manifest_rows(
                persisted_manifest.rows,
                pipeline_ids,
                fieldnames=persisted_manifest.fieldnames,
            ),
            work_manifest,
        )

        pipeline_command = build_production_pipeline_command(
            _pipeline_paths(spec, deps, workspace_input_dir, work_dir, work_manifest, mapping_file),
            _pipeline_options(spec, deps),
        )
        pipeline_result = run_production_pipeline_command(
            _with_pipeline_runtime_context(
                pipeline_command,
                spec=spec,
                shard_dir=shard_dir,
                batch_index=batch.batch_index,
            ),
            deps.pipeline_runner or _run_pipeline_subprocess,
        )
        if pipeline_result.exit_code != 0:
            reason = f"pipeline_exit_{pipeline_result.exit_code}"
            flush_local_failed_models(
                work_dir / "failed_models.tsv",
                global_failed_path=global_failed_path,
                shard_failed_path=shard_failed_path,
            )
            record_batch_failed(
                pipeline_ids,
                reason=reason,
                global_failed_path=global_failed_path,
                shard_failed_path=shard_failed_path,
            )
            clean_batch_outputs(work_dir, pipeline_ids)
            clean_metadata_json_outputs(work_dir, pipeline_ids)
            _append_processing_log_rows(
                spec.paths.output_dir / "processing_log.csv",
                _tracking_rows(
                    original_ids=prefilter.good_model_ids,
                    model_ids=pipeline_ids,
                    status_by_id={model_id: "pipeline_failed" for model_id in pipeline_ids},
                    failure_reasons={model_id: reason for model_id in pipeline_ids},
                    archive_name=archive_name,
                    logical_shard_id=logical_shard_id,
                    task_id=task.array_task_id,
                    now=deps.now,
                ),
            )
            batch_finished_at = deps.now().isoformat()
            _append_analysis_metadata_rows(
                spec.analysis_metadata.csv_path if spec.analysis_metadata.enabled else None,
                _analysis_metadata_rows(
                    deps,
                    spec,
                    work_dir=work_dir,
                    original_ids=prefilter.good_model_ids,
                    model_ids=pipeline_ids,
                    status_by_id={model_id: "pipeline_failed" for model_id in pipeline_ids},
                    failure_reasons={model_id: reason for model_id in pipeline_ids},
                    archive_name=archive_name,
                    logical_shard_id=logical_shard_id,
                    task_id=task.array_task_id,
                    batch_id=batch.batch_index,
                    batch_started_at=batch_started_at,
                    batch_finished_at=batch_finished_at,
                    include_scores=False,
                ),
            )
            had_failures = True
            exit_code = 1
            continue

        failure_reasons = flush_local_failed_models(
            work_dir / "failed_models.tsv",
            global_failed_path=global_failed_path,
            shard_failed_path=shard_failed_path,
        )
        if failure_reasons:
            had_failures = True
        failed_ids = frozenset(failure_reasons)
        success_ids = tuple(model_id for model_id in pipeline_ids if model_id not in failed_ids)
        if failed_ids:
            copy_batch_flat_to_success(work_dir, success_outputs_dir(shard_dir), failed_ids)
            clean_batch_outputs(work_dir, failed_ids)
            clean_metadata_json_outputs(work_dir, failed_ids)
        scores_by_id = _collect_analysis_scores(work_dir) if spec.analysis_metadata.enabled else {}
        sync_result = _sync_successful_batch(
            spec,
            deps,
            work_dir=work_dir,
            shard_dir=shard_dir,
            archive_name=archive_name,
            logical_shard_id=logical_shard_id,
            task_id=task.array_task_id,
            batch_index=batch.batch_index,
            success_ids=success_ids,
        )
        uploaded_file_count += len(sync_result.uploaded_files)
        if sync_result.status in {"uploaded", "local_tarred"}:
            write_batch_marker(
                shard_dir,
                batch.batch_index,
                sync_result.uploaded_files,
                failed_model_count=len(failed_ids),
                retry_failed_only=spec.worker.retry_failed_only,
            )
        status_by_id = {model_id: "model_failed" for model_id in failed_ids}
        status_by_id.update({model_id: sync_result.status or "pipeline_failed" for model_id in success_ids})
        _append_processing_log_rows(
            spec.paths.output_dir / "processing_log.csv",
            _tracking_rows(
                original_ids=prefilter.good_model_ids,
                model_ids=pipeline_ids,
                status_by_id=status_by_id,
                failure_reasons=failure_reasons,
                archive_name=archive_name,
                logical_shard_id=logical_shard_id,
                task_id=task.array_task_id,
                now=deps.now,
            ),
        )
        batch_finished_at = deps.now().isoformat()
        _append_analysis_metadata_rows(
            spec.analysis_metadata.csv_path if spec.analysis_metadata.enabled else None,
            _analysis_metadata_rows(
                deps,
                spec,
                work_dir=work_dir,
                original_ids=prefilter.good_model_ids,
                model_ids=pipeline_ids,
                status_by_id=status_by_id,
                failure_reasons=failure_reasons,
                archive_name=archive_name,
                logical_shard_id=logical_shard_id,
                task_id=task.array_task_id,
                batch_id=batch.batch_index,
                batch_started_at=batch_started_at,
                batch_finished_at=batch_finished_at,
                include_scores=True,
                scores_by_id=scores_by_id,
            ),
        )
        processed_models += len(prefilter.good_model_ids)

    remove_shard_failed_file_if_clean(shard_failed_path, had_failures=had_failures)
    metadata = _finalize_and_transfer_metadata(
        spec,
        deps,
        work_dir=work_dir,
        shard_dir=shard_dir,
        archive_name=archive_name,
        logical_shard_id=logical_shard_id,
        task_id=task.array_task_id,
        total_shards=total_shards,
    )
    if not metadata.success:
        return _SubShardResult(
            processed_models=processed_models,
            failed_model_ids=load_failed_model_ids(shard_failed_path),
            uploaded_files=uploaded_file_count,
            exit_code=metadata.exit_code or 1,
        )
    metadata_files = metadata.metadata_files
    uploaded_file_count += len(metadata_files)
    if spec.worker.self_upload or spec.storage.local_tar_dir is not None:
        write_uploaded_marker(
            shard_dir,
            s3_prefix=spec.storage.s3_output_prefix if spec.worker.self_upload else "",
            upload_mode=spec.storage.upload_mode,
            tar_prefix=spec.storage.s3_tar_prefix or "",
            total_batches=len(batches),
            metadata_files=metadata_files,
            shard_id=logical_shard_id,
            model_count=len(runnable_original_ids),
            timestamp=deps.now().isoformat(),
        )

    return _SubShardResult(
        processed_models=processed_models,
        failed_model_ids=load_failed_model_ids(shard_failed_path),
        uploaded_files=uploaded_file_count,
        exit_code=exit_code,
    )


def load_retry_failed_ids_from_state(processing_log: Path, shard_failed_path: Path) -> frozenset[str]:
    """Load IDs whose latest processing state still needs retry."""

    latest_status: dict[str, str] = {}
    if processing_log.exists() and processing_log.stat().st_size > 0:
        with processing_log.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                model_id = (row.get("model_id") or "").strip()
                original_id = (row.get("original_id") or "").strip()
                status = (row.get("upload_status") or "").strip()
                if not model_id or not status:
                    continue
                latest_status[model_id] = status
                if original_id:
                    latest_status[original_id] = status

    retry_ids = frozenset(model_id for model_id, status in latest_status.items() if status in RETRY_FAILED_STATUSES)
    if retry_ids:
        return retry_ids
    return load_failed_model_ids(shard_failed_path)


def append_tar_manifest_rows(manifest_csv: Path | None, rows: Iterable[Mapping[str, object]]) -> None:
    """Append rows to a tar manifest CSV using the legacy field order."""

    if manifest_csv is not None:
        _append_csv_rows_locked(manifest_csv, TAR_MANIFEST_FIELDS, rows)


def _fail_on_stale_local_tar_ledgers(spec: RunSpec, logical_shard_id: int) -> None:
    """Fail closed before rerunning a local-tar shard with stale append-only rows."""
    stale_paths = [
        path
        for path in (
            spec.storage.local_tar_manifest_csv,
            spec.paths.output_dir / "processing_log.csv",
            spec.analysis_metadata.csv_path if spec.analysis_metadata.enabled else None,
        )
        if path is not None and _csv_has_shard_rows(path, logical_shard_id)
    ]
    if stale_paths:
        rendered = ", ".join(str(path) for path in stale_paths)
        msg = f"Refusing local-tar restart for shard_{logical_shard_id}; stale ledger rows exist in {rendered}"
        raise RuntimeError(msg)


def _csv_has_shard_rows(path: Path, logical_shard_id: int) -> bool:
    if not path.exists():
        return False
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return any((row.get("shard_id") or "").strip() == str(logical_shard_id) for row in reader)
    except (OSError, csv.Error, UnicodeDecodeError):
        return True


def _sync_successful_batch(
    spec: RunSpec,
    deps: WorkerDependencies,
    *,
    work_dir: Path,
    shard_dir: Path,
    archive_name: str,
    logical_shard_id: int,
    task_id: int,
    batch_index: int,
    success_ids: tuple[str, ...],
) -> BatchSyncResult:
    if not success_ids:
        return BatchSyncResult(status="", uploaded_files=())

    if spec.storage.upload_mode == "tar" and (spec.worker.self_upload or spec.storage.local_tar_dir):
        return _sync_batch_tar(
            spec,
            deps,
            work_dir=work_dir,
            shard_dir=shard_dir,
            archive_name=archive_name,
            logical_shard_id=logical_shard_id,
            task_id=task_id,
            batch_index=batch_index,
            success_ids=success_ids,
        )

    if spec.worker.self_upload:
        with (
            gpu_keepalive(deps.gpu_keepalive_policy, process_factory=deps.keepalive_process_factory),
            upload_slot(spec.paths.output_dir / ".upload_slots", spec.worker.upload_slots),
        ):
            upload = upload_files_to_s3(
                work_dir,
                spec.storage.s3_output_prefix,
                s5cmd_path=spec.worker.s5cmd_path,
                batch_ids=success_ids,
                numworkers=spec.worker.s5cmd_numworkers,
                runner=deps.upload_runner,
                sleep=deps.sleep or _noop_sleep,
            )
        if upload.success:
            clean_batch_outputs(work_dir, success_ids)
            return BatchSyncResult(status="uploaded", uploaded_files=tuple(str(path) for path in upload.uploaded_files))
        copied = copy_batch_flat_to_success(work_dir, success_outputs_dir(shard_dir), success_ids)
        clean_batch_outputs(work_dir, success_ids)
        clean_metadata_json_outputs(work_dir, success_ids)
        return BatchSyncResult(
            status="upload_failed_lustre_fallback",
            uploaded_files=tuple(str(path) for path in copied),
        )

    copied = copy_batch_flat_to_success(work_dir, success_outputs_dir(shard_dir), success_ids)
    clean_batch_outputs(work_dir, success_ids)
    return BatchSyncResult(status="uploaded", uploaded_files=tuple(str(path) for path in copied))


def _sync_batch_tar(
    spec: RunSpec,
    deps: WorkerDependencies,
    *,
    work_dir: Path,
    shard_dir: Path,
    archive_name: str,
    logical_shard_id: int,
    task_id: int,
    batch_index: int,
    success_ids: tuple[str, ...],
) -> BatchSyncResult:
    tar_name = batch_tar_name(logical_shard_id, batch_index, spec.storage.tar_compression)
    if spec.worker.self_upload:
        tar_dir = work_dir / "_upload_tars"
    else:
        if spec.storage.local_tar_dir is None:
            return BatchSyncResult(status="", uploaded_files=())
        tar_dir = spec.storage.local_tar_dir / f"shard_{logical_shard_id}"
    tar_path = tar_dir / tar_name
    result = create_outputs_tar(
        work_dir,
        tar_path,
        batch_ids=success_ids,
        compression=spec.storage.tar_compression,
        zstd_compressor=deps.zstd_compressor,
    )
    if result.member_count == 0:
        return BatchSyncResult(status="local_tarred" if not spec.worker.self_upload else "uploaded", uploaded_files=())

    if spec.worker.self_upload:
        tar_prefix = spec.storage.s3_tar_prefix or f"{spec.storage.s3_output_prefix.rstrip('/')}/tars"
        s3_uri = join_destination_prefix(f"{tar_prefix.rstrip('/')}/shard_{logical_shard_id}", tar_name)
        with (
            gpu_keepalive(deps.gpu_keepalive_policy, process_factory=deps.keepalive_process_factory),
            upload_slot(spec.paths.output_dir / ".upload_slots", spec.worker.upload_slots),
        ):
            upload = upload_single_file_to_s3(
                tar_path,
                s3_uri,
                s5cmd_path=spec.worker.s5cmd_path,
                numworkers=spec.worker.s5cmd_numworkers,
                runner=deps.upload_runner,
                sleep=deps.sleep or _noop_sleep,
            )
        tar_path.unlink(missing_ok=True)
        if upload.success:
            append_tar_manifest_rows(
                spec.storage.s3_tar_manifest_csv,
                (
                    _tar_manifest_row(
                        deps,
                        run_name=spec.dataset.name,
                        tar_type="batch",
                        s3_uri=s3_uri,
                        tar_name=tar_name,
                        archive_name=archive_name,
                        logical_shard_id=logical_shard_id,
                        task_id=task_id,
                        batch_id=str(batch_index),
                        member_count=result.member_count,
                        size_bytes=result.size_bytes,
                        compression=spec.storage.tar_compression,
                    ),
                ),
            )
            clean_batch_outputs(work_dir, success_ids)
            return BatchSyncResult(status="uploaded", uploaded_files=(tar_name,))
        copied = copy_batch_flat_to_success(work_dir, success_outputs_dir(shard_dir), success_ids)
        clean_batch_outputs(work_dir, success_ids)
        clean_metadata_json_outputs(work_dir, success_ids)
        return BatchSyncResult(
            status="upload_failed_lustre_fallback",
            uploaded_files=tuple(str(path) for path in copied),
        )

    append_tar_manifest_rows(
        spec.storage.local_tar_manifest_csv,
        (
            _tar_manifest_row(
                deps,
                run_name=spec.dataset.name,
                tar_type="batch",
                s3_uri=tar_path.resolve().as_uri(),
                tar_name=tar_name,
                archive_name=archive_name,
                logical_shard_id=logical_shard_id,
                task_id=task_id,
                batch_id=str(batch_index),
                member_count=result.member_count,
                size_bytes=result.size_bytes,
                compression=spec.storage.tar_compression,
            ),
        ),
    )
    clean_batch_outputs(work_dir, success_ids)
    return BatchSyncResult(status="local_tarred", uploaded_files=(str(tar_path),))


def _finalize_and_transfer_metadata(
    spec: RunSpec,
    deps: WorkerDependencies,
    *,
    work_dir: Path,
    shard_dir: Path,
    archive_name: str,
    logical_shard_id: int,
    task_id: int,
    total_shards: int,
) -> MetadataTransferResult:
    s3_enabled = spec.worker.self_upload
    destination = metadata_destination_dir(work_dir, success_outputs_dir(shard_dir), s3_upload_enabled=s3_enabled)
    metadata_tag = _metadata_dataset_tag(spec)
    plan = plan_metadata_finalization(
        work_dir=work_dir,
        output_base_dir=destination,
        logical_shard_id=logical_shard_id,
        total_shards=total_shards,
        dataset_tag=metadata_tag,
    )
    result = finalize_metadata(plan, runner=deps.metadata_runner)
    if result.exit_code != 0:
        return MetadataTransferResult(success=False, metadata_files=(), exit_code=result.exit_code)

    if spec.storage.upload_mode == "tar" and (spec.worker.self_upload or spec.storage.local_tar_dir is not None):
        return _transfer_metadata_tar(
            spec,
            deps,
            source_dir=destination,
            shard_dir=shard_dir,
            archive_name=archive_name,
            logical_shard_id=logical_shard_id,
            task_id=task_id,
        )

    if spec.worker.self_upload:
        upload = upload_files_to_s3(
            destination,
            spec.storage.s3_output_prefix,
            s5cmd_path=spec.worker.s5cmd_path,
            numworkers=spec.worker.s5cmd_numworkers,
            runner=deps.upload_runner,
            sleep=deps.sleep or _noop_sleep,
        )
        if upload.success:
            cleanup_success_outputs(destination)
            return MetadataTransferResult(
                success=True,
                metadata_files=tuple(path.name for path in upload.uploaded_files),
            )
        return MetadataTransferResult(success=True, metadata_files=(), exit_code=0)

    return MetadataTransferResult(success=True, metadata_files=tuple(str(path) for path in result.output_files))


def _transfer_metadata_tar(
    spec: RunSpec,
    deps: WorkerDependencies,
    *,
    source_dir: Path,
    shard_dir: Path,
    archive_name: str,
    logical_shard_id: int,
    task_id: int,
) -> MetadataTransferResult:
    tar_name = metadata_tar_name(logical_shard_id, spec.storage.tar_compression)
    tar_path = (
        (spec.storage.local_tar_dir / "metadata" / tar_name)
        if spec.storage.local_tar_dir is not None and not spec.worker.self_upload
        else shard_dir / tar_name
    )
    result = create_outputs_tar(
        source_dir,
        tar_path,
        compression=spec.storage.tar_compression,
        zstd_compressor=deps.zstd_compressor,
    )
    if result.member_count == 0:
        return MetadataTransferResult(success=True, metadata_files=())

    if spec.worker.self_upload:
        tar_prefix = spec.storage.s3_tar_prefix or f"{spec.storage.s3_output_prefix.rstrip('/')}/tars"
        s3_uri = join_destination_prefix(f"{tar_prefix.rstrip('/')}/metadata", tar_name)
        upload = upload_single_file_to_s3(
            tar_path,
            s3_uri,
            s5cmd_path=spec.worker.s5cmd_path,
            numworkers=spec.worker.s5cmd_numworkers,
            runner=deps.upload_runner,
            sleep=deps.sleep or _noop_sleep,
        )
        if upload.success:
            tar_path.unlink(missing_ok=True)
            cleanup_success_outputs(source_dir)
            append_tar_manifest_rows(
                spec.storage.s3_tar_manifest_csv,
                (
                    _tar_manifest_row(
                        deps,
                        run_name=spec.dataset.name,
                        tar_type="metadata",
                        s3_uri=s3_uri,
                        tar_name=tar_name,
                        archive_name=archive_name,
                        logical_shard_id=logical_shard_id,
                        task_id=logical_shard_id,
                        batch_id="metadata",
                        member_count=result.member_count,
                        size_bytes=result.size_bytes,
                        compression=spec.storage.tar_compression,
                    ),
                ),
            )
            return MetadataTransferResult(success=True, metadata_files=(tar_name,))
        return MetadataTransferResult(success=True, metadata_files=(), exit_code=0)

    append_tar_manifest_rows(
        spec.storage.local_tar_manifest_csv,
        (
            _tar_manifest_row(
                deps,
                run_name=spec.dataset.name,
                tar_type="metadata",
                s3_uri=tar_path.resolve().as_uri(),
                tar_name=tar_name,
                archive_name=archive_name,
                logical_shard_id=logical_shard_id,
                task_id=logical_shard_id,
                batch_id="metadata",
                member_count=result.member_count,
                size_bytes=result.size_bytes,
                compression=spec.storage.tar_compression,
            ),
        ),
    )
    cleanup_success_outputs(source_dir)
    return MetadataTransferResult(success=True, metadata_files=(str(tar_path),))


def _pipeline_paths(
    spec: RunSpec,
    deps: WorkerDependencies,
    input_dir: Path,
    work_dir: Path,
    work_manifest: Path,
    mapping_file: Path | None,
) -> ProductionPipelinePaths:
    toolkit_source = resolve_toolkit(spec)
    toolkit_root = toolkit_source.root
    return ProductionPipelinePaths(
        pipeline_script=deps.pipeline_script or toolkit_root / "scripts" / "production_pipeline.py",
        input_dir=input_dir,
        output_dir=work_dir,
        chain_mapping=work_manifest,
        uniprot_db=spec.references.uniprot_duckdb,
        mapping_file=mapping_file,
        dataset_config=deps.dataset_config or (spec.paths.output_dir / "dataset_config.json"),
        provider_json=deps.provider_json or (spec.paths.output_dir / "provider.json"),
    )


def _pipeline_options(spec: RunSpec, deps: WorkerDependencies) -> ProductionPipelineOptions:
    return ProductionPipelineOptions(
        python_cmd=deps.python_cmd or sys.executable,
        workers=spec.worker.workers,
        clash_device=spec.worker.clash_device,
        analysis_batch_size=spec.worker.clash_batch_size,
        dssp_algorithm=spec.worker.dssp_algorithm,  # type: ignore[arg-type]
        tool_used=spec.worker.tool_used,
        homodimer_tool_used=spec.worker.homodimer_tool_used,
        heterodimers=spec.worker.heterodimers,
        parallel_stages=spec.worker.parallel_stages,
        no_cache=True,
    )


def _pipeline_model_ids(
    original_ids: Iterable[str],
    rename_map: Mapping[str, str],
    *,
    retry_failed_only: bool,
) -> tuple[str, ...]:
    _ = retry_failed_only
    return tuple(rename_map.get(model_id, model_id) for model_id in original_ids)


def _with_pipeline_runtime_context(
    command: PipelineCommand,
    *,
    spec: RunSpec,
    shard_dir: Path,
    batch_index: int,
) -> PipelineCommand:
    log_dir = shard_dir / "pipeline-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    prefix = log_dir / f"batch_{batch_index:04d}"
    argv_path = prefix.with_suffix(".argv.txt")
    stdout_path = prefix.with_suffix(".stdout.txt")
    stderr_path = prefix.with_suffix(".stderr.txt")
    argv_path.write_text(shlex.join(command.argv) + "\n", encoding="utf-8")
    return PipelineCommand(
        argv=command.argv,
        env=(
            *command.env,
            ("DUCKDB_MEMORY_LIMIT", spec.worker.duckdb_memory_limit),
            ("BSPP_PIPELINE_STDOUT", str(stdout_path)),
            ("BSPP_PIPELINE_STDERR", str(stderr_path)),
        ),
        working_dir=command.working_dir,
    )


def _tracking_rows(
    *,
    original_ids: Iterable[str],
    model_ids: Iterable[str],
    status_by_id: Mapping[str, str],
    failure_reasons: Mapping[str, str],
    archive_name: str,
    logical_shard_id: int,
    task_id: int,
    now: Callable[[], datetime],
) -> tuple[dict[str, object], ...]:
    timestamp = now().isoformat()
    return tuple(
        {
            "model_id": model_id,
            "original_id": original_id,
            "upload_status": status_by_id.get(model_id, "pipeline_failed"),
            "tar_of_origin": archive_name,
            "shard_id": logical_shard_id,
            "task_id": task_id,
            "failure_reason": failure_reasons.get(model_id, ""),
            "timestamp": timestamp,
        }
        for original_id, model_id in zip(original_ids, model_ids, strict=True)
    )


def _processing_rows(
    bad_models: Iterable[tuple[str, str]],
    *,
    status: str,
    archive_name: str,
    logical_shard_id: int,
    task_id: int,
    now: Callable[[], datetime],
) -> tuple[dict[str, object], ...]:
    timestamp = now().isoformat()
    return tuple(
        {
            "model_id": model_id,
            "original_id": model_id,
            "upload_status": status,
            "tar_of_origin": archive_name,
            "shard_id": logical_shard_id,
            "task_id": task_id,
            "failure_reason": reason,
            "timestamp": timestamp,
        }
        for model_id, reason in bad_models
    )


def _append_processing_log_rows(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    _append_csv_rows_locked(path, PROCESSING_LOG_FIELDS, rows)


def _append_analysis_metadata_rows(path: Path | None, rows: Iterable[Mapping[str, object]]) -> None:
    if path is not None:
        _append_csv_rows_locked(path, ANALYSIS_METADATA_FIELDS, rows)


def _append_csv_rows_locked(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    row_tuple = tuple(rows)
    if not row_tuple:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(".lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            write_header = not path.exists() or path.stat().st_size == 0
            with path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerows(row_tuple)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _tar_manifest_row(
    deps: WorkerDependencies,
    *,
    run_name: str,
    tar_type: str,
    s3_uri: str,
    tar_name: str,
    archive_name: str,
    logical_shard_id: int,
    task_id: int,
    batch_id: str,
    member_count: int,
    size_bytes: int,
    compression: str,
) -> dict[str, object]:
    return {
        "timestamp": deps.now().isoformat(),
        "run_name": run_name,
        "tar_type": tar_type,
        "s3_uri": s3_uri,
        "tar_name": tar_name,
        "source_archive": archive_name,
        "shard_id": logical_shard_id,
        "task_id": task_id,
        "batch_id": batch_id,
        "member_count": member_count,
        "size_bytes": size_bytes,
        "compression": compression,
    }


def _metadata_dataset_tag(spec: RunSpec) -> str:
    if not spec.worker.retry_failed_only:
        return spec.dataset.name
    delta_tag = spec.worker.retry_metadata_delta_tag.strip()
    if not delta_tag:
        return spec.dataset.name
    return f"{spec.dataset.name}-{delta_tag}" if spec.dataset.name else delta_tag


def _analysis_metadata_rows(
    deps: WorkerDependencies,
    spec: RunSpec,
    *,
    work_dir: Path,
    original_ids: Iterable[str],
    model_ids: Iterable[str],
    status_by_id: Mapping[str, str],
    failure_reasons: Mapping[str, str],
    archive_name: str,
    logical_shard_id: int,
    task_id: int,
    batch_id: int | str,
    batch_started_at: str,
    batch_finished_at: str,
    include_scores: bool,
    scores_by_id: Mapping[str, Mapping[str, object]] | None = None,
) -> tuple[dict[str, object], ...]:
    score_map = (
        scores_by_id if scores_by_id is not None else (_collect_analysis_scores(work_dir) if include_scores else {})
    )
    timestamp = deps.now().isoformat()
    rows: list[dict[str, object]] = []
    for original_id, model_id in zip(original_ids, model_ids, strict=True):
        scores = dict(score_map.get(model_id, {}))
        ipsae_max = _max_numeric_score(scores, _is_ipsae_threshold_col)
        pdockq2_max = _max_numeric_score(scores, _is_pdockq2_threshold_col)
        passes = (
            ipsae_max is not None
            and pdockq2_max is not None
            and ipsae_max >= spec.analysis_metadata.ipsae_threshold
            and pdockq2_max >= spec.analysis_metadata.pdockq2_threshold
        )
        rows.append(
            {
                "timestamp": timestamp,
                "run_name": spec.dataset.name,
                "model_id": model_id,
                "original_id": original_id,
                "upload_status": status_by_id.get(model_id, "pipeline_failed"),
                "passes_quality_threshold": "true" if passes else "false",
                "ipsae_max": "" if ipsae_max is None else ipsae_max,
                "pdockq2_max": "" if pdockq2_max is None else pdockq2_max,
                "quality_ipsae_threshold": spec.analysis_metadata.ipsae_threshold,
                "quality_pdockq2_threshold": spec.analysis_metadata.pdockq2_threshold,
                "source_archive": archive_name,
                "shard_id": logical_shard_id,
                "task_id": task_id,
                "batch_id": batch_id,
                "batch_started_at": batch_started_at,
                "batch_finished_at": batch_finished_at,
                "failure_reason": failure_reasons.get(model_id, ""),
                "expected_output_files_json": json.dumps(
                    [f"{model_id}{suffix}" for suffix in EXPECTED_OUTPUT_SUFFIXES],
                    separators=(",", ":"),
                ),
                "scores_json": json.dumps(scores, sort_keys=True, separators=(",", ":")),
            },
        )
    return tuple(rows)


def _collect_analysis_scores(work_dir: Path) -> dict[str, dict[str, object]]:
    scores_by_id = _parse_ipsae_score_rows(work_dir / "ipsae" / "ipsae_summary.csv")
    for model_id, scores in _parse_clash_interface_scores(work_dir).items():
        scores_by_id.setdefault(model_id, {}).update(scores)
    return scores_by_id


def _parse_ipsae_score_rows(ipsae_csv: Path) -> dict[str, dict[str, object]]:
    if not ipsae_csv.exists() or ipsae_csv.stat().st_size == 0:
        return {}
    scores: dict[str, dict[str, object]] = {}
    with ipsae_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            model_id = _model_id_from_pdb_path(row.get("pdb_path", ""))
            if not model_id:
                continue
            model_scores: dict[str, object] = {}
            for key, value in row.items():
                if not key or key in {"pdb_path", "processing_time_ms"}:
                    continue
                parsed = _try_float(value)
                model_scores[key] = parsed if parsed is not None else value
            scores[model_id] = model_scores
    return scores


def _parse_clash_interface_scores(work_dir: Path) -> dict[str, dict[str, object]]:
    analysis_dir = work_dir / "clash_interface_analysis"
    if not analysis_dir.exists():
        return {}
    scores_by_id: dict[str, dict[str, object]] = {}
    clash_suffix = "-model_v1_clashes.json"
    interface_suffix = "-model_v1_interface.json"
    for path in analysis_dir.glob(f"*{clash_suffix}"):
        model_id = _strip_suffix(path.name, clash_suffix)
        if not model_id:
            continue
        scores = scores_by_id.setdefault(model_id, {})
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for site in data.get("sites", []):
            label = site.get("label")
            annotations = site.get("additional_site_annotations", {})
            if label == "backbone_clashes":
                scores["N_clash_backbone"] = annotations.get("n_clashes", 0)
            elif label in {"heavy_atom_clashes", "side_chain_clashes"}:
                scores["N_clash_heavyAtom"] = annotations.get("n_clashes", 0)
        scores.setdefault("N_clash_backbone", 0)
        scores.setdefault("N_clash_heavyAtom", 0)
    for path in analysis_dir.glob(f"*{interface_suffix}"):
        model_id = _strip_suffix(path.name, interface_suffix)
        if not model_id:
            continue
        scores = scores_by_id.setdefault(model_id, {})
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        scores["N_interface_interactions"] = sum(
            len(site.get("additional_site_annotations", {}).get("interactions", [])) for site in data.get("sites", [])
        )
    return scores_by_id


def _model_id_from_pdb_path(pdb_path: str | None) -> str | None:
    if not pdb_path:
        return None
    name = Path(pdb_path).stem
    for suffix in ("-model_v1", "-model-v1"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def _strip_suffix(value: str, suffix: str) -> str | None:
    return value[: -len(suffix)] if value.endswith(suffix) else None


def _try_float(value: object) -> float | None:
    if not isinstance(value, str | bytes | int | float):
        return None
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_ipsae_threshold_col(column: str) -> bool:
    return column == "ipsae" or (column.startswith("ipsae_") and len(column) == len("ipsae_AB"))


def _is_pdockq2_threshold_col(column: str) -> bool:
    return column == "pDockQ2" or (column.startswith("pDockQ2_") and len(column) == len("pDockQ2_AB"))


def _max_numeric_score(scores: Mapping[str, object], predicate: Callable[[str], bool]) -> float | None:
    values = [value for key, value in scores.items() if predicate(key) and isinstance(value, int | float)]
    return max(values) if values else None


def _discover_workspace_model_ids(input_dir: Path) -> tuple[str, ...]:
    members = tuple(sorted(path.relative_to(input_dir).as_posix() for path in input_dir.rglob("*") if path.is_file()))
    return discover_model_ids(members)


def _load_archive_names(path: Path) -> tuple[str, ...]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        msg = f"Archive list must be a JSON list of strings: {path}"
        raise ValueError(msg)
    return tuple(data)


def _load_allowlist(path: Path) -> tuple[str, ...] | None:
    if not path.exists():
        return None
    return tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _run_pipeline_subprocess(command: PipelineCommand) -> PipelineResult:
    started = datetime.now(UTC)
    env: dict[str, str] | None = None
    stdout_path: Path | None = None
    stderr_path: Path | None = None
    if command.env:
        env = os.environ.copy()
        env.update(dict(command.env))
        stdout_raw = env.get("BSPP_PIPELINE_STDOUT")
        stderr_raw = env.get("BSPP_PIPELINE_STDERR")
        stdout_path = Path(stdout_raw) if stdout_raw else None
        stderr_path = Path(stderr_raw) if stderr_raw else None

    result = subprocess.run(
        command.argv,
        capture_output=True,
        text=True,
        cwd=command.working_dir,
        env=env,
    )
    if stdout_path is not None:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text(result.stdout, encoding="utf-8")
    if stderr_path is not None:
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.write_text(result.stderr, encoding="utf-8")
    elapsed = (datetime.now(UTC) - started).total_seconds()
    return PipelineResult(
        exit_code=result.returncode,
        elapsed_seconds=elapsed,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


def _noop_sleep(_seconds: float) -> None:
    return None


__all__ = [
    "ANALYSIS_METADATA_FIELDS",
    "PROCESSING_LOG_FIELDS",
    "RETRY_FAILED_STATUSES",
    "TAR_MANIFEST_FIELDS",
    "BatchSyncResult",
    "MetadataTransferResult",
    "WorkerDependencies",
    "append_tar_manifest_rows",
    "load_retry_failed_ids_from_state",
    "run_archive_task",
]
