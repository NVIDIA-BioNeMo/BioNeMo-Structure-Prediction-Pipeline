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

"""Pure, non-executing folding lifecycle composition.

Port Baseline: ``3864d0eda67e70979b8e48f00ed6a08f9e71c59e`` from
``folding/openfold-pipeline/docs/openfoldctl.md:1-582`` and
``folding/openfold-pipeline/scripts/openfoldctl.py:392-552``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from bspp.orchestration.contract.folding_archive import ArchivePlanOptions
from bspp.orchestration.contract.folding_checkpoint import (
    FoldingCheckpointState,
    FoldingRemainingWorkPlan,
)
from bspp.orchestration.contract.folding_index import FoldingIndexRecord
from bspp.orchestration.contract.folding_lifecycle import (
    FoldingArchiveLifecycleDecision,
    FoldingArchiveLifecycleRequest,
    FoldingCheckpointInputs,
    FoldingIndexLifecycleDecision,
    FoldingIndexLifecycleRequest,
    FoldingLifecycleConfig,
    FoldingLifecycleState,
    FoldingMergeLifecycleDecision,
    FoldingMergeLifecycleRequest,
    FoldingPreflightCheck,
    FoldingPreflightLifecycleDecision,
    FoldingPreflightLifecycleRequest,
    FoldingQueueWriteDeclaration,
    FoldingResumeLifecycleDecision,
    FoldingResumeLifecycleRequest,
    FoldingStatusEvidenceKind,
    FoldingStatusLifecycleDecision,
    FoldingStatusLifecycleRequest,
    FoldingSubmitLifecycleDecision,
    FoldingSubmitLifecycleRequest,
)
from bspp.orchestration.contract.folding_queue import FoldingQueueConfig
from bspp.orchestration.runtime.folding.archive import plan_folding_archives, render_archive_manifest_jsonl
from bspp.orchestration.runtime.folding.checkpoint import (
    load_checkpoint_state,
    merge_checkpoint_records,
    normalize_checkpoint_protein_id,
)
from bspp.orchestration.runtime.folding.indexing import index_folding_a3ms
from bspp.orchestration.runtime.folding.queue import plan_folding_queue
from bspp.orchestration.runtime.folding.queue_render import render_folding_queues
from bspp.orchestration.runtime.folding.results import scan_folding_results
from bspp.orchestration.runtime.folding.resume import plan_checkpoint_resume


def plan_folding_index_lifecycle(request: FoldingIndexLifecycleRequest) -> FoldingIndexLifecycleDecision:
    """Delegate the baseline controller's local index intent to the #49 seam."""
    if not isinstance(request, FoldingIndexLifecycleRequest):
        msg = "request must be a FoldingIndexLifecycleRequest"
        raise ValueError(msg)
    config = request.config
    _require_passing_preflight(
        FoldingPreflightLifecycleRequest(
            action="index",
            config=config,
            index=None,
            checkpoint_inputs=None,
            checkpoint_state=None,
            result_inventory=None,
            scan_predictions_root=None,
        )
    )
    index = index_folding_a3ms(
        tuple(Path(path).resolve(strict=True) for path in config.declared_a3m_inputs),
        sort_by_length=config.sort_by_length,
    )
    return FoldingIndexLifecycleDecision(
        index=index,
        index_worker_count=config.index_worker_count,
        sort_by_length=config.sort_by_length,
    )


def plan_folding_submit_lifecycle(request: FoldingSubmitLifecycleRequest) -> FoldingSubmitLifecycleDecision:
    """Plan queue artifacts while explicitly withholding submission authority."""
    if not isinstance(request, FoldingSubmitLifecycleRequest):
        msg = "request must be a FoldingSubmitLifecycleRequest"
        raise ValueError(msg)
    _require_passing_preflight(
        FoldingPreflightLifecycleRequest(
            action="submit",
            config=request.config,
            index=request.index,
            checkpoint_inputs=None,
            checkpoint_state=None,
            result_inventory=None,
            scan_predictions_root=None,
        )
    )
    return _plan_submit_for_records(request.config, request.index.records)


def _plan_submit_for_records(
    config: FoldingLifecycleConfig,
    records: tuple[FoldingIndexRecord, ...],
) -> FoldingSubmitLifecycleDecision:
    if not isinstance(config, FoldingLifecycleConfig):
        msg = "config must be a FoldingLifecycleConfig"
        raise ValueError(msg)
    queue_config = FoldingQueueConfig(
        strategy=config.queue_strategy,
        layout=config.queue_layout,
        worker_count=config.worker_count,
        max_proteins_per_batch=config.max_proteins_per_batch,
    )
    queue_plan = plan_folding_queue(records, queue_config)
    rendered = render_folding_queues(queue_plan)
    queue_writes = tuple(
        FoldingQueueWriteDeclaration(
            path=str(Path(config.work_dir) / artifact.file_name),
            artifact=artifact,
        )
        for artifact in rendered.artifacts
    )
    return FoldingSubmitLifecycleDecision(
        queue_plan=queue_plan,
        queue_writes=queue_writes,
        submission_planned=bool(queue_plan.assignments),
    )


def plan_folding_resume_lifecycle(request: FoldingResumeLifecycleRequest) -> FoldingResumeLifecycleDecision:
    """Combine #51 checkpoint policy with only #52 complete result pairs."""
    if not isinstance(request, FoldingResumeLifecycleRequest):
        msg = "request must be a FoldingResumeLifecycleRequest"
        raise ValueError(msg)
    _require_passing_preflight(
        FoldingPreflightLifecycleRequest(
            action="resume",
            config=request.config,
            index=request.index,
            checkpoint_inputs=None,
            checkpoint_state=request.checkpoint_state,
            result_inventory=request.result_inventory,
            scan_predictions_root=None,
        )
    )
    source_records = tuple(sorted(request.index.records, key=lambda record: record.source_ordinal))
    planned_ids = tuple(record.protein_id for record in source_records)
    checkpoint_plan = plan_checkpoint_resume(
        planned_ids,
        checkpoint_state=request.checkpoint_state,
        retry_failed=request.retry_failed,
    )
    recovered_set = {
        association.protein_id
        for association in request.result_inventory.associations
        if association.status == "complete"
    }
    recovered_ids = tuple(protein_id for protein_id in planned_ids if protein_id in recovered_set)
    completed_set = set(checkpoint_plan.completed_model_ids) | recovered_set
    remaining_work = FoldingRemainingWorkPlan(
        planned_model_ids=planned_ids,
        completed_model_ids=tuple(protein_id for protein_id in planned_ids if protein_id in completed_set),
        transient_failure_ids=tuple(
            protein_id for protein_id in checkpoint_plan.transient_failure_ids if protein_id not in recovered_set
        ),
        permanent_failure_ids=tuple(
            protein_id for protein_id in checkpoint_plan.permanent_failure_ids if protein_id not in recovered_set
        ),
        remaining_model_ids=tuple(
            protein_id for protein_id in checkpoint_plan.remaining_model_ids if protein_id not in recovered_set
        ),
        unplanned_checkpoint_ids=checkpoint_plan.unplanned_checkpoint_ids,
        retry_failed=request.retry_failed,
    )
    remaining_set = set(remaining_work.remaining_model_ids)
    submit_decision = _plan_submit_for_records(
        request.config,
        tuple(record for record in source_records if record.protein_id in remaining_set),
    )
    return FoldingResumeLifecycleDecision(
        remaining_work=remaining_work,
        recovered_result_ids=recovered_ids,
        submit_decision=submit_decision,
    )


def plan_folding_archive_lifecycle(request: FoldingArchiveLifecycleRequest) -> FoldingArchiveLifecycleDecision:
    """Delegate explicit result association and archive planning to #52."""
    if not isinstance(request, FoldingArchiveLifecycleRequest):
        msg = "request must be a FoldingArchiveLifecycleRequest"
        raise ValueError(msg)
    config = request.config
    _require_passing_preflight(
        FoldingPreflightLifecycleRequest(
            action="archive",
            config=config,
            index=request.index,
            checkpoint_inputs=None,
            checkpoint_state=None,
            result_inventory=request.result_inventory,
            scan_predictions_root=request.scan_predictions_root,
        )
    )
    inventory = request.result_inventory
    if inventory is None:
        if request.scan_predictions_root is None:
            msg = "scan_predictions_root is required when result_inventory is absent"
            raise ValueError(msg)
        inventory = scan_folding_results(request.index, Path(request.scan_predictions_root))
        _require_passing_preflight(
            FoldingPreflightLifecycleRequest(
                action="archive",
                config=config,
                index=request.index,
                checkpoint_inputs=None,
                checkpoint_state=None,
                result_inventory=inventory,
                scan_predictions_root=None,
            )
        )
    options = ArchivePlanOptions(
        run_tag=_required_archive_text(config.archive_run_tag, "archive_run_tag"),
        stage_root=_required_archive_text(config.archive_stage_root, "archive_stage_root"),
        archive_root=_required_archive_text(config.archive_root, "archive_root"),
        proteins_per_archive=config.proteins_per_archive,
        lz4_executable=config.lz4_executable,
        start_index=config.archive_start_index,
        max_archives=config.archive_max_archives,
        shuffle=config.archive_shuffle,
        shuffle_seed=config.archive_shuffle_seed,
    )
    archive_plan = plan_folding_archives(
        inventory,
        options,
        selected_protein_ids=request.selected_protein_ids,
        prior_manifest=request.prior_manifest,
        force=request.force,
    )
    return FoldingArchiveLifecycleDecision(
        result_inventory=inventory,
        archive_plan=archive_plan,
        manifest_jsonl=render_archive_manifest_jsonl(archive_plan),
    )


def preflight_folding_lifecycle(request: FoldingPreflightLifecycleRequest) -> FoldingPreflightLifecycleDecision:
    """Interpret caller-supplied local prerequisites without probing tools or a scheduler."""
    if not isinstance(request, FoldingPreflightLifecycleRequest):
        msg = "request must be a FoldingPreflightLifecycleRequest"
        raise ValueError(msg)
    decision, _checkpoint_state = _evaluate_preflight(request)
    return decision


def _evaluate_preflight(
    request: FoldingPreflightLifecycleRequest,
) -> tuple[FoldingPreflightLifecycleDecision, FoldingCheckpointState | None]:
    checkpoint_check, checkpoint_state = _checkpoint_preflight(
        request,
        required=request.action in {"resume", "merge"},
    )
    checks = (
        _config_preflight(request),
        _index_preflight(request, required=request.action in {"submit", "resume", "archive"}),
        checkpoint_check,
        _result_preflight(request, required=request.action in {"resume", "archive"}),
        _archive_preflight(request, required=request.action == "archive"),
    )
    return (
        FoldingPreflightLifecycleDecision(
            action=request.action,
            checks=checks,
            can_plan=all(check.status != "fail" for check in checks),
        ),
        checkpoint_state,
    )


def interpret_folding_status_lifecycle(request: FoldingStatusLifecycleRequest) -> FoldingStatusLifecycleDecision:
    """Summarize supplied manifest/checkpoint/result/archive evidence only."""
    if not isinstance(request, FoldingStatusLifecycleRequest):
        msg = "request must be a FoldingStatusLifecycleRequest"
        raise ValueError(msg)
    source_records = tuple(sorted(request.index.records, key=lambda record: record.source_ordinal))
    planned_ids = tuple(record.protein_id for record in source_records)
    planned_normalized_ids = tuple(normalize_checkpoint_protein_id(protein_id) for protein_id in planned_ids)
    planned_normalized = set(planned_normalized_ids)
    invalid: list[str] = []

    if request.manifest is not None:
        if request.manifest.total_proteins != len(planned_ids):
            invalid.append("manifest total does not match folding index")
        if request.manifest.queue_layout != request.config.queue_layout:
            invalid.append("manifest queue layout does not match folding configuration")
        if request.manifest.queue_artifact_count > min(request.config.worker_count, len(planned_ids)):
            invalid.append("manifest queue artifact count exceeds the configured run shape")
        if bool(request.manifest.total_proteins) != bool(request.manifest.queue_artifact_count):
            invalid.append("manifest queue artifact count contradicts whether indexed work exists")

    checkpoint_completed: set[str] = set()
    checkpoint_ids: set[str] = set()
    if request.checkpoint_state is not None:
        checkpoint_completed = {
            normalize_checkpoint_protein_id(record.protein_id) for record in request.checkpoint_state.completions
        }
        checkpoint_ids = checkpoint_completed | {
            normalize_checkpoint_protein_id(record.protein_id) for record in request.checkpoint_state.failures
        }
        if checkpoint_ids - planned_normalized:
            invalid.append("checkpoint contains identities outside folding index")

    recovered: set[str] = set()
    incomplete_result_count = 0
    if request.result_inventory is not None:
        associated_ids = tuple(item.protein_id for item in request.result_inventory.associations)
        if associated_ids != planned_ids:
            invalid.append("result associations do not cover folding index in source order")
        if request.result_inventory.unmatched or any(
            item.status in {"duplicate", "identity-collision"} for item in request.result_inventory.associations
        ):
            invalid.append("result inventory contains conflicting or unmatched evidence")
        recovered = {
            normalize_checkpoint_protein_id(item.protein_id)
            for item in request.result_inventory.associations
            if item.status == "complete"
        }
        incomplete_result_count = sum(item.status != "complete" for item in request.result_inventory.associations)
        if checkpoint_completed - recovered:
            invalid.append("checkpoint completion lacks a complete result pair in supplied result evidence")

    planned_archive_count = 0
    if request.archive_plan is not None:
        planned_archive_count = len(request.archive_plan.batches)
        archived_ids = {
            normalize_checkpoint_protein_id(protein_id)
            for batch in request.archive_plan.batches
            for protein_id in batch.protein_ids
        }
        if archived_ids - planned_normalized:
            invalid.append("archive plan contains identities outside folding index")

    completed_count = len((checkpoint_completed | recovered) & planned_normalized)
    evidence_kinds_list: list[FoldingStatusEvidenceKind] = []
    if request.manifest is not None:
        evidence_kinds_list.append("manifest")
    if request.checkpoint_state is not None:
        evidence_kinds_list.append("checkpoint")
    if request.result_inventory is not None:
        evidence_kinds_list.append("result")
    if request.archive_plan is not None:
        evidence_kinds_list.append("archive")
    any_evidence = bool(evidence_kinds_list)
    if any_evidence and len(planned_normalized) != len(planned_normalized_ids):
        invalid.append("folding index identities collide after checkpoint normalization")
    state: FoldingLifecycleState
    if invalid:
        state = "invalid"
    elif not any_evidence:
        state = "initial"
    elif completed_count == len(planned_ids):
        state = "complete"
    else:
        state = "partial"
    return FoldingStatusLifecycleDecision(
        state=state,
        evidence_kinds=tuple(evidence_kinds_list),
        planned_model_count=len(planned_ids),
        manifest_model_count=None if request.manifest is None else request.manifest.total_proteins,
        checkpoint_completed_count=(
            0 if request.checkpoint_state is None else len(request.checkpoint_state.completions)
        ),
        checkpoint_failed_count=0 if request.checkpoint_state is None else len(request.checkpoint_state.failures),
        recovered_result_count=len(recovered),
        completed_model_count=completed_count,
        incomplete_result_count=incomplete_result_count,
        planned_archive_count=planned_archive_count,
        invalid_evidence=tuple(invalid),
    )


def _config_preflight(request: FoldingPreflightLifecycleRequest) -> FoldingPreflightCheck:
    config = request.config
    required_files: tuple[tuple[str, str], ...] = ()
    required_dirs: tuple[tuple[str, str], ...] = ()
    if request.action == "index":
        invalid_a3m_inputs = [
            value for value in config.declared_a3m_inputs if not Path(value).is_file() and not Path(value).is_dir()
        ]
        if invalid_a3m_inputs:
            return FoldingPreflightCheck(
                name="config",
                status="fail",
                detail="declared A3M inputs must be existing files or directories: " + ", ".join(invalid_a3m_inputs),
            )
    elif request.action in {"submit", "resume"}:
        required_files = (
            ("container_image", config.container_image),
            ("trt_checkpoint_path", config.trt_checkpoint_path),
        )
        required_dirs = (
            ("trt_bionemo_dir", config.trt_bionemo_dir),
            ("trt_engines_dir", config.trt_engines_dir),
        )

    invalid = [f"{name}: {value}" for name, value in required_files if not Path(value).is_file()]
    invalid.extend(f"{name}: {value}" for name, value in required_dirs if not Path(value).is_dir())
    if invalid:
        return FoldingPreflightCheck(
            name="config",
            status="fail",
            detail="missing or invalid required local paths: " + ", ".join(invalid),
        )

    planned_dirs: list[tuple[str, str]] = [
        ("output_dir", config.output_dir),
        ("work_dir", config.work_dir),
        ("predictions_root", config.predictions_root),
        ("checkpoint_dir", config.checkpoint_dir),
    ]
    if request.action == "archive" and config.archive_enabled:
        planned_dirs.extend(
            (
                ("archive_stage_root", _required_archive_text(config.archive_stage_root, "archive_stage_root")),
                ("archive_root", _required_archive_text(config.archive_root, "archive_root")),
            )
        )
    wrong_kind = [
        f"{name}: {value}" for name, value in planned_dirs if Path(value).exists() and not Path(value).is_dir()
    ]
    if wrong_kind:
        return FoldingPreflightCheck(
            name="config",
            status="fail",
            detail="planned directory paths exist with the wrong type: " + ", ".join(wrong_kind),
        )
    absent = [f"{name}: {value}" for name, value in planned_dirs if not Path(value).exists()]
    if absent:
        return FoldingPreflightCheck(
            name="config",
            status="warning",
            detail="planned directories do not exist yet: " + ", ".join(absent),
        )
    return FoldingPreflightCheck(name="config", status="pass", detail="required local paths are valid")


def _index_preflight(request: FoldingPreflightLifecycleRequest, *, required: bool) -> FoldingPreflightCheck:
    if not required:
        return FoldingPreflightCheck(name="index", status="not-required", detail="index not required for action")
    if request.index is None:
        return FoldingPreflightCheck(name="index", status="fail", detail="folding index is missing")
    if request.action in {"submit", "resume"}:
        missing = [record.msa_path for record in request.index.records if not Path(record.msa_path).is_file()]
        if missing:
            return FoldingPreflightCheck(
                name="index",
                status="fail",
                detail="index references missing local A3M files: " + ", ".join(missing),
            )
    return FoldingPreflightCheck(
        name="index",
        status="pass",
        detail=f"{len(request.index.records)} indexed model identities",
    )


def _checkpoint_preflight(
    request: FoldingPreflightLifecycleRequest,
    *,
    required: bool,
) -> tuple[FoldingPreflightCheck, FoldingCheckpointState | None]:
    if not required:
        return (
            FoldingPreflightCheck(
                name="checkpoint",
                status="not-required",
                detail="checkpoint not required for action",
            ),
            None,
        )
    if request.action == "merge":
        if request.checkpoint_inputs is None:
            return (
                FoldingPreflightCheck(
                    name="checkpoint",
                    status="fail",
                    detail="explicit checkpoint input slots are missing",
                ),
                None,
            )
        try:
            state = load_lifecycle_checkpoint_state(request.checkpoint_inputs)
        except ValueError as exc:
            return FoldingPreflightCheck(name="checkpoint", status="fail", detail=str(exc)), None
        return (
            FoldingPreflightCheck(
                name="checkpoint",
                status="pass",
                detail=f"{len(state.completions) + len(state.failures)} checkpoint identities are coherent",
            ),
            state,
        )
    if request.checkpoint_state is None:
        return (
            FoldingPreflightCheck(name="checkpoint", status="fail", detail="checkpoint state is missing"),
            None,
        )
    if request.index is None:
        return FoldingPreflightCheck(name="checkpoint", status="fail", detail="checkpoint requires an index"), None
    planned_ids = {normalize_checkpoint_protein_id(record.protein_id) for record in request.index.records}
    checkpoint_ids = {
        normalize_checkpoint_protein_id(record.protein_id) for record in request.checkpoint_state.completions
    } | {normalize_checkpoint_protein_id(record.protein_id) for record in request.checkpoint_state.failures}
    unknown = sorted(checkpoint_ids - planned_ids)
    if unknown:
        return (
            FoldingPreflightCheck(
                name="checkpoint",
                status="fail",
                detail="checkpoint contains unplanned identities: " + ", ".join(unknown),
            ),
            None,
        )
    return (
        FoldingPreflightCheck(
            name="checkpoint",
            status="pass",
            detail=f"{len(checkpoint_ids)} checkpoint identities are coherent",
        ),
        request.checkpoint_state,
    )


def _result_preflight(request: FoldingPreflightLifecycleRequest, *, required: bool) -> FoldingPreflightCheck:
    if not required:
        return FoldingPreflightCheck(name="result", status="not-required", detail="results not required for action")
    if (
        request.action == "archive"
        and request.result_inventory is not None
        and request.scan_predictions_root is not None
    ):
        return FoldingPreflightCheck(
            name="result",
            status="fail",
            detail="archive preflight requires exactly one supplied inventory or explicit scan root",
        )
    if request.action == "archive" and request.result_inventory is None:
        if request.scan_predictions_root is None or not Path(request.scan_predictions_root).is_dir():
            return FoldingPreflightCheck(
                name="result",
                status="fail",
                detail="archive requires supplied result evidence or an existing explicit scan root",
            )
        return FoldingPreflightCheck(
            name="result",
            status="pass",
            detail=f"explicit result scan root is available: {request.scan_predictions_root}",
        )
    if request.result_inventory is None:
        return FoldingPreflightCheck(name="result", status="fail", detail="result inventory is missing")
    if request.index is None:
        return FoldingPreflightCheck(name="result", status="fail", detail="results cannot be checked without an index")
    planned_ids = tuple(
        record.protein_id for record in sorted(request.index.records, key=lambda item: item.source_ordinal)
    )
    associated_ids = tuple(item.protein_id for item in request.result_inventory.associations)
    invalid = [
        item.protein_id
        for item in request.result_inventory.associations
        if item.status in {"duplicate", "identity-collision"}
    ]
    if associated_ids != planned_ids or invalid:
        return FoldingPreflightCheck(
            name="result",
            status="fail",
            detail="result inventory has missing association coverage or conflicting planned evidence",
        )
    missing_complete_paths = [
        path
        for item in request.result_inventory.associations
        if item.status == "complete"
        for path in (*item.pdb_paths, *item.json_paths)
        if not Path(path).is_file()
    ]
    if missing_complete_paths:
        return FoldingPreflightCheck(
            name="result",
            status="fail",
            detail="complete result associations reference missing local files: " + ", ".join(missing_complete_paths),
        )
    complete = sum(item.status == "complete" for item in request.result_inventory.associations)
    unmatched_count = len(request.result_inventory.unmatched)
    return FoldingPreflightCheck(
        name="result",
        status="warning" if unmatched_count else "pass",
        detail=(
            f"{complete} valid complete result associations; "
            f"{unmatched_count} unplanned or malformed result files remain informational"
            if unmatched_count
            else f"{complete} valid complete result associations"
        ),
    )


def _archive_preflight(request: FoldingPreflightLifecycleRequest, *, required: bool) -> FoldingPreflightCheck:
    if not required:
        return FoldingPreflightCheck(name="archive", status="not-required", detail="archive not required for action")
    if not request.config.archive_enabled:
        return FoldingPreflightCheck(name="archive", status="fail", detail="archive configuration is disabled")
    return FoldingPreflightCheck(name="archive", status="pass", detail="archive configuration is complete")


def _require_passing_preflight(request: FoldingPreflightLifecycleRequest) -> None:
    decision, _checkpoint_state = _evaluate_preflight(request)
    _raise_for_failed_preflight(decision)


def _require_passing_merge_preflight(request: FoldingPreflightLifecycleRequest) -> FoldingCheckpointState:
    decision, checkpoint_state = _evaluate_preflight(request)
    _raise_for_failed_preflight(decision)
    if checkpoint_state is None:
        msg = "passing merge preflight must return a canonical checkpoint state"
        raise ValueError(msg)
    return checkpoint_state


def _raise_for_failed_preflight(decision: FoldingPreflightLifecycleDecision) -> None:
    failed = tuple(check for check in decision.checks if check.status == "fail")
    if failed:
        details = "; ".join(f"{check.name}: {check.detail}" for check in failed)
        msg = f"folding {decision.action} preflight failed: {details}"
        raise ValueError(msg)


def _required_archive_text(value: str | None, field_name: str) -> str:
    if value is None:
        msg = f"{field_name} is required for archive planning"
        raise ValueError(msg)
    return value


def load_lifecycle_checkpoint_state(inputs: FoldingCheckpointInputs) -> FoldingCheckpointState:
    """Load explicit node/GPU/global slots with global evidence always last."""
    if not isinstance(inputs, FoldingCheckpointInputs):
        msg = "inputs must be FoldingCheckpointInputs"
        raise ValueError(msg)
    _reject_repeated_checkpoint_files(inputs)
    shard_state = load_checkpoint_state(
        current_node_completion_shards=_paths(inputs.node_completion_shards),
        current_gpu_completion_shards=_paths(inputs.current_gpu_completion_shards),
        legacy_completion_shards=_paths(inputs.historical_gpu_completion_shards),
        current_node_failure_shards=_paths(inputs.node_failure_shards),
        current_gpu_failure_shards=_paths(inputs.current_gpu_failure_shards),
        legacy_failure_shards=_paths(inputs.historical_gpu_failure_shards),
    )
    merged_state = load_checkpoint_state(
        current_node_completion_shards=_optional_path_tuple(inputs.merged_completion_view),
        current_node_failure_shards=_optional_path_tuple(inputs.merged_failure_view),
    )
    merged_ids = {record.protein_id for record in merged_state.completions} | {
        record.protein_id for record in merged_state.failures
    }
    shard_source_ordinals = tuple(record.source_ordinal for record in shard_state.completions) + tuple(
        record.source_ordinal for record in shard_state.failures
    )
    merged_source_offset = max(shard_source_ordinals, default=-1) + 1
    rebased_merged_completions = tuple(
        replace(record, source_ordinal=merged_source_offset + record.source_ordinal)
        for record in merged_state.completions
    )
    rebased_merged_failures = tuple(
        replace(record, source_ordinal=merged_source_offset + record.source_ordinal) for record in merged_state.failures
    )
    return merge_checkpoint_records(
        (
            *(record for record in shard_state.completions if record.protein_id not in merged_ids),
            *rebased_merged_completions,
        ),
        (
            *(record for record in shard_state.failures if record.protein_id not in merged_ids),
            *rebased_merged_failures,
        ),
    )


def plan_folding_merge_lifecycle(request: FoldingMergeLifecycleRequest) -> FoldingMergeLifecycleDecision:
    """Declare aggregate checkpoint views without writing either destination."""
    if not isinstance(request, FoldingMergeLifecycleRequest):
        msg = "request must be a FoldingMergeLifecycleRequest"
        raise ValueError(msg)
    state = _require_passing_merge_preflight(
        FoldingPreflightLifecycleRequest(
            action="merge",
            config=request.config,
            index=None,
            checkpoint_inputs=request.checkpoint_inputs,
            checkpoint_state=None,
            result_inventory=None,
            scan_predictions_root=None,
        )
    )
    checkpoint_dir = Path(request.config.checkpoint_dir)
    return FoldingMergeLifecycleDecision(
        state=state,
        completed_view_path=str(checkpoint_dir / "completed_all.csv"),
        failed_view_path=str(checkpoint_dir / "failed_all.csv"),
        # The lifecycle always declares both canonical views; an empty state
        # still calls for header-only completion and failure CSVs.
        write_planned=True,
    )


def _paths(values: tuple[str, ...]) -> tuple[Path, ...]:
    return tuple(Path(value) for value in values)


def _optional_path_tuple(value: str | None) -> tuple[Path, ...]:
    return () if value is None else (Path(value),)


def _reject_repeated_checkpoint_files(inputs: FoldingCheckpointInputs) -> None:
    declared = (
        *inputs.node_completion_shards,
        *inputs.node_failure_shards,
        *inputs.current_gpu_completion_shards,
        *inputs.current_gpu_failure_shards,
        *inputs.historical_gpu_completion_shards,
        *inputs.historical_gpu_failure_shards,
        *((inputs.merged_completion_view,) if inputs.merged_completion_view is not None else ()),
        *((inputs.merged_failure_view,) if inputs.merged_failure_view is not None else ()),
    )
    seen: set[Path] = set()
    for value in declared:
        try:
            identity = Path(value).resolve(strict=True)
        except OSError as exc:
            msg = f"Cannot resolve lifecycle checkpoint input {value}: {exc}"
            raise ValueError(msg) from exc
        if identity in seen:
            msg = f"Checkpoint file occupies more than one lifecycle checkpoint slot: {value}"
            raise ValueError(msg)
        seen.add(identity)


__all__ = [
    "interpret_folding_status_lifecycle",
    "load_lifecycle_checkpoint_state",
    "plan_folding_archive_lifecycle",
    "plan_folding_index_lifecycle",
    "plan_folding_merge_lifecycle",
    "plan_folding_resume_lifecycle",
    "plan_folding_submit_lifecycle",
    "preflight_folding_lifecycle",
]
