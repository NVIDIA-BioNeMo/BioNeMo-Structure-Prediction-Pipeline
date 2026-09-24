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

"""CLI entry point for bspp-orchestration."""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import click

from bspp.orchestration.runtime.config import load_config, resolve_path
from bspp.orchestration.runtime.constants import MAX_PROTEINS_PER_SHARD

if TYPE_CHECKING:
    from bspp.orchestration.contract.phase import PhaseRunSpec
    from bspp.orchestration.contract.phase_carry_forward import AttemptCarryForwardRecord
    from bspp.orchestration.contract.preprocessing_action import PreprocessingChunkActionEvidence
    from bspp.orchestration.contract.runspec import RunSpec
    from bspp.orchestration.runtime.preprocessing.database_placement import DatabasePlacementCommandResult


class _PhaseRunSpecLoader(Protocol):
    def __call__(self, path: Path) -> PhaseRunSpec: ...


class _CarryForwardRecordLoader(Protocol):
    def __call__(self, path: Path) -> AttemptCarryForwardRecord: ...


class _ChunkExecutor(Protocol):
    def __call__(
        self,
        runspec: PhaseRunSpec,
        *,
        action_id: str,
        evidence_path: Path,
        database_placement_result_path: Path,
        placement_process_status: int,
        database_placement_failure_path: Path | None = None,
        carry_forward_record: AttemptCarryForwardRecord | None = None,
        phase_submission_id: str | None = None,
    ) -> PreprocessingChunkActionEvidence: ...


class _DatabasePlacer(Protocol):
    def __call__(
        self,
        runspec: PhaseRunSpec,
        *,
        action_id: str,
        source_manifest_path: Path,
        result_path: Path,
        failure_path: Path | None = None,
    ) -> DatabasePlacementCommandResult: ...


@dataclass(frozen=True)
class ExecutionCoordinator:
    """Fixed preprocessing application dependencies for private CLI composition."""

    load_phase_runspec: _PhaseRunSpecLoader
    load_carry_forward_record: _CarryForwardRecordLoader
    execute_chunk: _ChunkExecutor
    place_database: _DatabasePlacer


def _production_load_phase_runspec(path: Path) -> PhaseRunSpec:
    from bspp.orchestration.runtime.preprocessing.execution import load_preprocessing_phase_runspec

    return load_preprocessing_phase_runspec(path)


def _production_load_carry_forward_record(path: Path) -> AttemptCarryForwardRecord:
    from bspp.orchestration.runtime.preprocessing.carry_forward import load_attempt_carry_forward_record

    return load_attempt_carry_forward_record(path)


def _production_execute_chunk(
    runspec: PhaseRunSpec,
    *,
    action_id: str,
    evidence_path: Path,
    database_placement_result_path: Path,
    placement_process_status: int,
    database_placement_failure_path: Path | None = None,
    carry_forward_record: AttemptCarryForwardRecord | None = None,
    phase_submission_id: str | None = None,
) -> PreprocessingChunkActionEvidence:
    from bspp.orchestration.runtime.preprocessing.execution import execute_preprocessing_chunk_action

    return execute_preprocessing_chunk_action(
        runspec,
        action_id=action_id,
        evidence_path=evidence_path,
        database_placement_result_path=database_placement_result_path,
        placement_process_status=placement_process_status,
        database_placement_failure_path=database_placement_failure_path,
        carry_forward_record=carry_forward_record,
        phase_submission_id=phase_submission_id,
    )


def _production_place_database(
    runspec: PhaseRunSpec,
    *,
    action_id: str,
    source_manifest_path: Path,
    result_path: Path,
    failure_path: Path | None = None,
) -> DatabasePlacementCommandResult:
    from bspp.orchestration.runtime.preprocessing.database_placement import place_database

    return place_database(
        runspec,
        action_id=action_id,
        source_manifest_path=source_manifest_path,
        result_path=result_path,
        failure_path=failure_path,
    )


PRODUCTION_EXECUTION_COORDINATOR = ExecutionCoordinator(
    load_phase_runspec=_production_load_phase_runspec,
    load_carry_forward_record=_production_load_carry_forward_record,
    execute_chunk=_production_execute_chunk,
    place_database=_production_place_database,
)


def _run_preprocessing_place_database(
    coordinator: ExecutionCoordinator,
    *,
    phase_runspec_path: Path,
    action_id: str,
    source_manifest_path: Path,
    result_path: Path,
    failure_path: Path,
) -> None:
    from bspp.orchestration.runtime.preprocessing.database_placement import DatabasePlacementError

    try:
        runspec = coordinator.load_phase_runspec(phase_runspec_path)
        result = coordinator.place_database(
            runspec,
            action_id=action_id,
            source_manifest_path=source_manifest_path,
            result_path=result_path,
            failure_path=failure_path,
        )
    except (OSError, TypeError, ValueError, DatabasePlacementError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(result.to_mapping(), indent=2, sort_keys=True))


def _run_preprocessing_execute_chunk(
    coordinator: ExecutionCoordinator,
    *,
    phase_runspec_path: Path,
    action_id: str,
    evidence_path: Path,
    database_placement_result_path: Path,
    database_placement_failure_path: Path | None,
    placement_process_status: int,
    carry_forward_record_path: Path | None,
    phase_submission_id: str | None,
) -> None:
    from bspp.orchestration.runtime.preprocessing.execution import PreprocessingExecutionError

    try:
        runspec = coordinator.load_phase_runspec(phase_runspec_path)
        carry_record = (
            coordinator.load_carry_forward_record(carry_forward_record_path)
            if carry_forward_record_path is not None
            else None
        )
        evidence = coordinator.execute_chunk(
            runspec,
            action_id=action_id,
            evidence_path=evidence_path,
            database_placement_result_path=database_placement_result_path,
            database_placement_failure_path=database_placement_failure_path,
            placement_process_status=placement_process_status,
            carry_forward_record=carry_record,
            phase_submission_id=phase_submission_id,
        )
    except (OSError, TypeError, ValueError, PreprocessingExecutionError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(evidence.to_mapping(), indent=2, sort_keys=True))


def _place_database_callback(coordinator: ExecutionCoordinator) -> Any:
    def callback(
        phase_runspec_path: Path,
        action_id: str,
        source_manifest_path: Path,
        result_path: Path,
        failure_path: Path,
    ) -> None:
        _run_preprocessing_place_database(
            coordinator,
            phase_runspec_path=phase_runspec_path,
            action_id=action_id,
            source_manifest_path=source_manifest_path,
            result_path=result_path,
            failure_path=failure_path,
        )

    return callback


def _execute_chunk_callback(coordinator: ExecutionCoordinator) -> Any:
    def callback(
        phase_runspec_path: Path,
        action_id: str,
        evidence_path: Path,
        database_placement_result_path: Path,
        database_placement_failure_path: Path | None,
        placement_process_status: int,
        carry_forward_record_path: Path | None,
        phase_submission_id: str | None,
    ) -> None:
        _run_preprocessing_execute_chunk(
            coordinator,
            phase_runspec_path=phase_runspec_path,
            action_id=action_id,
            evidence_path=evidence_path,
            database_placement_result_path=database_placement_result_path,
            database_placement_failure_path=database_placement_failure_path,
            placement_process_status=placement_process_status,
            carry_forward_record_path=carry_forward_record_path,
            phase_submission_id=phase_submission_id,
        )

    return callback


@click.group()
def cli() -> None:
    """BSPP Pipelines CLI."""


@cli.group("preprocessing")
def preprocessing_group() -> None:
    """Execute declared preprocessing Runtime Actions."""


@preprocessing_group.group("database-set")
def preprocessing_database_set_group() -> None:
    """Provision immutable preprocessing Database Set source manifests."""


@preprocessing_group.group("database-cache")
def preprocessing_database_cache_group() -> None:
    """Operate the dedicated acceptance Database Replica cache namespace."""


@preprocessing_database_cache_group.command("clear")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="User Cluster Profile YAML containing the dedicated acceptance profile.",
)
@click.option("--profile", "profile_name", required=True, help="Exact dedicated acceptance Cluster Profile name.")
@click.option(
    "--write-evidence",
    "evidence_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Immutable terminal cache-maintenance evidence destination outside the cache.",
)
def preprocessing_database_cache_clear_cmd(
    config_path: Path,
    profile_name: str,
    evidence_path: Path,
) -> None:
    """Clear only the effective user's explicitly marked acceptance cache."""
    from bspp.orchestration.runtime.preprocessing.database_cache_maintenance import (
        DatabaseCacheMaintenanceError,
        clear_database_acceptance_cache,
        load_database_acceptance_cache_authority,
    )

    try:
        authority = load_database_acceptance_cache_authority(config_path, profile_name)
        evidence = clear_database_acceptance_cache(
            authority.profile,
            evidence_path=evidence_path,
            configured_sibling_cache_roots=authority.configured_sibling_cache_roots,
        )
    except (OSError, TypeError, ValueError, DatabaseCacheMaintenanceError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(evidence.to_mapping(), indent=2, sort_keys=True))


@preprocessing_database_set_group.command("provision")
@click.option(
    "--declaration",
    "declaration_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Explicit Database Set declaration JSON/YAML.",
)
@click.option(
    "--manifest-root",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Cluster-local root for immutable Database Source Manifests.",
)
@click.option(
    "--write-evidence",
    "evidence_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Durable Database Set Provisioning evidence destination.",
)
def preprocessing_database_set_provision_cmd(
    declaration_path: Path,
    manifest_root: Path,
    evidence_path: Path,
) -> None:
    """Inventory one declared Database Set version and publish its manifest."""
    from bspp.orchestration.contract.database_set_provisioning import load_database_set_declaration
    from bspp.orchestration.runtime.preprocessing.database_intake import (
        DatabaseIntakeError,
        download_database_set_from_s3,
    )
    from bspp.orchestration.runtime.preprocessing.database_provisioning import (
        DatabaseProvisioningError,
        provision_database_set,
    )

    try:
        declaration = load_database_set_declaration(declaration_path)
        download_database_set_from_s3(declaration)
        evidence = provision_database_set(
            declaration,
            manifest_root=manifest_root,
            evidence_path=evidence_path,
        )
    except (OSError, TypeError, ValueError, DatabaseIntakeError, DatabaseProvisioningError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(evidence.to_mapping(), indent=2, sort_keys=True))


@preprocessing_group.command("fetch-input")
@click.option(
    "--phase-runspec",
    "phase_runspec_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Immutable preprocessing Phase RunSpec JSON/YAML.",
)
@click.option(
    "--workspace-root",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Attempt workspace root where the relative input path is materialized.",
)
def preprocessing_fetch_input_cmd(phase_runspec_path: Path, workspace_root: Path) -> None:
    """Download and verify the remote FASTA input for a preprocessing Phase RunSpec."""
    from bspp.orchestration.contract.phase import VerifiedRemoteInputLocation
    from bspp.orchestration.runtime.preprocessing.execution import load_preprocessing_phase_runspec
    from bspp.orchestration.runtime.preprocessing.input_intake import (
        InputIntakeError,
        download_remote_fasta,
        normalize_fasta_to_identity_headers,
    )

    try:
        runspec = load_preprocessing_phase_runspec(phase_runspec_path)
        location = runspec.input_location
        if not isinstance(location, VerifiedRemoteInputLocation):
            raise click.ClickException(
                f"input_location is {location.kind!r}, not 'verified-remote-file'; "
                "fetch-input only downloads remote inputs"
            )
        workspace_resolved = workspace_root.resolve()
        destination = (workspace_root / location.path).resolve()
        if destination != workspace_resolved and workspace_resolved not in destination.parents:
            raise click.ClickException("remote input destination escapes the workspace root")
        verified = download_remote_fasta(location, destination)
        normalize_fasta_to_identity_headers(verified)
    except (OSError, TypeError, ValueError, InputIntakeError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(str(verified))


@preprocessing_group.command("stage-input")
@click.option(
    "--phase-runspec",
    "phase_runspec_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Immutable preprocessing Phase RunSpec JSON/YAML.",
)
@click.option(
    "--workspace-root",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Attempt workspace root where the relative input path is materialized.",
)
def preprocessing_stage_input_cmd(phase_runspec_path: Path, workspace_root: Path) -> None:
    """Download (if remote) and split the FASTA into per-chunk .fa files."""
    from bspp.orchestration.runtime.preprocessing.execution import load_preprocessing_phase_runspec
    from bspp.orchestration.runtime.preprocessing.input_intake import InputIntakeError
    from bspp.orchestration.runtime.preprocessing.input_staging import (
        InputStagingError,
        stage_preprocessing_input,
    )

    try:
        runspec = load_preprocessing_phase_runspec(phase_runspec_path)
        materialized = stage_preprocessing_input(runspec, workspace_root=workspace_root)
    except (OSError, TypeError, ValueError, InputIntakeError, InputStagingError) as exc:
        raise click.ClickException(str(exc)) from exc
    for path in materialized:
        click.echo(str(path))


@preprocessing_group.command("place-database")
@click.option(
    "--phase-runspec",
    "phase_runspec_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Immutable preprocessing Phase RunSpec JSON/YAML.",
)
@click.option("--action-id", required=True, help="Exact sole Runtime Action id.")
@click.option(
    "--source-manifest",
    "source_manifest_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Canonical staged Database Source Manifest.",
)
@click.option(
    "--write-result",
    "result_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Exclusive immutable Database Placement Result destination.",
)
@click.option(
    "--write-failure-evidence",
    "failure_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Exclusive immutable classified Database Placement failure destination.",
)
def preprocessing_place_database_cmd(
    phase_runspec_path: Path,
    action_id: str,
    source_manifest_path: Path,
    result_path: Path,
    failure_path: Path,
) -> None:
    """Validate or populate the policy-selected database before science starts."""
    _run_preprocessing_place_database(
        PRODUCTION_EXECUTION_COORDINATOR,
        phase_runspec_path=phase_runspec_path,
        action_id=action_id,
        source_manifest_path=source_manifest_path,
        result_path=result_path,
        failure_path=failure_path,
    )


@preprocessing_group.command("select-database-science-branch")
@click.option(
    "--phase-runspec",
    "phase_runspec_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Immutable preprocessing Phase RunSpec JSON/YAML.",
)
@click.option("--action-id", required=True, help="Exact sole Runtime Action id.")
@click.option(
    "--database-placement-result",
    "database_placement_result_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Exclusive Database Placement Result authority path.",
)
@click.option(
    "--database-placement-failure",
    "database_placement_failure_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Classified Database Placement failure authority path.",
)
def preprocessing_select_database_science_branch_cmd(
    phase_runspec_path: Path,
    action_id: str,
    database_placement_result_path: Path,
    database_placement_failure_path: Path,
) -> None:
    """Print the sole strict stage-preferred science branch token."""
    from bspp.orchestration.runtime.preprocessing._database_action_placement import (
        DatabaseActionPlacementError,
        resolve_database_science_branch,
    )
    from bspp.orchestration.runtime.preprocessing._database_placement_errors import DatabasePlacementError
    from bspp.orchestration.runtime.preprocessing.execution import load_preprocessing_phase_runspec

    try:
        runspec = load_preprocessing_phase_runspec(phase_runspec_path)
        branch = resolve_database_science_branch(
            runspec,
            action_id=action_id,
            result_path=database_placement_result_path,
            failure_path=database_placement_failure_path,
        )
    except (OSError, TypeError, ValueError, DatabaseActionPlacementError, DatabasePlacementError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(branch)


@preprocessing_group.command("execute-chunk")
@click.option(
    "--phase-runspec",
    "phase_runspec_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Immutable preprocessing Phase RunSpec JSON/YAML.",
)
@click.option("--action-id", required=True, help="Exact sole Runtime Action id.")
@click.option(
    "--write-evidence",
    "evidence_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Exclusive destination for structured action evidence.",
)
@click.option(
    "--database-placement-result",
    "database_placement_result_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Exclusive Database Placement Result authority path.",
)
@click.option(
    "--database-placement-failure",
    "database_placement_failure_path",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Classified Database Placement failure path when no Result exists.",
)
@click.option(
    "--placement-process-status",
    required=True,
    type=click.IntRange(min=0, max=255),
    help="Exact exit status captured from the Database Placement command.",
)
@click.option(
    "--carry-forward-record",
    "carry_forward_record_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Canonical staged Attempt carry-forward record.",
)
@click.option(
    "--phase-submission-id",
    default=None,
    help="Exact carried Phase Submission identity bound by the workspace sentinel.",
)
def preprocessing_execute_chunk_cmd(
    phase_runspec_path: Path,
    action_id: str,
    evidence_path: Path,
    database_placement_result_path: Path,
    database_placement_failure_path: Path | None,
    placement_process_status: int,
    carry_forward_record_path: Path | None,
    phase_submission_id: str | None,
) -> None:
    """Execute one exact chunk without lifecycle or scheduler ownership."""
    _run_preprocessing_execute_chunk(
        PRODUCTION_EXECUTION_COORDINATOR,
        phase_runspec_path=phase_runspec_path,
        action_id=action_id,
        evidence_path=evidence_path,
        database_placement_result_path=database_placement_result_path,
        database_placement_failure_path=database_placement_failure_path,
        placement_process_status=placement_process_status,
        carry_forward_record_path=carry_forward_record_path,
        phase_submission_id=phase_submission_id,
    )


@preprocessing_group.command("finalize-chunk")
@click.option(
    "--phase-runspec",
    "phase_runspec_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Immutable preprocessing Phase RunSpec JSON/YAML.",
)
@click.option(
    "--action-evidence",
    "action_evidence_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Successful execute-chunk evidence.",
)
@click.option(
    "--write-handoff",
    "handoff_path",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Exclusive destination for the validated handoff directory.",
)
def preprocessing_finalize_chunk_cmd(
    phase_runspec_path: Path,
    action_evidence_path: Path,
    handoff_path: Path,
) -> None:
    """Validate durable bundle bytes without scheduling or lifecycle ownership."""
    from bspp.orchestration.runtime.preprocessing.execution import load_preprocessing_phase_runspec
    from bspp.orchestration.runtime.preprocessing.finalization import (
        PreprocessingFinalizationError,
        finalize_preprocessing_chunk,
        load_preprocessing_action_evidence,
    )

    try:
        runspec = load_preprocessing_phase_runspec(phase_runspec_path)
        evidence = load_preprocessing_action_evidence(action_evidence_path)
        handoff = finalize_preprocessing_chunk(runspec, evidence, handoff_path=handoff_path)
    except (OSError, TypeError, ValueError, PreprocessingFinalizationError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(handoff.to_mapping(), indent=2, sort_keys=True))


@cli.group("worker")
def worker_group() -> None:
    """Native worker entry points."""


@worker_group.command("archive-task")
@click.option("--runspec", "--spec", "spec_path", required=True, type=click.Path(exists=True), help="RunSpec YAML.")
@click.option("--dry-run", is_flag=True, help="Build the task context in dry-run mode.")
def worker_archive_task_cmd(spec_path: str, dry_run: bool) -> None:
    """Run one native archive worker task from SLURM task environment."""
    from bspp.orchestration.contract.runspec import load_runspec
    from bspp.orchestration.runtime.worker import TaskContext, WorkerDependencies, run_archive_task

    task = TaskContext(
        job_id=os.environ.get("SLURM_JOB_ID", str(os.getpid())),
        array_task_id=_required_env_int("SLURM_ARRAY_TASK_ID"),
        array_task_count=_optional_env_int("SLURM_ARRAY_TASK_COUNT"),
        node_name=os.environ.get("SLURMD_NODENAME"),
        dry_run=dry_run,
    )
    spec = load_runspec(Path(spec_path))
    result = run_archive_task(spec, task, WorkerDependencies())
    click.echo(
        "archive={archive} shards={shards} processed={processed} failed={failed} uploaded={uploaded}".format(
            archive=result.archive_name,
            shards=",".join(str(shard) for shard in result.logical_shards),
            processed=result.processed_models,
            failed=len(result.failed_models),
            uploaded=result.uploaded_files,
        ),
    )
    if result.exit_code != 0:
        sys.exit(result.exit_code)


def _required_env_int(name: str) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        raise click.ClickException(f"{name} is required")
    try:
        return int(value)
    except ValueError as exc:
        raise click.ClickException(f"{name} must be an integer, got {value!r}") from exc


def _optional_env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise click.ClickException(f"{name} must be an integer, got {value!r}") from exc


@cli.group("runspec")
def runspec_group() -> None:
    """Canonical run specification operations."""


@runspec_group.command("validate")
@click.argument("spec", type=click.Path(exists=True))
@click.option("--strict-secrets", is_flag=True, help="Fail if required secrets cannot be resolved.")
def runspec_validate_cmd(spec: str, strict_secrets: bool) -> None:
    """Validate a RunSpec and print a redacted dry-run summary."""
    from bspp.orchestration.contract.runspec import load_runspec, render_dry_run
    from bspp.orchestration.contract.runspec_validation import validate_active_workflow_static

    run_spec = load_runspec(Path(spec))
    static_validation = validate_active_workflow_static(run_spec)
    if not static_validation.ok:
        raise click.ClickException("RunSpec static validation failed: " + "; ".join(static_validation.blockers))
    statuses = run_spec.validate_required_secrets() if strict_secrets else run_spec.resolve_secrets()
    click.echo(render_dry_run(run_spec, statuses))


@cli.group("inputs")
def inputs_group() -> None:
    """RunSpec-driven reference and archive input preparation."""


@cli.group("hq-chunks")
def hq_chunks_group() -> None:
    """Build high-quality chunk artifacts from native local-tar outputs."""


@hq_chunks_group.command("build")
@click.option("--selected-ids", required=True, type=click.Path(exists=True), help="high_quality_model_ids.txt.")
@click.option("--model-tar-index", required=True, type=click.Path(exists=True), help="model_tar_index.csv.")
@click.option("--staging-root", required=True, type=click.Path(), help="Temporary extraction staging root.")
@click.option("--chunks-dir", required=True, type=click.Path(), help="Destination directory for chunk_NNNN.tar files.")
@click.option("--local-tar-root", type=click.Path(exists=True, file_okay=False), default=None)
@click.option("--chunk-size", type=click.IntRange(min=1), default=1000, show_default=True)
@click.option("--write-report", type=click.Path(file_okay=False), default=None)
def hq_chunks_build_cmd(
    selected_ids: str,
    model_tar_index: str,
    staging_root: str,
    chunks_dir: str,
    local_tar_root: str | None,
    chunk_size: int,
    write_report: str | None,
) -> None:
    """Build chunk_NNNN.tar files for selected high-quality models."""
    from bspp.orchestration.runtime.postprocessing.hq_chunks import (
        build_hq_chunks,
        render_hq_chunk_build_report,
        write_hq_chunk_build_report,
    )

    report = build_hq_chunks(
        selected_ids_path=Path(selected_ids),
        model_tar_index_path=Path(model_tar_index),
        staging_root=Path(staging_root),
        chunks_dir=Path(chunks_dir),
        local_tar_root=Path(local_tar_root) if local_tar_root else None,
        chunk_size=chunk_size,
    )
    click.echo(render_hq_chunk_build_report(report), nl=False)
    if write_report:
        write_hq_chunk_build_report(report, Path(write_report))


@hq_chunks_group.command("publish")
@click.option("--runspec", "spec_path", required=True, type=click.Path(exists=True), help="RunSpec YAML.")
@click.option("--chunks-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option(
    "--upload-manifest",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Prior upload manifest to use for resume classification.",
)
@click.option(
    "--remote-inventory",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Pre-collected remote inventory CSV evidence.",
)
@click.option("--write-report", type=click.Path(file_okay=False), default=None)
@click.option("--execute/--dry-run", default=False, help="Execute live upload when set; default only writes a plan.")
def hq_chunks_publish_cmd(
    spec_path: str,
    chunks_dir: str,
    upload_manifest: str | None,
    remote_inventory: str | None,
    write_report: str | None,
    execute: bool,
) -> None:
    """Plan or execute publication of local HQ chunk tar artifacts."""
    from bspp.orchestration.contract.runspec import load_runspec
    from bspp.orchestration.runtime.data_movement.s3.inventory import read_inventory_csv
    from bspp.orchestration.runtime.postprocessing.hq_publication import (
        HqChunkPublicationExecutionError,
        execute_hq_chunk_publication,
        plan_hq_chunk_publication,
        render_hq_publication_execution_report,
        render_hq_publication_plan,
        write_hq_publication_reports,
    )

    spec = load_runspec(Path(spec_path))
    if execute:
        if remote_inventory is not None:
            raise click.ClickException("--execute collects remote inventory itself; omit --remote-inventory")
        output_dir = _hq_publication_output_dir(spec, write_report)
        if output_dir is None:
            raise click.ClickException(
                "HQ chunk publication execution requires --write-report or RunSpec publication.evidence_dir"
            )
        try:
            report = execute_hq_chunk_publication(
                spec=spec,
                chunks_dir=Path(chunks_dir),
                output_dir=output_dir,
                upload_manifest_path=Path(upload_manifest) if upload_manifest else None,
            )
        except HqChunkPublicationExecutionError as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(render_hq_publication_execution_report(report), nl=False)
        if not report.ok:
            raise click.ClickException("HQ chunk publication validation gate failed")
        return

    inventory = read_inventory_csv(Path(remote_inventory)) if remote_inventory else None
    plan = plan_hq_chunk_publication(
        spec,
        chunks_dir=Path(chunks_dir),
        upload_manifest_path=Path(upload_manifest) if upload_manifest else None,
        remote_inventory=inventory,
    )
    click.echo(render_hq_publication_plan(plan), nl=False)
    if write_report:
        write_hq_publication_reports(plan, Path(write_report))
    if not plan.ok:
        raise click.ClickException("HQ chunk publication plan is blocked by collision or evidence failures")


def _hq_publication_output_dir(spec: RunSpec, write_report: str | None) -> Path | None:
    if write_report is not None:
        return Path(write_report)
    publication = spec.analysis_metadata.high_quality_from_tars.publication
    if publication.evidence_dir is not None:
        return publication.evidence_dir
    return publication.manifest_dir


@inputs_group.command("check")
@click.option("--runspec", "spec_path", required=True, type=click.Path(exists=True), help="Path to RunSpec YAML.")
@click.option("--write-report", is_flag=True, help="Write reports under <output_dir>/wp3.")
def inputs_check_cmd(spec_path: str, write_report: bool) -> None:
    """Check reference files declared by a RunSpec."""
    from bspp.orchestration.contract.runspec import load_runspec
    from bspp.orchestration.runtime.inputs.references import check_references
    from bspp.orchestration.runtime.inputs.reports import report_to_text, write_json_report

    spec = load_runspec(Path(spec_path))
    statuses = check_references(spec)
    report = _input_report(spec, {"references": statuses})
    click.echo(report_to_text(report))
    if write_report:
        write_json_report(report, spec.paths.output_dir / "wp3" / "reference_report.json")


@inputs_group.command("ensure-references")
@click.option("--runspec", "spec_path", required=True, type=click.Path(exists=True), help="Path to RunSpec YAML.")
@click.option("--execute/--dry-run", default=False, help="Execute local file copies; remote URIs remain planned.")
@click.option("--write-report", is_flag=True, help="Write reports under <output_dir>/wp3.")
def inputs_ensure_references_cmd(spec_path: str, execute: bool, write_report: bool) -> None:
    """Plan or ensure reference files declared by a RunSpec."""
    from bspp.orchestration.contract.runspec import load_runspec
    from bspp.orchestration.runtime.inputs.references import check_references, ensure_references
    from bspp.orchestration.runtime.inputs.reports import report_to_text, write_json_report

    spec = load_runspec(Path(spec_path))
    plans = ensure_references(spec, dry_run=not execute)
    statuses = check_references(spec)
    report = _input_report(spec, {"references": statuses, "planned_reference_transfers": plans, "dry_run": not execute})
    click.echo(report_to_text(report))
    if write_report:
        write_json_report(report, spec.paths.output_dir / "wp3" / "reference_report.json")


@inputs_group.command("coverage")
@click.option("--runspec", "spec_path", required=True, type=click.Path(exists=True), help="Path to RunSpec YAML.")
@click.option("--write-report", is_flag=True, help="Write reports under <output_dir>/wp3.")
def inputs_coverage_cmd(spec_path: str, write_report: bool) -> None:
    """Report dataset archive coverage from the configured RunSpec source."""
    from bspp.orchestration.contract.runspec import load_runspec
    from bspp.orchestration.runtime.inputs.archives import (
        archives_for_runspec,
        archives_for_runspec_staging,
        plan_archive_staging,
    )
    from bspp.orchestration.runtime.inputs.reports import report_to_text, write_json_report

    spec = load_runspec(Path(spec_path))
    coverage = archives_for_runspec(spec)
    staging_coverage = archives_for_runspec_staging(spec, coverage)
    staging = plan_archive_staging(
        staging_coverage,
        spec.paths.staging_dir,
        archive_prefix=spec.storage.s3_archive_prefix,
        dry_run=True,
    )
    report = _input_report(spec, {"archive_coverage": coverage, "staging": staging})
    click.echo(report_to_text(report))
    if write_report:
        write_json_report(report, spec.paths.output_dir / "wp3" / "archive_coverage_report.json")


@inputs_group.command("stage-archives")
@click.option("--runspec", "spec_path", required=True, type=click.Path(exists=True), help="Path to RunSpec YAML.")
@click.option("--execute/--dry-run", default=False, help="Download missing archives when set.")
@click.option("--write-report", is_flag=True, help="Write reports under <output_dir>/wp3.")
def inputs_stage_archives_cmd(spec_path: str, execute: bool, write_report: bool) -> None:
    """Plan or download archive files into staging."""
    from bspp.orchestration.contract.runspec import load_runspec
    from bspp.orchestration.runtime.inputs.archives import (
        archives_for_runspec,
        archives_for_runspec_staging,
        plan_archive_staging,
    )
    from bspp.orchestration.runtime.inputs.reports import report_to_text, write_json_report

    spec = load_runspec(Path(spec_path))
    coverage = archives_for_runspec(spec)
    staging_coverage = archives_for_runspec_staging(spec, coverage)
    staging = plan_archive_staging(
        staging_coverage,
        spec.paths.staging_dir,
        archive_prefix=spec.storage.s3_archive_prefix,
        dry_run=not execute,
    )
    if execute:
        _execute_archive_downloads(spec, staging)
        staging = plan_archive_staging(
            staging_coverage,
            spec.paths.staging_dir,
            archive_prefix=spec.storage.s3_archive_prefix,
            dry_run=False,
        )
    report = _input_report(spec, {"archive_coverage": coverage, "staging": staging})
    click.echo(report_to_text(report))
    if write_report:
        write_json_report(report, spec.paths.output_dir / "wp3" / "staging_report.json")


@inputs_group.command("prepare")
@click.option("--runspec", "spec_path", required=True, type=click.Path(exists=True), help="Path to RunSpec YAML.")
@click.option("--execute/--dry-run", default=False, help="Create missing tracking parquet and stage archives when set.")
@click.option("--strict-secrets", is_flag=True, help="Fail if configured secrets cannot be resolved.")
@click.option("--write-report", is_flag=True, help="Write reports under <output_dir>/wp3.")
def inputs_prepare_cmd(spec_path: str, execute: bool, strict_secrets: bool, write_report: bool) -> None:
    """Run WP3 reference checks, tracking creation plan, coverage, and staging plan."""
    from bspp.orchestration.contract.runspec import load_runspec
    from bspp.orchestration.runtime.inputs.archives import (
        archives_for_runspec,
        archives_for_runspec_staging,
        plan_archive_staging,
    )
    from bspp.orchestration.runtime.inputs.references import check_references, ensure_references
    from bspp.orchestration.runtime.inputs.reports import report_to_text, write_json_report
    from bspp.orchestration.runtime.postprocessing.tracking import create_tracking_parquet

    spec = load_runspec(Path(spec_path))
    secret_statuses = spec.validate_required_secrets() if strict_secrets else spec.resolve_secrets()
    plans = ensure_references(spec, dry_run=not execute)
    tracking_action = "reuse"
    if not spec.references.tracking_parquet.exists():
        tracking_action = "create" if execute else "plan_create"
        if execute:
            create_tracking_parquet(
                spec.references.master_parquet,
                spec.references.tracking_parquet,
                s3_output_prefix=spec.storage.s3_output_prefix,
                gcs_destination_prefix=spec.storage.gcs_destination_prefix,
            )
    statuses = check_references(spec)
    coverage = None
    staging = None
    if spec.dataset.archive_source == "staging_dir" or spec.references.tracking_parquet.exists():
        coverage = archives_for_runspec(spec)
        staging_coverage = archives_for_runspec_staging(spec, coverage)
        staging = plan_archive_staging(
            staging_coverage,
            spec.paths.staging_dir,
            archive_prefix=spec.storage.s3_archive_prefix,
            dry_run=not execute,
        )
        if execute:
            _execute_archive_downloads(spec, staging)
            staging = plan_archive_staging(
                staging_coverage,
                spec.paths.staging_dir,
                archive_prefix=spec.storage.s3_archive_prefix,
                dry_run=False,
            )
    report = _input_report(
        spec,
        {
            "dry_run": not execute,
            "secret_statuses": [status.as_redacted_mapping() for status in secret_statuses],
            "references": statuses,
            "planned_reference_transfers": plans,
            "tracking_action": tracking_action,
            "archive_coverage": coverage,
            "staging": staging,
        },
    )
    click.echo(report_to_text(report))
    if write_report:
        write_json_report(report, spec.paths.output_dir / "wp3" / "prepare_report.json")


@runspec_group.command("render-recipe")
@click.argument("spec", type=click.Path(exists=True))
@click.option("--execute/--dry-run", default=False, help="Write recipe config when set.")
def runspec_render_recipe_cmd(spec: str, execute: bool) -> None:
    """Render the archive-mode recipe config from a RunSpec."""
    from bspp.orchestration.contract.runspec import load_runspec
    from bspp.orchestration.runtime.inputs.archives import archives_for_runspec
    from bspp.orchestration.runtime.postprocessing.runspec_artifacts import (
        render_archive_preprocess_artifacts,
        render_archive_preprocess_artifacts_report,
    )

    run_spec = load_runspec(Path(spec))
    coverage = archives_for_runspec(run_spec)
    plan = render_archive_preprocess_artifacts(run_spec, coverage, dry_run=not execute, write_allowlists=False)
    click.echo(render_archive_preprocess_artifacts_report(run_spec, plan))


@runspec_group.command("render-preprocess")
@click.argument("spec", type=click.Path(exists=True))
@click.option("--execute/--dry-run", default=False, help="Write preprocess artifacts when set.")
@click.option("--allowlists/--no-allowlists", default=True, help="Render per-archive allowlists.")
def runspec_render_preprocess_cmd(spec: str, execute: bool, allowlists: bool) -> None:
    """Render archive-mode preprocess artifacts from a RunSpec."""
    from bspp.orchestration.contract.runspec import load_runspec
    from bspp.orchestration.runtime.inputs.archives import archives_for_runspec
    from bspp.orchestration.runtime.postprocessing.runspec_artifacts import (
        render_archive_preprocess_artifacts,
        render_archive_preprocess_artifacts_report,
    )

    run_spec = load_runspec(Path(spec))
    coverage = archives_for_runspec(run_spec)
    try:
        plan = render_archive_preprocess_artifacts(
            run_spec,
            coverage,
            dry_run=not execute,
            write_allowlists=allowlists,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(render_archive_preprocess_artifacts_report(run_spec, plan))


@runspec_group.command("preflight")
@click.argument("spec", type=click.Path(exists=True))
@click.option("--write-report", is_flag=True, help="Write reports under <submission.evidence_dir>/preflight.")
@click.option("--strict", is_flag=True, help="Exit non-zero when preflight has blockers.")
def runspec_preflight_cmd(spec: str, write_report: bool, strict: bool) -> None:
    """Check workflow validation and Phase 1 policies before running a concrete RunSpec."""
    from bspp.orchestration.contract.runspec import load_runspec
    from bspp.orchestration.runtime.runspec_preflight import (
        build_runspec_preflight_report,
        render_runspec_preflight_report,
        write_runspec_preflight_reports,
    )

    run_spec = load_runspec(Path(spec))
    try:
        report = build_runspec_preflight_report(run_spec)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(render_runspec_preflight_report(report), nl=False)
    if write_report:
        if run_spec.submission is None:
            raise click.ClickException("runspec preflight --write-report requires submission.evidence_dir")
        write_runspec_preflight_reports(report, run_spec.submission.evidence_dir)
    if strict and not report.ready:
        sys.exit(1)


@runspec_group.command("validate-archive-output")
@click.argument("spec", type=click.Path(exists=True))
@click.option(
    "--aggregate-manifest",
    type=click.Path(exists=True),
    default=None,
    help="Aggregate manifest parquet to count.",
)
@click.option("--write-report", is_flag=True, help="Write reports under <output_dir>/wp6.")
@click.option("--strict", is_flag=True, help="Exit non-zero when applicable checks fail.")
def runspec_validate_archive_output_cmd(
    spec: str,
    aggregate_manifest: str | None,
    write_report: bool,
    strict: bool,
) -> None:
    """Validate archive-mode outputs against RunSpec expectations."""
    from bspp.orchestration.contract.runspec import load_runspec
    from bspp.orchestration.runtime.postprocessing.runspec_reports import (
        build_archive_output_validation_report,
        render_report,
        write_archive_output_validation_report,
    )

    run_spec = load_runspec(Path(spec))
    report = build_archive_output_validation_report(
        run_spec,
        aggregate_manifest=Path(aggregate_manifest) if aggregate_manifest else None,
    )
    click.echo(render_report(report), nl=False)
    if write_report:
        write_archive_output_validation_report(report, run_spec.paths.output_dir)
    if strict and not report.valid:
        sys.exit(1)


@runspec_group.command("plan-cleanup")
@click.argument("spec", type=click.Path(exists=True))
@click.option("--execute/--dry-run", default=False, help="Remove success_outputs dirs when set.")
@click.option("--force", is_flag=True, help="Allow cleanup even when a shard lacks .uploaded.")
def runspec_plan_cleanup_cmd(spec: str, execute: bool, force: bool) -> None:
    """Plan or execute local cleanup of archive-mode success outputs."""
    from bspp.orchestration.contract.runspec import load_runspec
    from bspp.orchestration.runtime.postprocessing.runspec_reports import plan_cleanup, render_report

    run_spec = load_runspec(Path(spec))
    click.echo(render_report(plan_cleanup(run_spec, execute=execute, force=force)), nl=False)


@runspec_group.command("plan-cleanup-archives")
@click.argument("spec", type=click.Path(exists=True))
@click.option("--execute/--dry-run", default=False, help="Delete delete-ready .tar.lz4 archives when set.")
@click.option(
    "--expected-s3-prefix",
    default=None,
    help="Expected S3 output prefix (default: RunSpec s3_output_prefix).",
)
@click.option("--write-report", is_flag=True, help="Write archive_cleanup_report.json and archive_cleanup.tsv.")
def runspec_plan_cleanup_archives_cmd(
    spec: str, execute: bool, expected_s3_prefix: str | None, write_report: bool
) -> None:
    """Plan or execute deletion of self-uploaded compressed archives."""
    from bspp.orchestration.contract.runspec import load_runspec
    from bspp.orchestration.runtime.inputs.reports import report_to_json
    from bspp.orchestration.runtime.postprocessing.archive_cleanup import (
        plan_archive_cleanup,
        write_archive_cleanup_report,
    )

    run_spec = load_runspec(Path(spec))
    report = plan_archive_cleanup(run_spec, expected_s3_prefix=expected_s3_prefix, execute=execute)
    click.echo(report_to_json(report), nl=False)
    if write_report:
        write_archive_cleanup_report(report, run_spec.paths.output_dir)


@cli.group("data-movement")
def data_movement() -> None:
    """Data movement operations (GCS, S3)."""


@data_movement.command("check-gcs")
@click.option(
    "--config",
    "config_path",
    default="pipelines.toml",
    type=click.Path(exists=True),
    help="Path to config file.",
)
def check_gcs(config_path: str) -> None:
    """Check GCS connectivity and credential validity."""
    from bspp.orchestration.runtime.data_movement.gcs.check import check_connection

    cfg_path = Path(config_path)
    cfg = load_config(cfg_path)

    gcs_cfg = cfg.get("gcs")
    if not gcs_cfg:
        click.echo("No [gcs] section in config.", err=True)
        sys.exit(1)

    credentials = resolve_path(cfg_path, gcs_cfg["credentials"])
    if not credentials.exists():
        click.echo(f"Credentials file not found: {credentials}", err=True)
        sys.exit(1)

    bucket_name: str = gcs_cfg["bucket"]
    prefix: str = gcs_cfg.get("prefix", "")

    click.echo("GCS Connection Check")
    click.echo(f"  Credentials: {credentials.name}")

    try:
        status = check_connection(credentials, bucket_name, prefix=prefix)
    except Exception as exc:
        click.echo(f"  FAILED: {exc}", err=True)
        sys.exit(1)

    click.echo(f"  Service account: {status.service_account}")
    click.echo(f"  Project: {status.project}")
    click.echo(f"  Bucket: {status.bucket} ... {'OK' if status.bucket_accessible else 'FAILED'}")

    if not status.bucket_accessible:
        sys.exit(1)


# -------------------------------------------------------------------
# data-movement gcs / s3 / dm subgroups
# -------------------------------------------------------------------


def _echo_transfer(result_or_plan: object) -> None:
    from bspp.orchestration.runtime.data_movement.common import (
        PlannedTransfer,
        TransferResult,
        format_argv,
    )

    if isinstance(result_or_plan, PlannedTransfer):
        click.echo(f"[DRY RUN] {result_or_plan.tool}: {format_argv(result_or_plan.argv)}")
        if result_or_plan.note:
            click.echo(f"  ({result_or_plan.note})")
    elif isinstance(result_or_plan, TransferResult):
        suffix = "OK" if result_or_plan.ok else f"FAILED rc={result_or_plan.returncode}"
        click.echo(f"{result_or_plan.tool}: {format_argv(result_or_plan.argv)}")
        click.echo(f"  {suffix} ({result_or_plan.elapsed_s:.1f}s)")
        if not result_or_plan.ok and result_or_plan.stderr_tail.strip():
            click.echo(f"  stderr: {result_or_plan.stderr_tail.strip()}", err=True)
            sys.exit(result_or_plan.returncode)


def _input_report(spec: Any, payload: dict[str, object]) -> dict[str, object]:
    return {
        "run_id": spec.dataset.run_id,
        "dataset": spec.dataset.name,
        "source_runspec": str(spec.source_path) if spec.source_path is not None else None,
        "source_hash": spec.source_hash,
        **payload,
    }


def _execute_archive_downloads(spec: Any, staging: Any) -> None:
    from bspp.orchestration.runtime.data_movement.s3 import transfer as s3_transfer

    spec.validate_required_secrets()
    env = spec.secret_execution_env().apply_to(os.environ)
    for item in staging.missing:
        if item.action != "download" or not item.source:
            continue
        if not item.source.startswith("s3://"):
            raise click.ClickException(f"Unsupported archive source for execution: {item.source}")
        result = s3_transfer.cp(item.source, item.destination, env=env)
        if not getattr(result, "ok", False):
            raise click.ClickException(f"Archive download failed for {item.archive}")


@data_movement.group("gcs")
def gcs_group() -> None:
    """GCS transfers via gcloud storage."""


@gcs_group.command("download")
@click.argument("src")
@click.argument("dst", type=click.Path())
@click.option("--recursive/--no-recursive", default=False)
@click.option("--dry-run", is_flag=True)
def gcs_download_cmd(src: str, dst: str, recursive: bool, dry_run: bool) -> None:
    """Download from GCS (``gs://...``) to a local path."""
    from bspp.orchestration.runtime.data_movement.gcs import transfer as gcs_transfer

    _echo_transfer(gcs_transfer.cp(src, dst, recursive=recursive, dry_run=dry_run))


@gcs_group.command("upload")
@click.argument("src", type=click.Path(exists=True))
@click.argument("dst")
@click.option("--recursive/--no-recursive", default=False)
@click.option("--dry-run", is_flag=True)
def gcs_upload_cmd(src: str, dst: str, recursive: bool, dry_run: bool) -> None:
    """Upload a local path to GCS (``gs://...``)."""
    from bspp.orchestration.runtime.data_movement.gcs import transfer as gcs_transfer

    _echo_transfer(gcs_transfer.cp(src, dst, recursive=recursive, dry_run=dry_run))


@gcs_group.command("verify")
@click.option(
    "--config",
    "config_path",
    default="pipelines.toml",
    type=click.Path(exists=True),
    help="Path to pipelines.toml (provides GCS credentials + bucket).",
)
@click.option("--prefix", default="", help="Blob prefix to scope the listing.")
@click.option("--target-date", default=None, help="ISO date (YYYY-MM-DD) to count only blobs updated on this UTC day.")
def gcs_verify_cmd(config_path: str, prefix: str, target_date: str | None) -> None:
    """Verify GCS uploads by counting blobs under *prefix*."""
    from datetime import date

    from bspp.orchestration.runtime.data_movement.gcs import verify

    cfg_path = Path(config_path)
    cfg = load_config(cfg_path)
    gcs_cfg = cfg.get("gcs")
    if not gcs_cfg:
        click.echo("No [gcs] section in config.", err=True)
        sys.exit(1)

    credentials = resolve_path(cfg_path, gcs_cfg["credentials"])
    target = date.fromisoformat(target_date) if target_date else None
    count, size_bytes = verify.count_with_prefix(
        credentials,
        gcs_cfg["bucket"],
        prefix=prefix,
        target_date=target,
    )
    scope = f"updated on {target_date}" if target else "total"
    click.echo(f"{count:,} blobs ({size_bytes / 1e9:.2f} GB) under {gcs_cfg['bucket']}/{prefix} — {scope}")


@data_movement.group("s3")
def s3_group() -> None:
    """S3 transfers via s5cmd (S3 endpoint)."""


@s3_group.command("download")
@click.argument("src")
@click.argument("dst", type=click.Path())
@click.option("--numworkers", type=int, default=None)
@click.option("--dry-run", is_flag=True)
def s3_download_cmd(src: str, dst: str, numworkers: int | None, dry_run: bool) -> None:
    """Download from S3 (``s3://...``) to a local path via s5cmd."""
    from bspp.orchestration.runtime.data_movement.s3 import transfer as s3_transfer

    _echo_transfer(s3_transfer.cp(src, dst, numworkers=numworkers, dry_run=dry_run))


@s3_group.command("upload")
@click.argument("src")
@click.argument("dst")
@click.option("--numworkers", type=int, default=None)
@click.option("--dry-run", is_flag=True)
def s3_upload_cmd(src: str, dst: str, numworkers: int | None, dry_run: bool) -> None:
    """Upload to S3 (``s3://...``) via s5cmd. Glob patterns in src are supported."""
    from bspp.orchestration.runtime.data_movement.s3 import transfer as s3_transfer

    _echo_transfer(s3_transfer.cp(src, dst, numworkers=numworkers, dry_run=dry_run))


@data_movement.group("operator")
def data_movement_operator_group() -> None:
    """Operator-initiated data-movement execution and verification."""


@data_movement_operator_group.command("execute")
@click.option(
    "--plan-json",
    "plan_json_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="OperatorTransferPlan JSON file (must have dry_run=false).",
)
@click.option("--force", is_flag=True, help="Acknowledge clobber risk for existing destination objects.")
@click.option(
    "--write-evidence",
    "evidence_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Directory for atomically written evidence JSON files.",
)
@click.option(
    "--snapshot-root",
    "snapshot_root",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Staging directory for the source snapshot + verify download (default: $SLURM_TMPDIR or /tmp).",
)
def data_movement_operator_execute_cmd(
    plan_json_path: Path,
    force: bool,
    evidence_dir: Path | None,
    snapshot_root: Path | None,
) -> None:
    """Execute an operator transfer plan on the cluster."""
    from bspp.orchestration.contract.operator_data_movement import operator_transfer_plan_from_json
    from bspp.orchestration.runtime.data_movement.common import ToolMissingError
    from bspp.orchestration.runtime.data_movement.operator_transfer import (
        OperatorTransferError,
        execute_operator_transfer,
    )
    from bspp.orchestration.runtime.data_movement.s3.client import MissingS3CredentialsError

    try:
        plan = operator_transfer_plan_from_json(plan_json_path.read_text())
        evidence_records = execute_operator_transfer(
            plan,
            force=force,
            evidence_dir=evidence_dir,
            snapshot_root=snapshot_root,
        )
    except (
        OSError,
        TypeError,
        ValueError,
        OperatorTransferError,
        ToolMissingError,
        MissingS3CredentialsError,
    ) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps([e.to_mapping() for e in evidence_records], indent=2, sort_keys=True))


@data_movement_operator_group.command("verify")
@click.option(
    "--evidence",
    "evidence_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Previously written evidence JSON file.",
)
def data_movement_operator_verify_cmd(evidence_path: Path) -> None:
    """Re-verify a remote object's size+sha256 from a prior evidence record."""
    from bspp.orchestration.contract.operator_data_movement import operator_transfer_evidence_from_json
    from bspp.orchestration.runtime.data_movement.common import ToolMissingError
    from bspp.orchestration.runtime.data_movement.operator_transfer import (
        OperatorTransferError,
        verify_operator_transfer,
    )
    from bspp.orchestration.runtime.data_movement.s3.client import MissingS3CredentialsError

    try:
        evidence = operator_transfer_evidence_from_json(evidence_path.read_text())
        verified_size, verified_sha = verify_operator_transfer(evidence.item)
    except (
        OSError,
        TypeError,
        ValueError,
        OperatorTransferError,
        ToolMissingError,
        MissingS3CredentialsError,
    ) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        json.dumps(
            {
                "evidence_id": evidence.evidence_id,
                "verified_size_bytes": verified_size,
                "verified_sha256": verified_sha,
                "ok": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _register_internal_data_movement_cli(group: click.Group) -> None:
    from importlib.metadata import entry_points

    from bspp.orchestration.runtime.data_movement.backends import BACKEND_ENTRY_POINT_GROUP

    for ep in entry_points(group=BACKEND_ENTRY_POINT_GROUP):
        try:
            backend = ep.load()
        except Exception:
            continue
        register = getattr(backend, "register_cli", None)
        if callable(register):
            register(group)


_register_internal_data_movement_cli(data_movement)


# -------------------------------------------------------------------
# Stage dispatcher commands (upload-s3 / upload-gcs / upload-gcs-direct)
# -------------------------------------------------------------------


def _run_stage(plan_text: str, plan: object, execute: bool, evidence_path: Path | None = None) -> None:
    click.echo(plan_text)
    if evidence_path is not None:
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        terminal_status = "completed" if getattr(plan, "all_ok", False) else "failed" if execute else "planned"
        record = plan.to_data_placement_record(  # type: ignore[attr-defined]
            evidence_path=evidence_path,
            job_id=os.environ.get("SLURM_JOB_ID"),
            terminal_status=terminal_status,
        )
        evidence_path.write_text(json.dumps(record.to_mapping(), indent=2, sort_keys=True) + "\n")
    if not execute:
        return
    # plan is always a StagePlan at runtime; avoid a top-level import just for the type hint.
    if not getattr(plan, "all_ok", False):
        results = getattr(plan, "results", ())
        sys.exit(results[-1].returncode if results else 1)


@cli.command("upload-s3")
@click.option("--dataset", required=True)
@click.option("--output-base", required=True, type=click.Path(exists=True))
@click.option(
    "--dataset-output-dir",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Exact dataset output directory; overrides --output-base/--dataset joining.",
)
@click.option("--tool", type=click.Choice(["dm", "s5cmd"]), default="s5cmd")
@click.option("--data-dir", type=click.Path(), default=None, help="Directory for prefix-list/command-file artifacts.")
@click.option("--write-evidence", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option(
    "--s3-destination-prefix",
    required=True,
    help=("S3 destination prefix, e.g. s3://bucket/path/. Required: no default is provided."),
)
@click.option("--execute/--dry-run", default=False, help="Execute the plan (default: preview only).")
def upload_s3_cmd(
    dataset: str,
    output_base: str,
    dataset_output_dir: str | None,
    tool: str,
    data_dir: str | None,
    write_evidence: Path | None,
    s3_destination_prefix: str,
    execute: bool,
) -> None:
    """Stage 1: Lustre -> S3 (dm or s5cmd)."""
    from bspp.orchestration.runtime.postprocessing import upload_and_track as upload_and_track_mod

    plan = upload_and_track_mod.upload_s3(
        dataset,
        Path(dataset_output_dir) if dataset_output_dir is not None else Path(output_base) / dataset,
        tool=tool,  # type: ignore[arg-type]
        data_dir=Path(data_dir) if data_dir else None,
        s3_destination_prefix=s3_destination_prefix,
        execute=execute,
        dry_run=not execute,
    )
    _run_stage(upload_and_track_mod.render_plan(plan), plan, execute, write_evidence)


@cli.command("upload-gcs")
@click.option("--dataset", required=True)
@click.option("--data-dir", required=True, type=click.Path(), help="Directory for prefix-list artifacts.")
@click.option("--tool", type=click.Choice(["dm", "gcloud"]), default="gcloud")
@click.option("--write-evidence", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option(
    "--s3-source-prefix",
    required=True,
    help=("S3 source prefix, e.g. s3://bucket/path/. Required: no default is provided."),
)
@click.option(
    "--gcs-destination-prefix",
    required=True,
    help=("GCS destination prefix, e.g. gs://bucket/path/. Required: no default is provided."),
)
@click.option("--execute/--dry-run", default=False)
def upload_gcs_cmd(
    dataset: str,
    data_dir: str,
    tool: str,
    write_evidence: Path | None,
    s3_source_prefix: str,
    gcs_destination_prefix: str,
    execute: bool,
) -> None:
    """Stage 2: S3 -> GCS (dm or gcloud)."""
    from bspp.orchestration.runtime.postprocessing import upload_and_track as upload_and_track_mod

    plan = upload_and_track_mod.upload_gcs(
        dataset,
        tool=tool,  # type: ignore[arg-type]
        data_dir=Path(data_dir),
        s3_source_prefix=s3_source_prefix,
        gcs_destination_prefix=gcs_destination_prefix,
        execute=execute,
        dry_run=not execute,
    )
    _run_stage(upload_and_track_mod.render_plan(plan), plan, execute, write_evidence)


@cli.command("upload-gcs-direct")
@click.option("--dataset", required=True)
@click.option("--output-base", required=True, type=click.Path(exists=True))
@click.option(
    "--dataset-output-dir",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Exact dataset output directory; overrides --output-base/--dataset joining.",
)
@click.option("--write-evidence", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option(
    "--gcs-destination-prefix",
    required=True,
    help=("GCS destination prefix, e.g. gs://bucket/path/. Required: no default is provided."),
)
@click.option("--execute/--dry-run", default=False)
def upload_gcs_direct_cmd(
    dataset: str,
    output_base: str,
    dataset_output_dir: str | None,
    write_evidence: Path | None,
    gcs_destination_prefix: str,
    execute: bool,
) -> None:
    """Stage 2.5: Lustre -> GCS directly via gcloud rsync (bypasses S3)."""
    from bspp.orchestration.runtime.postprocessing import upload_and_track as upload_and_track_mod

    plan = upload_and_track_mod.upload_gcs_direct(
        dataset,
        Path(dataset_output_dir) if dataset_output_dir is not None else Path(output_base) / dataset,
        gcs_destination_prefix=gcs_destination_prefix,
        execute=execute,
        dry_run=not execute,
    )
    _run_stage(upload_and_track_mod.render_plan(plan), plan, execute, write_evidence)


# -------------------------------------------------------------------
# Extraction commands
# -------------------------------------------------------------------


@cli.command("extract")
@click.option("--staging-dir", required=True, type=click.Path(exists=True), help="Directory with .tar.lz4 archives.")
@click.option("--output-dir", required=True, type=click.Path(), help="Destination directory for extracted files.")
@click.option("--parallel", default=0, type=int, help="Parallel workers (0=sequential).")
@click.option("--keep-archives", is_flag=True, help="Keep .tar.lz4 files after extraction.")
@click.option("--dry-run", is_flag=True, help="Show what would be extracted without extracting.")
def extract_cmd(staging_dir: str, output_dir: str, parallel: int, keep_archives: bool, dry_run: bool) -> None:
    """Extract .tar.lz4 archives from staging directory."""
    from bspp.orchestration.runtime.extraction.archives import (
        EXTRACTED_MARKER_DIR,
        extract_parallel,
        extract_sequential,
        find_archives,
    )

    staging = Path(staging_dir)
    out = Path(output_dir)
    archives = find_archives(staging)

    if not archives:
        click.echo(f"No .tar.lz4 archives found in {staging}")
        sys.exit(1)

    click.echo(f"Found {len(archives)} archive(s) in {staging}")

    if dry_run:
        mode = "parallel" if parallel > 0 else "sequential"
        click.echo(f"[DRY RUN] Would extract {len(archives)} archives ({mode})")
        for a in archives[:10]:
            click.echo(f"  {a.name}")
        if len(archives) > 10:
            click.echo(f"  ... and {len(archives) - 10} more")
        return

    marker_dir = out / EXTRACTED_MARKER_DIR

    if parallel > 0:
        total = extract_parallel(
            archives,
            out,
            keep_archives=keep_archives,
            marker_dir=marker_dir,
            workers=parallel,
        )
    else:
        total = extract_sequential(
            archives,
            out,
            keep_archives=keep_archives,
            marker_dir=marker_dir,
        )

    click.echo(f"Extracted {total} files from {len(archives)} archives to {out}")


# -------------------------------------------------------------------
# Preprocess commands
# -------------------------------------------------------------------


@cli.command("preprocess")
@click.option("--input-dir", type=click.Path(exists=True), default=None, help="Model files dir (flat mode).")
@click.option("--staging-dir", type=click.Path(exists=True), default=None, help="Archives dir (archive mode).")
@click.option("--output-dir", required=True, type=click.Path(), help="Output directory for manifests and configs.")
@click.option("--manifest-csv", type=click.Path(exists=True), default=None, help="Chain-mapping manifest CSV.")
@click.option("--shards-per-archive", type=int, default=2, help="Sub-shards per archive.")
@click.option("--tool-used", default="ColabFold v1.6.0 / AlphaFold-Multimer", help="Prediction tool.")
@click.option("--provider-id", default="NVDA", help="Provider ID for config files.")
@click.option("--provider-name", default="NVIDIA", help="Provider name for config files.")
def preprocess_cmd(
    input_dir: str | None,
    staging_dir: str | None,
    output_dir: str,
    manifest_csv: str | None,
    shards_per_archive: int,
    tool_used: str,
    provider_id: str,
    provider_name: str,
) -> None:
    """Discover model IDs, filter manifest, and generate shard config."""
    from bspp.orchestration.runtime.discovery import build_file_index, discover
    from bspp.orchestration.runtime.extraction.archives import find_archives
    from bspp.orchestration.runtime.postprocessing.manifest import (
        filter_manifest_for_models,
        write_archive_list,
        write_dataset_config,
        write_file_index,
        write_model_ids,
        write_provider_json,
    )
    from bspp.orchestration.runtime.postprocessing.shard_config import (
        compute_archive_shard_config,
        compute_shard_config,
        write_shard_config,
    )

    if input_dir is None and staging_dir is None:
        click.echo("Either --input-dir or --staging-dir must be specified.", err=True)
        sys.exit(1)
    if input_dir is not None and staging_dir is not None:
        click.echo("--input-dir and --staging-dir are mutually exclusive.", err=True)
        sys.exit(1)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Archive mode
    if staging_dir is not None:
        staging = Path(staging_dir)
        archives = find_archives(staging)
        if not archives:
            click.echo(f"No .tar.lz4 archives found in {staging}", err=True)
            sys.exit(1)

        archive_names = [a.name for a in archives]
        write_archive_list(archive_names, out / "archive_list.json")
        click.echo(f"Found {len(archive_names)} archives")

        shard_cfg = compute_archive_shard_config(archive_names, shards_per_archive)
        write_shard_config(shard_cfg, out / "shard_config.json")

        write_dataset_config(tool_used, provider_id, out / "dataset_config.json")
        write_provider_json(provider_id, provider_name, out / "provider.json")

        click.echo(f"Archive mode: {len(archive_names)} archives, array range {shard_cfg['array_range']}")
        return

    # Flat-file mode
    inp = Path(input_dir)  # type: ignore[arg-type]
    model_ids = discover(inp)
    if not model_ids:
        click.echo(f"No model IDs found in {inp}", err=True)
        sys.exit(1)

    write_model_ids(model_ids, out / "model_ids.txt")
    click.echo(f"Discovered {len(model_ids)} model IDs")

    file_index = build_file_index(inp)
    write_file_index(file_index, out / "file_index.json")

    if manifest_csv is not None:
        n = filter_manifest_for_models(
            Path(manifest_csv),
            set(model_ids),
            out / "filtered_manifest.csv",
        )
        click.echo(f"Filtered manifest: {n} rows")

    shard_cfg = compute_shard_config(len(model_ids))
    write_shard_config(shard_cfg, out / "shard_config.json")

    write_dataset_config(tool_used, provider_id, out / "dataset_config.json")
    write_provider_json(provider_id, provider_name, out / "provider.json")

    click.echo(
        f"Shard config: {len(model_ids)} models -> {shard_cfg['required_shards']} shards "
        f"(max {MAX_PROTEINS_PER_SHARD}/shard)"
    )


# -------------------------------------------------------------------
# Recipe commands
# -------------------------------------------------------------------


@cli.command("recipe")
@click.option("--dataset", required=True, help="Dataset name for the recipe.")
@click.option("--output-dir", required=True, type=click.Path(), help="Parent directory for the recipe.")
@click.option("--template", type=click.Path(exists=True), default=None, help="Path to a template config.yaml.")
def recipe_cmd(dataset: str, output_dir: str, template: str | None) -> None:
    """Generate a recipe directory with config.yaml for a dataset."""
    from bspp.orchestration.runtime.postprocessing.recipe import generate_recipe

    template_path = Path(template) if template is not None else None
    recipe_dir = generate_recipe(
        dataset,
        Path(output_dir),
        template_path=template_path,
    )
    click.echo(f"Recipe created at {recipe_dir}")


# -------------------------------------------------------------------
# Process commands (shard + pipeline)
# -------------------------------------------------------------------


@cli.command("process")
@click.option("--shard-id", type=int, default=None, help="Shard index (0-based). Required for sharding.")
@click.option("--input-dir", required=True, type=click.Path(exists=True), help="Directory with model input files.")
@click.option("--output-dir", required=True, type=click.Path(), help="Pipeline output directory.")
@click.option("--manifest-csv", type=click.Path(exists=True), default=None, help="Chain-mapping manifest CSV.")
@click.option(
    "--model-ids-file", type=click.Path(exists=True), default=None, help="File with model IDs (one per line)."
)
@click.option("--staging-dir", type=click.Path(), default=None, help="Staging directory for shard symlinks.")
@click.option("--batch-size", type=int, default=MAX_PROTEINS_PER_SHARD, help="Max models per shard.")
@click.option("--workers", type=int, default=4, help="Parallel pipeline workers.")
@click.option(
    "--stages",
    default="ipsae dssp validation metadata_export modelcif_export",
    help="Pipeline stages to run (space-separated).",
)
@click.option("--uniprot-db", type=click.Path(exists=True), default=None, help="UniProt DuckDB database path.")
@click.option("--dry-run", is_flag=True, help="Show what would be processed without running the pipeline.")
def process_cmd(
    shard_id: int | None,
    input_dir: str,
    output_dir: str,
    manifest_csv: str | None,
    model_ids_file: str | None,
    staging_dir: str | None,
    batch_size: int,
    workers: int,
    stages: str,
    uniprot_db: str | None,
    dry_run: bool,
) -> None:
    """Process a shard through the pipeline (pre-filter, symlink, run)."""
    from bspp.orchestration.runtime.discovery import build_file_index, discover
    from bspp.orchestration.runtime.postprocessing.manifest import read_model_ids
    from bspp.orchestration.runtime.postprocessing.runner import prefilter_batch, run_pipeline
    from bspp.orchestration.runtime.postprocessing.sharding import (
        compute_shard_slice,
        create_symlink_shard,
        filter_manifest_for_shard,
    )

    inp = Path(input_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Resolve model IDs
    all_model_ids = read_model_ids(Path(model_ids_file)) if model_ids_file is not None else discover(inp)

    if not all_model_ids:
        click.echo("No model IDs found.", err=True)
        sys.exit(1)

    # Compute shard slice if shard_id is given
    if shard_id is not None:
        num_shards = max(1, -(-len(all_model_ids) // batch_size))  # ceil division
        start, end = compute_shard_slice(shard_id, len(all_model_ids), num_shards)
        shard_model_ids = all_model_ids[start:end]
        click.echo(f"Shard {shard_id}/{num_shards}: models [{start}:{end}] ({len(shard_model_ids)} models)")
    else:
        shard_model_ids = all_model_ids
        click.echo(f"Processing all {len(shard_model_ids)} models (no sharding)")

    # Pre-filter for null meta JSONs
    good_ids, bad_ids = prefilter_batch(shard_model_ids, inp)
    if bad_ids:
        click.echo(f"Pre-filter: {len(bad_ids)} models skipped (null meta JSON)")
        for mid, reason in bad_ids[:5]:
            click.echo(f"  {mid}: {reason}")
        if len(bad_ids) > 5:
            click.echo(f"  ... and {len(bad_ids) - 5} more")

    click.echo(f"Models to process: {len(good_ids)}")

    if dry_run:
        click.echo("[DRY RUN] Would process the above models. Exiting.")
        return

    # Set up shard staging directory with symlinks
    shard_input = inp
    if staging_dir is not None:
        shard_staging = Path(staging_dir)
        file_index = build_file_index(inp)
        linked = create_symlink_shard(good_ids, file_index, inp, shard_staging)
        click.echo(f"Created {linked} symlinks in {shard_staging}")
        shard_input = shard_staging

    # Filter manifest for shard
    shard_manifest: Path | None = None
    if manifest_csv is not None:
        shard_manifest = out / "shard_manifest.csv"
        n = filter_manifest_for_shard(Path(manifest_csv), set(good_ids), shard_manifest)
        click.echo(f"Shard manifest: {n} rows")

    if shard_manifest is None:
        click.echo("No manifest CSV provided; pipeline may not run all stages.", err=True)
        sys.exit(1)

    # Run pipeline
    uniprot_path = Path(uniprot_db) if uniprot_db is not None else None
    exit_code = run_pipeline(
        shard_input,
        out,
        shard_manifest,
        stages=stages,
        workers=workers,
        uniprot_db=uniprot_path,
    )

    if exit_code != 0:
        click.echo(f"Pipeline exited with code {exit_code}", err=True)
        sys.exit(exit_code)

    click.echo("Pipeline completed successfully.")


# -------------------------------------------------------------------
# Aggregate commands
# -------------------------------------------------------------------


@cli.command("aggregate")
@click.option("--dataset", required=True, help="Dataset name.")
@click.option(
    "--output-base",
    required=True,
    type=click.Path(exists=True),
    help="Base output directory containing shard dirs.",
)
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="Output parquet path (default: <output-base>/<dataset>_manifest.parquet).",
)
def aggregate_cmd(dataset: str, output_base: str, output: str | None) -> None:
    """Aggregate per-shard manifests into a single dataset parquet."""
    from bspp.orchestration.runtime.postprocessing.aggregate import aggregate_dataset

    output_path = Path(output) if output is not None else None
    result = aggregate_dataset(dataset, Path(output_base), output_path=output_path)
    click.echo(f"Aggregated manifest written to {result}")


# -------------------------------------------------------------------
# Status commands
# -------------------------------------------------------------------


@cli.command("status")
@click.option("--dataset", default=None, help="Filter to a single dataset.")
@click.option("--tracking", required=True, type=click.Path(exists=True), help="Path to tracking parquet.")
@click.option(
    "--output-base",
    type=click.Path(exists=True),
    default=None,
    help="Dataset output dir root; enables per-shard counts when set.",
)
@click.option("--show-shards", is_flag=True, help="Include per-shard output counts in the report.")
@click.option("--rich/--plain", default=False, help="Use rich progress-bar + tables when available.")
def status_cmd(
    dataset: str | None,
    tracking: str,
    output_base: str | None,
    show_shards: bool,
    rich: bool,
) -> None:
    """Show post-processing status from tracking parquet."""
    from bspp.orchestration.runtime.postprocessing.tracking import query_status
    from bspp.orchestration.runtime.status import render_full_status

    counts = query_status(Path(tracking), dataset=dataset)

    count_report = None
    if show_shards:
        if output_base is None or dataset is None:
            click.echo("--show-shards requires --output-base and --dataset.", err=True)
            sys.exit(1)
        from bspp.orchestration.runtime.validation.count_outputs import count_shard_outputs

        dataset_dir = Path(output_base) / dataset
        try:
            count_report = count_shard_outputs(dataset_dir)
        except FileNotFoundError as exc:
            click.echo(f"{exc}", err=True)
            sys.exit(1)

    click.echo(render_full_status(counts, dataset=dataset, count_report=count_report, use_rich=rich))


# -------------------------------------------------------------------
# Validation commands
# -------------------------------------------------------------------


@cli.group("validate")
def validate_group() -> None:
    """Output validation helpers (count, timing, coverage)."""


@validate_group.command("count")
@click.option("--dataset", required=True)
@click.option("--output-base", required=True, type=click.Path(exists=True))
@click.option("--failed-only", is_flag=True, help="Print only comma-separated failed shard IDs.")
@click.option("--compact", is_flag=True, help="Skip per-shard lines; print summary only.")
def validate_count_cmd(dataset: str, output_base: str, failed_only: bool, compact: bool) -> None:
    """Count per-shard success outputs vs expected."""
    from bspp.orchestration.runtime.validation.count_outputs import (
        count_shard_outputs,
        render_count_report,
    )

    dataset_dir = Path(output_base) / dataset
    try:
        report = count_shard_outputs(dataset_dir)
    except FileNotFoundError as exc:
        click.echo(f"{exc}", err=True)
        sys.exit(1)

    if failed_only:
        click.echo(",".join(str(i) for i in report.failed_ids))
        # Exit non-zero if there is anything wrong: either shards that
        # need resubmission (mismatches) or stale shard dirs that need
        # to be cleaned. Stale shards are deliberately absent from
        # failed_ids (they're not resubmit candidates) so --failed-only
        # would otherwise return an empty line + exit 0 even though the
        # dataset is contaminated.
        sys.exit(0 if report.valid else 1)

    click.echo(render_count_report(report, compact=compact))
    sys.exit(0 if report.valid else 1)


@validate_group.command("timing")
@click.option("--dataset", required=True)
@click.option("--output-base", required=True, type=click.Path(exists=True))
def validate_timing_cmd(dataset: str, output_base: str) -> None:
    """Aggregate per-stage timing across shards."""
    from bspp.orchestration.runtime.validation.timing import (
        aggregate,
        load_shard_results,
        render_timing_summary,
    )

    dataset_dir = Path(output_base) / dataset
    results = load_shard_results(dataset_dir)
    click.echo(render_timing_summary(aggregate(results)))


@validate_group.command("coverage")
@click.option("--dataset", required=True)
@click.option("--output-base", required=True, type=click.Path(exists=True))
@click.option("--model-ids-file", required=True, type=click.Path(exists=True))
def validate_coverage_cmd(dataset: str, output_base: str, model_ids_file: str) -> None:
    """Verify shard coverage: no gaps, no duplicates, manifests consistent."""
    from bspp.orchestration.runtime.validation.coverage import (
        render_coverage_report,
        validate_coverage,
    )

    dataset_dir = Path(output_base) / dataset
    report = validate_coverage(dataset_dir, model_ids_file=Path(model_ids_file))
    click.echo(render_coverage_report(report))
    sys.exit(0 if report.coverage_ok else 1)


@validate_group.command("phase2-parity")
@click.option("--baseline-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--candidate-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--write-report", type=click.Path(file_okay=False), default=None)
@click.option(
    "--profile",
    type=click.Choice(["fast", "exact"]),
    default="fast",
    show_default=True,
    help="Parity profile: fast skips volatile whole-file hashes; exact keeps byte-level ledger hashes.",
)
@click.option("--strict", is_flag=True, help="Exit non-zero when parity checks fail.")
def validate_phase2_parity_cmd(
    baseline_dir: str,
    candidate_dir: str,
    write_report: str | None,
    profile: str,
    strict: bool,
) -> None:
    """Compare Phase 2 task-278 outputs against the Phase 1 oracle."""
    from bspp.orchestration.runtime.validation.phase2_parity import (
        ParityProfile,
        compare_phase2_parity,
        render_phase2_parity_report,
        write_phase2_parity_report,
    )

    report = compare_phase2_parity(Path(baseline_dir), Path(candidate_dir), profile=cast(ParityProfile, profile))
    click.echo(render_phase2_parity_report(report), nl=False)
    if write_report:
        write_phase2_parity_report(report, Path(write_report))
    if strict and not report.ok:
        sys.exit(1)


@validate_group.command("tar-payload-parity")
@click.option("--baseline-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--candidate-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--relative-dir", default="local_tars", show_default=True)
@click.option("--baseline-run-name", default=None, help="Run-name token to normalize in baseline member names.")
@click.option("--candidate-run-name", default=None, help="Run-name token to normalize in candidate member names.")
@click.option("--zstd-path", default="zstd", show_default=True, help="zstd executable used for .zst members.")
@click.option("--workers", default=1, show_default=True, type=click.IntRange(min=1))
@click.option("--sample-limit", default=20, show_default=True, type=click.IntRange(min=0))
@click.option(
    "--payload-sample-count",
    default=None,
    type=click.IntRange(min=0),
    help="Hash a deterministic payload sample after checking full tar/member inventory.",
)
@click.option(
    "--match-mode",
    type=click.Choice(["by-tar", "aggregate"]),
    default="by-tar",
    show_default=True,
    help="by-tar requires each member in the same tar; aggregate ignores tar placement.",
)
@click.option(
    "--exclude",
    multiple=True,
    default=(),
    help="Relative path prefix to exclude from tar inventory (repeatable).",
)
@click.option("--write-report", type=click.Path(file_okay=False), default=None)
@click.option("--strict", is_flag=True, help="Exit non-zero when payload parity checks fail.")
def validate_tar_payload_parity_cmd(
    baseline_dir: str,
    candidate_dir: str,
    relative_dir: str,
    baseline_run_name: str | None,
    candidate_run_name: str | None,
    zstd_path: str,
    workers: int,
    sample_limit: int,
    payload_sample_count: int | None,
    match_mode: str,
    exclude: tuple[str, ...],
    write_report: str | None,
    strict: bool,
) -> None:
    """Compare local-tar payload bytes, decompressing .zst members."""
    from bspp.orchestration.runtime.validation.tar_payload_parity import (
        MatchMode,
        compare_tar_payload_parity,
        render_tar_payload_parity_report,
        write_tar_payload_parity_report,
    )

    report = compare_tar_payload_parity(
        Path(baseline_dir),
        Path(candidate_dir),
        relative_dir=relative_dir,
        baseline_run_name=baseline_run_name,
        candidate_run_name=candidate_run_name,
        zstd_path=zstd_path,
        workers=workers,
        sample_limit=sample_limit,
        payload_sample_count=payload_sample_count,
        match_mode=cast(MatchMode, match_mode),
        exclude=exclude,
    )
    click.echo(render_tar_payload_parity_report(report), nl=False)
    if write_report:
        write_tar_payload_parity_report(report, Path(write_report))
    if strict and not report.ok:
        sys.exit(1)


@validate_group.command("semantic-acceptance")
@click.option("--baseline-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--candidate-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--expected-tar-count", type=click.IntRange(min=0), default=None)
@click.option("--expected-local-tars-rows", type=click.IntRange(min=0), default=None)
@click.option("--expected-failed-rows", type=click.IntRange(min=0), default=None)
@click.option("--expected-analysis-rows", type=click.IntRange(min=0), default=None)
@click.option("--expected-selected-ids", type=click.IntRange(min=0), default=None)
@click.option(
    "--candidate-parquet-required/--candidate-parquet-optional",
    default=True,
    help="Require candidate analysis_metadata.parquet and validate its row count.",
)
@click.option("--compare-failed-sets/--no-compare-failed-sets", default=True)
@click.option("--compare-tar-manifest-rows/--no-compare-tar-manifest-rows", default=True)
@click.option("--compare-analysis-model-rows/--no-compare-analysis-model-rows", default=True)
@click.option("--write-report", type=click.Path(file_okay=False), default=None)
@click.option("--strict", is_flag=True, help="Exit non-zero when semantic acceptance fails.")
def validate_semantic_acceptance_cmd(
    baseline_dir: str,
    candidate_dir: str,
    expected_tar_count: int | None,
    expected_local_tars_rows: int | None,
    expected_failed_rows: int | None,
    expected_analysis_rows: int | None,
    expected_selected_ids: int | None,
    candidate_parquet_required: bool,
    compare_failed_sets: bool,
    compare_tar_manifest_rows: bool,
    compare_analysis_model_rows: bool,
    write_report: str | None,
    strict: bool,
) -> None:
    """Compare semantic outputs after tar payload parity passes."""
    from bspp.orchestration.runtime.validation.semantic_acceptance import (
        compare_semantic_acceptance,
        render_semantic_acceptance_report,
        write_semantic_acceptance_report,
    )

    report = compare_semantic_acceptance(
        Path(baseline_dir),
        Path(candidate_dir),
        expected_tar_count=expected_tar_count,
        expected_local_tars_rows=expected_local_tars_rows,
        expected_failed_rows=expected_failed_rows,
        expected_analysis_rows=expected_analysis_rows,
        expected_selected_ids=expected_selected_ids,
        require_candidate_parquet=candidate_parquet_required,
        compare_failed_sets=compare_failed_sets,
        compare_tar_manifest_rows=compare_tar_manifest_rows,
        compare_analysis_model_rows=compare_analysis_model_rows,
    )
    click.echo(render_semantic_acceptance_report(report), nl=False)
    if write_report:
        write_semantic_acceptance_report(report, Path(write_report))
    if strict and not report.ok:
        sys.exit(1)


@validate_group.command("hq-chunks")
@click.option("--chunks-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--selected-ids", required=True, type=click.Path(exists=True))
@click.option("--model-tar-index", required=True, type=click.Path(exists=True))
@click.option("--chunk-size", type=click.IntRange(min=1), default=1000, show_default=True)
@click.option("--sample-limit", type=click.IntRange(min=1), default=20, show_default=True)
@click.option("--write-report", type=click.Path(file_okay=False), default=None)
@click.option("--strict", is_flag=True, help="Exit non-zero when HQ chunk validation fails.")
def validate_hq_chunks_cmd(
    chunks_dir: str,
    selected_ids: str,
    model_tar_index: str,
    chunk_size: int,
    sample_limit: int,
    write_report: str | None,
    strict: bool,
) -> None:
    """Validate high-quality chunk tar outputs."""
    from bspp.orchestration.runtime.validation.hq_chunks import (
        render_hq_chunk_validation_report,
        validate_hq_chunks,
        write_hq_chunk_validation_report,
    )

    report = validate_hq_chunks(
        chunks_dir=Path(chunks_dir),
        selected_ids_path=Path(selected_ids),
        model_tar_index_path=Path(model_tar_index),
        chunk_size=chunk_size,
        sample_limit=sample_limit,
    )
    click.echo(render_hq_chunk_validation_report(report), nl=False)
    if write_report:
        write_hq_chunk_validation_report(report, Path(write_report))
    if strict and not report.ok:
        sys.exit(1)


@validate_group.command("hq-chunks-publication")
@click.option("--runspec", "spec_path", required=True, type=click.Path(exists=True), help="RunSpec YAML.")
@click.option("--chunks-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--upload-manifest", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--remote-inventory", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--sampled-hashes", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--write-report", type=click.Path(file_okay=False), default=None)
@click.option("--strict", is_flag=True, help="Exit non-zero when publication gate-A fails.")
def validate_hq_chunks_publication_cmd(
    spec_path: str,
    chunks_dir: str,
    upload_manifest: str,
    remote_inventory: str,
    sampled_hashes: str,
    write_report: str | None,
    strict: bool,
) -> None:
    """Validate HQ chunk publication from pre-collected evidence files."""
    from bspp.orchestration.contract.runspec import load_runspec
    from bspp.orchestration.runtime.validation.hq_publication import (
        render_hq_publication_validation_report,
        validate_hq_chunk_publication_evidence_files,
        write_hq_publication_validation_report,
    )

    spec = load_runspec(Path(spec_path))
    report = validate_hq_chunk_publication_evidence_files(
        spec,
        chunks_dir=Path(chunks_dir),
        upload_manifest_path=Path(upload_manifest),
        remote_inventory_path=Path(remote_inventory),
        sampled_hashes_path=Path(sampled_hashes),
    )
    click.echo(render_hq_publication_validation_report(report), nl=False)
    if write_report:
        write_hq_publication_validation_report(report, Path(write_report))
    if strict and not report.ok:
        sys.exit(1)


# -------------------------------------------------------------------
# SLURM submission commands (via submitit)
# -------------------------------------------------------------------


def _parse_array(value: str) -> list[int]:
    """Parse ``0-9`` / ``5,12,37`` / ``0-5,10,12-14`` array ranges."""
    ids: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            ids.extend(range(int(lo), int(hi) + 1))
        else:
            ids.append(int(part))
    return sorted(set(ids))


@cli.group("slurm")
def slurm_group() -> None:
    """SLURM submission helpers (submitit)."""


@slurm_group.command("submit")
@click.option("--recipe-dir", required=True, type=click.Path(exists=True))
@click.option("--array", required=True, help="Shard IDs: '0-9' or '5,12,37'.")
@click.option("--input-dir", required=True, type=click.Path(exists=True))
@click.option("--output-dir", required=True, type=click.Path())
# --manifest-csv is required here because the `process` worker invoked by each
# shard exits non-zero when no manifest is provided; silently accepting None
# would enqueue an array that fails at runtime instead of at submission.
@click.option("--manifest-csv", required=True, type=click.Path(exists=True))
@click.option("--uniprot-db", type=click.Path(exists=True), default=None)
@click.option("--stages", default="ipsae dssp validation metadata_export modelcif_export")
@click.option("--workers", type=int, default=24)
@click.option("--dry-run", is_flag=True)
def slurm_submit_cmd(
    recipe_dir: str,
    array: str,
    input_dir: str,
    output_dir: str,
    manifest_csv: str,
    uniprot_db: str | None,
    stages: str,
    workers: int,
    dry_run: bool,
) -> None:
    """Submit a SLURM array job over a list of shard IDs."""
    from bspp.orchestration.runtime.slurm import submit_array

    shard_ids = _parse_array(array)
    if not shard_ids:
        click.echo("No shard IDs parsed from --array.", err=True)
        sys.exit(1)

    summary = submit_array(
        Path(recipe_dir),
        shard_ids,
        input_dir=Path(input_dir),
        output_dir=Path(output_dir),
        manifest_csv=Path(manifest_csv),
        uniprot_db=Path(uniprot_db) if uniprot_db else None,
        stages=stages,
        workers=workers,
        dry_run=dry_run,
    )
    if dry_run:
        click.echo("[DRY RUN] Would submit:")
        click.echo(f"  shards: {len(summary.shard_ids)} (example argv below)")
        click.echo(f"  log folder: {summary.log_folder}")
        click.echo(f"  argv: {' '.join(summary.argv_preview)}")
    else:
        click.echo(f"Submitted {len(summary.shard_ids)} shards -> {len(summary.job_ids)} job(s)")
        for jid in summary.job_ids[:5]:
            click.echo(f"  job {jid}")
        if len(summary.job_ids) > 5:
            click.echo(f"  ... and {len(summary.job_ids) - 5} more")


@slurm_group.command("resubmit-failed")
@click.option("--recipe-dir", required=True, type=click.Path(exists=True))
@click.option("--dataset", required=True)
@click.option("--output-base", required=True, type=click.Path(exists=True))
@click.option("--input-dir", required=True, type=click.Path(exists=True))
# --manifest-csv is required for the same reason as in `slurm submit`: the
# `process` worker exits 1 without it.
@click.option("--manifest-csv", required=True, type=click.Path(exists=True))
@click.option("--uniprot-db", type=click.Path(exists=True), default=None)
@click.option("--stages", default="ipsae dssp validation metadata_export modelcif_export")
@click.option("--workers", type=int, default=24)
@click.option("--dry-run", is_flag=True)
def slurm_resubmit_cmd(
    recipe_dir: str,
    dataset: str,
    output_base: str,
    input_dir: str,
    manifest_csv: str,
    uniprot_db: str | None,
    stages: str,
    workers: int,
    dry_run: bool,
) -> None:
    """Resubmit only shards with mismatched success-output counts."""
    from bspp.orchestration.runtime.slurm import resubmit_failed_shards

    failed, summary = resubmit_failed_shards(
        Path(recipe_dir),
        dataset=dataset,
        output_base=Path(output_base),
        input_dir=Path(input_dir),
        manifest_csv=Path(manifest_csv),
        uniprot_db=Path(uniprot_db) if uniprot_db else None,
        stages=stages,
        workers=workers,
        dry_run=dry_run,
    )

    if not failed:
        click.echo("No failed shards detected — nothing to resubmit.")
        return

    click.echo(f"Failed shards: {failed}")
    if summary is None:
        return
    if dry_run:
        click.echo(f"[DRY RUN] Would resubmit {len(failed)} shards via {summary.log_folder}.")
    else:
        click.echo(f"Resubmitted {len(failed)} shards -> {len(summary.job_ids)} job(s)")


@slurm_group.command("status")
@click.option("--job-id", "job_ids", multiple=True, required=True)
@click.option("--log-folder", required=True, type=click.Path(exists=True))
def slurm_status_cmd(job_ids: tuple[str, ...], log_folder: str) -> None:
    """Query one or more submitit job IDs from their log folder."""
    from bspp.orchestration.runtime.slurm import query_jobs
    from bspp.orchestration.runtime.slurm.status import render_job_statuses

    statuses = query_jobs(job_ids, log_folder=Path(log_folder))
    click.echo(render_job_statuses(statuses))


# -------------------------------------------------------------------
# Tracking commands
# -------------------------------------------------------------------


@cli.group("tracking")
def tracking_group() -> None:
    """Tracking parquet lifecycle operations."""


@tracking_group.command("create")
@click.option("--master", required=True, type=click.Path(exists=True), help="Path to master parquet.")
@click.option("--output", required=True, type=click.Path(), help="Output tracking parquet path.")
@click.option(
    "--s3-output-prefix",
    required=True,
    help="Explicit s3:// destination prefix written to the s3_destination column.",
)
@click.option(
    "--gcs-destination-prefix",
    default=None,
    help="Optional gs:// destination prefix; omitted writes a null gcs_destination column.",
)
@click.option("--force", is_flag=True, help="Overwrite existing output file.")
def tracking_create_cmd(
    master: str,
    output: str,
    s3_output_prefix: str,
    gcs_destination_prefix: str | None,
    force: bool,
) -> None:
    """Create a tracking parquet from a master parquet."""
    from bspp.orchestration.runtime.postprocessing.tracking import create_tracking_parquet

    n = create_tracking_parquet(
        Path(master),
        Path(output),
        s3_output_prefix=s3_output_prefix,
        gcs_destination_prefix=gcs_destination_prefix,
        force=force,
    )
    click.echo(f"Tracking parquet created with {n:,} rows at {output}")


@tracking_group.command("update")
@click.option("--tracking", required=True, type=click.Path(exists=True), help="Path to tracking parquet.")
@click.option("--match-column", required=True, help="Column to match against.")
@click.option("--match-substring", required=True, help="Substring to search for in the match column.")
@click.option("--set-status", required=True, help="New postprocess_status value.")
@click.option("--dry-run", is_flag=True, help="Preview matched rows without writing.")
def tracking_update_cmd(tracking: str, match_column: str, match_substring: str, set_status: str, dry_run: bool) -> None:
    """Bulk-update status in the tracking parquet by column substring match."""
    from bspp.orchestration.runtime.postprocessing.tracking import update_status

    count = update_status(
        Path(tracking),
        match_column=match_column,
        match_substring=match_substring,
        new_status=set_status,
        dry_run=dry_run,
    )
    mode = "[DRY RUN] Would update" if dry_run else "Updated"
    click.echo(f"{mode} {count:,} rows to status={set_status!r}")


# -------------------------------------------------------------------
# Cleanup commands
# -------------------------------------------------------------------


@cli.command("cleanup")
@click.option("--dataset", default=None, help="Dataset name for tracking filter.")
@click.option(
    "--output-base",
    required=True,
    type=click.Path(exists=True),
    help="Base output directory containing shard dirs.",
)
@click.option("--force", is_flag=True, help="Skip tracking safety check.")
@click.option(
    "--tracking",
    type=click.Path(exists=True),
    default=None,
    help="Path to tracking parquet (required unless --force).",
)
def cleanup_cmd(dataset: str | None, output_base: str, force: bool, tracking: str | None) -> None:
    """Remove shard_*/success_outputs/ directories after upload."""
    from bspp.orchestration.runtime.postprocessing.cleanup import cleanup_shard_outputs

    tracking_path = Path(tracking) if tracking is not None else None

    if not force and tracking_path is None:
        click.echo("--tracking is required unless --force is used.", err=True)
        sys.exit(1)

    count = cleanup_shard_outputs(
        Path(output_base),
        force=force,
        tracking_path=tracking_path,
        dataset=dataset,
    )
    click.echo(f"Cleaned {count} shard directories.")


@cli.group("benchmark")
def benchmark_group() -> None:
    """Cluster-side folding benchmark validation worker commands."""


@benchmark_group.command("validate-run")
@click.option(
    "--run-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Completed folding run directory.",
)
@click.option(
    "--suite",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to validation-suite.json.",
)
@click.option(
    "--index",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the canonical-pair index.json.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Validation evidence output directory (defaults to run-dir/validation).",
)
@click.option("--corpus", required=True, help="S3 prefix whose contents are fetched recursively.")
@click.option("--fingerprint", required=True, help="Pinned dataset fingerprint (64 hex).")
def benchmark_validate_run_cmd(
    run_dir: Path,
    suite: Path,
    index: Path,
    output_dir: Path | None,
    corpus: str,
    fingerprint: str,
) -> None:
    """Fetch the pinned benchmark corpus and validate a completed folding run."""

    from bspp.orchestration.runtime.data_movement.common import ToolMissingError
    from bspp.orchestration.runtime.data_movement.s3.client import MissingS3CredentialsError
    from bspp.orchestration.runtime.folding.benchmark.corpus import fetch_pinned_corpus
    from bspp.orchestration.runtime.folding.benchmark.validation import validate_run

    with tempfile.TemporaryDirectory(prefix="bspp-benchmark-corpus-") as corpus_temp:
        corpus_dir = Path(corpus_temp)
        try:
            fetch_pinned_corpus(corpus, corpus_dir, credentials=None, expected_fingerprint=fingerprint)
            summary = validate_run(
                run_dir,
                suite,
                index,
                corpus_dir,
                output_dir=output_dir,
                expected_fingerprint=fingerprint,
            )
        except (MissingS3CredentialsError, ToolMissingError, OSError, TypeError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
    click.echo(
        json.dumps(
            {
                "case_count": summary["case_count"],
                "passed_count": summary["passed_count"],
                "cases": summary["cases"],
            },
            sort_keys=True,
        ),
        nl=False,
    )


def _write_evidence_atomic(dest: Path, payload: object) -> None:
    """Write evidence JSON atomically (temp file + rename) to avoid partial files."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(suffix=".tmp", dir=str(dest.parent))
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(content)
        os.replace(tmp_name, dest)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


@cli.group("phase")
def phase_publish_group() -> None:
    """Phase transport publication operations."""


@phase_publish_group.command("publish-preprocessing")
@click.option(
    "--input-json",
    "input_json_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="JSON file containing the artifact location mapping and s3 prefix.",
)
@click.option(
    "--evidence-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory to write artifact-location-remote.json and msa-set-upload-evidence.json.",
)
def phase_publish_preprocessing_cmd(input_json_path: Path, evidence_dir: Path) -> None:
    """Publish a verified local MSA-set bundle to S3 and write upload evidence.

    This is a transport-only operator command expected to run after Phase
    finalization. It does not itself check the Phase lifecycle state; the
    operator is trusted to invoke it post-finalization.
    """
    from bspp.orchestration.contract.preprocessing_handoff import (
        verified_local_bundled_artifact_location_from_mapping,
    )
    from bspp.orchestration.runtime.folding.seam_transport import (
        SeamTransportError,
        publish_msa_set_to_s3,
    )

    try:
        payload = json.loads(input_json_path.read_text())
        artifact_location = verified_local_bundled_artifact_location_from_mapping(payload["artifact_location"])
        s3_prefix = payload["s3_prefix"]
        phase_run_id = payload.get("phase_run_id")
        attempt_id = payload.get("attempt_id")
        remote, evidence = publish_msa_set_to_s3(
            location=artifact_location,
            s3_prefix=s3_prefix,
            phase_run_id=phase_run_id,
            attempt_id=attempt_id,
        )
        evidence_dir.mkdir(parents=True, exist_ok=True)
        _write_evidence_atomic(evidence_dir / "artifact-location-remote.json", remote.to_mapping())
        _write_evidence_atomic(evidence_dir / "msa-set-upload-evidence.json", evidence.to_mapping())
    except (SeamTransportError, OSError, TypeError, ValueError, KeyError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        json.dumps(
            {
                "remote_artifact_location_id": remote.artifact_location_id,
                "object_key": evidence.object_key,
            },
            sort_keys=True,
        ),
        nl=False,
    )


@phase_publish_group.command("publish-folding")
@click.option(
    "--input-json",
    "input_json_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="JSON file containing prediction bundle records, local paths, and s3 prefix.",
)
@click.option(
    "--evidence-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory to write prediction-bundle-upload-evidence.json.",
)
@click.option(
    "--bundles",
    "bundles_json",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Optional operator-attested prediction-bundle records JSON (defaults to input-json bundles).",
)
def phase_publish_folding_cmd(
    input_json_path: Path,
    evidence_dir: Path,
    bundles_json: Path | None,
) -> None:
    """Publish operator-attested prediction bundles to S3 and write upload evidence.

    This is a transport-only operator command expected to run after Phase
    finalization. It does not itself check the Phase lifecycle state; the
    operator is trusted to invoke it post-finalization.
    """
    from bspp.orchestration.contract.prediction_bundle import prediction_archive_bundle_from_mapping
    from bspp.orchestration.runtime.folding.seam_transport import (
        SeamTransportError,
        publish_prediction_bundles_to_s3,
    )

    try:
        payload = json.loads(input_json_path.read_text())
        bundles_source = json.loads(bundles_json.read_text()) if bundles_json is not None else payload["bundles"]
        bundles = tuple(prediction_archive_bundle_from_mapping(item) for item in bundles_source)
        local_paths = tuple(payload["local_paths"])
        s3_prefix = payload["s3_prefix"]
        phase_run_id = payload.get("phase_run_id")
        attempt_id = payload.get("attempt_id")
        all_evidence = publish_prediction_bundles_to_s3(
            bundles=bundles,
            local_paths=local_paths,
            s3_prefix=s3_prefix,
            phase_run_id=phase_run_id,
            attempt_id=attempt_id,
        )
        evidence_dir.mkdir(parents=True, exist_ok=True)
        flattened = [ev.to_mapping() for ev in all_evidence]
        _write_evidence_atomic(evidence_dir / "prediction-bundle-upload-evidence.json", flattened)
    except (SeamTransportError, OSError, TypeError, ValueError, KeyError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        json.dumps(
            {"bundle_count": len(flattened), "operator_attested": True},
            sort_keys=True,
        ),
        nl=False,
    )


@phase_publish_group.command("derive-seam-parquets")
@click.option(
    "--input-json",
    "input_json_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="JSON file containing the seam-derivation inputs.",
)
@click.option(
    "--evidence-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory to receive seam-derivation-result.json.",
)
def phase_derive_seam_parquets_cmd(input_json_path: Path, evidence_dir: Path) -> None:
    """Derive master + tracking parquets from a completed folding run.

    Pure local transform: reads already-fetched evidence files and writes the
    two parquets the postprocessing Phase Plan pins. The control command
    validates the Phase Run and writes the input JSON before delegating here,
    mirroring the publish-folding transport discipline.
    """
    from bspp.orchestration.runtime.folding.seam_derivation import (
        SeamDerivationError,
        derive_seam_parquets,
    )

    try:
        payload = json.loads(input_json_path.read_text())
        result = derive_seam_parquets(
            index_path=Path(payload["index_path"]),
            evidence_path=Path(payload["evidence_path"]),
            master_output=Path(payload["master_output"]),
            tracking_output=Path(payload["tracking_output"]),
            s3_output_prefix=payload["s3_output_prefix"],
            source_run=payload["source_run"],
            archive_name=payload["archive_name"],
            phase_run_id=payload.get("phase_run_id"),
            gcs_destination_prefix=payload.get("gcs_destination_prefix"),
            force=bool(payload.get("force", False)),
        )
        evidence_dir.mkdir(parents=True, exist_ok=True)
        _write_evidence_atomic(
            evidence_dir / "seam-derivation-result.json",
            {
                "master_path": str(result.master_path),
                "tracking_path": str(result.tracking_path),
                "row_count": result.row_count,
            },
        )
    except (SeamDerivationError, OSError, TypeError, ValueError, KeyError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        json.dumps(
            {"master": str(result.master_path), "tracking": str(result.tracking_path), "rows": result.row_count},
            sort_keys=True,
        ),
        nl=False,
    )


@cli.group("folding")
def folding_group() -> None:
    """Job-local folding Runtime Action execution."""


@folding_group.command("execute-action")
@click.option(
    "--phase-runspec",
    "phase_runspec_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Immutable folding Phase RunSpec JSON/YAML.",
)
@click.option("--action-id", required=True, help="Exact sole folding Runtime Action id.")
@click.option(
    "--action-evidence",
    "action_evidence_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Exclusive destination for structured action evidence.",
)
@click.option(
    "--handoff",
    "handoff_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Exclusive destination for the validated action handoff.",
)
def folding_execute_action_cmd(
    phase_runspec_path: Path,
    action_id: str,
    action_evidence_path: Path,
    handoff_path: Path,
) -> None:
    """Execute one exact folding Runtime Action without lifecycle or scheduler ownership."""
    from bspp.orchestration.runtime.folding.executor import (
        PRODUCTION_EXECUTOR_DEPS,
        FoldingExecutorError,
        run_execute_action,
    )

    try:
        handoff = run_execute_action(
            phase_runspec_path=phase_runspec_path,
            action_id=action_id,
            action_evidence_path=action_evidence_path,
            handoff_path=handoff_path,
            deps=PRODUCTION_EXECUTOR_DEPS,
        )
    except (FoldingExecutorError, OSError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(handoff, indent=2, sort_keys=True))


@folding_group.command("legacy-msa-import")
@click.option(
    "--handoff-root",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Legacy preprocessing MSA handoff directory (four-file layout).",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory for the enriched handoff records (must be empty or absent).",
)
@click.option(
    "--lz4",
    "lz4_executable",
    default="lz4",
    show_default=True,
    help="lz4 executable used for bundle/tar identity verification.",
)
def folding_legacy_msa_import_cmd(handoff_root: Path, output_dir: Path, lz4_executable: str) -> None:
    """Verify a legacy MSA handoff and publish an enriched length-bearing handoff."""
    from bspp.orchestration.runtime.folding.legacy_msa_import import (
        LegacyMsaImportError,
        run_legacy_msa_import,
    )

    try:
        result = run_legacy_msa_import(
            handoff_root,
            output_dir,
            lz4_argv=(lz4_executable, "-d", "-c"),
        )
    except (LegacyMsaImportError, OSError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        json.dumps(
            {
                "artifact_set_id": result.artifact_set_id,
                "artifact_location_id": result.artifact_location_id,
                "member_lengths": list(result.member_lengths),
            },
            indent=2,
            sort_keys=True,
        )
    )


_CLI_TEMPLATE = cli


def _build_cli(coordinator: ExecutionCoordinator) -> click.Group:
    """Build the application with one fixed preprocessing coordinator."""
    application = copy.deepcopy(_CLI_TEMPLATE)
    preprocessing = application.commands["preprocessing"]
    if not isinstance(preprocessing, click.Group):
        raise AssertionError("preprocessing command must remain a Click group")
    preprocessing.commands["place-database"].callback = _place_database_callback(coordinator)
    preprocessing.commands["execute-chunk"].callback = _execute_chunk_callback(coordinator)
    return application


cli = _build_cli(PRODUCTION_EXECUTION_COORDINATOR)


def main() -> None:
    """Dispatch the runtime CLI with the autorequeue transport-failure boundary.

    Every rendered postprocessing action body already enters Runtime through
    the CLI dispatch, so this is the single process boundary that can turn an
    audited group-A :class:`TransportFailure` into the reserved exit 85 after
    writing its create-once restart classification record.
    """
    from bspp.orchestration.runtime.postprocessing.autorequeue_boundary import (
        handle_autorequeue_transport_failure,
    )
    from bspp.orchestration.runtime.postprocessing.failure_adapter import TransportFailure

    try:
        cli()
    except TransportFailure as failure:
        handle_autorequeue_transport_failure(failure)
        raise


if __name__ == "__main__":
    main()
