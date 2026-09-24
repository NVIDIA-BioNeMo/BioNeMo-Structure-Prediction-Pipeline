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

"""Execute one exact preprocessing Runtime Action inside the Execution Runtime."""

from __future__ import annotations

import errno
import hashlib
import os
import shutil
import subprocess
import tarfile
import time
import uuid
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO, Never, Protocol

import yaml

from bspp.orchestration.contract.database_placement_result import (
    DatabasePostScienceEvidence,
    DatabasePostScienceObservationFailure,
)
from bspp.orchestration.contract.database_replica_lease import (
    DatabaseReplicaLeaseEvidence,
    PreprocessingStagedDatabasePlacementEvidence,
)
from bspp.orchestration.contract.phase import PhaseRunSpec, PreprocessingRuntimeAction, phase_runspec_from_mapping
from bspp.orchestration.contract.phase_carry_forward import (
    AttemptCarryForwardAdoptionEvidence,
    AttemptCarryForwardRecord,
)
from bspp.orchestration.contract.preprocessing_action import (
    PreprocessingChunkActionEvidence,
    PreprocessingCommandDisposition,
    PreprocessingCommandKind,
    PreprocessingCommandOutcome,
    PreprocessingDatabasePlacementCommandFailureEvidence,
    PreprocessingOutputHash,
    PreprocessingOutputRole,
    PreprocessingRawSearchEvidence,
    preprocessing_command_digest,
)
from bspp.orchestration.contract.preprocessing_runtime import PREPROCESSING_ADAPTER_VERSION
from bspp.orchestration.contract.preprocessing_state import (
    PreprocessingArchiveEvidence,
    PreprocessingPairedEvidence,
)
from bspp.orchestration.runtime.preprocessing._action_evidence_io import (
    load_preprocessing_action_evidence,
    publish_preprocessing_action_evidence,
)
from bspp.orchestration.runtime.preprocessing._database_action_placement import (
    ActionDatabasePlacementEvidence,
    DatabaseActionPlacementError,
    DirectActionDatabasePlacement,
    StagedActionDatabasePlacement,
    database_action_placement_evidence,
    lease_failure_evidence,
    reconcile_action_database_placement,
    resolve_action_database_placement,
)
from bspp.orchestration.runtime.preprocessing._database_placement_errors import DatabasePlacementError
from bspp.orchestration.runtime.preprocessing._database_placement_paths import (
    PRODUCTION_DATABASE_PLACEMENT_PATHS,
    DatabasePlacementPaths,
)
from bspp.orchestration.runtime.preprocessing._database_replica_lease import (
    DatabaseReplicaLeasePaths,
    HeldDatabaseReplicaLease,
    _database_replica_lease,
)
from bspp.orchestration.runtime.preprocessing.carry_forward import (
    adopt_carried_a3ms,
    prepare_carried_execution,
    reconcile_carry_forward_adoption,
)
from bspp.orchestration.runtime.preprocessing.content_validation import (
    normalize_preprocessing_tar_member,
    validate_preprocessing_a3m_bytes,
    validate_preprocessing_tar_headers,
)
from bspp.orchestration.runtime.preprocessing.database_placement import _observe_direct_database_source
from bspp.orchestration.runtime.preprocessing.raw_search import (
    PreprocessingRawSearchError,
    publish_validated_named_a3ms,
    reconcile_raw_search_evidence,
    searched_preprocessing_records,
    validate_raw_search_closure,
)
from bspp.orchestration.runtime.preprocessing.state import parse_preprocessing_record_a3m_members

GPUSERVER_WARMUP_SECONDS = 60.0
GPUSERVER_SHUTDOWN_TIMEOUT_SECONDS = 5.0
RECORD_LS_SCRIPT = 'exec /usr/bin/ls -lh "$1"/*.a3m'

Clock = Callable[[], datetime]
Sleeper = Callable[[float], None]


class ScientificKernelProcess(Protocol):
    """Process operations owned by the scientific-kernel lifecycle."""

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...


ScientificKernelStart = Callable[
    [tuple[str, ...], Mapping[str, str], BinaryIO],
    ScientificKernelProcess,
]
ScientificKernelShutdown = Callable[
    [ScientificKernelProcess, PreprocessingRuntimeAction],
    PreprocessingCommandOutcome,
]


@dataclass(frozen=True)
class ScientificKernelLauncher:
    """Typed scientific-kernel process boundary for private composition."""

    start: ScientificKernelStart
    shutdown: ScientificKernelShutdown


@dataclass(frozen=True)
class PreprocessingActionEvidenceStore:
    """Typed action-evidence persistence boundary for private composition."""

    load: Callable[[Path], PreprocessingChunkActionEvidence]
    publish: Callable[[PreprocessingChunkActionEvidence, Path], None]


@dataclass(frozen=True)
class ExecutionCoordinator:
    """Own the execution state machine under one immutable site composition."""

    database_placement_paths: DatabasePlacementPaths
    clock: Clock
    sleeper: Sleeper
    scientific_kernel_launcher: ScientificKernelLauncher
    evidence_store: PreprocessingActionEvidenceStore

    def execute(
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
        clock: Clock | None = None,
        sleeper: Sleeper | None = None,
    ) -> PreprocessingChunkActionEvidence:
        """Execute with fixed production authority and compatible clock overrides."""
        return _execute_preprocessing_chunk_action(
            runspec,
            action_id=action_id,
            evidence_path=evidence_path,
            database_placement_result_path=database_placement_result_path,
            placement_process_status=placement_process_status,
            database_placement_failure_path=database_placement_failure_path,
            carry_forward_record=carry_forward_record,
            phase_submission_id=phase_submission_id,
            clock=self.clock if clock is None else clock,
            sleeper=self.sleeper if sleeper is None else sleeper,
            database_placement_paths=self.database_placement_paths,
            scientific_kernel_launcher=self.scientific_kernel_launcher,
            evidence_store=self.evidence_store,
        )


def _start_scientific_kernel(
    argv: tuple[str, ...],
    environment: Mapping[str, str],
    stdout: BinaryIO,
) -> ScientificKernelProcess:
    return subprocess.Popen(
        argv,
        env=environment,
        stdout=stdout,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def _shutdown_scientific_kernel(
    server: ScientificKernelProcess,
    action: PreprocessingRuntimeAction,
) -> PreprocessingCommandOutcome:
    return _shutdown_gpuserver(server, action)


PRODUCTION_SCIENTIFIC_KERNEL_LAUNCHER = ScientificKernelLauncher(
    start=_start_scientific_kernel,
    shutdown=_shutdown_scientific_kernel,
)
PRODUCTION_PREPROCESSING_ACTION_EVIDENCE_STORE = PreprocessingActionEvidenceStore(
    load=load_preprocessing_action_evidence,
    publish=publish_preprocessing_action_evidence,
)


class PreprocessingExecutionError(RuntimeError):
    """One declared preprocessing action failed closed."""


def load_preprocessing_phase_runspec(path: Path) -> PhaseRunSpec:
    """Strict-load one JSON/YAML preprocessing Phase RunSpec."""
    try:
        payload = yaml.safe_load(path.read_bytes())
    except yaml.YAMLError as exc:
        msg = f"Invalid preprocessing Phase RunSpec YAML/JSON in {path}: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(payload, Mapping):
        msg = f"Expected preprocessing Phase RunSpec mapping in {path}"
        raise TypeError(msg)
    return phase_runspec_from_mapping(payload)


def execute_preprocessing_chunk_action(
    runspec: PhaseRunSpec,
    *,
    action_id: str,
    evidence_path: Path,
    database_placement_result_path: Path,
    placement_process_status: int,
    database_placement_failure_path: Path | None = None,
    carry_forward_record: AttemptCarryForwardRecord | None = None,
    phase_submission_id: str | None = None,
    clock: Clock | None = None,
    sleeper: Sleeper | None = None,
) -> PreprocessingChunkActionEvidence:
    """Execute the sole declared chunk action and write additive evidence."""
    return PRODUCTION_EXECUTION_COORDINATOR.execute(
        runspec,
        action_id=action_id,
        evidence_path=evidence_path,
        database_placement_result_path=database_placement_result_path,
        placement_process_status=placement_process_status,
        database_placement_failure_path=database_placement_failure_path,
        carry_forward_record=carry_forward_record,
        phase_submission_id=phase_submission_id,
        clock=clock,
        sleeper=sleeper,
    )


def _execute_preprocessing_chunk_action(
    runspec: PhaseRunSpec,
    *,
    action_id: str,
    evidence_path: Path,
    database_placement_result_path: Path,
    placement_process_status: int,
    database_placement_failure_path: Path | None = None,
    carry_forward_record: AttemptCarryForwardRecord | None = None,
    phase_submission_id: str | None = None,
    clock: Clock | None = None,
    sleeper: Sleeper | None = None,
    database_placement_paths: DatabasePlacementPaths,
    scientific_kernel_launcher: ScientificKernelLauncher,
    evidence_store: PreprocessingActionEvidenceStore,
) -> PreprocessingChunkActionEvidence:
    """Execute one action against an explicitly composed physical site."""
    now = clock or _utc_now
    started_at = _format_timestamp(now())
    action = runspec.payload.actions[0]
    foreground_outcomes: list[PreprocessingCommandOutcome] = []
    server_outcome: PreprocessingCommandOutcome | None = None
    server: ScientificKernelProcess | None = None
    failure: Exception | None = None
    adoption: AttemptCarryForwardAdoptionEvidence | None = None
    raw_search_evidence: PreprocessingRawSearchEvidence | None = None
    search_members: tuple[str, ...] | None = None
    post_science_observation: DatabasePostScienceEvidence | None = None
    lease_outcome: DatabaseReplicaLeaseEvidence | None = None
    held_lease: HeldDatabaseReplicaLease | None = None
    lease_stack = ExitStack()
    lease_released = False
    science_started = False
    reject_output_acceptance = False
    owns_scratch_log = False
    durable_log_published = False

    try:
        placement = resolve_action_database_placement(
            runspec,
            action_id=action_id,
            result_path=database_placement_result_path,
            failure_path=database_placement_failure_path,
            placement_process_status=placement_process_status,
        )
    except DatabaseActionPlacementError as exc:
        raise PreprocessingExecutionError(str(exc)) from exc
    if isinstance(placement, PreprocessingDatabasePlacementCommandFailureEvidence):
        placement_error = f"{placement.classification} (status={placement.placement_process_status}): {placement.error}"
        _publish_pre_science_placement_failure_action(
            runspec,
            action,
            placement=placement,
            evidence_path=evidence_path,
            started_at=started_at,
            finished_at=_format_timestamp(now()),
            error=placement_error,
            carry_forward_record=carry_forward_record,
            placement_process_status=placement_process_status,
            evidence_store=evidence_store,
        )
    direct_result = placement.result if isinstance(placement, DirectActionDatabasePlacement) else None
    direct_failure = placement.failure if isinstance(placement, DirectActionDatabasePlacement) else None
    staged_result = placement.result if isinstance(placement, StagedActionDatabasePlacement) else None
    staged_failure = placement.failure if isinstance(placement, StagedActionDatabasePlacement) else None
    placement_failure = direct_failure if direct_failure is not None else staged_failure
    if placement_failure is not None:
        placement_evidence = database_action_placement_evidence(
            placement,
            lease_outcome=None,
            science_started=False,
            post_science_observation=None,
        )
        _publish_pre_science_placement_failure_action(
            runspec,
            action,
            placement=placement_evidence,
            evidence_path=evidence_path,
            started_at=started_at,
            finished_at=_format_timestamp(now()),
            error=(
                f"{placement_failure.classification} (status={placement_process_status}): {placement_failure.error}"
            ),
            carry_forward_record=carry_forward_record,
            placement_process_status=placement_process_status,
            evidence_store=evidence_store,
        )

    try:
        if action_id != action.action_id:
            msg = f"action id {action_id!r} does not match the sole declared action {action.action_id!r}"
            raise PreprocessingExecutionError(msg)
        if direct_result is None and staged_result is None:
            raise PreprocessingExecutionError("Database Placement produced no successful Result")
        if (carry_forward_record is None) != (phase_submission_id is None):
            raise PreprocessingExecutionError("carry-forward record and Phase Submission id must be present together")
        if carry_forward_record is not None:
            assert phase_submission_id is not None
            search_members = prepare_carried_execution(
                runspec,
                carry_forward_record,
                phase_submission_id=phase_submission_id,
            )
        elif runspec.carry_forward is not None:
            raise PreprocessingExecutionError("carried RunSpec requires its staged carry record and submission id")
        _preflight(
            runspec,
            action,
            evidence_path=evidence_path,
            carry_forward_record=carry_forward_record,
        )
        payload = action.payload
        raw_search_output = Path(payload.evidence.raw_search_output_directory)
        scratch_output = Path(payload.evidence.scratch_output_directory)
        scratch_logs = Path(payload.evidence.scratch_log_directory)
        raw_search_output.mkdir(parents=True)
        scratch_output.mkdir(parents=True)
        scratch_logs.mkdir(parents=True)

        if staged_result is not None:
            try:
                held_lease = lease_stack.enter_context(
                    _database_replica_lease(
                        runspec,
                        staged_result,
                        paths=DatabaseReplicaLeasePaths(
                            selected_root=database_placement_paths.selected_root,
                            lease=database_placement_paths.lease,
                        ),
                    )
                )
            except DatabasePlacementError as exc:
                lease_outcome = lease_failure_evidence(staged_result, exc)
                raise PreprocessingExecutionError(f"Database Replica Lease failed before science: {exc}") from exc

        gpuserver_environment = _command_environment(payload.gpuserver_environment)
        try:
            with Path(payload.evidence.scratch_log_path).open("xb") as log_handle:
                owns_scratch_log = True
                server = scientific_kernel_launcher.start(
                    payload.gpuserver_argv,
                    gpuserver_environment,
                    log_handle,
                )
                science_started = True
                (sleeper or time.sleep)(GPUSERVER_WARMUP_SECONDS)
                observed_return_code = server.poll()
                if observed_return_code is not None:
                    server_outcome = _command_outcome(
                        "gpuserver",
                        payload.gpuserver_argv,
                        payload.gpuserver_environment,
                        "exited-unexpectedly",
                        observed_return_code,
                    )
                    msg = f"gpuserver exited before search with return code {observed_return_code}"
                    raise PreprocessingExecutionError(msg)

                _run_foreground(
                    "search",
                    payload.search_argv,
                    payload.search_environment,
                    foreground_outcomes,
                    stdout=log_handle,
                )
                log_handle.flush()
                os.fsync(log_handle.fileno())
                try:
                    server_outcome = scientific_kernel_launcher.shutdown(server, action)
                except Exception as cleanup_error:
                    server_outcome = _command_outcome(
                        "gpuserver",
                        payload.gpuserver_argv,
                        payload.gpuserver_environment,
                        "cleanup-failed",
                        None,
                    )
                    msg = f"gpuserver cleanup failed before publication: {cleanup_error}"
                    raise PreprocessingExecutionError(msg) from cleanup_error
                log_handle.flush()
                os.fsync(log_handle.fileno())
                if server_outcome.disposition == "exited-unexpectedly":
                    msg = "gpuserver exited before adapter-owned shutdown"
                    raise PreprocessingExecutionError(msg)
        except OSError as exc:
            if server is None:
                server_outcome = _command_outcome(
                    "gpuserver",
                    payload.gpuserver_argv,
                    payload.gpuserver_environment,
                    "failed-to-start",
                    None,
                )
            msg = f"failed to start preprocessing Scientific Kernel: {exc}"
            raise PreprocessingExecutionError(msg) from exc

        if staged_result is not None:
            assert held_lease is not None
            lease_outcome = held_lease.terminal_evidence(
                kernel_started=True,
                kernel_terminal=server.poll() is not None,
            )
            lease_stack.close()
            lease_released = True
        else:
            assert direct_result is not None
            post_science_observation = _observe_direct_database_source(
                runspec,
                direct_result,
                paths=database_placement_paths,
            )
            if isinstance(post_science_observation, DatabasePostScienceObservationFailure):
                reject_output_acceptance = True
                raise PreprocessingExecutionError(
                    f"post-science Database Source observation failed: {post_science_observation.error}"
                )
            if not post_science_observation.matches(direct_result.pre_science_observation):
                reject_output_acceptance = True
                raise PreprocessingExecutionError("direct Database Source drifted after the Scientific Kernel")

        _require_nonempty_regular_file(Path(payload.evidence.scratch_log_path), "scratch log")
        raw_search_evidence = validate_raw_search_closure(
            runspec,
            action,
            carry_forward_record=carry_forward_record,
        )
        publish_validated_named_a3ms(raw_search_evidence, staging_directory=scratch_output)
        _validate_declared_a3ms(action, expected_members=search_members)
        if carry_forward_record is not None:
            assert phase_submission_id is not None
            adoption = adopt_carried_a3ms(
                runspec,
                carry_forward_record,
                phase_submission_id=phase_submission_id,
                adopted_at=_format_timestamp(now()),
            )
        _validate_declared_a3ms(action)
        record_argv = _record_ls_argv(action)
        with Path(payload.evidence.scratch_record_path).open("xb") as record_handle:
            _run_foreground(
                "record-ls",
                record_argv,
                (),
                foreground_outcomes,
                stdout=record_handle,
            )
            record_handle.flush()
            os.fsync(record_handle.fileno())
        _require_nonempty_regular_file(Path(payload.evidence.scratch_record_path), "scratch record")

        _publish_copy_exclusive(Path(payload.evidence.scratch_log_path), Path(payload.evidence.durable_log_path))
        durable_log_published = True
        _publish_copy_exclusive(
            Path(payload.evidence.scratch_record_path),
            Path(payload.evidence.durable_record_path),
        )

        _run_foreground(
            "tar",
            payload.package.tar_argv,
            (),
            foreground_outcomes,
            stdout=subprocess.DEVNULL,
        )
        scratch_tar = Path(payload.package.scratch_tar_path)
        _require_nonempty_regular_file(scratch_tar, "scratch tar")
        _inspect_exact_tar(scratch_tar, action)

        _run_foreground(
            "lz4",
            payload.package.lz4_argv,
            (),
            foreground_outcomes,
            stdout=subprocess.DEVNULL,
        )
        scratch_lz4 = Path(payload.package.scratch_lz4_path)
        _require_nonempty_regular_file(scratch_lz4, "scratch tar.lz4")

        _move_exclusive(scratch_tar, Path(payload.package.durable_tar_path))
        _move_exclusive(scratch_lz4, Path(payload.package.durable_lz4_path))
        _validate_declared_inputs(
            runspec,
            action,
            carry_forward_record=carry_forward_record,
        )
        _move_exclusive(
            Path(payload.package.completed_input_source_path),
            Path(payload.package.completed_input_path),
        )
    except Exception as exc:  # evidence must retain every fail-closed outcome
        failure = exc
    finally:
        if server is not None and server_outcome is None:
            try:
                server_outcome = scientific_kernel_launcher.shutdown(server, action)
                if server_outcome.disposition == "exited-unexpectedly" and failure is None:
                    failure = PreprocessingExecutionError("gpuserver exited before adapter-owned shutdown")
            except Exception as cleanup_error:
                server_outcome = _command_outcome(
                    "gpuserver",
                    action.payload.gpuserver_argv,
                    action.payload.gpuserver_environment,
                    "cleanup-failed",
                    None,
                )
                original = "" if failure is None else f"{failure}; "
                failure = PreprocessingExecutionError(f"{original}gpuserver cleanup failed: {cleanup_error}")
        if staged_result is not None and server is not None:
            terminal_cleanup_errors = _hold_staged_lease_until_gpuserver_terminal(server)
            if terminal_cleanup_errors:
                original = "" if failure is None else f"{failure}; "
                details = "; ".join(terminal_cleanup_errors)
                failure = PreprocessingExecutionError(f"{original}staged terminal cleanup retried: {details}")
        if staged_result is not None and held_lease is not None and lease_outcome is None:
            try:
                lease_outcome = held_lease.terminal_evidence(
                    kernel_started=server is not None,
                    kernel_terminal=server is not None and server.poll() is not None,
                )
            except DatabasePlacementError as lease_error:
                original = "" if failure is None else f"{failure}; "
                failure = PreprocessingExecutionError(f"{original}terminal lease evidence failed: {lease_error}")
        if not lease_released:
            lease_stack.close()
            lease_released = True
        if science_started and direct_result is not None and post_science_observation is None:
            post_science_observation = _observe_direct_database_source(
                runspec,
                direct_result,
                paths=database_placement_paths,
            )
            if isinstance(post_science_observation, DatabasePostScienceObservationFailure):
                reject_output_acceptance = True
                observation_error = post_science_observation.error
                failure = PreprocessingExecutionError(
                    f"post-science Database Source observation failed: {observation_error}"
                    if failure is None
                    else f"{failure}; post-science Database Source observation failed: {observation_error}"
                )
            elif not post_science_observation.matches(direct_result.pre_science_observation):
                reject_output_acceptance = True
                drift = "direct Database Source drifted after the Scientific Kernel"
                failure = PreprocessingExecutionError(drift if failure is None else f"{failure}; {drift}")

    if failure is not None and owns_scratch_log and not durable_log_published and not reject_output_acceptance:
        try:
            if server is not None and server.poll() is None:
                raise PreprocessingExecutionError("gpuserver is still running; diagnostic log is not final")
            scratch_log = Path(action.payload.evidence.scratch_log_path)
            if not _is_regular_file(scratch_log):
                raise PreprocessingExecutionError("owned scratch log is no longer a regular non-symlink file")
            _publish_copy_exclusive(scratch_log, Path(action.payload.evidence.durable_log_path))
        except Exception as log_error:
            failure = PreprocessingExecutionError(
                f"{failure}; failed to retain preprocessing diagnostic log: {log_error}"
            )

    finished_at = _format_timestamp(now())
    command_outcomes = (() if server_outcome is None else (server_outcome,)) + tuple(foreground_outcomes)
    placement_evidence = database_action_placement_evidence(
        placement,
        lease_outcome=lease_outcome,
        science_started=science_started,
        post_science_observation=post_science_observation,
    )
    try:
        evidence = _build_evidence(
            runspec,
            action,
            action_id=action.action_id,
            started_at=started_at,
            finished_at=finished_at,
            command_outcomes=command_outcomes,
            failure=failure,
            carry_forward_adoption=adoption,
            raw_search_evidence=raw_search_evidence,
            database_placement=placement_evidence,
            placement_process_status=placement_process_status,
            accept_outputs=direct_failure is None and staged_failure is None and not reject_output_acceptance,
        )
    except (OSError, ValueError) as evidence_error:
        if failure is not None:
            raise PreprocessingExecutionError(
                f"{failure}; failed to construct action evidence: {evidence_error}"
            ) from evidence_error
        failure = PreprocessingExecutionError(f"failed to construct successful action evidence: {evidence_error}")
        evidence = _build_evidence(
            runspec,
            action,
            action_id=action.action_id,
            started_at=started_at,
            finished_at=finished_at,
            command_outcomes=command_outcomes,
            failure=failure,
            carry_forward_adoption=adoption,
            raw_search_evidence=raw_search_evidence,
            database_placement=placement_evidence,
            placement_process_status=placement_process_status,
            accept_outputs=direct_failure is None and staged_failure is None and not reject_output_acceptance,
        )
    if failure is None:
        try:
            reconcile_preprocessing_chunk_action_evidence(
                runspec,
                evidence,
                carry_forward_record=carry_forward_record,
            )
        except (OSError, ValueError) as reconciliation_error:
            failure = PreprocessingExecutionError(f"action evidence reconciliation failed: {reconciliation_error}")
            evidence = _build_evidence(
                runspec,
                action,
                action_id=action.action_id,
                started_at=started_at,
                finished_at=finished_at,
                command_outcomes=command_outcomes,
                failure=failure,
                carry_forward_adoption=adoption,
                raw_search_evidence=raw_search_evidence,
                database_placement=placement_evidence,
                placement_process_status=placement_process_status,
                accept_outputs=direct_failure is None and staged_failure is None and not reject_output_acceptance,
            )
    if evidence_path.exists():
        if failure is None:
            failure = PreprocessingExecutionError(f"refusing to replace existing evidence: {evidence_path}")
        raise PreprocessingExecutionError(str(failure)) from failure
    evidence_store.publish(evidence, evidence_path)
    if failure is not None:
        if isinstance(failure, PreprocessingExecutionError):
            raise failure
        raise PreprocessingExecutionError(str(failure)) from failure
    return evidence


def _publish_pre_science_placement_failure_action(
    runspec: PhaseRunSpec,
    action: PreprocessingRuntimeAction,
    *,
    placement: ActionDatabasePlacementEvidence,
    evidence_path: Path,
    started_at: str,
    finished_at: str,
    error: str,
    carry_forward_record: AttemptCarryForwardRecord | None,
    placement_process_status: int,
    evidence_store: PreprocessingActionEvidenceStore,
) -> Never:
    """Publish, reload, and reconcile a payload-free failed action before science."""
    bounded_error = (error or "Database Placement failed before science")[:2048]
    evidence = PreprocessingChunkActionEvidence(
        adapter_version=PREPROCESSING_ADAPTER_VERSION,
        phase_run_id=runspec.phase_run_id,
        attempt_id=runspec.attempt_id,
        phase_runspec_digest=runspec.digest,
        action_id=action.action_id,
        placement_process_status=placement_process_status,
        chunk_name=action.payload.chunk_name,
        started_at=started_at,
        finished_at=finished_at,
        outcome="failed",
        command_outcomes=(),
        paired_evidence=PreprocessingPairedEvidence(
            chunk_name=action.payload.chunk_name,
            durable_record_path=action.payload.evidence.durable_record_path,
            durable_log_path=action.payload.evidence.durable_log_path,
            record_lines=None,
            log_lines=None,
        ),
        archive_evidence=PreprocessingArchiveEvidence(
            chunk_name=action.payload.chunk_name,
            durable_tar_path=action.payload.package.durable_tar_path,
            durable_lz4_path=action.payload.package.durable_lz4_path,
            tar_size_bytes=None,
            lz4_size_bytes=None,
            tar_members=None,
        ),
        output_hashes=(),
        error=bounded_error,
        database_placement=placement,
        raw_search_evidence=None,
        carry_forward_adoption=None,
    )
    try:
        reconcile_preprocessing_chunk_action_evidence(
            runspec,
            evidence,
            carry_forward_record=carry_forward_record,
        )
        if evidence_path.exists() or evidence_path.is_symlink():
            raise PreprocessingExecutionError(f"refusing to replace existing evidence: {evidence_path}")
        evidence_store.publish(evidence, evidence_path)
        reloaded = evidence_store.load(evidence_path)
        if reloaded != evidence:
            raise PreprocessingExecutionError("reloaded failed action evidence differs from published authority")
        reconcile_preprocessing_chunk_action_evidence(
            runspec,
            reloaded,
            carry_forward_record=carry_forward_record,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise PreprocessingExecutionError(f"failed to publish reconciled placement failure action: {exc}") from exc
    raise PreprocessingExecutionError(bounded_error)


def reconcile_preprocessing_chunk_action_evidence(
    runspec: PhaseRunSpec,
    evidence: PreprocessingChunkActionEvidence,
    *,
    carry_forward_record: AttemptCarryForwardRecord | None = None,
) -> None:
    """Cross-check additive action evidence against its RunSpec and current files."""
    _reconcile_preprocessing_chunk_action_evidence(runspec, evidence, verify_scratch_a3ms=True)
    if _is_pre_science_placement_failure(evidence):
        if evidence.carry_forward_adoption is not None:
            raise ValueError("pre-science placement failure cannot claim carry-forward adoption")
        return
    reconcile_carry_forward_adoption(runspec, evidence.carry_forward_adoption, carry_forward_record)


def reconcile_preprocessing_chunk_action_evidence_for_finalization(
    runspec: PhaseRunSpec,
    evidence: PreprocessingChunkActionEvidence,
) -> None:
    """Reconcile durable outputs while retaining, but not opening, scratch attestations."""
    if evidence.outcome != "succeeded":
        raise ValueError("preprocessing finalization requires successful action evidence")
    _reconcile_preprocessing_chunk_action_evidence(runspec, evidence, verify_scratch_a3ms=False)
    action = runspec.payload.actions[0]
    completed_input = Path(action.payload.package.completed_input_path)
    if completed_input.read_bytes() != _expected_chunk_bytes(runspec, action):
        raise ValueError("completed preprocessing input does not match the exact RunSpec chunk")


def _reconcile_preprocessing_chunk_action_evidence(
    runspec: PhaseRunSpec,
    evidence: PreprocessingChunkActionEvidence,
    *,
    verify_scratch_a3ms: bool,
) -> None:
    action = runspec.payload.actions[0]
    payload = action.payload
    if (
        evidence.adapter_version != PREPROCESSING_ADAPTER_VERSION
        or evidence.phase_run_id != runspec.phase_run_id
        or evidence.attempt_id != runspec.attempt_id
        or evidence.phase_runspec_digest != runspec.digest
        or evidence.action_id != action.action_id
        or evidence.chunk_name != payload.chunk_name
    ):
        msg = "preprocessing action evidence identity does not match the sole Phase RunSpec action"
        raise ValueError(msg)
    placement = evidence.database_placement
    reconcile_action_database_placement(
        runspec,
        placement,
        action_id=action.action_id,
        require_success=evidence.outcome == "succeeded",
    )
    if evidence.outcome == "succeeded" and evidence.raw_search_evidence is None:
        raise ValueError("successful adapter-v3 action evidence requires raw-search evidence")
    if runspec.carry_forward is None:
        if evidence.carry_forward_adoption is not None:
            raise ValueError("no-carry RunSpec cannot retain carry-forward adoption evidence")
    else:
        adoption = evidence.carry_forward_adoption
        if adoption is None and _is_pre_science_placement_failure(evidence):
            pass
        elif (
            adoption is None
            or adoption.attempt_carry_forward_id != runspec.carry_forward.attempt_carry_forward_id
            or adoption.attempt_carry_forward_digest != runspec.carry_forward.digest
        ):
            raise ValueError("carried RunSpec requires matching passing adoption evidence")
    paired = evidence.paired_evidence
    archive = evidence.archive_evidence
    if (
        paired.durable_log_path != payload.evidence.durable_log_path
        or paired.durable_record_path != payload.evidence.durable_record_path
        or archive.durable_tar_path != payload.package.durable_tar_path
        or archive.durable_lz4_path != payload.package.durable_lz4_path
    ):
        msg = "nested preprocessing evidence paths do not match the declared action"
        raise ValueError(msg)

    expected_commands = (
        ("gpuserver", payload.gpuserver_argv, payload.gpuserver_environment),
        ("search", payload.search_argv, payload.search_environment),
        ("record-ls", _record_ls_argv(action), ()),
        ("tar", payload.package.tar_argv, ()),
        ("lz4", payload.package.lz4_argv, ()),
    )
    for observed, (kind, argv, environment) in zip(evidence.command_outcomes, expected_commands, strict=False):
        if observed.command_kind != kind or observed.command_digest != preprocessing_command_digest(argv, environment):
            msg = f"{kind} command attestation does not match the declared action"
            raise ValueError(msg)

    if paired.record_lines is not None:
        record_members = parse_preprocessing_record_a3m_members(paired.record_lines)
        expected_members = tuple(item.member_name for item in payload.expected_a3ms)
        if len(record_members) != len(expected_members) or set(record_members) != set(expected_members):
            msg = "real record-ls membership does not match declared A3Ms"
            raise ValueError(msg)
        if tuple(Path(paired.durable_record_path).read_text().splitlines()) != paired.record_lines:
            msg = "durable record bytes do not match nested paired evidence"
            raise ValueError(msg)
    if (
        paired.log_lines is not None
        and tuple(Path(paired.durable_log_path).read_text().splitlines()) != paired.log_lines
    ):
        msg = "durable log bytes do not match nested paired evidence"
        raise ValueError(msg)

    if archive.tar_members is not None:
        observed_members = _normalized_file_members(archive.tar_members)
        expected_members = tuple(item.member_name for item in payload.expected_a3ms)
        if len(observed_members) != len(expected_members) or set(observed_members) != set(expected_members):
            msg = "archive evidence does not contain the exact declared A3Ms"
            raise ValueError(msg)

    expected_hash_paths = _expected_hash_paths(action)
    observed_hash_paths = {(item.role, item.member_name): item.path for item in evidence.output_hashes}
    if evidence.outcome == "succeeded":
        if observed_hash_paths != expected_hash_paths:
            msg = "successful output hashes do not cover the exact declared outputs"
            raise ValueError(msg)
    elif any(expected_hash_paths.get(key) != path for key, path in observed_hash_paths.items()):
        msg = "failed output hashes must be a subset of the exact declared outputs"
        raise ValueError(msg)
    for item in evidence.output_hashes:
        if verify_scratch_a3ms or item.role != "a3m":
            _verify_output_hash(item)
    if evidence.raw_search_evidence is not None:
        reconcile_raw_search_evidence(
            runspec,
            evidence.raw_search_evidence,
            evidence.output_hashes,
            carry_forward_adoption=evidence.carry_forward_adoption,
            verify_raw_files=verify_scratch_a3ms,
        )
    by_role: dict[str, PreprocessingOutputHash] = {
        item.role: item for item in evidence.output_hashes if item.role != "a3m"
    }
    for role, observed_size in (("tar", archive.tar_size_bytes), ("tar-lz4", archive.lz4_size_bytes)):
        hashed = by_role.get(role)
        if observed_size is not None and (hashed is None or hashed.size_bytes != observed_size):
            msg = f"{role} hash size does not match archive evidence"
            raise ValueError(msg)
    if evidence.outcome == "succeeded":
        actual_tar_members = _inspect_exact_tar(Path(payload.package.durable_tar_path), action)
        if actual_tar_members != archive.tar_members:
            msg = "durable tar members do not match nested archive evidence"
            raise ValueError(msg)


def _is_pre_science_placement_failure(evidence: PreprocessingChunkActionEvidence) -> bool:
    placement = evidence.database_placement
    return evidence.outcome == "failed" and (
        isinstance(placement, PreprocessingDatabasePlacementCommandFailureEvidence) or placement.failure is not None
    )


def _preflight(
    runspec: PhaseRunSpec,
    action: PreprocessingRuntimeAction,
    *,
    evidence_path: Path,
    carry_forward_record: AttemptCarryForwardRecord | None,
) -> None:
    payload = action.payload
    raw_directory = payload.evidence.raw_search_output_directory
    scratch_directory = payload.evidence.scratch_output_directory
    log_directory = payload.evidence.scratch_log_directory
    if (
        not raw_directory
        or payload.search_argv[5] != raw_directory
        or len({os.path.normpath(raw_directory), os.path.normpath(scratch_directory), os.path.normpath(log_directory)})
        != 3
    ):
        raise PreprocessingExecutionError("raw-search, staging, and log paths must be present and distinct")
    expected_glob = f"{payload.evidence.scratch_output_directory.rstrip('/')}/*.a3m"
    if payload.evidence.a3m_record_glob != expected_glob:
        msg = "declared A3M record glob must exactly match the scratch output directory"
        raise PreprocessingExecutionError(msg)
    destinations = (
        evidence_path,
        Path(raw_directory),
        Path(payload.evidence.scratch_output_directory),
        Path(payload.evidence.scratch_log_directory),
        Path(payload.evidence.durable_log_path),
        Path(payload.evidence.durable_record_path),
        Path(payload.package.durable_tar_path),
        Path(payload.package.durable_lz4_path),
        Path(payload.package.completed_input_path),
    )
    existing = tuple(str(path) for path in destinations if path.exists() or path.is_symlink())
    if existing:
        msg = f"refusing to reuse existing preprocessing action path(s): {', '.join(existing)}"
        raise PreprocessingExecutionError(msg)

    _validate_declared_inputs(runspec, action, carry_forward_record=carry_forward_record)
    try:
        searched_preprocessing_records(runspec, carry_forward_record)
    except PreprocessingRawSearchError as exc:
        raise PreprocessingExecutionError(str(exc)) from exc


def _validate_declared_inputs(
    runspec: PhaseRunSpec,
    action: PreprocessingRuntimeAction,
    *,
    carry_forward_record: AttemptCarryForwardRecord | None = None,
) -> None:
    payload = action.payload
    expected_input = _expected_chunk_bytes(runspec, action)
    search_input = Path(payload.search_argv[3])
    split_input = Path(payload.package.completed_input_source_path)
    expected_search = (
        _remaining_chunk_bytes(runspec, carry_forward_record) if carry_forward_record is not None else expected_input
    )
    for label, path, expected in (
        ("staged search input", search_input, expected_search),
        ("split input", split_input, expected_input),
    ):
        _require_nonempty_regular_file(path, label)
        if path.read_bytes() != expected:
            msg = f"{label} bytes do not match the exact declared chunk: {path}"
            raise PreprocessingExecutionError(msg)


def _remaining_chunk_bytes(runspec: PhaseRunSpec, record: AttemptCarryForwardRecord) -> bytes:
    records = {item.source_ordinal: item for item in runspec.payload.work_plan.input.records}
    return "".join(
        f">{records[ordinal].identity}\n{records[ordinal].sequence}\n" for ordinal in record.remaining_record_ordinals
    ).encode()


def _expected_chunk_bytes(runspec: PhaseRunSpec, action: PreprocessingRuntimeAction) -> bytes:
    chunk = runspec.payload.work_plan.chunks[0]
    records_by_ordinal = {item.source_ordinal: item for item in runspec.payload.work_plan.input.records}
    records = tuple(records_by_ordinal[ordinal] for ordinal in chunk.record_ordinals)
    expected_identity = tuple((item.source_ordinal, item.record_identity) for item in action.payload.expected_a3ms)
    observed_identity = tuple((item.source_ordinal, item.identity) for item in records)
    if observed_identity != expected_identity:
        msg = "declared action A3Ms do not match the work-plan source records"
        raise PreprocessingExecutionError(msg)
    return "".join(f">{item.identity}\n{item.sequence}\n" for item in records).encode()


def _validate_declared_a3ms(
    action: PreprocessingRuntimeAction,
    *,
    expected_members: tuple[str, ...] | None = None,
) -> None:
    scratch = Path(action.payload.evidence.scratch_output_directory)
    expected = expected_members or tuple(item.member_name for item in action.payload.expected_a3ms)
    entries = tuple(scratch.iterdir())
    observed = tuple(path.name for path in entries)
    if len(observed) != len(expected) or set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        msg = f"scratch A3M inventory mismatch; missing={missing}, extra={extra}"
        raise PreprocessingExecutionError(msg)
    for path in entries:
        _require_nonempty_regular_file(path, "declared A3M")
        _validate_a3m_text(path)


def _validate_a3m_text(path: Path) -> None:
    try:
        validate_preprocessing_a3m_bytes(path.read_bytes(), label=str(path))
    except ValueError as exc:
        raise PreprocessingExecutionError(str(exc)) from exc


def _record_ls_argv(action: PreprocessingRuntimeAction) -> tuple[str, ...]:
    scratch = action.payload.evidence.scratch_output_directory
    return ("/bin/bash", "-c", RECORD_LS_SCRIPT, "record-ls", scratch)


def _run_foreground(
    kind: PreprocessingCommandKind,
    argv: tuple[str, ...],
    environment: tuple[tuple[str, str], ...],
    outcomes: list[PreprocessingCommandOutcome],
    *,
    stdout: int | BinaryIO,
) -> None:
    try:
        result = subprocess.run(
            argv,
            env=_command_environment(environment),
            stdout=stdout,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except OSError as exc:
        outcomes.append(_command_outcome(kind, argv, environment, "failed-to-start", None))
        msg = f"{kind} command failed to start: {exc}"
        raise PreprocessingExecutionError(msg) from exc
    outcomes.append(_command_outcome(kind, argv, environment, "completed", result.returncode))
    if result.returncode != 0:
        msg = f"{kind} command failed with return code {result.returncode}"
        raise PreprocessingExecutionError(msg)


def _shutdown_gpuserver(
    server: ScientificKernelProcess,
    action: PreprocessingRuntimeAction,
) -> PreprocessingCommandOutcome:
    disposition: PreprocessingCommandDisposition
    return_code = server.poll()
    if return_code is not None:
        disposition = "exited-unexpectedly"
    else:
        server.terminate()
        try:
            return_code = server.wait(timeout=GPUSERVER_SHUTDOWN_TIMEOUT_SECONDS)
            disposition = "terminated-by-adapter"
        except subprocess.TimeoutExpired:
            server.kill()
            return_code = server.wait(timeout=GPUSERVER_SHUTDOWN_TIMEOUT_SECONDS)
            disposition = "killed-by-adapter"
    return _command_outcome(
        "gpuserver",
        action.payload.gpuserver_argv,
        action.payload.gpuserver_environment,
        disposition,
        return_code,
    )


def _hold_staged_lease_until_gpuserver_terminal(server: ScientificKernelProcess) -> tuple[str, ...]:
    """Force termination without releasing the lease until terminality is proven."""
    observed_errors: dict[str, str] = {}
    while server.poll() is None:
        try:
            server.kill()
        except OSError as exc:
            observed_errors["force-kill failed"] = str(exc)[:512]
        try:
            server.wait(timeout=GPUSERVER_SHUTDOWN_TIMEOUT_SECONDS)
        except OSError as exc:
            observed_errors["terminal wait failed"] = str(exc)[:512]
            time.sleep(0.05)
        except subprocess.TimeoutExpired as exc:
            observed_errors["terminal wait timed out"] = str(exc)[:512]
            time.sleep(0.05)
    return tuple(f"{kind}: {message}" for kind, message in observed_errors.items())


def _command_outcome(
    kind: PreprocessingCommandKind,
    argv: tuple[str, ...],
    environment: tuple[tuple[str, str], ...],
    disposition: PreprocessingCommandDisposition,
    return_code: int | None,
) -> PreprocessingCommandOutcome:
    return PreprocessingCommandOutcome(
        command_kind=kind,
        command_digest=preprocessing_command_digest(argv, environment),
        disposition=disposition,
        return_code=return_code,
    )


def _inspect_exact_tar(path: Path, action: PreprocessingRuntimeAction) -> tuple[str, ...]:
    try:
        with tarfile.open(path, mode="r:") as archive:
            try:
                members = validate_preprocessing_tar_headers(archive, action)
            except ValueError as exc:
                raise PreprocessingExecutionError(str(exc)) from exc
    except (OSError, tarfile.TarError) as exc:
        msg = f"invalid preprocessing tar archive {path}: {exc}"
        raise PreprocessingExecutionError(msg) from exc
    return tuple(member.name for member in members)


def _build_evidence(
    runspec: PhaseRunSpec,
    action: PreprocessingRuntimeAction,
    *,
    action_id: str,
    started_at: str,
    finished_at: str,
    command_outcomes: tuple[PreprocessingCommandOutcome, ...],
    failure: Exception | None,
    carry_forward_adoption: AttemptCarryForwardAdoptionEvidence | None,
    raw_search_evidence: PreprocessingRawSearchEvidence | None,
    database_placement: ActionDatabasePlacementEvidence,
    placement_process_status: int,
    accept_outputs: bool,
) -> PreprocessingChunkActionEvidence:
    reject_candidate_science = failure is not None and isinstance(
        database_placement, PreprocessingStagedDatabasePlacementEvidence
    )
    try:
        paired = _paired_evidence(action)
        archive = _archive_evidence(action)
        outputs = _existing_output_hashes(action) if accept_outputs and not reject_candidate_science else ()
    except OSError:
        if failure is None:
            raise
        paired = PreprocessingPairedEvidence(
            chunk_name=action.payload.chunk_name,
            durable_record_path=action.payload.evidence.durable_record_path,
            durable_log_path=action.payload.evidence.durable_log_path,
            record_lines=None,
            log_lines=None,
        )
        archive = PreprocessingArchiveEvidence(
            chunk_name=action.payload.chunk_name,
            durable_tar_path=action.payload.package.durable_tar_path,
            durable_lz4_path=action.payload.package.durable_lz4_path,
            tar_size_bytes=None,
            lz4_size_bytes=None,
            tar_members=None,
        )
        outputs = ()
    return PreprocessingChunkActionEvidence(
        adapter_version=PREPROCESSING_ADAPTER_VERSION,
        phase_run_id=runspec.phase_run_id,
        attempt_id=runspec.attempt_id,
        phase_runspec_digest=runspec.digest,
        action_id=action_id,
        placement_process_status=placement_process_status,
        chunk_name=action.payload.chunk_name,
        started_at=started_at,
        finished_at=finished_at,
        outcome="succeeded" if failure is None else "failed",
        command_outcomes=command_outcomes,
        paired_evidence=paired,
        archive_evidence=archive,
        output_hashes=outputs,
        error=None if failure is None else str(failure),
        database_placement=database_placement,
        raw_search_evidence=None if reject_candidate_science else raw_search_evidence,
        carry_forward_adoption=carry_forward_adoption,
    )


def _paired_evidence(action: PreprocessingRuntimeAction) -> PreprocessingPairedEvidence:
    plan = action.payload.evidence
    record_path = Path(plan.durable_record_path)
    log_path = Path(plan.durable_log_path)
    return PreprocessingPairedEvidence(
        chunk_name=action.payload.chunk_name,
        durable_record_path=plan.durable_record_path,
        durable_log_path=plan.durable_log_path,
        record_lines=tuple(record_path.read_text(errors="replace").splitlines())
        if _is_regular_file(record_path)
        else None,
        log_lines=tuple(log_path.read_text(errors="replace").splitlines()) if _is_regular_file(log_path) else None,
    )


def _archive_evidence(action: PreprocessingRuntimeAction) -> PreprocessingArchiveEvidence:
    plan = action.payload.package
    tar_path = Path(plan.durable_tar_path)
    lz4_path = Path(plan.durable_lz4_path)
    tar_present = _is_regular_file(tar_path)
    tar_members: tuple[str, ...] | None = None
    if tar_present:
        try:
            tar_members = _inspect_exact_tar(tar_path, action)
        except PreprocessingExecutionError:
            tar_members = ()
    return PreprocessingArchiveEvidence(
        chunk_name=action.payload.chunk_name,
        durable_tar_path=plan.durable_tar_path,
        durable_lz4_path=plan.durable_lz4_path,
        tar_size_bytes=tar_path.stat().st_size if tar_present else None,
        lz4_size_bytes=lz4_path.stat().st_size if _is_regular_file(lz4_path) else None,
        tar_members=tar_members,
    )


def _existing_output_hashes(action: PreprocessingRuntimeAction) -> tuple[PreprocessingOutputHash, ...]:
    payload = action.payload
    candidates: list[tuple[PreprocessingOutputRole, str, str | None]] = [
        ("a3m", _join(payload.evidence.scratch_output_directory, item.member_name), item.member_name)
        for item in payload.expected_a3ms
    ]
    candidates.extend(
        [
            ("log", payload.evidence.durable_log_path, None),
            ("record", payload.evidence.durable_record_path, None),
            ("tar", payload.package.durable_tar_path, None),
            ("tar-lz4", payload.package.durable_lz4_path, None),
            ("completed-input", payload.package.completed_input_path, None),
        ]
    )
    return tuple(
        _hash_output(role, path, member_name) for role, path, member_name in candidates if _is_regular_file(Path(path))
    )


def _expected_hash_paths(action: PreprocessingRuntimeAction) -> dict[tuple[str, str | None], str]:
    payload = action.payload
    result: dict[tuple[str, str | None], str] = {
        ("a3m", item.member_name): _join(payload.evidence.scratch_output_directory, item.member_name)
        for item in payload.expected_a3ms
    }
    result.update(
        {
            ("log", None): payload.evidence.durable_log_path,
            ("record", None): payload.evidence.durable_record_path,
            ("tar", None): payload.package.durable_tar_path,
            ("tar-lz4", None): payload.package.durable_lz4_path,
            ("completed-input", None): payload.package.completed_input_path,
        }
    )
    return result


def _hash_output(role: PreprocessingOutputRole, path: str, member_name: str | None) -> PreprocessingOutputHash:
    source = Path(path)
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return PreprocessingOutputHash(
        role=role,
        path=path,
        size_bytes=size,
        sha256=digest.hexdigest(),
        member_name=member_name,
    )


def _verify_output_hash(output: PreprocessingOutputHash) -> None:
    observed = _hash_output(output.role, output.path, output.member_name)
    if observed != output:
        msg = f"preprocessing output hash does not match current file: {output.path}"
        raise ValueError(msg)


def _normalized_file_members(raw_members: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(normalized for item in raw_members if (normalized := normalize_preprocessing_tar_member(item)) != ".")


def _require_nonempty_regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        msg = f"{label} must be an existing regular non-symlink file: {path}"
        raise PreprocessingExecutionError(msg)
    if path.stat().st_size <= 0:
        msg = f"{label} must be non-empty: {path}"
        raise PreprocessingExecutionError(msg)


def _is_regular_file(path: Path) -> bool:
    return not path.is_symlink() and path.is_file()


def _publish_copy_exclusive(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    try:
        with source.open("rb") as reader, temporary.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        os.link(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _move_exclusive(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        _publish_copy_exclusive(source, destination)
    source.unlink()
    _fsync_directory(destination.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _command_environment(environment: tuple[tuple[str, str], ...]) -> dict[str, str]:
    result = dict(os.environ)
    result.update(environment)
    return result


def _join(root: str, member: str) -> str:
    return f"{root.rstrip('/')}/{member}"


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _format_timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        msg = "preprocessing action evidence clock must be timezone-aware"
        raise ValueError(msg)
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


PRODUCTION_EXECUTION_COORDINATOR = ExecutionCoordinator(
    database_placement_paths=PRODUCTION_DATABASE_PLACEMENT_PATHS,
    clock=_utc_now,
    sleeper=time.sleep,
    scientific_kernel_launcher=PRODUCTION_SCIENTIFIC_KERNEL_LAUNCHER,
    evidence_store=PRODUCTION_PREPROCESSING_ACTION_EVIDENCE_STORE,
)


__all__ = [
    "GPUSERVER_WARMUP_SECONDS",
    "PREPROCESSING_ADAPTER_VERSION",
    "PreprocessingExecutionError",
    "execute_preprocessing_chunk_action",
    "load_preprocessing_phase_runspec",
    "reconcile_preprocessing_chunk_action_evidence",
    "reconcile_preprocessing_chunk_action_evidence_for_finalization",
]
