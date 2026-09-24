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

"""Whole-directory durable authority for preprocessing Phase Materialization."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Literal, cast

from bspp.orchestration.contract.database_set_provisioning import canonical_database_source_manifest_bytes
from bspp.orchestration.contract.folding_carry_forward import (
    FoldingCarryForwardRecord,
    FoldingCarryForwardReference,
)
from bspp.orchestration.contract.folding_shard import (
    FoldShardProjection,
    FoldShardProjectionBinding,
    fold_shard_projection_document_bytes,
    fold_shard_projection_from_mapping,
)
from bspp.orchestration.contract.phase import (
    FoldingPhasePlan,
    FoldingPhaseRunSpec,
    PhasePlan,
    PhaseRunSpec,
    canonical_mapping_digest,
    phase_plan_family_from_mapping,
    phase_runspec_family_from_mapping,
)
from bspp.orchestration.contract.phase_action_evidence_attestation import (
    PhaseActionEvidenceAttestedEvent,
    phase_action_evidence_attested_event_from_mapping,
)
from bspp.orchestration.contract.phase_cancellation import (
    PhaseCancellationActionDisposition,
    PhaseCancellationActionTarget,
    PhaseCancellationActionView,
    PhaseCancellationCompletedEvent,
    PhaseCancellationIntendedEvent,
    PhaseCancellationLifecycleView,
    PhaseCancellationTerminalReference,
    PhaseJobCancellationRequestIntendedEvent,
    PhaseJobCancellationRequestResultEvent,
    PhaseJobCancellationRequestView,
    phase_cancellation_completed_event_from_mapping,
    phase_cancellation_intended_event_from_mapping,
    phase_job_cancellation_request_intended_event_from_mapping,
    phase_job_cancellation_request_result_event_from_mapping,
)
from bspp.orchestration.contract.phase_carry_forward import (
    AttemptCarryForwardAdoptionEvidence,
    AttemptCarryForwardReceiptReference,
    AttemptCarryForwardRecord,
    AttemptCarryForwardRequest,
    AttemptCarryForwardSelection,
)
from bspp.orchestration.contract.phase_receipt import (
    PhaseFinalizedEvent,
    PhaseReceipt,
    phase_finalized_event_from_mapping,
)
from bspp.orchestration.contract.phase_reconciliation import (
    FoldingActionTerminalObservationView,
    FoldingActionTerminalObservedEvent,
    PhaseActionTerminalObservationView,
    PhaseActionTerminalObservedEvent,
    folding_action_terminal_observed_event_from_mapping,
    phase_action_terminal_observed_event_from_mapping,
)
from bspp.orchestration.contract.phase_retry import (
    PhaseAttemptHistoryView,
    PhaseAttemptRetriedEvent,
    compare_retry_invariants,
    phase_attempt_retried_event_from_mapping,
    phase_input_set_identity_digest,
    phase_retry_id,
    phase_scientific_identity_digest,
)
from bspp.orchestration.contract.phase_state import (
    PhaseAttempt,
    PhaseMaterializedEvent,
    PhaseRun,
    PhaseRunLifecycleView,
    phase_materialized_event_from_mapping,
    phase_run_from_mapping,
)
from bspp.orchestration.contract.phase_submission import (
    PhaseActionDispatchIntendedEvent,
    PhaseActionDispatchRejectedEvent,
    PhaseActionSatisfiedWithoutDispatchEvent,
    PhaseActionSubmissionView,
    PhaseActionSubmittedEvent,
    PhaseSubmissionIntendedEvent,
    PhaseSubmissionIntendedPayload,
    PhaseSubmissionLifecycleView,
    PhaseSubmissionStatus,
    phase_action_dispatch_intended_event_from_mapping,
    phase_action_dispatch_rejected_event_from_mapping,
    phase_action_satisfied_without_dispatch_event_from_mapping,
    phase_action_submitted_event_from_mapping,
    phase_submission_intended_event_from_mapping,
)
from bspp.orchestration.contract.preprocessing_execution import (
    PreprocessingChunkExecutionPlan,
    materialize_preprocessing_chunk_execution_plan,
)
from bspp.orchestration.contract.preprocessing_runtime import preprocessing_runtime_tuple_id
from bspp.orchestration.control.folding_phase_adapter import validate_folding_plan_runspec_binding
from bspp.orchestration.control.folding_shard import fold_shard_projection_from_runspec
from bspp.orchestration.control.phase_carry_forward import derive_attempt_carry_forward
from bspp.orchestration.control.phase_lifecycle import classify_phase_lifecycle
from bspp.orchestration.control.transport import LEGACY_WIRE_SHAPE_CUTOVER_AT, command_argv, legacy_command_argv

_PHASE_PLAN = Path("phase-plan.json")
_PHASE_RUN = Path("phase-run.json")
_MATERIALIZED_EVENT = Path("events/000001-phase-materialized.json")
_EVENT_NAME = re.compile(r"(?P<sequence>[0-9]{6})-(?P<event_type>[a-z][a-z0-9-]*)\.json")
_PHASE_RUN_ID = re.compile(r"phase-run-[0-9a-f]{32}")

PhasePlanRecord = PhasePlan | FoldingPhasePlan
PhaseRunSpecRecord = PhaseRunSpec | FoldingPhaseRunSpec


class PhaseAuthorityCollisionError(FileExistsError):
    """Raised when publication would replace an existing Phase Run authority."""

    def __init__(self, phase_run_id: str) -> None:
        super().__init__(f"Phase Run authority already exists for {phase_run_id}")
        self.phase_run_id = phase_run_id


class PhaseAuthoritySealedError(RuntimeError):
    """Raised when an append targets an already accepted Phase Run."""


def validate_phase_run_id(phase_run_id: str) -> str:
    """Return one canonical Phase Run id before any authority path is derived."""
    if _PHASE_RUN_ID.fullmatch(phase_run_id) is None:
        raise ValueError("Phase operation requires a valid Phase Run id")
    return phase_run_id


def require_complete_current_runspec(authority: PhaseAuthorityValidation, *, operation: str) -> None:
    """Reject external-effect operations during Retry projection recovery."""
    if not authority.current_runspec_projection_complete:
        raise ValueError(f"{operation} requires rerunning Phase Retry to complete the current RunSpec projection")


@dataclass(frozen=True)
class PhaseAuthorityValidation:
    """Strictly replayed records from one complete Phase Run authority."""

    authority_path: Path
    phase_plan: PhasePlanRecord
    phase_run: PhaseRun
    current_attempt: PhaseAttempt
    phase_runspec: PhaseRunSpecRecord
    materialized_event: PhaseMaterializedEvent
    lifecycle: PhaseRunLifecycleView
    events: tuple[object, ...]
    prior_attempts: tuple[PhaseAttemptHistoryView, ...] = ()
    current_runspec_projection_complete: bool = True
    current_carry_forward: AttemptCarryForwardRecord | FoldingCarryForwardRecord | None = None
    current_action_evidence_attestation: PhaseActionEvidenceAttestedEvent | None = None
    submission: PhaseSubmissionLifecycleView | None = None
    cancellation: PhaseCancellationLifecycleView | None = None
    terminal_observations: tuple[PhaseActionTerminalObservationView, ...] = ()
    array_terminal_observations: tuple[FoldingActionTerminalObservationView, ...] = ()
    finalized_event: PhaseFinalizedEvent | None = None
    receipt: PhaseReceipt | None = None


PhaseActionTerminalView = PhaseActionTerminalObservationView | FoldingActionTerminalObservationView


def _select_terminal_observations_by_action(
    terminal_observations: tuple[PhaseActionTerminalObservationView, ...],
    array_terminal_observations: tuple[FoldingActionTerminalObservationView, ...],
) -> dict[str, PhaseActionTerminalView]:
    """Project two validated terminal tuples into one array-preferred action view.

    Replay already enforces uniqueness within each terminal collection, so the
    scalar and array maps are each injective here. Overlap between the two
    collections is legal and backward compatible: a cancelled packed fold can
    carry both a scalar parent-row fact (cancellation) and the exact
    parent-plus-task array evidence (Resume). On overlap the array observation
    wins because it is the only view carrying the parent job and exact task
    indexes.
    """
    selected: dict[str, PhaseActionTerminalView] = {item.action_id: item for item in terminal_observations}
    selected.update({item.action_id: item for item in array_terminal_observations})
    return selected


PhaseAuthorityEventLoader = Callable[[Mapping[str, object]], object]
PhaseAuthorityEventReplay = Callable[[object, PhaseAuthorityValidation], PhaseAuthorityValidation]


@dataclass(frozen=True)
class PhaseAuthorityEventHandler:
    """Strict loader and replay transition for one registered event discriminator."""

    event_type: str
    loader: PhaseAuthorityEventLoader
    replay: PhaseAuthorityEventReplay
    terminal: bool = False

    def __post_init__(self) -> None:
        if re.fullmatch(r"[a-z][a-z0-9-]*", self.event_type) is None:
            raise ValueError("Phase authority event handler requires a canonical event type")


class PhaseAuthorityStore:
    """Publish and replay one Phase Run as a same-filesystem atomic directory."""

    def __init__(
        self,
        authority_root: Path,
        *,
        before_publish: Callable[[Path], None] | None = None,
        before_event_publish: Callable[[Path], None] | None = None,
        after_retry_event_publish: Callable[[PhaseAuthorityValidation], None] | None = None,
        after_retry_attempt_directory_create: Callable[[Path], None] | None = None,
        after_retry_carry_forward_link: Callable[[Path], None] | None = None,
        after_retry_runspec_link: Callable[[Path], None] | None = None,
        before_retry_projection_replay: Callable[[Path], None] | None = None,
        event_handlers: tuple[PhaseAuthorityEventHandler, ...] = (),
    ) -> None:
        self.authority_root = authority_root
        self._before_publish = before_publish
        self._before_event_publish = before_event_publish
        self._after_retry_event_publish = after_retry_event_publish
        self._after_retry_attempt_directory_create = after_retry_attempt_directory_create
        self._after_retry_carry_forward_link = after_retry_carry_forward_link
        self._after_retry_runspec_link = after_retry_runspec_link
        self._before_retry_projection_replay = before_retry_projection_replay
        self._event_handlers = _event_handler_registry(event_handlers)

    @staticmethod
    def _runspec_family(runspec: PhaseRunSpec | FoldingPhaseRunSpec) -> Literal["preprocessing", "folding"]:
        """Return the exact Phase family for one RunSpec record."""
        return "folding" if isinstance(runspec, FoldingPhaseRunSpec) else "preprocessing"

    def publish(
        self,
        *,
        phase_plan: PhasePlanRecord,
        phase_run: PhaseRun,
        phase_runspec: PhaseRunSpecRecord,
        materialized_event: PhaseMaterializedEvent,
    ) -> PhaseAuthorityValidation:
        """Durably stage, validate, and atomically reveal one complete authority."""
        _validate_records(phase_plan, phase_run, phase_runspec, materialized_event)
        self.authority_root.mkdir(parents=True, exist_ok=True)
        if not self.authority_root.is_dir():
            msg = f"Phase authority root is not a directory: {self.authority_root}"
            raise ValueError(msg)

        staging_path = Path(tempfile.mkdtemp(prefix=f".{phase_run.phase_run_id}.staging-", dir=self.authority_root))
        try:
            runspec_path = staging_path / phase_run.attempts[0].phase_runspec_location
            _write_json_durable(phase_plan.to_mapping(), staging_path / _PHASE_PLAN)
            _write_json_durable(phase_run.to_mapping(), staging_path / _PHASE_RUN)
            _write_json_durable(materialized_event.to_mapping(), staging_path / _MATERIALIZED_EVENT)
            _write_json_durable(phase_runspec.to_mapping(), runspec_path)
            if isinstance(phase_runspec, FoldingPhaseRunSpec):
                if not isinstance(phase_plan, FoldingPhasePlan):
                    raise ValueError("folding Phase RunSpec requires a folding Phase Plan")
                _write_fold_shard_projection(staging_path, phase_plan, phase_runspec)
            else:
                _write_bytes_durable(
                    canonical_database_source_manifest_bytes(phase_runspec.payload.database.source_manifest),
                    staging_path / phase_runspec.payload.database.source_manifest_projection,
                    mode=0o444,
                )
            _fsync_tree_directories(staging_path)
            staged = _validate_authority_directory(
                staging_path,
                expected_phase_run_id=phase_run.phase_run_id,
                event_handlers=self._event_handlers,
            )
            if staged.phase_plan != phase_plan or staged.phase_run != phase_run:
                msg = "staged Phase authority does not replay to its source records"
                raise ValueError(msg)
            if staged.phase_runspec != phase_runspec or staged.materialized_event != materialized_event:
                msg = "staged Phase attempt or event does not replay to its source records"
                raise ValueError(msg)
            if self._before_publish is not None:
                self._before_publish(staging_path)
            destination = self.authority_root / phase_run.phase_run_id
            self._publish_directory_no_replace(staging_path, destination, phase_run.phase_run_id)
            _fsync_directory(self.authority_root)
        except BaseException:
            if staging_path.exists():
                shutil.rmtree(staging_path)
            raise
        return self.validate(phase_run.phase_run_id)

    def validate(self, phase_run_id: str) -> PhaseAuthorityValidation:
        """Fail closed unless the run has the exact complete replayable layout."""
        validate_phase_run_id(phase_run_id)
        return _validate_authority_directory(
            self.authority_root / phase_run_id,
            expected_phase_run_id=phase_run_id,
            event_handlers=self._event_handlers,
        )

    def append_finalized_event(
        self,
        phase_run_id: str,
        event_factory: Callable[[int], PhaseFinalizedEvent],
    ) -> PhaseAuthorityValidation:
        """Append one complete terminal event with atomic no-replace publication."""
        return self.append_event(phase_run_id, event_factory)

    def append_event(
        self,
        phase_run_id: str,
        event_factory: Callable[[int], object],
    ) -> PhaseAuthorityValidation:
        """Append one registered event with atomic no-replace publication."""
        validate_phase_run_id(phase_run_id)
        self.authority_root.mkdir(parents=True, exist_ok=True)
        with self._authority_lock():
            current = _validate_authority_directory(
                self.authority_root / phase_run_id,
                expected_phase_run_id=phase_run_id,
                event_handlers=self._event_handlers,
            )
            if not current.current_runspec_projection_complete:
                raise ValueError("Phase operation requires rerunning Retry to complete the current RunSpec projection")
            if current.lifecycle.sealed:
                raise PhaseAuthoritySealedError(f"Phase Run is already sealed: {phase_run_id}")
            event_files = _discover_event_files(current.authority_path, event_handlers=self._event_handlers)
            next_sequence = len(event_files) + 1
            event = event_factory(next_sequence)
            event_sequence = getattr(event, "sequence", None)
            event_type = getattr(event, "event_type", None)
            to_mapping = getattr(event, "to_mapping", None)
            if event_sequence != next_sequence:
                raise ValueError("Phase event factory returned a non-contiguous sequence")
            if not isinstance(event_type, str) or event_type not in self._event_handlers or not callable(to_mapping):
                raise ValueError("Phase event factory returned an unregistered event")
            if event_type == "phase-materialized":
                raise ValueError("phase-materialized may appear only as the first authority event")
            self._event_handlers[event_type].replay(event, current)
            destination = current.authority_path / "events" / f"{event_sequence:06d}-{event_type}.json"
            temporary = self.authority_root / f".phase-event-{uuid.uuid4().hex}.json"
            try:
                payload = to_mapping()
                if not isinstance(payload, Mapping):
                    raise TypeError("Phase event to_mapping() must return a mapping")
                _write_json_durable(payload, temporary)
                # The test hook runs only after complete staged bytes are durable.
                if self._before_event_publish is not None:
                    self._before_event_publish(temporary)
                try:
                    os.link(temporary, destination)
                except FileExistsError as exc:
                    raise PhaseAuthorityCollisionError(phase_run_id) from exc
                _fsync_directory(destination.parent)
            finally:
                temporary.unlink(missing_ok=True)
                _fsync_directory(self.authority_root)
            return _validate_authority_directory(
                current.authority_path,
                expected_phase_run_id=phase_run_id,
                event_handlers=self._event_handlers,
            )

    def append_attempt_retried_event(
        self,
        phase_run_id: str,
        event_factory: Callable[[int], PhaseAttemptRetriedEvent],
    ) -> PhaseAuthorityValidation:
        """Stage successor bytes, append Retry authority, then publish projections."""
        staged_runspec = self.authority_root / f".phase-runspec-{uuid.uuid4().hex}.json"
        staged_carry = self.authority_root / f".phase-carry-forward-{uuid.uuid4().hex}.json"
        staged = False

        def staged_event_factory(sequence: int) -> PhaseAttemptRetriedEvent:
            nonlocal staged
            event = event_factory(sequence)
            _write_json_durable(event.payload.successor_phase_runspec.to_mapping(), staged_runspec)
            if event.payload.carry_forward_record is not None:
                _write_json_durable(event.payload.carry_forward_record.to_mapping(), staged_carry)
            staged = True
            return event

        try:
            retried = self.append_event(phase_run_id, staged_event_factory)
            if not staged:
                raise AssertionError("Retry RunSpec bytes were not staged before event publication")
            if retried.current_runspec_projection_complete:
                raise AssertionError("Retry event unexpectedly exposed a preexisting successor projection")
            if self._after_retry_event_publish is not None:
                self._after_retry_event_publish(retried)
            return self._publish_current_attempt_runspec_projection(
                phase_run_id,
                staged_runspec=staged_runspec,
                staged_carry=staged_carry if staged_carry.exists() else None,
            )
        finally:
            staged_runspec.unlink(missing_ok=True)
            staged_carry.unlink(missing_ok=True)
            if self.authority_root.is_dir():
                _fsync_directory(self.authority_root)

    def publish_current_attempt_runspec_projection(self, phase_run_id: str) -> PhaseAuthorityValidation:
        """Complete only the exact latest Retry projection embedded in authority."""
        return self._publish_current_attempt_runspec_projection(
            phase_run_id,
            staged_runspec=None,
            staged_carry=None,
        )

    def _publish_current_attempt_runspec_projection(
        self,
        phase_run_id: str,
        *,
        staged_runspec: Path | None,
        staged_carry: Path | None,
    ) -> PhaseAuthorityValidation:
        validate_phase_run_id(phase_run_id)
        self.authority_root.mkdir(parents=True, exist_ok=True)
        with self._authority_lock():
            current = _validate_authority_directory(
                self.authority_root / phase_run_id,
                expected_phase_run_id=phase_run_id,
                event_handlers=self._event_handlers,
            )
            if current.current_runspec_projection_complete:
                return current
            destination = current.authority_path / current.current_attempt.phase_runspec_location
            attempt_directory = destination.parent
            if os.path.lexists(attempt_directory):
                if attempt_directory.is_symlink() or not attempt_directory.is_dir():
                    raise ValueError("Phase Retry Attempt projection parent is not a regular directory")
            else:
                attempt_directory.mkdir()
                _fsync_directory(attempt_directory.parent)
                if self._after_retry_attempt_directory_create is not None:
                    self._after_retry_attempt_directory_create(attempt_directory)
            if current.current_carry_forward is not None:
                carry_destination = attempt_directory / _carry_forward_file_name(current.current_carry_forward)
                carry_temporary = staged_carry or self.authority_root / f".phase-carry-forward-{uuid.uuid4().hex}.json"
                owns_carry_temporary = staged_carry is None
                carry_linked = False
                try:
                    expected_carry = _canonical_json_bytes(current.current_carry_forward.to_mapping())
                    if owns_carry_temporary:
                        _write_json_durable(current.current_carry_forward.to_mapping(), carry_temporary)
                    elif carry_temporary.read_bytes() != expected_carry:
                        raise ValueError("staged carry-forward record differs from embedded authority")
                    try:
                        os.link(carry_temporary, carry_destination)
                        carry_linked = True
                    except FileExistsError:
                        if (
                            carry_destination.is_symlink()
                            or not carry_destination.is_file()
                            or carry_destination.read_bytes() != expected_carry
                        ):
                            raise ValueError(
                                "existing carry-forward projection differs from embedded authority"
                            ) from None
                    _fsync_directory(attempt_directory)
                    if carry_linked and self._after_retry_carry_forward_link is not None:
                        self._after_retry_carry_forward_link(carry_destination)
                finally:
                    if owns_carry_temporary:
                        carry_temporary.unlink(missing_ok=True)
                    _fsync_directory(self.authority_root)
            if not isinstance(current.phase_runspec, FoldingPhaseRunSpec):
                manifest_destination = (
                    current.authority_path / current.phase_runspec.payload.database.source_manifest_projection
                )
                expected_manifest = canonical_database_source_manifest_bytes(
                    current.phase_runspec.payload.database.source_manifest
                )
                manifest_temporary = self.authority_root / f".database-source-manifest-{uuid.uuid4().hex}.json"
                try:
                    _write_bytes_durable(expected_manifest, manifest_temporary, mode=0o444)
                    try:
                        os.link(manifest_temporary, manifest_destination)
                    except FileExistsError:
                        if (
                            manifest_destination.is_symlink()
                            or not manifest_destination.is_file()
                            or stat.S_IMODE(manifest_destination.stat().st_mode) != 0o444
                            or manifest_destination.read_bytes() != expected_manifest
                        ):
                            raise ValueError(
                                "existing database source-manifest projection differs from embedded authority"
                            ) from None
                    _fsync_directory(attempt_directory)
                finally:
                    manifest_temporary.unlink(missing_ok=True)
                    _fsync_directory(self.authority_root)
            if isinstance(current.phase_runspec, FoldingPhaseRunSpec):
                if not isinstance(current.phase_plan, FoldingPhasePlan):
                    raise ValueError("folding Phase RunSpec requires a folding Phase Plan")
                binding = current.phase_runspec.payload.fold_shard_projection
                if binding is not None:
                    fold_shard_destination = current.authority_path / binding.location
                    expected_projection, expected_binding = fold_shard_projection_from_runspec(
                        current.phase_plan, current.phase_runspec
                    )
                    if expected_binding != binding:
                        raise ValueError("fold shard projection binding differs from embedded authority")
                    expected_fold_shard = fold_shard_projection_document_bytes(expected_projection)
                    fold_shard_temporary = self.authority_root / f".fold-shard-projection-{uuid.uuid4().hex}.json"
                    try:
                        _write_bytes_durable(expected_fold_shard, fold_shard_temporary)
                        try:
                            os.link(fold_shard_temporary, fold_shard_destination)
                        except FileExistsError:
                            if (
                                fold_shard_destination.is_symlink()
                                or not fold_shard_destination.is_file()
                                or fold_shard_destination.read_bytes() != expected_fold_shard
                            ):
                                raise ValueError(
                                    "existing fold shard projection differs from embedded authority"
                                ) from None
                        _fsync_directory(attempt_directory)
                    finally:
                        fold_shard_temporary.unlink(missing_ok=True)
                        _fsync_directory(self.authority_root)
            temporary = staged_runspec or self.authority_root / f".phase-runspec-{uuid.uuid4().hex}.json"
            owns_temporary = staged_runspec is None
            linked = False
            try:
                if owns_temporary:
                    _write_json_durable(current.phase_runspec.to_mapping(), temporary)
                elif temporary.read_bytes() != _canonical_json_bytes(current.phase_runspec.to_mapping()):
                    raise ValueError("staged Phase Retry RunSpec differs from embedded authority")
                try:
                    os.link(temporary, destination)
                    linked = True
                except FileExistsError:
                    expected = _canonical_json_bytes(current.phase_runspec.to_mapping())
                    if destination.is_symlink() or not destination.is_file() or destination.read_bytes() != expected:
                        raise ValueError(
                            "existing Phase Retry RunSpec projection differs from embedded authority"
                        ) from None
                _fsync_directory(attempt_directory)
                if linked and self._after_retry_runspec_link is not None:
                    self._after_retry_runspec_link(destination)
            finally:
                if owns_temporary:
                    temporary.unlink(missing_ok=True)
                _fsync_directory(self.authority_root)
        if self._before_retry_projection_replay is not None:
            self._before_retry_projection_replay(destination)
        complete = self.validate(phase_run_id)
        if not complete.current_runspec_projection_complete:
            raise ValueError("Phase Retry RunSpec projection did not become complete")
        return complete

    @contextmanager
    def phase_operation_lock(self, phase_run_id: str) -> Iterator[None]:
        """Serialize external-effect lifecycle operations for one Phase Run.

        The empty sentinel intentionally persists so every opener addresses one
        stable inode. Unlinking it could let an existing waiter lock the old
        inode while a new opener locks a replacement inode.
        """
        validate_phase_run_id(phase_run_id)
        self.authority_root.mkdir(parents=True, exist_ok=True)
        lock_path = self.authority_root / f".phase-operation-{phase_run_id}.lock"
        with lock_path.open("a+b") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def _publish_directory_no_replace(self, staging_path: Path, destination: Path, phase_run_id: str) -> None:
        with self._authority_lock():
            # The lock makes the existence check and rename one authority-store
            # operation. The explicit check is required because POSIX rename can
            # replace an existing empty directory.
            if os.path.lexists(destination):
                raise PhaseAuthorityCollisionError(phase_run_id)
            try:
                os.rename(staging_path, destination)
            except OSError as exc:
                if exc.errno in {errno.EEXIST, errno.ENOTEMPTY} or os.path.lexists(destination):
                    raise PhaseAuthorityCollisionError(phase_run_id) from exc
                raise

    @contextmanager
    def _authority_lock(self) -> Iterator[None]:
        """Serialize publish and append with the authority-root global lock."""
        lock_path = self.authority_root / ".phase-authority.lock"
        with lock_path.open("a+b") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _validate_authority_directory(
    path: Path,
    *,
    expected_phase_run_id: str,
    event_handlers: Mapping[str, PhaseAuthorityEventHandler],
) -> PhaseAuthorityValidation:
    if path.is_symlink() or not path.is_dir():
        msg = f"missing Phase Run authority directory: {path}"
        raise ValueError(msg)

    # Strictly discover and type every event before any event-derived Attempt
    # id is permitted to influence the expected filesystem layout.
    event_files = _discover_event_files(path, event_handlers=event_handlers)
    typed_events = tuple(_load_registered_event(path / event_file, event_handlers) for event_file in event_files)
    first = typed_events[0]
    if not isinstance(first, PhaseMaterializedEvent):
        raise ValueError("the first registered Phase event must load as PhaseMaterializedEvent")
    declared_attempts: list[PhaseAttempt] = [first.payload.phase_run.attempts[0]]
    for event in typed_events[1:]:
        if not isinstance(event, PhaseAttemptRetriedEvent):
            continue
        predecessor = declared_attempts[-1]
        successor = event.payload.successor_attempt
        if (
            event.payload.predecessor_attempt_id != predecessor.attempt_id
            or successor.ordinal != predecessor.ordinal + 1
            or successor.attempt_id != f"attempt-{successor.ordinal:04d}"
        ):
            raise ValueError("Phase Retry events must declare one contiguous predecessor/successor chain")
        declared_attempts.append(successor)

    runspec_relatives = {Path(attempt.phase_runspec_location) for attempt in declared_attempts}
    embedded_runspecs = [first.payload.phase_runspec]
    embedded_runspecs.extend(
        event.payload.successor_phase_runspec
        for event in typed_events[1:]
        if isinstance(event, PhaseAttemptRetriedEvent)
    )
    manifest_relatives = {
        Path(runspec.payload.database.source_manifest_projection)
        for runspec in embedded_runspecs
        if not isinstance(runspec, FoldingPhaseRunSpec)
    }
    fold_shard_relatives = {
        Path(runspec.payload.fold_shard_projection.location)
        for runspec in embedded_runspecs
        if isinstance(runspec, FoldingPhaseRunSpec) and runspec.payload.fold_shard_projection is not None
    }
    carry_relatives = {
        Path(event.payload.successor_attempt.phase_runspec_location).parent
        / _carry_forward_file_name(event.payload.carry_forward_record)
        for event in typed_events
        if isinstance(event, PhaseAttemptRetriedEvent) and event.payload.carry_forward_record is not None
    }
    fixed_files = {
        _PHASE_PLAN,
        _PHASE_RUN,
        *runspec_relatives,
        *manifest_relatives,
        *carry_relatives,
        *fold_shard_relatives,
    }
    expected_directories = {
        Path("events"),
        Path("attempts"),
        *(Path(attempt.phase_runspec_location).parent for attempt in declared_attempts),
    }
    observed_files: set[Path] = set()
    observed_directories: set[Path] = set()
    for candidate in path.rglob("*"):
        relative = candidate.relative_to(path)
        if candidate.is_symlink():
            msg = f"Phase authority must not contain symlinks: {relative}"
            raise ValueError(msg)
        if candidate.is_file():
            observed_files.add(relative)
        elif candidate.is_dir():
            observed_directories.add(relative)
        else:
            msg = f"Phase authority contains unsupported entry: {relative}"
            raise ValueError(msg)
    latest_attempt = declared_attempts[-1]
    latest_runspec = Path(latest_attempt.phase_runspec_location)
    latest_directory = latest_runspec.parent
    latest_runspec_record = embedded_runspecs[-1]
    latest_manifest = (
        Path(latest_runspec_record.payload.database.source_manifest_projection)
        if not isinstance(latest_runspec_record, FoldingPhaseRunSpec)
        else None
    )
    latest_retry = typed_events[-1] if isinstance(typed_events[-1], PhaseAttemptRetriedEvent) else None
    latest_carry = latest_directory / "attempt-carry-forward.json"
    latest_requires_carry = latest_retry is not None and latest_retry.payload.carry_forward_record is not None
    if latest_retry is not None and latest_retry.payload.carry_forward_record is not None:
        latest_carry = latest_directory / _carry_forward_file_name(latest_retry.payload.carry_forward_record)
    latest_fold_shard = (
        Path(latest_runspec_record.payload.fold_shard_projection.location)
        if isinstance(latest_runspec_record, FoldingPhaseRunSpec)
        and latest_runspec_record.payload.fold_shard_projection is not None
        else None
    )
    carry_prefix_valid = not latest_requires_carry or (
        latest_carry in observed_files or latest_runspec not in observed_files
    )
    if not carry_prefix_valid:
        raise ValueError("carried Retry RunSpec projection cannot precede its carry-forward record")
    manifest_incomplete = latest_manifest is not None and latest_manifest not in observed_files
    fold_shard_incomplete = latest_fold_shard is not None and latest_fold_shard not in observed_files
    terminal_retry_incomplete = (
        len(declared_attempts) > 1
        and latest_retry is not None
        and latest_retry.payload.successor_attempt == latest_attempt
        and (
            latest_runspec not in observed_files
            or manifest_incomplete
            or fold_shard_incomplete
            or latest_directory not in observed_directories
            or (latest_requires_carry and latest_carry not in observed_files)
        )
    )
    allowed_files = fixed_files | set(event_files)
    allowed_directories = expected_directories
    if terminal_retry_incomplete:
        allowed_files = allowed_files - {latest_runspec}
        if latest_manifest is not None and latest_manifest not in observed_files:
            allowed_files = allowed_files - {latest_manifest}
        if latest_fold_shard is not None and latest_fold_shard not in observed_files:
            allowed_files = allowed_files - {latest_fold_shard}
        if latest_requires_carry and latest_carry not in observed_files:
            allowed_files = allowed_files - {latest_carry}
        if latest_directory not in observed_directories:
            allowed_directories = allowed_directories - {latest_directory}
    if observed_files != allowed_files or observed_directories != allowed_directories:
        expected_entries = allowed_files | allowed_directories
        observed_entries = observed_files | observed_directories
        missing = sorted(str(item) for item in expected_entries - observed_entries)
        extra = sorted(str(item) for item in observed_entries - expected_entries)
        msg = f"Phase authority layout mismatch; missing={missing}, extra={extra}"
        raise ValueError(msg)

    phase_plan = phase_plan_family_from_mapping(_read_json(path / _PHASE_PLAN))
    phase_run = phase_run_from_mapping(_read_json(path / _PHASE_RUN))
    initial_runspec_relative = Path(declared_attempts[0].phase_runspec_location)
    phase_runspec = phase_runspec_family_from_mapping(_read_json(path / initial_runspec_relative))
    if not isinstance(phase_plan, PhasePlan | FoldingPhasePlan):
        raise ValueError("Phase authority rejects a postprocessing Phase Plan")
    if not isinstance(phase_runspec, PhaseRunSpec | FoldingPhaseRunSpec):
        raise ValueError("Phase authority rejects a postprocessing Phase RunSpec")
    if not isinstance(phase_runspec, FoldingPhaseRunSpec):
        _validate_database_manifest_projection(path, phase_runspec)
    else:
        if not isinstance(phase_plan, FoldingPhasePlan):
            raise ValueError("folding Phase RunSpec requires a folding Phase Plan")
        _validate_fold_shard_projection(path, phase_plan, phase_runspec)
    materialized = first
    if phase_run.phase_run_id != expected_phase_run_id:
        msg = "Phase authority directory identity does not match its Phase Run"
        raise ValueError(msg)
    _validate_records(phase_plan, phase_run, phase_runspec, materialized)
    validation = PhaseAuthorityValidation(
        authority_path=path,
        phase_plan=phase_plan,
        phase_run=phase_run,
        current_attempt=phase_run.attempts[0],
        phase_runspec=phase_runspec,
        materialized_event=materialized,
        lifecycle=PhaseRunLifecycleView(
            phase_run_id=phase_run.phase_run_id,
            current_attempt_id=phase_run.current_attempt_id,
            attempt_status="materialized",
            run_status="materialized",
            sealed=False,
            phase_receipt_id=None,
        ),
        events=(materialized,),
    )
    for event_file, event in zip(event_files[1:], typed_events[1:], strict=True):
        event_type = _event_type_from_path(event_file)
        handler = event_handlers[event_type]
        validation = handler.replay(event, validation)
    if terminal_retry_incomplete != (not validation.current_runspec_projection_complete):
        raise ValueError("Phase Retry projection completeness does not match the strict authority layout")
    return validation


def _replay_materialized_event(
    _event: object,
    _authority: PhaseAuthorityValidation,
) -> PhaseAuthorityValidation:
    raise ValueError("phase-materialized may appear only as the first authority event")


def _replay_attempt_retried_event(
    event: object,
    authority: PhaseAuthorityValidation,
) -> PhaseAuthorityValidation:
    if not isinstance(event, PhaseAttemptRetriedEvent):
        raise TypeError("phase-attempt-retried handler requires PhaseAttemptRetriedEvent")
    if not authority.current_runspec_projection_complete:
        raise ValueError("Phase Retry cannot follow an incomplete RunSpec projection")
    classification = classify_phase_lifecycle(authority.lifecycle)
    if classification not in {"failed", "cancelled"}:
        raise ValueError(f"Phase Retry cannot cross a {classification} Phase Attempt boundary")
    if authority.lifecycle.sealed or authority.receipt is not None or authority.finalized_event is not None:
        raise ValueError("Phase Retry requires an unsealed, unreceipted Phase Run")
    payload = event.payload
    predecessor = authority.current_attempt
    successor = payload.successor_attempt
    runspec = payload.successor_phase_runspec
    if (
        event.phase_run_id != authority.phase_run.phase_run_id
        or payload.predecessor_attempt_id != predecessor.attempt_id
        or payload.predecessor_phase_runspec_digest != authority.phase_runspec.digest
        or payload.predecessor_outcome != classification
        or successor.ordinal != predecessor.ordinal + 1
        or successor.attempt_id != f"attempt-{successor.ordinal:04d}"
        or runspec.phase_run_id != authority.phase_run.phase_run_id
    ):
        raise ValueError("Phase Retry event does not bind the exact contiguous predecessor authority")
    if payload.phase_plan_digest != authority.phase_plan.digest:
        raise ValueError("Phase Retry event does not bind the exact stored Phase Plan")
    input_digest = phase_input_set_identity_digest(authority.phase_plan)
    scientific_digest = phase_scientific_identity_digest(authority.phase_plan)
    if payload.input_set_identity_digest != input_digest or payload.scientific_identity_digest != scientific_digest:
        raise ValueError("Phase Retry event does not preserve input and scientific identity")
    compare_retry_invariants(authority.phase_plan, authority.phase_runspec, runspec)
    _validate_carry_retry(event, authority)
    expected_retry_id = phase_retry_id(
        phase_run_id=authority.phase_run.phase_run_id,
        predecessor_attempt_id=predecessor.attempt_id,
        successor_attempt_id=successor.attempt_id,
        predecessor_outcome=payload.predecessor_outcome,
        predecessor_phase_runspec_digest=authority.phase_runspec.digest,
        phase_plan_digest=authority.phase_plan.digest,
        input_set_identity_digest=input_digest,
        scientific_identity_digest=scientific_digest,
        selected_cluster_profile=payload.selected_cluster_profile,
        successor_phase_runspec_digest=runspec.digest,
    )
    if event.payload.retry_id != expected_retry_id:
        raise ValueError("Phase Retry event id does not match replayed authority")
    projection_path = authority.authority_path / successor.phase_runspec_location
    manifest_projection_path = (
        authority.authority_path / runspec.payload.database.source_manifest_projection
        if not isinstance(runspec, FoldingPhaseRunSpec)
        else None
    )
    fold_shard_projection_path = (
        authority.authority_path / runspec.payload.fold_shard_projection.location
        if isinstance(runspec, FoldingPhaseRunSpec) and runspec.payload.fold_shard_projection is not None
        else None
    )
    carry_projection_path = projection_path.parent / (
        _carry_forward_file_name(payload.carry_forward_record)
        if payload.carry_forward_record is not None
        else "attempt-carry-forward.json"
    )
    carry_projection_complete = payload.carry_forward_record is None or (
        carry_projection_path.is_file() and not carry_projection_path.is_symlink()
    )
    manifest_projection_complete = manifest_projection_path is None or (
        manifest_projection_path.is_file() and not manifest_projection_path.is_symlink()
    )
    fold_shard_projection_complete = fold_shard_projection_path is None or (
        fold_shard_projection_path.is_file() and not fold_shard_projection_path.is_symlink()
    )
    projection_complete = (
        projection_path.is_file()
        and not projection_path.is_symlink()
        and carry_projection_complete
        and manifest_projection_complete
        and fold_shard_projection_complete
    )
    if carry_projection_complete and payload.carry_forward_record is not None:
        expected_carry_bytes = _canonical_json_bytes(payload.carry_forward_record.to_mapping())
        if carry_projection_path.read_bytes() != expected_carry_bytes:
            raise ValueError("carry-forward projection differs from embedded Retry authority")
    if projection_complete:
        expected_bytes = _canonical_json_bytes(runspec.to_mapping())
        if projection_path.read_bytes() != expected_bytes:
            raise ValueError("successor Phase RunSpec projection differs from embedded Retry authority")
        projected = phase_runspec_family_from_mapping(_read_json(projection_path))
        if projected != runspec or projected.digest != successor.phase_runspec_digest:
            raise ValueError("successor Phase RunSpec projection does not bind its Retry event")
        if not isinstance(runspec, FoldingPhaseRunSpec):
            _validate_database_manifest_projection(authority.authority_path, runspec)
        else:
            if not isinstance(authority.phase_plan, FoldingPhasePlan):
                raise ValueError("folding Phase Retry requires a folding Phase Plan")
            _validate_fold_shard_projection(authority.authority_path, authority.phase_plan, runspec)
    history = PhaseAttemptHistoryView(
        attempt=predecessor,
        phase_runspec=authority.phase_runspec,
        outcome=payload.predecessor_outcome,
        submission=authority.submission,
        terminal_observations=authority.terminal_observations,
        cancellation=authority.cancellation,
        finalized_event=authority.finalized_event,
        receipt=authority.receipt,
        retry_id=payload.retry_id,
        carry_forward=authority.current_carry_forward,
    )
    lifecycle = PhaseRunLifecycleView(
        phase_run_id=authority.phase_run.phase_run_id,
        current_attempt_id=successor.attempt_id,
        attempt_status="materialized",
        run_status="materialized",
        sealed=False,
        phase_receipt_id=None,
    )
    return replace(
        authority,
        current_attempt=successor,
        phase_runspec=runspec,
        lifecycle=lifecycle,
        events=(*authority.events, event),
        prior_attempts=(*authority.prior_attempts, history),
        current_runspec_projection_complete=projection_complete,
        current_carry_forward=payload.carry_forward_record,
        current_action_evidence_attestation=None,
        submission=None,
        cancellation=None,
        terminal_observations=(),
        array_terminal_observations=(),
        finalized_event=None,
        receipt=None,
    )


def _validate_carry_retry(event: PhaseAttemptRetriedEvent, authority: PhaseAuthorityValidation) -> None:
    record = event.payload.carry_forward_record
    if record is None:
        return
    if isinstance(authority.phase_runspec, FoldingPhaseRunSpec):
        _validate_folding_carry_retry(event, authority, record)
        return
    successor = event.payload.successor_phase_runspec
    if not isinstance(successor, PhaseRunSpec):
        raise ValueError("carried Retry requires a preprocessing successor RunSpec")
    if not isinstance(record, AttemptCarryForwardRecord):
        raise ValueError("preprocessing carried Retry requires a preprocessing carry-forward record")
    if authority.current_action_evidence_attestation is None:
        raise ValueError("carried Retry requires a durable current evidence attestation")
    attestation = authority.current_action_evidence_attestation
    verification = record.verification
    if (
        verification.attestation_id != attestation.payload.attestation_id
        or verification.attestation_digest != attestation.payload.digest
        or verification.attestation_event_sequence != attestation.sequence
        or verification.evidence_path != attestation.payload.action_evidence_path
        or verification.evidence_document_sha256 != attestation.payload.evidence_document_sha256
        or verification.evidence_mapping_digest != attestation.payload.evidence_mapping_digest
        or verification.terminal_event_digest != attestation.payload.terminal_event_digest
        or verification.evidence_finished_at != attestation.payload.evidence.finished_at
    ):
        raise ValueError("carry-forward verification does not match its durable attestation")
    if (
        record.phase_run_id != authority.phase_run.phase_run_id
        or record.phase_plan_digest != authority.phase_plan.digest
        or record.source_attempt_id != authority.current_attempt.attempt_id
        or record.source_runspec_digest != authority.phase_runspec.digest
        or record.target_attempt_id != event.payload.successor_attempt.attempt_id
        or record.input_set_identity_digest != phase_input_set_identity_digest(authority.phase_plan)
        or record.scientific_identity_digest != phase_scientific_identity_digest(authority.phase_plan)
    ):
        raise ValueError("carry-forward record identity does not match replayed Retry authority")
    if (
        record.source_attempt_ordinal != authority.current_attempt.ordinal
        or record.target_attempt_ordinal != event.payload.successor_attempt.ordinal
        or record.declared_at != event.payload.successor_attempt.created_at
    ):
        raise ValueError("carry-forward record does not bind exact predecessor/successor Attempt authority")
    if (
        authority.phase_runspec.cluster.transport != successor.cluster.transport
        or authority.phase_runspec.cluster.ssh_target != successor.cluster.ssh_target
    ):
        raise ValueError("carried Retry must preserve transport and SSH endpoint")
    expected = {item.member_name: item for item in authority.phase_runspec.payload.actions[0].payload.expected_a3ms}
    evidence_hashes = {
        item.member_name: item
        for item in attestation.payload.evidence.output_hashes
        if item.role == "a3m" and item.member_name is not None
    }
    if len(record.content) >= len(expected):
        raise ValueError("carry-forward content must be a proper ExpectedA3M subset")
    for item in record.content:
        declared = expected.get(item.member_name)
        observed = evidence_hashes.get(item.member_name)
        if (
            declared is None
            or declared.source_ordinal != item.source_ordinal
            or declared.record_identity != item.record_identity
            or declared.source_header != item.source_header
            or observed is None
            or observed.size_bytes != item.size_bytes
            or observed.sha256 != item.sha256
            or observed.path != item.source_declared_path
        ):
            raise ValueError("carry-forward content is not exactly supported by attested evidence")
    carried_members = {item.member_name for item in record.content}
    expected_remaining = tuple(
        item.source_ordinal for item in expected.values() if item.member_name not in carried_members
    )
    if record.remaining_record_ordinals != expected_remaining:
        raise ValueError("carry-forward remaining-record projection is not the exact complement")
    uncarried_successor = replace(successor, carry_forward=None)
    request = AttemptCarryForwardRequest(
        source_attempt_id=authority.current_attempt.attempt_id,
        content=tuple(
            AttemptCarryForwardSelection(
                member_name=item.member_name,
                size_bytes=item.size_bytes,
                sha256=item.sha256,
            )
            for item in record.content
        ),
    )
    expected_record, expected_runspec = derive_attempt_carry_forward(
        authority=authority,
        successor_runspec=uncarried_successor,
        target_attempt_ordinal=event.payload.successor_attempt.ordinal,
        request=request,
        declared_at=event.payload.successor_attempt.created_at,
    )
    if record != expected_record or successor != expected_runspec:
        raise ValueError("carry-forward Retry projection does not match independently derived authority")
    inbound_record = authority.current_carry_forward
    if inbound_record is not None:
        if not isinstance(inbound_record, AttemptCarryForwardRecord):
            raise ValueError("preprocessing carried Retry requires a preprocessing predecessor carry record")
        adoption = attestation.payload.evidence.carry_forward_adoption
        if adoption is None or (
            adoption.attempt_carry_forward_id != inbound_record.attempt_carry_forward_id
            or adoption.attempt_carry_forward_digest != inbound_record.digest
            or adoption.phase_submission_id != attestation.payload.submission_id
        ):
            raise ValueError("carried predecessor lacks passing inbound adoption evidence")
        by_member = {item.member_name: item for item in record.content}
        adopted = {item.member_name: item for item in adoption.content}
        for inbound in inbound_record.content:
            propagated = by_member.get(inbound.member_name)
            prior = adopted.get(inbound.member_name)
            if (
                propagated is None
                or prior is None
                or propagated.size_bytes != prior.size_bytes
                or propagated.sha256 != prior.sha256
                or propagated.sha256 != inbound.sha256
            ):
                raise ValueError("carried Retry must propagate every inbound member byte-identically")


def _validate_folding_carry_retry(
    event: PhaseAttemptRetriedEvent,
    authority: PhaseAuthorityValidation,
    record: AttemptCarryForwardRecord | FoldingCarryForwardRecord,
) -> None:
    """Structurally bind one replayed folding carry record to its Retry authority.

    Replay has no transport, so this never re-scans predecessor journals: the
    scan is correctness-enforced at derivation time and the record is immutable
    once embedded in the atomic Retry event.
    """
    successor = event.payload.successor_phase_runspec
    if not isinstance(successor, FoldingPhaseRunSpec):
        raise ValueError("folding carried Retry requires a folding successor RunSpec")
    if not isinstance(record, FoldingCarryForwardRecord):
        raise ValueError("folding carried Retry requires a folding carry-forward record")
    if (
        record.phase_run_id != authority.phase_run.phase_run_id
        or record.phase_plan_digest != authority.phase_plan.digest
        or record.source_attempt_id != authority.current_attempt.attempt_id
        or record.source_runspec_digest != authority.phase_runspec.digest
        or record.target_attempt_id != event.payload.successor_attempt.attempt_id
    ):
        raise ValueError("folding carry-forward record identity does not match replayed Retry authority")
    if (
        record.source_attempt_ordinal != authority.current_attempt.ordinal
        or record.target_attempt_ordinal != event.payload.successor_attempt.ordinal
        or record.declared_at != event.payload.successor_attempt.created_at
    ):
        raise ValueError("folding carry-forward record does not bind exact predecessor/successor Attempt authority")
    if not isinstance(authority.phase_plan, FoldingPhasePlan) or record.backend != authority.phase_plan.payload.backend:
        raise ValueError("folding carry-forward record does not bind the exact folding Phase Plan backend")
    reference = successor.carry_forward
    if not isinstance(reference, FoldingCarryForwardReference):
        raise ValueError("folding carried Retry requires its successor RunSpec carry reference")
    if (
        reference.folding_carry_forward_id != record.folding_carry_forward_id
        or reference.digest != record.digest
        or record.target_attempt_id != successor.attempt_id
    ):
        raise ValueError("folding carry-forward record does not match its successor RunSpec reference")


def _replay_submission_intended_event(
    event: object,
    authority: PhaseAuthorityValidation,
) -> PhaseAuthorityValidation:
    if not isinstance(event, PhaseSubmissionIntendedEvent):
        raise TypeError("phase-submission-intended handler requires PhaseSubmissionIntendedEvent")
    if authority.submission is not None:
        raise ValueError("Phase authority contains more than one submission intent")
    _validate_submission_event_envelope(event, authority)
    runspec = authority.phase_runspec
    if isinstance(runspec, FoldingPhaseRunSpec):
        return _replay_folding_submission_intended_event(event, authority, runspec)
    payload = event.payload
    attempt = authority.current_attempt
    document_path = authority.authority_path / attempt.phase_runspec_location
    document_sha256 = hashlib.sha256(document_path.read_bytes()).hexdigest()
    tuple_id = preprocessing_runtime_tuple_id(runspec.cluster.preprocessing_runtime.qualification_tuple)
    if (
        payload.phase_runspec_location != attempt.phase_runspec_location
        or payload.phase_runspec_digest != runspec.digest
        or payload.phase_runspec_document_sha256 != document_sha256
        or payload.qualification_tuple_id != tuple_id
    ):
        raise ValueError("Phase Submission intent does not bind the authoritative RunSpec bytes and runtime")
    planned_action_ids = tuple(plan.action_id for plan in payload.actions)
    runspec_action_ids = tuple(action.action_id for action in runspec.payload.actions)
    if planned_action_ids != runspec_action_ids:
        raise ValueError("Phase Submission intent action order does not match the authoritative RunSpec")
    cluster_action_root = (
        PurePosixPath(runspec.cluster.output_root) / "bspp-phase-runs" / runspec.phase_run_id / runspec.attempt_id
    )
    legacy_cluster_action_root = (
        PurePosixPath(runspec.cluster.output_root) / "afcdb-phase-runs" / runspec.phase_run_id / runspec.attempt_id
    )
    cluster_script_root = (
        PurePosixPath(runspec.cluster.staging_root)
        / "bspp-phase-runs"
        / runspec.phase_run_id
        / runspec.attempt_id
        / "actions"
    )
    legacy_cluster_script_root = (
        PurePosixPath(runspec.cluster.staging_root)
        / "afcdb-phase-runs"
        / runspec.phase_run_id
        / runspec.attempt_id
        / "actions"
    )
    for plan, action in zip(payload.actions, runspec.payload.actions, strict=True):
        action_root = cluster_action_root / action.action_id
        legacy_action_root = legacy_cluster_action_root / action.action_id
        if (
            plan.runtime_action_digest != canonical_mapping_digest(action.to_mapping())
            or plan.dependency_action_ids != action.dependencies
            or plan.cluster_script_path
            not in (
                str(cluster_script_root / f"{action.action_id}.sbatch"),
                str(legacy_cluster_script_root / f"{action.action_id}.sbatch"),
            )
            or plan.action_evidence_path
            not in (str(action_root / "action-evidence.json"), str(legacy_action_root / "action-evidence.json"))
            or plan.handoff_path not in (str(action_root / "handoff"), str(legacy_action_root / "handoff"))
        ):
            raise ValueError("Phase Submission action plan does not bind its authoritative Runtime Action")
        record = authority.current_carry_forward
        if record is None:
            if plan.carry_forward_record_path is not None or plan.carry_forward_mounts:
                raise ValueError("no-carry Phase Attempt cannot declare carried submission inputs")
            continue
        if not isinstance(record, AttemptCarryForwardRecord):
            raise ValueError("preprocessing Phase Submission requires a preprocessing carry-forward record")
        reference = runspec.carry_forward
        if reference is None:
            raise ValueError("carried Phase Submission requires its RunSpec record reference")
        expected_record_path = str(
            PurePosixPath(runspec.cluster.staging_root)
            / "bspp-phase-runs"
            / runspec.phase_run_id
            / runspec.attempt_id
            / "attempt-carry-forward.json"
        )
        projected_record = authority.authority_path / reference.location
        record_sha256 = hashlib.sha256(projected_record.read_bytes()).hexdigest()
        if (
            plan.carry_forward_record_path != expected_record_path
            or plan.carry_forward_record_sha256 != record_sha256
            or plan.carry_forward_submission_id != payload.submission_id
        ):
            raise ValueError("carried Phase Submission does not bind its exact staged record")
        expected_mounts = {
            (expected_record_path, expected_record_path, "file", True, "carry-record"),
            (
                record.workspace.workspace_root,
                record.workspace.private_workspace_mount_path,
                "directory",
                False,
                "carry-workspace",
            ),
            *(
                (item.physical_root, item.logical_root, "directory", False, "carry-workspace")
                for item in record.workspace.roots
            ),
            *(
                (
                    item.source_physical_path,
                    item.source_private_mount_path,
                    "file",
                    True,
                    "carry-source",
                )
                for item in record.content
            ),
        }
        observed_mounts = {
            (item.source, item.target, item.source_kind, item.read_only, item.origin)
            for item in plan.carry_forward_mounts
        }
        if observed_mounts != expected_mounts or len(observed_mounts) != len(plan.carry_forward_mounts):
            raise ValueError("carried Phase Submission mount projection does not match Retry authority")
    submission = PhaseSubmissionLifecycleView(
        phase_run_id=event.phase_run_id,
        attempt_id=event.attempt_id,
        submission_id=payload.submission_id,
        phase_runspec_location=payload.phase_runspec_location,
        phase_runspec_digest=payload.phase_runspec_digest,
        phase_runspec_document_sha256=payload.phase_runspec_document_sha256,
        qualification_tuple_id=payload.qualification_tuple_id,
        actions=tuple(PhaseActionSubmissionView(plan=plan, status="planned") for plan in payload.actions),
        status="submitting",
    )
    return replace(authority, events=(*authority.events, event), submission=submission)


def _replay_folding_submission_intended_event(
    event: PhaseSubmissionIntendedEvent,
    authority: PhaseAuthorityValidation,
    runspec: FoldingPhaseRunSpec,
) -> PhaseAuthorityValidation:
    """Replay one folding submission intent against its exact re-rendered authority."""
    from bspp.orchestration.control.folding_phase_adapter import render_folding_submission_intent

    payload = event.payload
    attempt = authority.current_attempt
    document_path = authority.authority_path / attempt.phase_runspec_location
    document_sha256 = hashlib.sha256(document_path.read_bytes()).hexdigest()
    carry_record = authority.current_carry_forward
    carry_document_sha256: str | None = None
    if carry_record is not None:
        if not isinstance(carry_record, FoldingCarryForwardRecord):
            raise ValueError("folding Phase Submission requires a folding carry-forward record")
        reference = runspec.carry_forward
        if reference is None:
            raise ValueError("carried folding RunSpec lacks its carry-forward reference")
        carry_document_sha256 = hashlib.sha256((authority.authority_path / reference.location).read_bytes()).hexdigest()
    re_rendered = render_folding_submission_intent(
        phase_runspec=runspec,
        phase_runspec_location=attempt.phase_runspec_location,
        phase_runspec_document_sha256=document_sha256,
        carry_forward_record=carry_record if isinstance(carry_record, FoldingCarryForwardRecord) else None,
        carry_forward_document_sha256=carry_document_sha256,
    )
    if not isinstance(re_rendered, PhaseSubmissionIntendedPayload):
        raise ValueError("folding submission renderer must return PhaseSubmissionIntendedPayload")
    if (
        payload.phase_runspec_location != attempt.phase_runspec_location
        or payload.phase_runspec_digest != runspec.digest
        or payload.phase_runspec_document_sha256 != document_sha256
        or payload != re_rendered
    ):
        raise ValueError("Phase Submission intent does not bind the authoritative folding RunSpec bytes")
    submission = PhaseSubmissionLifecycleView(
        phase_run_id=event.phase_run_id,
        attempt_id=event.attempt_id,
        submission_id=payload.submission_id,
        phase_runspec_location=payload.phase_runspec_location,
        phase_runspec_digest=payload.phase_runspec_digest,
        phase_runspec_document_sha256=payload.phase_runspec_document_sha256,
        qualification_tuple_id=payload.qualification_tuple_id,
        actions=tuple(PhaseActionSubmissionView(plan=plan, status="planned") for plan in payload.actions),
        status="submitting",
    )
    return replace(authority, events=(*authority.events, event), submission=submission)


def _replay_cancellation_intended_event(
    event: object,
    authority: PhaseAuthorityValidation,
) -> PhaseAuthorityValidation:
    if not isinstance(event, PhaseCancellationIntendedEvent):
        raise TypeError("phase-cancellation-intended handler requires PhaseCancellationIntendedEvent")
    if classify_phase_lifecycle(authority.lifecycle) != "active":
        raise ValueError("Phase Cancellation intent requires the current active Phase Attempt")
    if authority.cancellation is not None:
        raise ValueError("Phase authority contains more than one cancellation intent")
    if event.phase_run_id != authority.phase_run.phase_run_id or event.attempt_id != authority.phase_runspec.attempt_id:
        raise ValueError("Phase Cancellation intent does not bind the authoritative run and attempt")
    if event.payload.phase_runspec_digest != authority.phase_runspec.digest:
        raise ValueError("Phase Cancellation intent does not bind the authoritative RunSpec")
    declared_ids = tuple(action.action_id for action in authority.phase_runspec.payload.actions)
    if authority.submission is None:
        expected_targets = tuple(
            PhaseCancellationActionTarget(action_id=action_id, initial_submission_status="not-submitted")
            for action_id in declared_ids
        )
        if event.payload.submission_id is not None:
            raise ValueError("Phase Cancellation intent references absent submission authority")
    else:
        if event.payload.submission_id != authority.submission.submission_id:
            raise ValueError("Phase Cancellation intent does not bind the durable submission")
        if any(action.status == "rejected" for action in authority.submission.actions):
            raise ValueError("active Phase Cancellation cannot target a rejected submission")
        expected_targets = tuple(
            PhaseCancellationActionTarget(
                action_id=action.action_id,
                initial_submission_status=cast(
                    "Literal['planned', 'dispatching', 'submitted', 'satisfied']", action.status
                ),
                scheduler_correlation_token=action.scheduler_correlation_token,
                job_id=action.job_id,
            )
            for action in authority.submission.actions
        )
    if (
        event.payload.targets != expected_targets
        or tuple(target.action_id for target in expected_targets) != declared_ids
    ):
        raise ValueError("Phase Cancellation targets do not freeze the exact authoritative action order and state")
    terminal_by_action = {item.action_id: item for item in authority.terminal_observations}
    actions: list[PhaseCancellationActionView] = []
    for target in expected_targets:
        terminal = terminal_by_action.get(target.action_id)
        if terminal is not None:
            terminal_event = _terminal_event(authority, target.action_id)
            actions.append(
                PhaseCancellationActionView(
                    target=target,
                    disposition="terminal-confirmed",
                    bound_job_id=terminal.job_id,
                    terminal_observation_digest=canonical_mapping_digest(terminal_event.to_mapping()),
                )
            )
        elif target.initial_submission_status in {"not-submitted", "planned", "satisfied"}:
            actions.append(PhaseCancellationActionView(target=target, disposition="no-job"))
        elif target.initial_submission_status == "dispatching":
            actions.append(PhaseCancellationActionView(target=target, disposition="correlating"))
        else:
            actions.append(
                PhaseCancellationActionView(
                    target=target,
                    disposition="cancel-pending",
                    bound_job_id=target.job_id,
                )
            )
    cancellation = PhaseCancellationLifecycleView(
        phase_run_id=event.phase_run_id,
        attempt_id=event.attempt_id,
        cancellation_id=event.payload.cancellation_id,
        submission_id=event.payload.submission_id,
        phase_runspec_digest=event.payload.phase_runspec_digest,
        actions=tuple(actions),
        status="cancelling",
    )
    return replace(
        authority,
        lifecycle=_cancelling_lifecycle(authority),
        events=(*authority.events, event),
        cancellation=cancellation,
    )


def _replay_job_cancellation_request_intended_event(
    event: object,
    authority: PhaseAuthorityValidation,
) -> PhaseAuthorityValidation:
    if not isinstance(event, PhaseJobCancellationRequestIntendedEvent):
        raise TypeError("cancellation request intent handler requires PhaseJobCancellationRequestIntendedEvent")
    _require_cancellation_event_envelope(event, authority)
    cancellation, index, action = _cancellation_action(authority, event.payload.action_id)
    if action.disposition != "cancel-pending" or action.bound_job_id is None:
        raise ValueError("scancel request intent requires one pending job-bound cancellation target")
    expected_ordinal = len(action.requests) + 1
    expected_argv = command_argv(
        ("scancel", action.bound_job_id),
        transport=authority.phase_runspec.cluster.transport,
        ssh_target=authority.phase_runspec.cluster.ssh_target,
    )
    if (
        event.payload.job_id != action.bound_job_id
        or event.payload.request_ordinal != expected_ordinal
        or event.payload.scancel_argv != expected_argv
    ):
        raise ValueError("scancel request intent does not bind the exact next target command")
    request = PhaseJobCancellationRequestView(
        request_ordinal=event.payload.request_ordinal,
        scancel_argv=event.payload.scancel_argv,
    )
    updated = replace(action, disposition="cancel-requesting", requests=(*action.requests, request))
    return _replace_cancellation_action(authority, event, cancellation, index, updated)


def _replay_job_cancellation_request_result_event(
    event: object,
    authority: PhaseAuthorityValidation,
) -> PhaseAuthorityValidation:
    if not isinstance(event, PhaseJobCancellationRequestResultEvent):
        raise TypeError("cancellation request result handler requires PhaseJobCancellationRequestResultEvent")
    _require_cancellation_event_envelope(event, authority)
    cancellation, index, action = _cancellation_action(authority, event.payload.action_id)
    if action.disposition != "cancel-requesting" or action.bound_job_id is None or not action.requests:
        raise ValueError("scancel request result requires one unresolved durable request intent")
    intended = action.requests[-1]
    if (
        event.payload.job_id != action.bound_job_id
        or event.payload.request_ordinal != intended.request_ordinal
        or event.payload.scancel_argv != intended.scancel_argv
    ):
        raise ValueError("scancel request result does not bind its exact durable request intent")
    result = replace(
        intended,
        return_code=event.payload.return_code,
        stdout=event.payload.stdout,
        stderr=event.payload.stderr,
    )
    disposition: PhaseCancellationActionDisposition = (
        "cancel-requested" if result.return_code == 0 else "cancel-pending"
    )
    updated = replace(action, disposition=disposition, requests=(*action.requests[:-1], result))
    return _replace_cancellation_action(authority, event, cancellation, index, updated)


def _replay_cancellation_completed_event(
    event: object,
    authority: PhaseAuthorityValidation,
) -> PhaseAuthorityValidation:
    if not isinstance(event, PhaseCancellationCompletedEvent):
        raise TypeError("phase-cancellation-completed handler requires PhaseCancellationCompletedEvent")
    _require_cancellation_event_envelope(event, authority)
    assert authority.cancellation is not None
    if any(action.disposition not in {"no-job", "terminal-confirmed"} for action in authority.cancellation.actions):
        raise ValueError("Phase Cancellation cannot complete before every target is terminal or job-free")
    expected = tuple(
        PhaseCancellationTerminalReference(
            action_id=action.action_id,
            job_id=action.bound_job_id,
            terminal_observation_digest=action.terminal_observation_digest,
        )
        for action in authority.cancellation.actions
        if action.disposition == "terminal-confirmed"
        and action.bound_job_id is not None
        and action.terminal_observation_digest is not None
    )
    if event.payload.terminal_references != expected:
        raise ValueError("Phase Cancellation completion does not reference exact terminal evidence in target order")
    cancellation = replace(authority.cancellation, status="cancelled")
    return replace(
        authority,
        lifecycle=_cancelled_lifecycle(authority),
        events=(*authority.events, event),
        cancellation=cancellation,
    )


def _replay_action_dispatch_intended_event(
    event: object,
    authority: PhaseAuthorityValidation,
) -> PhaseAuthorityValidation:
    if not isinstance(event, PhaseActionDispatchIntendedEvent):
        raise TypeError("phase-action-dispatch-intended handler requires PhaseActionDispatchIntendedEvent")
    _validate_submission_event_envelope(event, authority)
    submission, index, view = _submission_action(authority, event.payload.submission_id, event.payload.action_id)
    if view.status != "planned":
        raise ValueError("Runtime Action may be dispatch-intended exactly once from planned state")
    expected_dependency_job_ids = _dependency_job_ids(submission, view)
    if (
        event.payload.script_sha256 != view.plan.script_sha256
        or event.payload.scheduler_correlation_token != view.plan.scheduler_correlation_token
        or event.payload.dependency_job_ids != expected_dependency_job_ids
    ):
        raise ValueError("dispatch intent does not bind its rendered action and declared dependency jobs")
    updated = replace(view, status="dispatching", dependency_job_ids=expected_dependency_job_ids)
    return _replace_submission_action(authority, event, submission, index, updated)


def _replay_action_submitted_event(
    event: object,
    authority: PhaseAuthorityValidation,
) -> PhaseAuthorityValidation:
    if not isinstance(event, PhaseActionSubmittedEvent):
        raise TypeError("phase-action-submitted handler requires PhaseActionSubmittedEvent")
    classification = classify_phase_lifecycle(authority.lifecycle)
    if classification == "active":
        _validate_submission_event_envelope(event, authority)
    elif classification == "cancelling":
        if (
            event.phase_run_id != authority.phase_run.phase_run_id
            or event.attempt_id != authority.phase_runspec.attempt_id
        ):
            raise ValueError("Phase Submission event does not bind the authoritative run and attempt")
        cancellation, _cancel_index, cancel_action = _cancellation_action(authority, event.payload.action_id)
        if (
            cancel_action.target.initial_submission_status != "dispatching"
            or cancel_action.disposition != "correlating"
            or cancellation.submission_id != event.payload.submission_id
        ):
            raise ValueError("post-cancellation assignment is permitted only for exact dispatch correlation recovery")
    else:
        raise ValueError(f"Phase Submission event cannot cross a {classification} Phase Attempt boundary")
    submission, index, view = _submission_action(authority, event.payload.submission_id, event.payload.action_id)
    _validate_dispatch_outcome(event.payload, view, authority=authority, occurred_at=event.occurred_at)
    if event.payload.expected_task_indexes != view.plan.expected_task_indexes:
        raise ValueError("submitted payload expected task indexes do not bind the frozen action plan")
    if any(item.job_id == event.payload.job_id for item in submission.actions if item.action_id != view.action_id):
        raise ValueError("Slurm job id is already assigned to another Runtime Action")
    updated = replace(
        view,
        status="submitted",
        job_id=event.payload.job_id,
        sbatch_argv=event.payload.sbatch_argv,
    )
    assigned = _replace_submission_action(authority, event, submission, index, updated)
    if classification == "cancelling":
        assert assigned.cancellation is not None
        cancellation, cancel_index, cancel_action = _cancellation_action(assigned, event.payload.action_id)
        rebound = replace(cancel_action, disposition="cancel-pending", bound_job_id=event.payload.job_id)
        actions = (
            *cancellation.actions[:cancel_index],
            rebound,
            *cancellation.actions[cancel_index + 1 :],
        )
        assigned = replace(assigned, cancellation=replace(cancellation, actions=actions))
    return assigned


def _replay_action_dispatch_rejected_event(
    event: object,
    authority: PhaseAuthorityValidation,
) -> PhaseAuthorityValidation:
    if not isinstance(event, PhaseActionDispatchRejectedEvent):
        raise TypeError("phase-action-dispatch-rejected handler requires PhaseActionDispatchRejectedEvent")
    _validate_submission_event_envelope(event, authority)
    submission, index, view = _submission_action(authority, event.payload.submission_id, event.payload.action_id)
    _validate_dispatch_outcome(event.payload, view, authority=authority, occurred_at=event.occurred_at)
    updated = replace(
        view,
        status="rejected",
        sbatch_argv=event.payload.sbatch_argv,
        return_code=event.payload.return_code,
        stdout=event.payload.stdout,
        stderr=event.payload.stderr,
    )
    rejected = _replace_submission_action(authority, event, submission, index, updated)
    return replace(rejected, lifecycle=_failed_lifecycle(authority))


def _load_fold_shard_projection(
    authority: PhaseAuthorityValidation,
    binding: FoldShardProjectionBinding,
) -> FoldShardProjection:
    """Load and verify the canonical fold shard projection from its binding."""
    projection_path = authority.authority_path / binding.location
    try:
        raw = projection_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read fold shard projection {projection_path}: {exc}") from exc
    if hashlib.sha256(raw).hexdigest() != binding.sha256 or len(raw) != binding.size_bytes:
        raise ValueError("fold shard projection does not bind its authoritative bytes")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed fold shard projection {projection_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("fold shard projection must be a JSON object")
    projection = fold_shard_projection_from_mapping(payload)
    if projection.worker_count != binding.worker_count or projection.lpt_version != binding.lpt_version:
        raise ValueError("fold shard projection binding mismatch")
    return projection


def _replay_action_satisfied_without_dispatch_event(
    event: object,
    authority: PhaseAuthorityValidation,
) -> PhaseAuthorityValidation:
    if not isinstance(event, PhaseActionSatisfiedWithoutDispatchEvent):
        raise TypeError(
            "phase-action-satisfied-without-dispatch handler requires PhaseActionSatisfiedWithoutDispatchEvent"
        )
    _validate_submission_event_envelope(event, authority)
    submission, index, view = _submission_action(authority, event.payload.submission_id, event.payload.action_id)
    if view.status != "planned":
        raise ValueError("Runtime Action may be satisfied-without-dispatch exactly once from planned state")
    runspec = authority.phase_runspec
    if not isinstance(runspec, FoldingPhaseRunSpec):
        raise ValueError("no-dispatch satisfaction requires a folding Phase RunSpec")
    if event.payload.phase_runspec_digest != runspec.digest:
        raise ValueError("satisfied action does not bind the authoritative RunSpec")
    if (
        event.payload.runtime_action_digest != view.plan.runtime_action_digest
        or event.payload.scheduler_correlation_token != view.plan.scheduler_correlation_token
    ):
        raise ValueError("satisfied action does not bind its rendered action plan")
    record = authority.current_carry_forward
    if not isinstance(record, FoldingCarryForwardRecord):
        raise ValueError("satisfied action requires a sealed folding carry record")
    if event.payload.carry_record_digest != record.digest:
        raise ValueError("satisfied action does not bind the sealed carry record")
    binding = runspec.payload.fold_shard_projection
    if binding is None:
        raise ValueError("satisfied action requires a fold shard projection binding")
    if event.payload.shard_manifest_digest != binding.sha256:
        raise ValueError("satisfied action does not bind the fold shard projection")
    projection = _load_fold_shard_projection(authority, binding)
    expected_targets = tuple(target.target_id for rank in projection.ranks for target in rank.targets)
    if event.payload.carried_target_ids != expected_targets:
        raise ValueError("satisfied action carried closure does not match the canonical shard projection")
    updated = replace(view, status="satisfied", dependency_job_ids=(), job_id=None, sbatch_argv=())
    return _replace_submission_action(authority, event, submission, index, updated)


def _replay_action_terminal_observed_event(
    event: object,
    authority: PhaseAuthorityValidation,
) -> PhaseAuthorityValidation:
    if not isinstance(event, PhaseActionTerminalObservedEvent):
        raise TypeError("phase-action-terminal-observed handler requires PhaseActionTerminalObservedEvent")
    classification = classify_phase_lifecycle(authority.lifecycle)
    if classification not in {"active", "cancelling"}:
        raise ValueError(f"terminal observation cannot cross a {classification} Phase Attempt boundary")
    if event.phase_run_id != authority.phase_run.phase_run_id or event.attempt_id != authority.phase_runspec.attempt_id:
        raise ValueError("terminal observation event does not bind the authoritative run and attempt")
    submission, _index, action = _submission_action(
        authority,
        event.payload.submission_id,
        event.payload.action_id,
    )
    if action.status != "submitted" or action.job_id is None:
        raise ValueError("terminal observation requires an exact durable submitted action")
    if any(item.action_id == action.action_id for item in authority.terminal_observations):
        raise ValueError("Runtime Action may have exactly one durable terminal observation")
    if (
        event.payload.phase_runspec_digest != authority.phase_runspec.digest
        or event.payload.runtime_action_digest != action.plan.runtime_action_digest
        or event.payload.scheduler_correlation_token != action.scheduler_correlation_token
        or event.payload.job_id != action.job_id
        or submission.submission_id != event.payload.submission_id
    ):
        raise ValueError("terminal observation does not bind the exact submitted Runtime Action")
    terminal = PhaseActionTerminalObservationView.from_event(event)
    lifecycle = (
        _failed_lifecycle(authority)
        if classification == "active" and terminal.outcome == "failed"
        else authority.lifecycle
    )
    observed = replace(
        authority,
        lifecycle=lifecycle,
        events=(*authority.events, event),
        terminal_observations=(*authority.terminal_observations, terminal),
    )
    if classification == "cancelling":
        cancellation, index, cancel_action = _cancellation_action(authority, event.payload.action_id)
        if cancel_action.bound_job_id != event.payload.job_id or cancel_action.disposition in {
            "no-job",
            "correlating",
            "terminal-confirmed",
        }:
            raise ValueError("cancellation-owned terminal observation requires its exact bound pending job")
        updated = replace(
            cancel_action,
            disposition="terminal-confirmed",
            terminal_observation_digest=canonical_mapping_digest(event.to_mapping()),
        )
        actions = (*cancellation.actions[:index], updated, *cancellation.actions[index + 1 :])
        observed = replace(observed, cancellation=replace(cancellation, actions=actions))
    return observed


def _replay_folding_array_terminal_observed_event(
    event: object,
    authority: PhaseAuthorityValidation,
) -> PhaseAuthorityValidation:
    if not isinstance(event, FoldingActionTerminalObservedEvent):
        raise TypeError("phase-action-array-terminal-observed handler requires FoldingActionTerminalObservedEvent")
    classification = classify_phase_lifecycle(authority.lifecycle)
    if classification != "active":
        raise ValueError(f"folding array terminal observation cannot cross a {classification} Phase Attempt boundary")
    if event.phase_run_id != authority.phase_run.phase_run_id or event.attempt_id != authority.phase_runspec.attempt_id:
        raise ValueError("folding array terminal observation event does not bind the authoritative run and attempt")
    submission, _index, action = _submission_action(
        authority,
        event.payload.submission_id,
        event.payload.action_id,
    )
    if action.status != "submitted" or action.job_id is None:
        raise ValueError("folding array terminal observation requires an exact durable submitted action")
    if any(item.action_id == action.action_id for item in authority.array_terminal_observations):
        raise ValueError("Runtime Action may have exactly one durable folding array terminal observation")
    if (
        event.payload.phase_runspec_digest != authority.phase_runspec.digest
        or event.payload.runtime_action_digest != action.plan.runtime_action_digest
        or event.payload.parent_job_id != action.job_id
        or event.payload.expected_task_indexes != action.plan.expected_task_indexes
        or submission.submission_id != event.payload.submission_id
    ):
        raise ValueError("folding array terminal observation does not bind the exact submitted Runtime Action")
    view = FoldingActionTerminalObservationView.from_event(event)
    lifecycle = _failed_lifecycle(authority) if view.outcome == "failed" else authority.lifecycle
    return replace(
        authority,
        lifecycle=lifecycle,
        events=(*authority.events, event),
        array_terminal_observations=(*authority.array_terminal_observations, view),
    )


def _replay_action_evidence_attested_event(
    event: object,
    authority: PhaseAuthorityValidation,
) -> PhaseAuthorityValidation:
    if not isinstance(event, PhaseActionEvidenceAttestedEvent):
        raise TypeError("phase-action-evidence-attested handler requires PhaseActionEvidenceAttestedEvent")
    if authority.current_action_evidence_attestation is not None:
        raise ValueError("Phase Attempt may have exactly one durable action-evidence attestation")
    if classify_phase_lifecycle(authority.lifecycle) not in {"failed", "cancelled"}:
        raise ValueError("action-evidence attestation requires the current retryable Phase Attempt")
    if (
        event.phase_run_id != authority.phase_run.phase_run_id
        or event.attempt_id != authority.current_attempt.attempt_id
    ):
        raise ValueError("action-evidence attestation does not bind the authoritative run and attempt")
    _submission, _index, action = _submission_action(
        authority,
        event.payload.submission_id,
        event.payload.action_id,
    )
    if action.status != "submitted" or action.job_id is None:
        raise ValueError("action-evidence attestation requires an exact durable submitted action")
    terminal = _terminal_event(authority, action.action_id)
    payload = event.payload
    if (
        payload.phase_runspec_digest != authority.phase_runspec.digest
        or payload.runtime_action_digest != action.plan.runtime_action_digest
        or payload.scheduler_correlation_token != action.scheduler_correlation_token
        or payload.job_id != action.job_id
        or payload.action_evidence_path != action.plan.action_evidence_path
        or payload.terminal_event_sequence != terminal.sequence
        or payload.terminal_event_digest != canonical_mapping_digest(terminal.to_mapping())
        or payload.terminal_state != terminal.payload.state
        or payload.terminal_outcome != terminal.payload.outcome
        or payload.terminal_observed_at != terminal.occurred_at
        or payload.evidence.phase_run_id != authority.phase_run.phase_run_id
        or payload.evidence.attempt_id != authority.current_attempt.attempt_id
        or payload.evidence.phase_runspec_digest != authority.phase_runspec.digest
        or payload.evidence.action_id != action.action_id
    ):
        raise ValueError("action-evidence attestation does not match durable submission and terminal authority")
    return replace(
        authority,
        events=(*authority.events, event),
        current_action_evidence_attestation=event,
    )


def _validate_submission_event_envelope(
    event: object,
    authority: PhaseAuthorityValidation,
) -> None:
    if (
        getattr(event, "phase_run_id", None) != authority.phase_run.phase_run_id
        or getattr(event, "attempt_id", None) != authority.phase_runspec.attempt_id
    ):
        raise ValueError("Phase Submission event does not bind the authoritative run and attempt")
    _require_active_lifecycle(authority, operation="Phase Submission event")


def _require_active_lifecycle(authority: PhaseAuthorityValidation, *, operation: str) -> None:
    lifecycle = authority.lifecycle
    if (
        lifecycle.attempt_status != "materialized"
        or lifecycle.run_status != "materialized"
        or lifecycle.sealed
        or lifecycle.phase_receipt_id is not None
    ):
        raise ValueError(f"{operation} requires the current active Phase Attempt")


def _failed_lifecycle(authority: PhaseAuthorityValidation) -> PhaseRunLifecycleView:
    return PhaseRunLifecycleView(
        phase_run_id=authority.phase_run.phase_run_id,
        current_attempt_id=authority.current_attempt.attempt_id,
        attempt_status="failed",
        run_status="failed",
        sealed=False,
        phase_receipt_id=None,
    )


def _cancelling_lifecycle(authority: PhaseAuthorityValidation) -> PhaseRunLifecycleView:
    return PhaseRunLifecycleView(
        phase_run_id=authority.phase_run.phase_run_id,
        current_attempt_id=authority.current_attempt.attempt_id,
        attempt_status="cancelling",
        run_status="cancelling",
        sealed=False,
        phase_receipt_id=None,
    )


def _cancelled_lifecycle(authority: PhaseAuthorityValidation) -> PhaseRunLifecycleView:
    return PhaseRunLifecycleView(
        phase_run_id=authority.phase_run.phase_run_id,
        current_attempt_id=authority.current_attempt.attempt_id,
        attempt_status="cancelled",
        run_status="cancelled",
        sealed=False,
        phase_receipt_id=None,
    )


def _require_cancellation_event_envelope(event: object, authority: PhaseAuthorityValidation) -> None:
    if classify_phase_lifecycle(authority.lifecycle) != "cancelling" or authority.cancellation is None:
        raise ValueError("Phase Cancellation event requires the current cancelling Phase Attempt")
    if (
        getattr(event, "phase_run_id", None) != authority.phase_run.phase_run_id
        or getattr(event, "attempt_id", None) != authority.phase_runspec.attempt_id
        or getattr(getattr(event, "payload", None), "cancellation_id", None) != authority.cancellation.cancellation_id
    ):
        raise ValueError("Phase Cancellation event does not bind the exact cancellation authority")


def _cancellation_action(
    authority: PhaseAuthorityValidation,
    action_id: str,
) -> tuple[PhaseCancellationLifecycleView, int, PhaseCancellationActionView]:
    cancellation = authority.cancellation
    if cancellation is None:
        raise ValueError("Phase authority has no durable cancellation intent")
    for index, action in enumerate(cancellation.actions):
        if action.action_id == action_id:
            return cancellation, index, action
    raise ValueError("Phase Cancellation event references an undeclared target")


def _replace_cancellation_action(
    authority: PhaseAuthorityValidation,
    event: object,
    cancellation: PhaseCancellationLifecycleView,
    index: int,
    action: PhaseCancellationActionView,
) -> PhaseAuthorityValidation:
    actions = (*cancellation.actions[:index], action, *cancellation.actions[index + 1 :])
    return replace(
        authority,
        events=(*authority.events, event),
        cancellation=replace(cancellation, actions=actions),
    )


def _terminal_event(authority: PhaseAuthorityValidation, action_id: str) -> PhaseActionTerminalObservedEvent:
    submission = authority.submission
    if submission is None:
        raise ValueError("durable terminal observation requires current submission authority")
    matches = tuple(
        event
        for event in authority.events
        if (
            isinstance(event, PhaseActionTerminalObservedEvent)
            and event.phase_run_id == authority.phase_run.phase_run_id
            and event.attempt_id == authority.current_attempt.attempt_id
            and event.payload.submission_id == submission.submission_id
            and event.payload.action_id == action_id
        )
    )
    if len(matches) != 1:
        raise ValueError("durable terminal observation view lacks exactly one source event")
    return matches[0]


def _submission_action(
    authority: PhaseAuthorityValidation,
    submission_id: str,
    action_id: str,
) -> tuple[PhaseSubmissionLifecycleView, int, PhaseActionSubmissionView]:
    submission = authority.submission
    if submission is None or submission.submission_id != submission_id:
        raise ValueError("Phase Action event does not bind the durable submission intent")
    for index, view in enumerate(submission.actions):
        if view.action_id == action_id:
            return submission, index, view
    raise ValueError("Phase Action event references an undeclared action")


def _dependency_job_ids(
    submission: PhaseSubmissionLifecycleView,
    view: PhaseActionSubmissionView,
) -> tuple[str, ...]:
    by_id = {item.action_id: item for item in submission.actions}
    result: list[str] = []
    for dependency_id in view.plan.dependency_action_ids:
        dependency = by_id[dependency_id]
        if dependency.status == "satisfied":
            continue
        if dependency.status != "submitted" or dependency.job_id is None:
            raise ValueError("Runtime Action cannot dispatch before every declared dependency is assigned")
        result.append(dependency.job_id)
    return tuple(result)


_LEGACY_WIRE_SHAPE_CUTOVER = datetime.fromisoformat(LEGACY_WIRE_SHAPE_CUTOVER_AT.removesuffix("Z") + "+00:00")


def _legacy_wire_shape_era(occurred_at: str) -> bool:
    """True only when the event predates the base64 wire-shape cutover.

    Fails closed: an unparseable timestamp is never in the legacy era.
    """
    try:
        occurred = datetime.fromisoformat(occurred_at.removesuffix("Z") + "+00:00")
    except ValueError:
        return False
    return occurred < _LEGACY_WIRE_SHAPE_CUTOVER


def _validate_dispatch_outcome(
    payload: object,
    view: PhaseActionSubmissionView,
    *,
    authority: PhaseAuthorityValidation,
    occurred_at: str,
) -> None:
    if view.status != "dispatching":
        raise ValueError("Runtime Action outcome requires exactly one prior dispatch intent")
    if (
        getattr(payload, "script_sha256", None) != view.plan.script_sha256
        or getattr(payload, "scheduler_correlation_token", None) != view.plan.scheduler_correlation_token
        or getattr(payload, "dependency_job_ids", None) != view.dependency_job_ids
    ):
        raise ValueError("Runtime Action outcome does not bind its exact dispatch intent")
    sbatch: tuple[str, ...] = ("sbatch", "--parsable")
    if view.dependency_job_ids:
        sbatch = (*sbatch, "--dependency=afterok:" + ":".join(view.dependency_job_ids))
    expected_argv = command_argv(
        (*sbatch, view.plan.cluster_script_path),
        transport=authority.phase_runspec.cluster.transport,
        ssh_target=authority.phase_runspec.cluster.ssh_target,
    )
    recorded_argv = getattr(payload, "sbatch_argv", None)
    if recorded_argv == expected_argv:
        return
    # Sealed legacy acceptance evidence records the legacy login-shell wire
    # shape.  Tolerate that shape only for events recorded before the cutover,
    # so current authorities keep the exact current-shape invocation guarantee.
    legacy_argv = legacy_command_argv(
        (*sbatch, view.plan.cluster_script_path),
        transport=authority.phase_runspec.cluster.transport,
        ssh_target=authority.phase_runspec.cluster.ssh_target,
    )
    if not (_legacy_wire_shape_era(occurred_at) and recorded_argv == legacy_argv):
        raise ValueError("Runtime Action outcome does not record the exact invoked sbatch argv")


def _replace_submission_action(
    authority: PhaseAuthorityValidation,
    event: object,
    submission: PhaseSubmissionLifecycleView,
    index: int,
    action: PhaseActionSubmissionView,
) -> PhaseAuthorityValidation:
    actions = (*submission.actions[:index], action, *submission.actions[index + 1 :])
    status: PhaseSubmissionStatus = "failed" if any(item.status == "rejected" for item in actions) else "submitting"
    if all(item.status in {"submitted", "satisfied"} for item in actions):
        status = "submitted"
    updated = replace(submission, actions=actions, status=status)
    return replace(authority, events=(*authority.events, event), submission=updated)


def _replay_finalized_event(
    event: object,
    authority: PhaseAuthorityValidation,
) -> PhaseAuthorityValidation:
    if not isinstance(event, PhaseFinalizedEvent):
        raise TypeError("phase-finalized handler requires PhaseFinalizedEvent")
    if authority.lifecycle.sealed or authority.finalized_event is not None or authority.receipt is not None:
        raise ValueError("Phase authority contains more than one terminal finalization event")
    _require_active_lifecycle(authority, operation="Phase Finalization")
    _validate_finalized_against_authority(event, authority)
    receipt = event.payload.receipt
    lifecycle = PhaseRunLifecycleView(
        phase_run_id=authority.phase_run.phase_run_id,
        current_attempt_id=authority.current_attempt.attempt_id,
        attempt_status="succeeded",
        run_status="accepted",
        sealed=True,
        phase_receipt_id=receipt.phase_receipt_id,
    )
    return replace(
        authority,
        lifecycle=lifecycle,
        events=(*authority.events, event),
        finalized_event=event,
        receipt=receipt,
    )


def _discover_event_files(
    path: Path,
    *,
    event_handlers: Mapping[str, PhaseAuthorityEventHandler],
    observed_files: set[Path] | None = None,
) -> tuple[Path, ...]:
    files = observed_files
    if files is None:
        events_path = path / "events"
        if events_path.is_symlink() or not events_path.is_dir():
            raise ValueError("Phase authority events must be an immediate regular directory")
        files = set()
        for candidate in events_path.iterdir():
            if candidate.is_symlink() or not candidate.is_file():
                raise ValueError(f"unexpected Phase event path: {candidate.relative_to(path)}")
            files.add(candidate.relative_to(path))
    candidates = tuple(sorted(item for item in files if item.parts and item.parts[0] == "events"))
    by_sequence: dict[int, Path] = {}
    event_types: dict[int, str] = {}
    for relative in candidates:
        if len(relative.parts) != 2:
            raise ValueError(f"unexpected Phase event path: {relative}")
        match = _EVENT_NAME.fullmatch(relative.name)
        if match is None:
            raise ValueError(f"unknown or malformed Phase event filename: {relative.name}")
        sequence = int(match.group("sequence"))
        event_type = match.group("event_type")
        if event_type not in event_handlers:
            raise ValueError(f"unknown Phase event type: {event_type!r}")
        if sequence in by_sequence:
            raise ValueError(f"duplicate Phase event sequence: {sequence:06d}")
        by_sequence[sequence] = relative
        event_types[sequence] = event_type
    if not by_sequence or by_sequence.get(1) != _MATERIALIZED_EVENT:
        raise ValueError("Phase authority must begin with 000001-phase-materialized.json")
    expected_sequences = tuple(range(1, max(by_sequence) + 1))
    if tuple(sorted(by_sequence)) != expected_sequences:
        raise ValueError("Phase authority event sequence must be contiguous from 000001")
    terminal_seen = False
    for sequence in expected_sequences:
        event_type = event_types[sequence]
        if sequence == 1 and event_type != "phase-materialized":
            raise ValueError("the first Phase event must be phase-materialized")
        if sequence > 1 and event_type == "phase-materialized":
            raise ValueError("phase-materialized may appear only at sequence 000001")
        if terminal_seen:
            raise ValueError("Phase authority contains an event after a terminal event")
        mapping = _read_json(path / by_sequence[sequence])
        if mapping.get("event_type") != event_type or mapping.get("sequence") != sequence:
            raise ValueError("Phase event filename does not match its document discriminator or sequence")
        terminal_seen = event_handlers[event_type].terminal
    return tuple(by_sequence[sequence] for sequence in expected_sequences)


def _event_handler_registry(
    additional: tuple[PhaseAuthorityEventHandler, ...],
) -> Mapping[str, PhaseAuthorityEventHandler]:
    handlers = {
        "phase-materialized": PhaseAuthorityEventHandler(
            event_type="phase-materialized",
            loader=phase_materialized_event_from_mapping,
            replay=_replay_materialized_event,
        ),
        "phase-submission-intended": PhaseAuthorityEventHandler(
            event_type="phase-submission-intended",
            loader=phase_submission_intended_event_from_mapping,
            replay=_replay_submission_intended_event,
        ),
        "phase-action-dispatch-intended": PhaseAuthorityEventHandler(
            event_type="phase-action-dispatch-intended",
            loader=phase_action_dispatch_intended_event_from_mapping,
            replay=_replay_action_dispatch_intended_event,
        ),
        "phase-action-submitted": PhaseAuthorityEventHandler(
            event_type="phase-action-submitted",
            loader=phase_action_submitted_event_from_mapping,
            replay=_replay_action_submitted_event,
        ),
        "phase-action-dispatch-rejected": PhaseAuthorityEventHandler(
            event_type="phase-action-dispatch-rejected",
            loader=phase_action_dispatch_rejected_event_from_mapping,
            replay=_replay_action_dispatch_rejected_event,
        ),
        "phase-action-satisfied-without-dispatch": PhaseAuthorityEventHandler(
            event_type="phase-action-satisfied-without-dispatch",
            loader=phase_action_satisfied_without_dispatch_event_from_mapping,
            replay=_replay_action_satisfied_without_dispatch_event,
        ),
        "phase-action-terminal-observed": PhaseAuthorityEventHandler(
            event_type="phase-action-terminal-observed",
            loader=phase_action_terminal_observed_event_from_mapping,
            replay=_replay_action_terminal_observed_event,
        ),
        "phase-action-array-terminal-observed": PhaseAuthorityEventHandler(
            event_type="phase-action-array-terminal-observed",
            loader=folding_action_terminal_observed_event_from_mapping,
            replay=_replay_folding_array_terminal_observed_event,
        ),
        "phase-action-evidence-attested": PhaseAuthorityEventHandler(
            event_type="phase-action-evidence-attested",
            loader=phase_action_evidence_attested_event_from_mapping,
            replay=_replay_action_evidence_attested_event,
        ),
        "phase-cancellation-intended": PhaseAuthorityEventHandler(
            event_type="phase-cancellation-intended",
            loader=phase_cancellation_intended_event_from_mapping,
            replay=_replay_cancellation_intended_event,
        ),
        "phase-job-cancellation-request-intended": PhaseAuthorityEventHandler(
            event_type="phase-job-cancellation-request-intended",
            loader=phase_job_cancellation_request_intended_event_from_mapping,
            replay=_replay_job_cancellation_request_intended_event,
        ),
        "phase-job-cancellation-request-result": PhaseAuthorityEventHandler(
            event_type="phase-job-cancellation-request-result",
            loader=phase_job_cancellation_request_result_event_from_mapping,
            replay=_replay_job_cancellation_request_result_event,
        ),
        "phase-cancellation-completed": PhaseAuthorityEventHandler(
            event_type="phase-cancellation-completed",
            loader=phase_cancellation_completed_event_from_mapping,
            replay=_replay_cancellation_completed_event,
        ),
        "phase-attempt-retried": PhaseAuthorityEventHandler(
            event_type="phase-attempt-retried",
            loader=phase_attempt_retried_event_from_mapping,
            replay=_replay_attempt_retried_event,
            terminal=False,
        ),
        "phase-finalized": PhaseAuthorityEventHandler(
            event_type="phase-finalized",
            loader=phase_finalized_event_from_mapping,
            replay=_replay_finalized_event,
            terminal=True,
        ),
    }
    for handler in additional:
        if handler.event_type in handlers:
            raise ValueError(f"duplicate Phase authority event handler: {handler.event_type}")
        handlers[handler.event_type] = handler
    return handlers


def _event_type_from_path(path: Path) -> str:
    match = _EVENT_NAME.fullmatch(path.name)
    if match is None:
        raise ValueError(f"unknown or malformed Phase event filename: {path.name}")
    return match.group("event_type")


def _load_registered_event(
    path: Path,
    event_handlers: Mapping[str, PhaseAuthorityEventHandler],
) -> object:
    event_type = _event_type_from_path(path)
    event = event_handlers[event_type].loader(_read_json(path))
    sequence = int(path.name.split("-", maxsplit=1)[0])
    if getattr(event, "event_type", None) != event_type or getattr(event, "sequence", None) != sequence:
        raise ValueError("registered Phase event loader returned a mismatched event")
    return event


def _validate_finalized_against_authority(
    event: PhaseFinalizedEvent,
    authority: PhaseAuthorityValidation,
) -> None:
    runspec = authority.phase_runspec
    if isinstance(runspec, FoldingPhaseRunSpec):
        _validate_folding_finalized_against_authority(event, authority, runspec)
        return
    action = runspec.payload.actions[0]
    receipt = event.payload.receipt
    submission = authority.submission
    if submission is None or submission.status != "submitted":
        raise ValueError("Phase Finalization requires a complete durable Phase Submission")
    submitted_action = next((item for item in submission.actions if item.action_id == action.action_id), None)
    if submitted_action is None or submitted_action.status != "submitted" or submitted_action.job_id is None:
        raise ValueError("Phase Finalization action has no durable Slurm assignment")
    if event.phase_run_id != authority.phase_run.phase_run_id or event.attempt_id != runspec.attempt_id:
        raise ValueError("phase-finalized event does not bind the authoritative run and attempt")
    if (
        receipt.phase_plan_digest != authority.phase_plan.digest
        or receipt.phase_runspec_digest != runspec.digest
        or receipt.phase_runspec_location != authority.current_attempt.phase_runspec_location
    ):
        raise ValueError("Phase Receipt does not bind the authoritative Plan and RunSpec")
    if (
        receipt.input_sha256 != runspec.input_location.sha256
        or receipt.input_size_bytes != runspec.input_location.size_bytes
        or receipt.input_location_digest != canonical_mapping_digest(runspec.input_location.to_mapping())
    ):
        raise ValueError("Phase Receipt does not bind the authoritative input identity")
    if receipt.cluster_snapshot_digest != canonical_mapping_digest(runspec.cluster.to_mapping()):
        raise ValueError("Phase Receipt does not bind the authoritative cluster snapshot")
    if receipt.runtime_action_digest != canonical_mapping_digest(action.to_mapping()):
        raise ValueError("Phase Receipt does not bind the authoritative Runtime Action")
    scheduler = event.payload.scheduler_evidence
    action_evidence = event.payload.action_evidence
    content = event.payload.content_validation
    if action_evidence is None or content is None:
        raise ValueError("preprocessing Phase Finalization requires chunk action and content evidence")
    if (
        scheduler.phase_runspec_digest != runspec.digest
        or action_evidence.phase_runspec_digest != runspec.digest
        or content.phase_runspec_digest != runspec.digest
        or scheduler.action_id != action.action_id
        or action_evidence.action_id != action.action_id
        or content.action_id != action.action_id
    ):
        raise ValueError("finalization evidence does not bind the authoritative RunSpec action")
    if scheduler.job_id != submitted_action.job_id or receipt.slurm_job_id != submitted_action.job_id:
        raise ValueError("Phase Finalization scheduler evidence does not match Control's assigned Slurm job")
    terminal = next(
        (item for item in authority.terminal_observations if item.action_id == action.action_id),
        None,
    )
    if terminal is not None and (
        terminal.outcome != "succeeded"
        or terminal.job_id != scheduler.job_id
        or terminal.source != scheduler.source
        or terminal.state != scheduler.state
        or terminal.exit_code != scheduler.exit_code
    ):
        raise ValueError("Phase Finalization evidence conflicts with durable terminal accounting")
    expected_closure = _replayed_carry_closure(authority, action_evidence.carry_forward_adoption)
    if receipt.carry_forward_closure != expected_closure:
        raise ValueError("Phase Receipt carry-forward closure does not match replayed authority")


def _validate_folding_finalized_against_authority(
    event: PhaseFinalizedEvent,
    authority: PhaseAuthorityValidation,
    runspec: FoldingPhaseRunSpec,
) -> None:
    """Validate one folding terminal receipt against its authoritative folding RunSpec."""
    receipt = event.payload.receipt
    action = next((item for item in runspec.payload.actions if item.action_kind == "canonical-pair"), None)
    if action is None:
        raise ValueError("folding Phase Finalization requires the terminal canonical-pair action")
    submission = authority.submission
    if submission is None or submission.status != "submitted":
        raise ValueError("Phase Finalization requires a complete durable Phase Submission")
    submitted_action = next((item for item in submission.actions if item.action_id == action.action_id), None)
    if submitted_action is None or submitted_action.status != "submitted" or submitted_action.job_id is None:
        raise ValueError("Phase Finalization action has no durable Slurm assignment")
    if event.phase_run_id != authority.phase_run.phase_run_id or event.attempt_id != runspec.attempt_id:
        raise ValueError("phase-finalized event does not bind the authoritative run and attempt")
    if (
        receipt.phase_plan_digest != authority.phase_plan.digest
        or receipt.phase_runspec_digest != runspec.digest
        or receipt.phase_runspec_location != authority.current_attempt.phase_runspec_location
    ):
        raise ValueError("Phase Receipt does not bind the authoritative Plan and RunSpec")
    if receipt.action_id != action.action_id:
        raise ValueError("Phase Receipt does not bind the terminal canonical-pair action")
    if receipt.input_location_digest != canonical_mapping_digest(runspec.input_location.to_mapping()):
        raise ValueError("Phase Receipt does not bind the authoritative input identity")
    if receipt.cluster_snapshot_digest != canonical_mapping_digest(runspec.cluster.to_mapping()):
        raise ValueError("Phase Receipt does not bind the authoritative cluster snapshot")
    if receipt.runtime_action_digest != canonical_mapping_digest(action.to_mapping()):
        raise ValueError("Phase Receipt does not bind the authoritative Runtime Action")
    if receipt.carry_forward_closure:
        raise ValueError("folding Phase Receipt cannot carry forward")
    scheduler = event.payload.scheduler_evidence
    if scheduler.phase_runspec_digest != runspec.digest or scheduler.action_id != action.action_id:
        raise ValueError("finalization evidence does not bind the authoritative RunSpec action")
    if scheduler.job_id != submitted_action.job_id or receipt.slurm_job_id != submitted_action.job_id:
        raise ValueError("Phase Finalization scheduler evidence does not match Control's assigned Slurm job")
    terminal = next(
        (item for item in authority.terminal_observations if item.action_id == action.action_id),
        None,
    )
    if terminal is not None and (
        terminal.outcome != "succeeded"
        or terminal.job_id != scheduler.job_id
        or terminal.source != scheduler.source
        or terminal.state != scheduler.state
        or terminal.exit_code != scheduler.exit_code
    ):
        raise ValueError("Phase Finalization evidence conflicts with durable terminal accounting")
    if event.payload.folding_action_evidence_digests != receipt.folding_action_evidence_digests:
        raise ValueError("Phase Receipt does not bind the exact folding action evidence")
    if event.payload.canonical_pair_index_digest != receipt.canonical_pair_index_digest:
        raise ValueError("Phase Receipt does not bind the exact canonical-pair index digest")


def _replayed_carry_closure(
    authority: PhaseAuthorityValidation,
    current_adoption: AttemptCarryForwardAdoptionEvidence | None,
) -> tuple[AttemptCarryForwardReceiptReference, ...]:
    raw_records = tuple(
        record
        for record in (
            *(item.carry_forward for item in authority.prior_attempts),
            authority.current_carry_forward,
        )
        if record is not None
    )
    if not raw_records:
        if current_adoption is not None:
            raise ValueError("no-carry receipt cannot contain adoption evidence")
        return ()
    if any(not isinstance(record, AttemptCarryForwardRecord) for record in raw_records):
        raise ValueError("carry-forward receipt closure requires preprocessing carry records")
    records = tuple(cast("AttemptCarryForwardRecord", record) for record in raw_records)
    if authority.current_carry_forward is None:
        raise ValueError("carry-forward receipt lineage does not end at the current Attempt")
    result: list[AttemptCarryForwardReceiptReference] = []
    for index, record in enumerate(records):
        if index and records[index - 1].target_attempt_id != record.source_attempt_id:
            raise ValueError("carry-forward receipt lineage is not contiguous")
        adoption = (
            current_adoption
            if record.target_attempt_id == authority.current_attempt.attempt_id
            else _replayed_historical_adoption(authority, record)
        )
        if adoption is None:
            raise ValueError("carry-forward receipt lineage lacks target adoption")
        expected_content = tuple(
            (item.member_name, item.source_private_mount_path, item.target_declared_path, item.size_bytes, item.sha256)
            for item in record.content
        )
        observed_content = tuple(
            (item.member_name, item.source_path, item.target_path, item.size_bytes, item.sha256)
            for item in adoption.content
        )
        if (
            adoption.phase_run_id != authority.phase_run.phase_run_id
            or adoption.attempt_id != record.target_attempt_id
            or adoption.attempt_carry_forward_id != record.attempt_carry_forward_id
            or adoption.attempt_carry_forward_digest != record.digest
            or adoption.remaining_search_input_sha256 != record.remaining_search_input_sha256
            or expected_content != observed_content
        ):
            raise ValueError("carry-forward receipt target adoption differs from Retry authority")
        if record.target_attempt_id == authority.current_attempt.attempt_id and (
            authority.submission is None or adoption.phase_submission_id != authority.submission.submission_id
        ):
            raise ValueError("carry-forward receipt adoption differs from current submission authority")
        result.append(
            AttemptCarryForwardReceiptReference(
                attempt_carry_forward_id=record.attempt_carry_forward_id,
                attempt_carry_forward_digest=record.digest,
                source_attempt_id=record.source_attempt_id,
                target_attempt_id=record.target_attempt_id,
                adopted_content_digest=record.content_digest,
                source_verification_evidence_digest=record.verification.evidence_mapping_digest,
                target_adoption_evidence_digest=adoption.digest,
            )
        )
    return tuple(result)


def _replayed_historical_adoption(
    authority: PhaseAuthorityValidation,
    record: AttemptCarryForwardRecord,
) -> AttemptCarryForwardAdoptionEvidence:
    matches = tuple(
        (event.payload.evidence.carry_forward_adoption, event.payload.submission_id)
        for event in authority.events
        if isinstance(event, PhaseActionEvidenceAttestedEvent)
        and event.attempt_id == record.target_attempt_id
        and event.payload.evidence.carry_forward_adoption is not None
        and event.payload.evidence.carry_forward_adoption.attempt_carry_forward_id == record.attempt_carry_forward_id
    )
    if len(matches) != 1 or matches[0][0] is None:
        raise ValueError("carry-forward receipt ancestor lacks unique target adoption")
    adoption, submission_id = matches[0]
    assert adoption is not None
    if adoption.phase_submission_id != submission_id:
        raise ValueError("carry-forward receipt ancestor differs from attested submission")
    return adoption


def _validate_records(
    phase_plan: PhasePlanRecord,
    phase_run: PhaseRun,
    phase_runspec: PhaseRunSpecRecord,
    event: PhaseMaterializedEvent,
) -> None:
    if phase_run.phase_kind != phase_plan.phase_kind:
        msg = (
            "Phase Run phase_kind does not match its Phase Plan phase_kind: "
            f"{phase_run.phase_kind!r} != {phase_plan.phase_kind!r}"
        )
        raise ValueError(msg)
    if isinstance(phase_runspec, FoldingPhaseRunSpec):
        if not isinstance(phase_plan, FoldingPhasePlan):
            raise ValueError("folding Phase RunSpec requires a folding Phase Plan")
        _validate_folding_records(phase_plan, phase_run, phase_runspec, event)
        return
    if not isinstance(phase_plan, PhasePlan):
        raise ValueError("preprocessing Phase RunSpec requires a preprocessing Phase Plan")
    _validate_preprocessing_records(phase_plan, phase_run, phase_runspec, event)


def _validate_preprocessing_records(
    phase_plan: PhasePlan,
    phase_run: PhaseRun,
    phase_runspec: PhaseRunSpec,
    event: PhaseMaterializedEvent,
) -> None:
    attempt = phase_run.attempts[0]
    if phase_plan.digest != phase_run.phase_plan_digest:
        msg = "Phase Plan digest does not match the Phase Run"
        raise ValueError(msg)
    if phase_runspec.phase_plan_digest != phase_plan.digest:
        msg = "Phase RunSpec does not preserve the exact Phase Plan digest"
        raise ValueError(msg)
    if phase_runspec.digest != attempt.phase_runspec_digest:
        msg = "Phase RunSpec digest does not match the immutable Phase Attempt"
        raise ValueError(msg)
    if phase_runspec.phase_run_id != phase_run.phase_run_id or phase_runspec.attempt_id != attempt.attempt_id:
        msg = "Phase RunSpec run/attempt references do not match durable Phase state"
        raise ValueError(msg)
    if phase_plan.target_cluster != phase_runspec.cluster.profile_name:
        msg = "Phase RunSpec cluster snapshot does not match the Phase Plan target"
        raise ValueError(msg)
    if phase_plan.input_location != phase_runspec.input_location:
        msg = "Phase RunSpec input location does not match the Phase Plan"
        raise ValueError(msg)
    if phase_plan.payload.work_plan != phase_runspec.payload.work_plan:
        msg = "Phase RunSpec must preserve the exact authored preprocessing work plan"
        raise ValueError(msg)
    database = phase_runspec.payload.database
    expected_execution = materialize_preprocessing_chunk_execution_plan(
        phase_plan.payload.chunk_execution_intent,
        selected_database_root=database.selected_container_root,
        primary_database_name=database.primary_database_name,
        metagenomic_database_name=database.metagenomic_database_name,
    )
    if not _execution_plan_equivalent(expected_execution, phase_runspec.payload.actions[0].payload):
        msg = "Phase RunSpec action must exactly materialize the authored chunk execution intent"
        raise ValueError(msg)
    if phase_plan.payload.database.database_set != database.database_set:
        raise ValueError("Phase RunSpec database binding must preserve the Phase Plan Database Set identity")
    if phase_plan.payload.database.requested_policy != database.requested_policy:
        raise ValueError("Phase RunSpec database binding must preserve the requested access policy")
    if phase_plan.payload.transport != phase_runspec.payload.transport:
        raise ValueError("Phase RunSpec payload transport does not bind its Phase Plan payload transport")
    if phase_plan.payload.s3_publish_prefix != phase_runspec.payload.s3_publish_prefix:
        raise ValueError("Phase RunSpec payload s3_publish_prefix does not bind its Phase Plan prefix")
    if event.payload.phase_run != phase_run or event.payload.phase_runspec != phase_runspec:
        msg = "phase-materialized event does not replay the authoritative run and RunSpec"
        raise ValueError(msg)


def _validate_folding_records(
    phase_plan: FoldingPhasePlan,
    phase_run: PhaseRun,
    phase_runspec: FoldingPhaseRunSpec,
    event: PhaseMaterializedEvent,
) -> None:
    attempt = phase_run.attempts[0]
    if phase_plan.digest != phase_run.phase_plan_digest:
        msg = "Phase Plan digest does not match the Phase Run"
        raise ValueError(msg)
    if phase_runspec.phase_plan_digest != phase_plan.digest:
        msg = "Phase RunSpec does not preserve the exact Phase Plan digest"
        raise ValueError(msg)
    if phase_runspec.digest != attempt.phase_runspec_digest:
        msg = "Phase RunSpec digest does not match the immutable Phase Attempt"
        raise ValueError(msg)
    if phase_runspec.phase_run_id != phase_run.phase_run_id or phase_runspec.attempt_id != attempt.attempt_id:
        msg = "Phase RunSpec run/attempt references do not match durable Phase state"
        raise ValueError(msg)
    if phase_plan.target_cluster != phase_runspec.cluster.profile_name:
        msg = "Phase RunSpec cluster snapshot does not match the Phase Plan target"
        raise ValueError(msg)
    if phase_plan.input_location != phase_runspec.input_location:
        msg = "Phase RunSpec input location does not match the Phase Plan"
        raise ValueError(msg)
    validate_folding_plan_runspec_binding(phase_plan, phase_runspec)
    if event.payload.phase_run != phase_run or event.payload.phase_runspec != phase_runspec:
        msg = "phase-materialized event does not replay the authoritative run and RunSpec"
        raise ValueError(msg)


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_bytes())
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        msg = f"invalid JSON in Phase authority record {path}"
        raise ValueError(msg) from exc
    if not isinstance(payload, Mapping):
        msg = f"expected JSON object in Phase authority record {path}"
        raise ValueError(msg)
    return payload


def _write_json_durable(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_json_bytes(payload)
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _write_bytes_durable(encoded: bytes, path: Path, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(encoded)
        if mode is not None:
            os.fchmod(handle.fileno(), mode)
        handle.flush()
        os.fsync(handle.fileno())


def _validate_database_manifest_projection(authority_path: Path, runspec: PhaseRunSpec) -> None:
    projection = authority_path / runspec.payload.database.source_manifest_projection
    if projection.is_symlink() or not projection.is_file():
        raise ValueError("Phase authority requires an exact regular database source-manifest projection")
    if stat.S_IMODE(projection.stat().st_mode) != 0o444:
        raise ValueError("Phase authority requires a read-only database source-manifest projection")
    expected = canonical_database_source_manifest_bytes(runspec.payload.database.source_manifest)
    if projection.read_bytes() != expected:
        raise ValueError("database source-manifest projection differs from embedded RunSpec authority")


def _write_fold_shard_projection(
    staging_path: Path,
    phase_plan: FoldingPhasePlan,
    phase_runspec: FoldingPhaseRunSpec,
) -> None:
    if phase_runspec.payload.fold_shard_projection is None:
        return
    projection, binding = fold_shard_projection_from_runspec(phase_plan, phase_runspec)
    _write_bytes_durable(
        fold_shard_projection_document_bytes(projection),
        staging_path / binding.location,
    )


def _validate_fold_shard_projection(
    authority_path: Path,
    phase_plan: FoldingPhasePlan,
    phase_runspec: FoldingPhaseRunSpec,
) -> None:
    binding = phase_runspec.payload.fold_shard_projection
    if binding is None:
        return
    projection_path = authority_path / binding.location
    if projection_path.is_symlink() or not projection_path.is_file():
        raise ValueError("Phase authority requires an exact regular fold shard projection")
    expected_projection, expected_binding = fold_shard_projection_from_runspec(phase_plan, phase_runspec)
    if binding != expected_binding:
        raise ValueError("fold shard projection binding differs from the canonical Plan-derived projection")
    if projection_path.read_bytes() != fold_shard_projection_document_bytes(expected_projection):
        raise ValueError("fold shard projection document differs from embedded RunSpec authority")


def _execution_plan_equivalent(
    expected: PreprocessingChunkExecutionPlan,
    observed: PreprocessingChunkExecutionPlan,
) -> bool:
    """Return True if two execution plans are equivalent modulo --pair-mode backward compat.

    Old sealed evidence was produced with ``--pair-mode paired``; new
    materialization uses ``--pair-mode unpaired_paired``.  This helper accepts
    both values at the ``--pair-mode`` position so old evidence remains
    replayable.
    """
    if expected == observed:
        return True
    # Check if the only difference is the --pair-mode value
    expected_argv = expected.search_argv
    observed_argv = observed.search_argv
    if "--pair-mode" not in expected_argv or "--pair-mode" not in observed_argv:
        return False
    idx = expected_argv.index("--pair-mode")
    if observed_argv.index("--pair-mode") != idx:
        return False
    if expected_argv[idx + 1] not in {"paired", "unpaired_paired"}:
        return False
    if observed_argv[idx + 1] not in {"paired", "unpaired_paired"}:
        return False
    expected_rest = (*expected_argv[:idx], *expected_argv[idx + 2 :])
    observed_rest = (*observed_argv[:idx], *observed_argv[idx + 2 :])
    if expected_rest != observed_rest:
        return False
    # Compare all other fields
    return (
        expected.chunk_name == observed.chunk_name
        and expected.scientific == observed.scientific
        and expected.site == observed.site
        and expected.runtime == observed.runtime
        and expected.expected_a3ms == observed.expected_a3ms
        and expected.evidence == observed.evidence
        and expected.package == observed.package
        and expected.gpuserver_argv == observed.gpuserver_argv
        and expected.gpuserver_environment == observed.gpuserver_environment
        and expected.search_environment == observed.search_environment
        and expected.schema_version == observed.schema_version
    )


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def _carry_forward_file_name(record: AttemptCarryForwardRecord | FoldingCarryForwardRecord) -> str:
    """Return the authority filename for one carry record, by type."""
    if isinstance(record, FoldingCarryForwardRecord):
        return "folding-carry-forward.json"
    return "attempt-carry-forward.json"


def _fsync_tree_directories(root: Path) -> None:
    directories = sorted(
        (candidate for candidate in root.rglob("*") if candidate.is_dir()),
        key=lambda candidate: len(candidate.parts),
        reverse=True,
    )
    for directory in (*directories, root):
        _fsync_directory(directory)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "PhaseAuthorityCollisionError",
    "PhaseAuthorityEventHandler",
    "PhaseAuthorityEventLoader",
    "PhaseAuthorityEventReplay",
    "PhaseAuthoritySealedError",
    "PhaseAuthorityStore",
    "PhaseAuthorityValidation",
    "require_complete_current_runspec",
    "validate_phase_run_id",
]
