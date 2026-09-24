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

"""Restart-safe family-specific operator for one postprocessing Phase Run."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shlex
import stat
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from bspp.orchestration.contract.postprocessing_submission_events import (
    PostprocessingMaterializedPayload,
)
from bspp.orchestration.contract.source_package import source_package_identity_from_mapping
from bspp.orchestration.control.postprocessing_authority_reader import require_postprocessing_v2_authority
from bspp.orchestration.control.postprocessing_authority_store import postprocessing_phase_coordinator_lock
from bspp.orchestration.control.postprocessing_evidence_transfer import (
    fetch_postprocessing_finalization_evidence,
)
from bspp.orchestration.control.postprocessing_phase_diagnostics import (
    capture_postprocessing_phase_diagnostics,
)
from bspp.orchestration.control.postprocessing_phase_finalization import finalize_postprocessing_phase
from bspp.orchestration.control.postprocessing_phase_materialization import (
    _postprocessing_phase_plan_from_document,
    _verified_document,
    materialize_postprocessing_phase,
)
from bspp.orchestration.control.postprocessing_phase_observation import (
    resume_postprocessing_phase,
    status_postprocessing_phase,
)
from bspp.orchestration.control.postprocessing_phase_submission import submit_postprocessing_phase
from bspp.orchestration.control.postprocessing_scheduler_evidence import (
    export_postprocessing_scheduler_evidence,
)
from bspp.orchestration.control.profiles import resolve_cluster_profile
from bspp.orchestration.control.runtime_qualification_validation import validate_runtime_qualification_source

_PHASE_RUN_ID = re.compile(r"phase-run-[0-9a-f]{32}")
_MAX_INTENT_BYTES = 64 * 1024
_MAX_INPUT_DOCUMENT_BYTES = 16 * 1024 * 1024
_MAX_ERROR_CHARS = 2048
_HEARTBEAT_SECONDS = 300.0


class _LifecycleResult(Protocol):
    @property
    def phase_run_id(self) -> str: ...

    @property
    def attempt_id(self) -> str: ...

    @property
    def status(self) -> str: ...

    @property
    def details(self) -> Mapping[str, object]: ...


class _MaterializationResult(Protocol):
    @property
    def phase_run_id(self) -> str: ...

    @property
    def attempt_id(self) -> str: ...


class _OperatorMismatchError(ValueError):
    pass


class _DiagnosticsResult(Protocol):
    @property
    def output(self) -> Path: ...


@dataclass(frozen=True)
class PostprocessingPhaseOperatorDependencies:
    """Direct-service and clock seams for deterministic restart/failure tests."""

    materialize: Callable[..., _MaterializationResult]
    submit: Callable[..., _LifecycleResult]
    status: Callable[..., _LifecycleResult]
    resume: Callable[..., _LifecycleResult]
    fetch: Callable[..., object]
    export_scheduler: Callable[..., object]
    finalize: Callable[..., object]
    diagnostics: Callable[..., _DiagnosticsResult]
    wall_clock: Callable[[], datetime]
    monotonic_clock: Callable[[], float]
    sleeper: Callable[[float], None]
    interrupted: Callable[[], bool]
    phase_run_id_factory: Callable[[], str]
    emit: Callable[[Mapping[str, object]], None]
    bind_initial_attempt: Callable[..., str] = lambda authority_root, phase_run_id, intent: (
        _bind_operator_v1_initial_attempt(authority_root, phase_run_id=phase_run_id, intent=intent)
    )


DEFAULT_POSTPROCESSING_PHASE_OPERATOR_DEPENDENCIES = PostprocessingPhaseOperatorDependencies(
    materialize=materialize_postprocessing_phase,
    submit=submit_postprocessing_phase,
    status=status_postprocessing_phase,
    resume=resume_postprocessing_phase,
    fetch=fetch_postprocessing_finalization_evidence,
    export_scheduler=export_postprocessing_scheduler_evidence,
    finalize=finalize_postprocessing_phase,
    diagnostics=capture_postprocessing_phase_diagnostics,
    wall_clock=lambda: datetime.now(UTC),
    monotonic_clock=time.monotonic,
    sleeper=time.sleep,
    interrupted=lambda: False,
    phase_run_id_factory=lambda: f"phase-run-{uuid.uuid4().hex}",
    bind_initial_attempt=lambda authority_root, phase_run_id, intent: _bind_operator_v1_initial_attempt(
        authority_root, phase_run_id=phase_run_id, intent=intent
    ),
    emit=lambda _event: None,
)


PostprocessingPhaseOperatorStatus = Literal[
    "accepted",
    "lifecycle-failed",
    "timed-out",
    "interrupted",
    "cancelling",
    "cancelled",
    "lock-conflict",
    "configuration-mismatch",
]


@dataclass(frozen=True)
class PostprocessingPhaseOperatorResult:
    status: PostprocessingPhaseOperatorStatus
    execution_root: Path
    phase_run_id: str | None
    attempt_id: str | None
    recovery_command: str | None
    detail: str | None = None
    diagnostics: str | None = None
    explicit_cancel_command: str | None = None

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "phase_kind": "postprocessing",
            "operation": "run-postprocessing",
            "status": self.status,
            "execution_root": str(self.execution_root),
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "recovery_command": self.recovery_command,
            "explicit_cancel_command": self.explicit_cancel_command,
            "diagnostics": self.diagnostics,
            "detail": self.detail,
        }

    def render_json(self) -> str:
        return json.dumps(self.to_mapping(), indent=2, sort_keys=True) + "\n"


@dataclass
class _TransitionWriter:
    root: Path
    phase_run_id: str
    dependencies: PostprocessingPhaseOperatorDependencies
    operation: str = "run-postprocessing"
    last_state: str | None = None
    last_detail: str | None = None
    next_heartbeat: float = 0.0

    def record(self, state: str, *, attempt_id: str | None, detail: str | None = None) -> None:
        now = self.dependencies.monotonic_clock()
        normalized_detail = _bounded_detail(detail) if detail is not None else None
        heartbeat = now >= self.next_heartbeat
        changed = state != self.last_state
        detail_changed = normalized_detail != self.last_detail
        payload: dict[str, object] = {
            "schema_version": 1,
            "operation": self.operation,
            "phase_run_id": self.phase_run_id,
            "attempt_id": attempt_id,
            "state": state,
            "updated_at": _timestamp(self.dependencies.wall_clock()),
        }
        if normalized_detail is not None:
            payload["detail"] = normalized_detail
        _atomic_json_replace(self.root / "current.json", payload)
        if changed or detail_changed or heartbeat:
            self.dependencies.emit(payload)
            self.next_heartbeat = now + _HEARTBEAT_SECONDS
        self.last_state = state
        self.last_detail = normalized_detail


def run_postprocessing_phase(
    phase_plan_path: Path,
    *,
    authority_root: Path,
    execution_root: Path,
    source_repo: Path,
    config_path: Path,
    poll_interval_seconds: float = 30.0,
    timeout_seconds: float = 24 * 60 * 60,
    dependencies: PostprocessingPhaseOperatorDependencies = DEFAULT_POSTPROCESSING_PHASE_OPERATOR_DEPENDENCIES,
) -> PostprocessingPhaseOperatorResult:
    """Drive one postprocessing Phase through finalization using only derived evidence paths."""
    if poll_interval_seconds <= 0 or timeout_seconds <= 0:
        raise ValueError("postprocessing operator poll interval and timeout must be positive")
    plan_path = phase_plan_path.resolve(strict=True)
    source_root = source_repo.resolve(strict=True)
    config = config_path.resolve(strict=True)
    authority = authority_root.resolve()
    execution = execution_root.resolve()
    _reject_overlapping_roots(authority, execution, source_root)
    recovery = _recovery_command(
        plan_path=plan_path,
        authority_root=authority,
        execution_root=execution,
        source_repo=source_root,
        config_path=config,
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
    )
    started = dependencies.monotonic_clock()
    try:
        with ExitStack() as locks:
            locks.enter_context(_operation_lock(execution))
            plan_document = _stable_regular_bytes(plan_path, maximum=_MAX_INPUT_DOCUMENT_BYTES)
            config_document = _stable_regular_bytes(config, maximum=_MAX_INPUT_DOCUMENT_BYTES)
            base_intent = _intent_identity(
                plan_path=plan_path,
                plan_document=plan_document,
                authority_root=authority,
                execution_root=execution,
                source_repo=source_root,
                config_path=config,
                config_document=config_document,
                poll_interval_seconds=poll_interval_seconds,
                timeout_seconds=timeout_seconds,
            )
            intent_path = execution / "operation-intent.json"
            if os.path.lexists(intent_path):
                persisted = _load_intent(intent_path)
                recovery = _recovery_command_from_intent(persisted)
                phase_run_id = _phase_run_id_from_intent(persisted)
                expected = {**base_intent, "phase_run_id": phase_run_id}
                if persisted != expected:
                    raise _OperatorMismatchError(
                        "postprocessing operator invocation differs from its create-once intent"
                    )
            else:
                phase_run_id = dependencies.phase_run_id_factory()
                if not isinstance(phase_run_id, str) or _PHASE_RUN_ID.fullmatch(phase_run_id) is None:
                    raise ValueError("postprocessing operator Phase Run id factory returned an invalid id")
                persisted = {**base_intent, "phase_run_id": phase_run_id}
                _revalidate_operator_input_snapshot(plan_path, plan_document, label="Phase Plan")
                _revalidate_operator_input_snapshot(config, config_document, label="Cluster Profile")
                _write_create_once_json(intent_path, persisted)
            locks.enter_context(postprocessing_phase_coordinator_lock(authority, phase_run_id))
            authority_path = _confined_authority_path(authority, phase_run_id)
            if os.path.lexists(authority_path):
                _verify_existing_authority_plan(
                    authority_path,
                    phase_run_id=phase_run_id,
                    phase_plan_digest=_phase_plan_digest_from_intent(persisted),
                )
            else:
                _revalidate_operator_input_snapshot(plan_path, plan_document, label="Phase Plan")
                _revalidate_operator_input_snapshot(config, config_document, label="Cluster Profile")
            writer = _TransitionWriter(execution, phase_run_id, dependencies)
            attempt_id: str | None = None
            expected_attempt_id = "attempt-0001"
            try:
                _stop_if_requested(dependencies, started, timeout_seconds)
                if not os.path.lexists(authority_path):
                    writer.record("materializing", attempt_id=None)
                    materialized = dependencies.materialize(
                        plan_path,
                        authority_root=authority,
                        config_path=config,
                        source_repo=source_root,
                        clock=dependencies.wall_clock,
                        phase_run_id_factory=lambda: phase_run_id,
                        phase_plan_document=plan_document,
                        config_document=config_document,
                    )
                    if materialized.phase_run_id != phase_run_id:
                        raise ValueError("postprocessing materialization returned a different Phase Run id")
                    _require_expected_attempt(materialized, expected_attempt_id=expected_attempt_id)
                    attempt_id = expected_attempt_id
                    _verify_existing_authority_plan(
                        authority_path,
                        phase_run_id=phase_run_id,
                        phase_plan_digest=_phase_plan_digest_from_intent(persisted),
                    )
                expected_attempt_id = dependencies.bind_initial_attempt(
                    authority,
                    phase_run_id=phase_run_id,
                    intent=persisted,
                )
                attempt_id = expected_attempt_id
                return _drive_postprocessing_attempt(
                    phase_run_id=phase_run_id,
                    expected_attempt_id=expected_attempt_id,
                    authority_root=authority,
                    execution_root=execution,
                    recovery_command=recovery,
                    poll_interval_seconds=poll_interval_seconds,
                    timeout_seconds=timeout_seconds,
                    started=started,
                    writer=writer,
                    dependencies=dependencies,
                )
            except _OperatorMismatchError as exc:
                result = PostprocessingPhaseOperatorResult(
                    status="configuration-mismatch",
                    execution_root=execution,
                    phase_run_id=phase_run_id,
                    attempt_id=expected_attempt_id,
                    recovery_command=recovery,
                    detail=_bounded_detail(str(exc)),
                )
                writer.record(result.status, attempt_id=expected_attempt_id, detail=result.detail)
                return result
            except KeyboardInterrupt:
                result = PostprocessingPhaseOperatorResult(
                    status="interrupted",
                    execution_root=execution,
                    phase_run_id=phase_run_id,
                    attempt_id=attempt_id,
                    recovery_command=recovery,
                    detail="postprocessing operator interrupted; no cancellation was requested",
                )
                writer.record(result.status, attempt_id=attempt_id, detail=result.detail)
                return _with_diagnostics(
                    result,
                    phase_run_id=phase_run_id,
                    authority_root=authority,
                    execution_root=execution,
                    dependencies=dependencies,
                )
            except _OperatorStopError as exc:
                result = PostprocessingPhaseOperatorResult(
                    status=exc.status,
                    execution_root=execution,
                    phase_run_id=phase_run_id,
                    attempt_id=attempt_id,
                    recovery_command=recovery,
                    detail=exc.detail,
                )
                writer.record(result.status, attempt_id=attempt_id, detail=result.detail)
                return _with_diagnostics(
                    result,
                    phase_run_id=phase_run_id,
                    authority_root=authority,
                    execution_root=execution,
                    dependencies=dependencies,
                )
            except (OSError, TypeError, ValueError) as exc:
                result = PostprocessingPhaseOperatorResult(
                    status="lifecycle-failed",
                    execution_root=execution,
                    phase_run_id=phase_run_id,
                    attempt_id=attempt_id,
                    recovery_command=recovery,
                    detail=_bounded_detail(str(exc)),
                )
                writer.record(result.status, attempt_id=attempt_id, detail=result.detail)
                return _with_diagnostics(
                    result,
                    phase_run_id=phase_run_id,
                    authority_root=authority,
                    execution_root=execution,
                    dependencies=dependencies,
                )
    except BlockingIOError:
        return PostprocessingPhaseOperatorResult(
            status="lock-conflict",
            execution_root=execution,
            phase_run_id=None,
            attempt_id=None,
            recovery_command=recovery,
            detail="another postprocessing coordinator owns an operation lock; no cancellation was requested",
        )
    except _OperatorMismatchError as exc:
        return PostprocessingPhaseOperatorResult(
            status="configuration-mismatch",
            execution_root=execution,
            phase_run_id=None,
            attempt_id=None,
            recovery_command=recovery,
            detail=_bounded_detail(str(exc)),
        )


@dataclass(frozen=True)
class _OperatorStopError(Exception):
    status: Literal["timed-out", "interrupted"]
    detail: str


def _require_expected_attempt(result: object, *, expected_attempt_id: str) -> None:
    observed_attempt_id = getattr(result, "attempt_id", None)
    if observed_attempt_id != expected_attempt_id:
        raise _OperatorMismatchError(
            "postprocessing coordinator expected "
            f"{expected_attempt_id}, but durable authority selected {observed_attempt_id}"
        )


def _require_expected_lifecycle_identity(
    result: object,
    *,
    expected_phase_run_id: str,
    expected_attempt_id: str,
) -> None:
    observed_phase_run_id = getattr(result, "phase_run_id", None)
    observed_attempt_id = getattr(result, "attempt_id", None)
    if observed_phase_run_id != expected_phase_run_id or observed_attempt_id != expected_attempt_id:
        raise _OperatorMismatchError(
            "postprocessing coordinator expected Phase Run "
            f"{expected_phase_run_id} Attempt {expected_attempt_id}, but the lifecycle result selected "
            f"Phase Run {observed_phase_run_id} Attempt {observed_attempt_id}"
        )


def _drive_postprocessing_attempt(
    *,
    phase_run_id: str,
    expected_attempt_id: str,
    authority_root: Path,
    execution_root: Path,
    recovery_command: str,
    poll_interval_seconds: float,
    timeout_seconds: float,
    started: float,
    writer: _TransitionWriter,
    dependencies: PostprocessingPhaseOperatorDependencies,
) -> PostprocessingPhaseOperatorResult:
    """Drive exactly one already-bound Attempt through accepted finalization."""
    _stop_if_requested(dependencies, started, timeout_seconds)
    observed = dependencies.status(phase_run_id, authority_root=authority_root)
    _require_expected_lifecycle_identity(
        observed,
        expected_phase_run_id=phase_run_id,
        expected_attempt_id=expected_attempt_id,
    )
    if observed.status == "materialized" or _initial_submission_incomplete(observed):
        writer.record("submitting", attempt_id=expected_attempt_id)
        submitted = dependencies.submit(
            phase_run_id,
            authority_root=authority_root,
            clock=dependencies.wall_clock,
        )
        _require_expected_lifecycle_identity(
            submitted,
            expected_phase_run_id=phase_run_id,
            expected_attempt_id=expected_attempt_id,
        )
        observed = dependencies.status(phase_run_id, authority_root=authority_root)
        _require_expected_lifecycle_identity(
            observed,
            expected_phase_run_id=phase_run_id,
            expected_attempt_id=expected_attempt_id,
        )
        if _initial_submission_incomplete(observed):
            raise ValueError("postprocessing initial submission remains incomplete after Submit recovery")
    while True:
        _require_expected_lifecycle_identity(
            observed,
            expected_phase_run_id=phase_run_id,
            expected_attempt_id=expected_attempt_id,
        )
        if observed.status == "accepted":
            break
        terminal_result = _terminal_operator_result(
            observed,
            execution=execution_root,
            recovery=recovery_command,
            authority_root=authority_root,
        )
        if terminal_result is not None:
            writer.record(
                terminal_result.status,
                attempt_id=expected_attempt_id,
                detail=terminal_result.detail,
            )
            if terminal_result.status in {"lifecycle-failed", "cancelled"}:
                return _with_diagnostics(
                    terminal_result,
                    phase_run_id=phase_run_id,
                    authority_root=authority_root,
                    execution_root=execution_root,
                    dependencies=dependencies,
                )
            return terminal_result
        if _all_actions_terminal(observed):
            break
        _stop_if_requested(dependencies, started, timeout_seconds)
        writer.record("polling", attempt_id=expected_attempt_id, detail=_compact_progress(observed))
        resumed = dependencies.resume(
            phase_run_id,
            authority_root=authority_root,
            clock=dependencies.wall_clock,
        )
        _require_expected_lifecycle_identity(
            resumed,
            expected_phase_run_id=phase_run_id,
            expected_attempt_id=expected_attempt_id,
        )
        observed = dependencies.status(phase_run_id, authority_root=authority_root)
        _require_expected_lifecycle_identity(
            observed,
            expected_phase_run_id=phase_run_id,
            expected_attempt_id=expected_attempt_id,
        )
        if _all_actions_terminal(observed) or observed.status != "submitted":
            continue
        _stop_if_requested(dependencies, started, timeout_seconds)
        dependencies.sleeper(poll_interval_seconds)

    handoff = execution_root / "handoff"
    scheduler_evidence = execution_root / "scheduler-evidence.json"
    writer.record("fetching", attempt_id=expected_attempt_id)
    dependencies.fetch(phase_run_id, authority_root=authority_root, destination=handoff)
    _stop_if_requested(dependencies, started, timeout_seconds)
    writer.record("exporting-scheduler-evidence", attempt_id=expected_attempt_id)
    dependencies.export_scheduler(
        phase_run_id,
        authority_root=authority_root,
        output=scheduler_evidence,
    )
    _stop_if_requested(dependencies, started, timeout_seconds)
    writer.record("finalizing", attempt_id=expected_attempt_id)
    finalized = dependencies.finalize(
        phase_run_id,
        authority_root=authority_root,
        scheduler_evidence_path=scheduler_evidence,
        aggregate_action_evidence_path=handoff / "aggregate-action-evidence.json",
        handoff_path=handoff,
        acceptance_adjudication_path=handoff / "acceptance/adjudication.json",
        clock=dependencies.wall_clock,
    )
    _require_expected_lifecycle_identity(
        finalized,
        expected_phase_run_id=phase_run_id,
        expected_attempt_id=expected_attempt_id,
    )
    writer.record("accepted", attempt_id=expected_attempt_id)
    return PostprocessingPhaseOperatorResult(
        status="accepted",
        execution_root=execution_root,
        phase_run_id=phase_run_id,
        attempt_id=expected_attempt_id,
        recovery_command=None,
    )


def _stop_if_requested(
    dependencies: PostprocessingPhaseOperatorDependencies,
    started: float,
    timeout_seconds: float,
) -> None:
    if dependencies.interrupted():
        raise _OperatorStopError(
            "interrupted",
            "postprocessing operator interrupted; no cancellation was requested",
        )
    if dependencies.monotonic_clock() - started >= timeout_seconds:
        raise _OperatorStopError(
            "timed-out",
            "postprocessing operator timed out; no cancellation was requested",
        )


def _terminal_operator_result(
    observed: _LifecycleResult,
    *,
    execution: Path,
    recovery: str,
    authority_root: Path,
) -> PostprocessingPhaseOperatorResult | None:
    if observed.status == "failed":
        return PostprocessingPhaseOperatorResult(
            status="lifecycle-failed",
            execution_root=execution,
            phase_run_id=observed.phase_run_id,
            attempt_id=observed.attempt_id,
            recovery_command=recovery,
            detail="postprocessing durable authority reports a failed Attempt",
        )
    if observed.status == "cancelling":
        return PostprocessingPhaseOperatorResult(
            status="cancelling",
            execution_root=execution,
            phase_run_id=observed.phase_run_id,
            attempt_id=observed.attempt_id,
            recovery_command=recovery,
            explicit_cancel_command=shlex.join(
                ("bsppctl", "phase", "cancel", observed.phase_run_id, "--authority-root", str(authority_root))
            ),
            detail="cancellation is already explicit in durable authority; rerun phase cancel separately",
        )
    if observed.status == "cancelled":
        return PostprocessingPhaseOperatorResult(
            status="cancelled",
            execution_root=execution,
            phase_run_id=observed.phase_run_id,
            attempt_id=observed.attempt_id,
            recovery_command=recovery,
            detail="postprocessing durable authority reports a cancelled Attempt",
        )
    return None


def _all_actions_terminal(observed: _LifecycleResult) -> bool:
    actions = observed.details.get("actions")
    return (
        isinstance(actions, list)
        and bool(actions)
        and all(isinstance(item, Mapping) and item.get("terminal") is not None for item in actions)
    )


def _initial_submission_incomplete(observed: _LifecycleResult) -> bool:
    """Identify durable initial-submission intent that still lacks assignments."""
    if observed.status != "submitted":
        return False
    actions = observed.details.get("actions")
    if not isinstance(actions, list):
        return False
    return any(
        isinstance(item, Mapping)
        and item.get("durable_status") in {"planned", "dispatching"}
        and item.get("job_id") is None
        for item in actions
    )


def _compact_progress(observed: _LifecycleResult) -> str:
    actions = observed.details.get("actions")
    if not isinstance(actions, list):
        return f"authority_status={observed.status}"
    terminal = sum(isinstance(item, Mapping) and item.get("terminal") is not None for item in actions)
    assigned = sum(isinstance(item, Mapping) and item.get("job_id") is not None for item in actions)
    progress = (
        f"authority_status={observed.status} assigned={assigned}/{len(actions)} terminal={terminal}/{len(actions)}"
    )
    gate = next(
        (item for item in actions if isinstance(item, Mapping) and item.get("terminal") is None),
        None,
    )
    if gate is None:
        return progress
    action_id = _progress_value(gate, "action_id")
    scheduler = gate.get("scheduler")
    if isinstance(scheduler, Mapping) and _nonempty_string(scheduler.get("state")):
        return f"{progress} gate={action_id} state={scheduler['state']}"
    tasks = gate.get("tasks")
    if isinstance(tasks, list):
        task_mappings = [task for task in tasks if isinstance(task, Mapping)]
        unresolved = next(
            (task for task in task_mappings if task.get("observation_status") not in {"observed", "not-applicable"}),
            None,
        )
        selected_task = unresolved if unresolved is not None else next(iter(task_mappings), None)
        if selected_task is not None:
            task_index = _progress_value(selected_task, "task_index")
            task_state = selected_task.get("state") or selected_task.get("observation_status")
            return f"{progress} gate={action_id} task_index={task_index} state={_progress_scalar(task_state)}"
    return f"{progress} gate={action_id} state={_progress_value(gate, 'durable_status')}"


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _progress_value(mapping: Mapping[str, object], key: str) -> str:
    if key not in mapping:
        return "missing"
    return _progress_scalar(mapping[key])


def _progress_scalar(value: object) -> str:
    return "null" if value is None else str(value)


def _with_diagnostics(
    result: PostprocessingPhaseOperatorResult,
    *,
    phase_run_id: str,
    authority_root: Path,
    execution_root: Path,
    dependencies: PostprocessingPhaseOperatorDependencies,
) -> PostprocessingPhaseOperatorResult:
    authority_path = authority_root / phase_run_id
    if not authority_path.is_dir() or authority_path.is_symlink():
        return result
    try:
        captured = dependencies.diagnostics(
            phase_run_id,
            authority_root=authority_root,
            diagnostics_root=execution_root / "diagnostics",
            clock=dependencies.wall_clock,
        )
        output = str(captured.output)
    except (OSError, TypeError, ValueError) as exc:
        diagnostic = f"diagnostics warning: {_bounded_detail(str(exc))}"
    else:
        diagnostic = output
    return PostprocessingPhaseOperatorResult(
        status=result.status,
        execution_root=result.execution_root,
        phase_run_id=result.phase_run_id,
        attempt_id=result.attempt_id,
        recovery_command=result.recovery_command,
        detail=result.detail,
        diagnostics=diagnostic,
        explicit_cancel_command=result.explicit_cancel_command,
    )


def _intent_identity(
    *,
    plan_path: Path,
    plan_document: bytes,
    authority_root: Path,
    execution_root: Path,
    source_repo: Path,
    config_path: Path,
    config_document: bytes,
    poll_interval_seconds: float,
    timeout_seconds: float,
) -> dict[str, object]:
    phase_plan = _postprocessing_phase_plan_from_document(plan_document, source=plan_path)
    profile = resolve_cluster_profile(
        phase_plan.target_cluster,
        config_path=config_path,
        config_document=config_document,
    )
    qualification = _verified_document(phase_plan.runtime_qualification, relative_to=plan_path.parent)
    try:
        qualification_mapping = json.loads(qualification)
        tuple_mapping = qualification_mapping["tuple"]
        source_mapping = tuple_mapping["source_package_identity"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("postprocessing operator could not read pinned source identity") from exc
    if not isinstance(source_mapping, Mapping):
        raise ValueError("postprocessing operator pinned source identity must be a mapping")
    source = source_package_identity_from_mapping(cast("Mapping[str, object]", source_mapping))
    if source.package_role != "orchestration":
        raise ValueError("postprocessing operator requires an orchestration source identity")
    validate_runtime_qualification_source(source_repo, commit=source.commit, tree=source.tree)
    return {
        "format_version": 1,
        "operation_kind": "postprocessing-phase-operator-v1",
        "phase_plan": {
            "path": str(plan_path),
            "sha256": hashlib.sha256(plan_document).hexdigest(),
            "size_bytes": len(plan_document),
            "phase_plan_digest": phase_plan.digest,
        },
        "authority_root": str(authority_root),
        "execution_root": str(execution_root),
        "source": {
            "repository": str(source_repo),
            "commit": source.commit,
            "tree": source.tree,
            "source_package_sha256": source.package_sha256,
            "source_package_manifest_sha256": source.manifest_sha256,
        },
        "profile": {
            "name": profile.name,
            "config_path": str(config_path),
            "config_sha256": hashlib.sha256(config_document).hexdigest(),
            "config_size_bytes": len(config_document),
        },
        "options": {
            "poll_interval_seconds": poll_interval_seconds,
            "timeout_seconds": timeout_seconds,
        },
    }


def _recovery_command(
    *,
    plan_path: Path,
    authority_root: Path,
    execution_root: Path,
    source_repo: Path,
    config_path: Path,
    poll_interval_seconds: float,
    timeout_seconds: float,
) -> str:
    return shlex.join(
        (
            "bsppctl",
            "--config",
            str(config_path),
            "phase",
            "run-postprocessing",
            str(plan_path),
            "--authority-root",
            str(authority_root),
            "--execution-root",
            str(execution_root),
            "--source-repo",
            str(source_repo),
            "--poll-interval",
            f"{poll_interval_seconds:g}",
            "--timeout",
            f"{timeout_seconds:g}",
        )
    )


def _recovery_command_from_intent(intent: Mapping[str, object]) -> str:
    phase_plan = _intent_path(intent, "phase_plan", "path")
    authority_root = _intent_path(intent, None, "authority_root")
    execution_root = _intent_path(intent, None, "execution_root")
    source_repo = _intent_path(intent, "source", "repository")
    config_path = _intent_path(intent, "profile", "config_path")
    options = _required_mapping(intent, "options")
    poll_interval = _required_positive_number(options, "poll_interval_seconds")
    timeout = _required_positive_number(options, "timeout_seconds")
    return _recovery_command(
        plan_path=phase_plan,
        authority_root=authority_root,
        execution_root=execution_root,
        source_repo=source_repo,
        config_path=config_path,
        poll_interval_seconds=poll_interval,
        timeout_seconds=timeout,
    )


def _reject_overlapping_roots(authority_root: Path, execution_root: Path, source_repo: Path) -> None:
    if (
        authority_root == execution_root
        or authority_root in execution_root.parents
        or execution_root in authority_root.parents
    ):
        raise ValueError("postprocessing authority and coordinator execution roots must not overlap")
    if execution_root == source_repo or source_repo in execution_root.parents or execution_root in source_repo.parents:
        raise ValueError("postprocessing coordinator execution root must be outside the source repository")
    if authority_root == source_repo or source_repo in authority_root.parents or authority_root in source_repo.parents:
        raise ValueError("postprocessing authority root must be outside the source repository")


@contextmanager
def _operation_lock(execution_root: Path) -> Any:
    if os.path.lexists(execution_root):
        metadata = execution_root.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or execution_root.is_symlink():
            raise ValueError("postprocessing coordinator execution root must be a real directory")
    else:
        execution_root.mkdir(parents=True, mode=0o700)
    lock_path = execution_root / "operation.lock"
    descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError("postprocessing operation lock must be one regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(descriptor)


def _load_intent(path: Path) -> dict[str, object]:
    document = _stable_regular_bytes(path, maximum=_MAX_INTENT_BYTES)
    try:
        payload = json.loads(document)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _OperatorMismatchError("postprocessing operation intent is not valid JSON") from exc
    if not isinstance(payload, dict) or _canonical_json(payload) != document:
        raise _OperatorMismatchError("postprocessing operation intent is not canonical JSON")
    if set(payload) != {
        "format_version",
        "operation_kind",
        "phase_plan",
        "phase_run_id",
        "authority_root",
        "execution_root",
        "source",
        "profile",
        "options",
    }:
        raise _OperatorMismatchError("postprocessing operation intent has missing or extra fields")
    if payload.get("format_version") != 1 or payload.get("operation_kind") != "postprocessing-phase-operator-v1":
        raise _OperatorMismatchError("postprocessing operation intent has an unsupported format")
    intent = cast("dict[str, object]", payload)
    _phase_run_id_from_intent(intent)
    return intent


def _phase_run_id_from_intent(intent: Mapping[str, object]) -> str:
    phase_run_id = _required_string(intent, "phase_run_id")
    if _PHASE_RUN_ID.fullmatch(phase_run_id) is None:
        raise _OperatorMismatchError("postprocessing operation intent phase_run_id is invalid")
    return phase_run_id


def _phase_plan_digest_from_intent(intent: Mapping[str, object]) -> str:
    phase_plan = _required_mapping(intent, "phase_plan")
    digest = _required_string(phase_plan, "phase_plan_digest")
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise _OperatorMismatchError("postprocessing operation intent Phase Plan digest is invalid")
    return digest


def _confined_authority_path(authority_root: Path, phase_run_id: str) -> Path:
    if _PHASE_RUN_ID.fullmatch(phase_run_id) is None:
        raise _OperatorMismatchError("postprocessing operation intent phase_run_id is invalid")
    authority_path = authority_root / phase_run_id
    if authority_path.parent != authority_root:
        raise _OperatorMismatchError("postprocessing authority path escaped its configured root")
    return authority_path


def _verify_existing_authority_plan(
    authority_path: Path,
    *,
    phase_run_id: str,
    phase_plan_digest: str,
) -> None:
    try:
        metadata = authority_path.lstat()
    except OSError as exc:
        raise _OperatorMismatchError("existing postprocessing authority is unavailable") from exc
    if not stat.S_ISDIR(metadata.st_mode) or authority_path.is_symlink():
        raise _OperatorMismatchError("existing postprocessing authority must be one real directory")
    try:
        resolved = authority_path.resolve(strict=True)
    except OSError as exc:
        raise _OperatorMismatchError("existing postprocessing authority is unavailable") from exc
    if resolved != authority_path or resolved.parent != authority_path.parent:
        raise _OperatorMismatchError("existing postprocessing authority escaped its configured root")
    try:
        document = _stable_regular_bytes(authority_path / "phase-run.json", maximum=_MAX_INPUT_DOCUMENT_BYTES)
    except (OSError, ValueError) as exc:
        raise _OperatorMismatchError("existing postprocessing authority phase-run record is unsafe") from exc
    try:
        phase_run = json.loads(document)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _OperatorMismatchError("existing postprocessing authority phase-run record is invalid") from exc
    if not isinstance(phase_run, Mapping):
        raise _OperatorMismatchError("existing postprocessing authority phase-run record must be a mapping")
    if phase_run.get("phase_run_id") != phase_run_id or phase_run.get("phase_plan_digest") != phase_plan_digest:
        raise _OperatorMismatchError(
            "existing postprocessing authority does not match the persisted Phase Plan identity"
        )


def _bind_operator_v1_initial_attempt(
    authority_root: Path,
    *,
    phase_run_id: str,
    intent: Mapping[str, object],
) -> str:
    """Bind operator-v1 to the immutable Attempt created at materialization."""
    try:
        authority = require_postprocessing_v2_authority(authority_root, phase_run_id)
    except (OSError, TypeError, ValueError) as exc:
        raise _OperatorMismatchError("existing postprocessing authority is invalid") from exc
    initial = authority.phase_run.initial_attempt
    if initial.attempt_id != "attempt-0001" or initial.ordinal != 1:
        raise _OperatorMismatchError("postprocessing operator-v1 requires initial attempt-0001")
    if authority.phase_plan.digest != _phase_plan_digest_from_intent(intent):
        raise _OperatorMismatchError("postprocessing initial Attempt has a different Phase Plan identity")
    if not authority.events:
        raise _OperatorMismatchError("postprocessing initial Attempt lacks its materialization event")
    materialized = authority.events[0]
    if (
        materialized.sequence != 1
        or materialized.event_type != "phase-materialized"
        or materialized.phase_run_id != phase_run_id
        or materialized.attempt_id != initial.attempt_id
        or not isinstance(materialized.payload, PostprocessingMaterializedPayload)
        or materialized.payload.phase_plan_digest != authority.phase_plan.digest
        or materialized.payload.phase_runspec_digest != initial.phase_runspec_digest
    ):
        raise _OperatorMismatchError("postprocessing initial Attempt and materialization event identities differ")
    if authority.attempt_id != initial.attempt_id:
        raise _OperatorMismatchError(
            "postprocessing operator-v1 is bound to "
            f"{initial.attempt_id}, but durable authority selected {authority.attempt_id}"
        )
    if not authority.current_attempt_projection_complete or authority.runspec.digest != initial.phase_runspec_digest:
        raise _OperatorMismatchError("postprocessing initial Attempt projection is incomplete or differs")
    profile = _required_mapping(intent, "profile")
    if authority.runspec.cluster.profile_name != _required_string(profile, "name"):
        raise _OperatorMismatchError("postprocessing initial Attempt has a different Cluster Profile")
    try:
        qualification = json.loads(authority.runtime_qualification_bytes)
        tuple_mapping = qualification["tuple"]
        source_mapping = tuple_mapping["source_package_identity"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise _OperatorMismatchError("postprocessing initial Attempt qualification is invalid") from exc
    if not isinstance(source_mapping, Mapping):
        raise _OperatorMismatchError("postprocessing initial Attempt source identity is invalid")
    source = source_package_identity_from_mapping(cast("Mapping[str, object]", source_mapping))
    intended_source = _required_mapping(intent, "source")
    if (
        source.commit != _required_string(intended_source, "commit")
        or source.tree != _required_string(intended_source, "tree")
        or source.package_sha256 != _required_string(intended_source, "source_package_sha256")
        or source.manifest_sha256 != _required_string(intended_source, "source_package_manifest_sha256")
    ):
        raise _OperatorMismatchError("postprocessing initial Attempt has a different source identity")
    return initial.attempt_id


def _intent_path(intent: Mapping[str, object], section: str | None, key: str) -> Path:
    values = intent if section is None else _required_mapping(intent, section)
    rendered = _required_string(values, key)
    path = Path(rendered)
    if not path.is_absolute() or str(path) != rendered or ".." in path.parts:
        raise _OperatorMismatchError(f"postprocessing operation intent {key} path is not canonical absolute")
    return path


def _required_mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise _OperatorMismatchError(f"postprocessing operation intent {key} must be a mapping")
    return value


def _required_positive_number(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise _OperatorMismatchError(f"postprocessing operation intent {key} must be positive")
    return float(value)


def _write_create_once_json(path: Path, payload: Mapping[str, object]) -> None:
    document = _canonical_json(payload)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(document)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    _fsync_directory(path.parent)


def _atomic_json_replace(path: Path, payload: Mapping[str, object]) -> None:
    if os.path.lexists(path) and (path.is_symlink() or not path.is_file()):
        raise ValueError("postprocessing current checkpoint must be a regular file")
    document = _canonical_json(payload)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".current.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(document)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _stable_regular_bytes(path: Path, *, maximum: int) -> bytes:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink() or metadata.st_nlink != 1:
        raise ValueError(f"postprocessing operator input must be one regular file: {path}")
    if metadata.st_size > maximum:
        raise ValueError(f"postprocessing operator input exceeds its size bound: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as handle:
        opened = os.fstat(handle.fileno())
        if _input_stat_signature(opened) != _input_stat_signature(metadata):
            raise ValueError("postprocessing operator input changed during open")
        document = handle.read(maximum + 1)
        after = os.fstat(handle.fileno())
        reached = path.stat(follow_symlinks=False)
    if len(document) > maximum:
        raise ValueError(f"postprocessing operator input exceeds its size bound: {path}")
    if _input_stat_signature(opened) != _input_stat_signature(after) or _input_stat_signature(
        opened
    ) != _input_stat_signature(reached):
        raise ValueError("postprocessing operator input changed while being captured")
    return document


def _revalidate_operator_input_snapshot(path: Path, expected: bytes, *, label: str) -> None:
    try:
        observed = _stable_regular_bytes(path, maximum=_MAX_INPUT_DOCUMENT_BYTES)
    except (OSError, ValueError) as exc:
        raise _OperatorMismatchError(f"postprocessing operator {label} snapshot is no longer readable") from exc
    if observed != expected:
        raise _OperatorMismatchError(f"postprocessing operator {label} changed after intent capture")


def _input_stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_nlink,
    )


def _required_string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise _OperatorMismatchError(f"postprocessing operation intent {key} must be non-empty")
    return value


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _bounded_detail(value: str) -> str:
    return value.replace("\x00", "")[:_MAX_ERROR_CHARS]


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("postprocessing operator wall clock must be timezone-aware")
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


__all__ = [
    "DEFAULT_POSTPROCESSING_PHASE_OPERATOR_DEPENDENCIES",
    "PostprocessingPhaseOperatorDependencies",
    "PostprocessingPhaseOperatorResult",
    "run_postprocessing_phase",
]
