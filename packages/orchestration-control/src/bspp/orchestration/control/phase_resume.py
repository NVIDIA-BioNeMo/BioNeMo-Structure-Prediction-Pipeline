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

"""Bounded idempotent reconciliation for the current Phase Attempt."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Literal, cast

from bspp.orchestration.contract.phase_reconciliation import (
    TERMINAL_PHASE_SLURM_STATES,
    FoldingActionTerminalObservationView,
    FoldingActionTerminalObservedEvent,
    FoldingActionTerminalObservedPayload,
    FoldingTaskTerminalEvidence,
    PhaseActionTerminalObservationView,
    PhaseActionTerminalObservedEvent,
    PhaseActionTerminalObservedPayload,
    phase_action_terminal_outcome,
)
from bspp.orchestration.control.phase_adapters import phase_authority_family
from bspp.orchestration.control.phase_authority import (
    PhaseAuthorityStore,
    PhaseAuthorityValidation,
    _select_terminal_observations_by_action,
    require_complete_current_runspec,
    validate_phase_run_id,
)
from bspp.orchestration.control.phase_lifecycle import require_active_phase_attempt
from bspp.orchestration.control.phase_submission import (
    _dispatch_planned_action,
    _prepare_submission_material,
    _recover_dispatched_action,
    _require_existing_submission_intent,
    _submission_intended_at,
    _topological_actions,
)
from bspp.orchestration.control.postprocessing_authority_reader import (
    reject_historical_postprocessing_mutation,
)
from bspp.orchestration.control.postprocessing_phase_adapter import (
    PostprocessingLifecycleResult,
    resume_postprocessing_phase,
)
from bspp.orchestration.control.transport import (
    CommandRunner,
    RemoteSlurmTransport,
    SlurmSubmissionRejected,
    default_command_runner,
)

Clock = Callable[[], datetime]
PhaseResumeOutcome = Literal["no-op", "reconciled", "failed"]

PHASE_RESUME_NEW_ACTION_LIMIT = 1


@dataclass(frozen=True)
class PhaseResumeActionResult:
    """Stable reconciliation projection for the sole frozen Runtime Action."""

    action_id: str
    durable_status: str
    job_id: str | None
    terminal: PhaseActionTerminalObservationView | FoldingActionTerminalObservationView | None

    def to_mapping(self) -> dict[str, object]:
        return {
            "action_id": self.action_id,
            "durable_status": self.durable_status,
            "job_id": self.job_id,
            "terminal": self.terminal.to_mapping() if self.terminal is not None else None,
        }


@dataclass(frozen=True)
class PhaseResumeResult:
    """Bounded Phase Resume result for one current immutable attempt."""

    phase_run_id: str
    attempt_id: str
    phase_runspec_digest: str
    submission_id: str
    outcome: PhaseResumeOutcome
    recovered_action_ids: tuple[str, ...]
    submitted_action_ids: tuple[str, ...]
    terminal_action_ids: tuple[str, ...]
    actions: tuple[PhaseResumeActionResult, ...]
    warnings: tuple[str, ...] = ()

    def to_mapping(self) -> dict[str, object]:
        return {
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "phase_runspec_digest": self.phase_runspec_digest,
            "submission_id": self.submission_id,
            "outcome": self.outcome,
            "recovered_action_ids": list(self.recovered_action_ids),
            "submitted_action_ids": list(self.submitted_action_ids),
            "terminal_action_ids": list(self.terminal_action_ids),
            "actions": [action.to_mapping() for action in self.actions],
            "warnings": list(self.warnings),
        }

    def render_json(self) -> str:
        return json.dumps(self.to_mapping(), indent=2, sort_keys=True) + "\n"


def resume_phase(
    phase_run_id: str,
    *,
    authority_root: Path,
    clock: Clock | None = None,
    authority_store: PhaseAuthorityStore | None = None,
    runner: CommandRunner = default_command_runner,
) -> PhaseResumeResult | PostprocessingLifecycleResult:
    """Run one bounded reconciliation cycle under exactly one operation lock."""
    family = phase_authority_family(authority_root, phase_run_id)
    if family == "postprocessing":
        reject_historical_postprocessing_mutation(authority_root, phase_run_id)
        if authority_store is not None:
            raise ValueError("postprocessing Phase Resume does not accept a preprocessing authority store")
        return resume_postprocessing_phase(
            phase_run_id,
            authority_root=authority_root,
            clock=clock,
            runner=runner,
        )
    is_folding = family == "folding"
    validate_phase_run_id(phase_run_id)
    store = authority_store or PhaseAuthorityStore(authority_root)
    with store.phase_operation_lock(phase_run_id):
        return _resume_phase_under_lock(
            phase_run_id,
            store=store,
            now=clock or _utc_now,
            runner=runner,
            is_folding=is_folding,
        )


def _resume_phase_under_lock(
    phase_run_id: str,
    *,
    store: PhaseAuthorityStore,
    now: Clock,
    runner: CommandRunner,
    is_folding: bool,
) -> PhaseResumeResult:
    """Reconcile once; the caller already owns the per-run operation lock."""
    authority = store.validate(phase_run_id)
    require_complete_current_runspec(authority, operation="Phase Resume")
    require_active_phase_attempt(authority.lifecycle, operation="Phase Resume")
    runspec_path, document_sha256, intended = _prepare_submission_material(authority)
    _require_existing_submission_intent(authority, intended)
    _require_action_shape(authority, is_folding=is_folding)
    assert authority.submission is not None

    transport = RemoteSlurmTransport(
        kind=authority.phase_runspec.cluster.transport,
        ssh_target=authority.phase_runspec.cluster.ssh_target,
        runner=runner,
    )

    if is_folding:
        return _resume_folding_actions(
            phase_run_id,
            store=store,
            authority=authority,
            runspec_path=runspec_path,
            runspec_document_sha256=document_sha256,
            transport=transport,
            now=now,
        )

    recovered: list[str] = []
    submitted: list[str] = []
    terminal: list[str] = []
    warnings: list[str] = []

    view = authority.submission.actions[0]
    if view.status == "dispatching":
        authority = _recover_dispatched_action(
            store,
            authority,
            view.plan,
            transport=transport,
            intended_at=_submission_intended_at(authority),
            now=now,
        )
        recovered.append(view.action_id)

    # Recompute from the authority returned by correlation repair so a recovered
    # assignment can be terminally observed during this same bounded cycle.
    assert authority.submission is not None
    view = authority.submission.actions[0]
    terminal_by_action = {item.action_id: item for item in authority.terminal_observations}
    if view.status == "submitted" and view.job_id is not None and view.action_id not in terminal_by_action:
        observation = transport.query_observation_best_effort((view.job_id,), require_exact_terminal_exit=True)
        warnings.extend(observation.warnings)
        selected = observation.selected_state_by_job_id().get(view.job_id)
        if selected is None:
            if not observation.warnings:
                warnings.append(f"submitted job {view.job_id} is missing from current scheduler observation")
        elif selected.source != "sacct":
            if selected.state in TERMINAL_PHASE_SLURM_STATES:
                warnings.append(f"job {view.job_id} has terminal queue state without sacct confirmation")
        else:
            outcome = phase_action_terminal_outcome(selected.state, selected.exit_code)
            if outcome is None:
                if selected.state == "COMPLETED" and selected.exit_code is None:
                    warnings.append(f"job {view.job_id} COMPLETED accounting lacks an exit code")
            else:
                payload = PhaseActionTerminalObservedPayload(
                    submission_id=authority.submission.submission_id,
                    phase_runspec_digest=authority.phase_runspec.digest,
                    action_id=view.action_id,
                    runtime_action_digest=view.plan.runtime_action_digest,
                    scheduler_correlation_token=view.scheduler_correlation_token,
                    job_id=view.job_id,
                    state=selected.state,
                    exit_code=selected.exit_code,
                    outcome=outcome,
                )
                observed_at = _format_timestamp(now())
                authority = store.append_event(
                    phase_run_id,
                    lambda sequence: PhaseActionTerminalObservedEvent(
                        sequence=sequence,
                        phase_run_id=phase_run_id,
                        attempt_id=authority.phase_runspec.attempt_id,
                        occurred_at=observed_at,
                        payload=payload,
                    ),
                )
                terminal.append(view.action_id)

    if authority.lifecycle.attempt_status == "failed":
        return _result(
            authority,
            outcome="failed",
            recovered=recovered,
            submitted=submitted,
            terminal=terminal,
            warnings=warnings,
        )

    assert authority.submission is not None
    view = authority.submission.actions[0]
    remaining_dispatch_budget = PHASE_RESUME_NEW_ACTION_LIMIT
    if view.status == "planned" and remaining_dispatch_budget > 0:
        try:
            authority = _dispatch_planned_action(
                store,
                authority,
                view.plan,
                runspec_path=runspec_path,
                runspec_document_sha256=document_sha256,
                transport=transport,
                now=now,
            )
        except SlurmSubmissionRejected as exc:
            authority = store.validate(phase_run_id)
            if (
                authority.submission is None
                or authority.submission.status != "failed"
                or authority.submission.actions[0].status != "rejected"
                or authority.lifecycle.attempt_status != "failed"
                or authority.lifecycle.run_status != "failed"
            ):
                raise AssertionError(
                    "durable Runtime Action rejection did not establish a failed Phase boundary"
                ) from exc
            return _result(
                authority,
                outcome="failed",
                recovered=recovered,
                submitted=submitted,
                terminal=terminal,
                warnings=warnings,
            )
        submitted.append(view.action_id)
        remaining_dispatch_budget -= 1
    if remaining_dispatch_budget < 0:
        raise AssertionError("Phase Resume dispatch budget underflow")

    changed = bool(recovered or submitted or terminal)
    return _result(
        authority,
        outcome="reconciled" if changed else "no-op",
        recovered=recovered,
        submitted=submitted,
        terminal=terminal,
        warnings=warnings,
    )


def _resume_folding_actions(
    phase_run_id: str,
    *,
    store: PhaseAuthorityStore,
    authority: PhaseAuthorityValidation,
    runspec_path: Path,
    runspec_document_sha256: str,
    transport: RemoteSlurmTransport,
    now: Clock,
) -> PhaseResumeResult:
    """Reconcile the ordered folding action graph one bounded cycle at a time."""
    recovered: list[str] = []
    submitted: list[str] = []
    terminal: list[str] = []
    warnings: list[str] = []
    remaining_dispatch_budget = PHASE_RESUME_NEW_ACTION_LIMIT

    for action in _topological_actions(authority.phase_runspec.payload.actions):
        assert authority.submission is not None
        view = next(item for item in authority.submission.actions if item.action_id == action.action_id)
        if view.status == "satisfied":
            # A fully carried fold action is durably terminal without a Slurm
            # job: it has no observation to reconcile and no dispatch to drive.
            continue
        if view.status == "dispatching":
            authority = _recover_dispatched_action(
                store,
                authority,
                view.plan,
                transport=transport,
                intended_at=_submission_intended_at(authority),
                now=now,
            )
            recovered.append(view.action_id)

        # Recompute from the authority returned by correlation repair so a recovered
        # assignment can be terminally observed during this same bounded cycle.
        assert authority.submission is not None
        submission = authority.submission
        view = next(item for item in submission.actions if item.action_id == action.action_id)
        terminal_by_action = _select_terminal_observations_by_action(
            authority.terminal_observations,
            authority.array_terminal_observations,
        )
        already_terminal = view.action_id in terminal_by_action
        if view.status == "submitted" and view.job_id is not None and not already_terminal:
            expected_indexes = view.plan.expected_task_indexes
            if expected_indexes:
                observation = transport.query_observation_best_effort(
                    (view.job_id,),
                    require_exact_terminal_exit=True,
                    expected_terminal_job_ids=tuple(f"{view.job_id}_{index}" for index in expected_indexes),
                )
                warnings.extend(observation.warnings)
                tasks = _complete_fold_array_task_set(view.job_id, expected_indexes, observation.sacct_jobs)
                if tasks is not None:
                    array_outcome: Literal["succeeded", "failed"] = (
                        "succeeded" if all(_fold_task_succeeded(task) for task in tasks) else "failed"
                    )
                    array_payload = FoldingActionTerminalObservedPayload(
                        submission_id=submission.submission_id,
                        phase_runspec_digest=authority.phase_runspec.digest,
                        action_id=view.action_id,
                        runtime_action_digest=view.plan.runtime_action_digest,
                        parent_job_id=view.job_id,
                        expected_task_indexes=expected_indexes,
                        tasks=tasks,
                        outcome=array_outcome,
                    )
                    observed_at = _format_timestamp(now())
                    authority = store.append_event(
                        phase_run_id,
                        partial(
                            FoldingActionTerminalObservedEvent,
                            phase_run_id=phase_run_id,
                            attempt_id=authority.phase_runspec.attempt_id,
                            occurred_at=observed_at,
                            payload=array_payload,
                        ),
                    )
                    terminal.append(view.action_id)
            else:
                observation = transport.query_observation_best_effort((view.job_id,), require_exact_terminal_exit=True)
                warnings.extend(observation.warnings)
                selected = observation.selected_state_by_job_id().get(view.job_id)
                if selected is None:
                    if not observation.warnings:
                        warnings.append(f"submitted job {view.job_id} is missing from current scheduler observation")
                elif selected.source != "sacct":
                    if selected.state in TERMINAL_PHASE_SLURM_STATES:
                        warnings.append(f"job {view.job_id} has terminal queue state without sacct confirmation")
                else:
                    outcome = phase_action_terminal_outcome(selected.state, selected.exit_code)
                    if outcome is None:
                        if selected.state == "COMPLETED" and selected.exit_code is None:
                            warnings.append(f"job {view.job_id} COMPLETED accounting lacks an exit code")
                    else:
                        payload = PhaseActionTerminalObservedPayload(
                            submission_id=submission.submission_id,
                            phase_runspec_digest=authority.phase_runspec.digest,
                            action_id=view.action_id,
                            runtime_action_digest=view.plan.runtime_action_digest,
                            scheduler_correlation_token=view.scheduler_correlation_token,
                            job_id=view.job_id,
                            state=selected.state,
                            exit_code=selected.exit_code,
                            outcome=outcome,
                        )
                        observed_at = _format_timestamp(now())
                        authority = store.append_event(
                            phase_run_id,
                            partial(
                                PhaseActionTerminalObservedEvent,
                                phase_run_id=phase_run_id,
                                attempt_id=authority.phase_runspec.attempt_id,
                                occurred_at=observed_at,
                                payload=payload,
                            ),
                        )
                        terminal.append(view.action_id)

        if authority.lifecycle.attempt_status == "failed":
            return _result(
                authority,
                outcome="failed",
                recovered=recovered,
                submitted=submitted,
                terminal=terminal,
                warnings=warnings,
            )

        assert authority.submission is not None
        view = next(item for item in authority.submission.actions if item.action_id == action.action_id)
        if view.status == "planned" and remaining_dispatch_budget > 0:
            try:
                authority = _dispatch_planned_action(
                    store,
                    authority,
                    view.plan,
                    runspec_path=runspec_path,
                    runspec_document_sha256=runspec_document_sha256,
                    transport=transport,
                    now=now,
                )
            except SlurmSubmissionRejected as exc:
                authority = store.validate(phase_run_id)
                if (
                    authority.submission is None
                    or authority.submission.status != "failed"
                    or next(item for item in authority.submission.actions if item.action_id == action.action_id).status
                    != "rejected"
                    or authority.lifecycle.attempt_status != "failed"
                    or authority.lifecycle.run_status != "failed"
                ):
                    raise AssertionError(
                        "durable Runtime Action rejection did not establish a failed Phase boundary"
                    ) from exc
                return _result(
                    authority,
                    outcome="failed",
                    recovered=recovered,
                    submitted=submitted,
                    terminal=terminal,
                    warnings=warnings,
                )
            submitted.append(view.action_id)
            remaining_dispatch_budget -= 1
    if remaining_dispatch_budget < 0:
        raise AssertionError("Phase Resume dispatch budget underflow")

    changed = bool(recovered or submitted or terminal)
    return _result(
        authority,
        outcome="reconciled" if changed else "no-op",
        recovered=recovered,
        submitted=submitted,
        terminal=terminal,
        warnings=warnings,
    )


def _require_action_shape(authority: PhaseAuthorityValidation, *, is_folding: bool) -> None:
    """Require the exact action shape for the current Phase family."""
    actions = authority.phase_runspec.payload.actions
    submission = authority.submission
    if is_folding:
        declared_ids = tuple(action.action_id for action in actions)
        if not declared_ids:
            raise ValueError("folding Phase Resume requires the complete declared action set")
        if submission is None or tuple(item.action_id for item in submission.actions) != declared_ids:
            raise ValueError("Phase Resume submission does not match the frozen folding action graph")
        return
    if len(actions) != 1 or actions[0].dependencies:
        raise ValueError("Phase Resume currently requires exactly one dependency-free Runtime Action")
    if submission is None or len(submission.actions) != 1 or submission.actions[0].action_id != actions[0].action_id:
        raise ValueError("Phase Resume submission does not match the frozen one-action RunSpec")


def _fold_task_succeeded(task: FoldingTaskTerminalEvidence) -> bool:
    return task.state == "COMPLETED" and task.exit_code in {"0", "0:0"}


def _complete_fold_array_task_set(
    parent_job_id: str,
    expected_indexes: tuple[int, ...],
    records: tuple[object, ...],
) -> tuple[FoldingTaskTerminalEvidence, ...] | None:
    """Return the exact per-task terminal evidence, or ``None`` when unresolved.

    Every expected ``<parent>_<index>`` task must have exactly one conclusive
    ``sacct`` record with a terminal state and an exact exit code; a missing,
    ambiguous, or non-terminal task leaves the whole set unresolved rather than
    collapsing to a bare parent or child row.
    """
    from bspp.orchestration.control.monitoring import SlurmJobRecord

    typed = tuple(item for item in records if isinstance(item, SlurmJobRecord) and item.source == "sacct")
    tasks: list[FoldingTaskTerminalEvidence] = []
    for task_index in expected_indexes:
        job_id = f"{parent_job_id}_{task_index}"
        matches = tuple(item for item in typed if item.job_id == job_id)
        if len(matches) != 1 or matches[0].state not in TERMINAL_PHASE_SLURM_STATES or matches[0].exit_code is None:
            return None
        record = matches[0]
        tasks.append(
            FoldingTaskTerminalEvidence(
                task_index=task_index,
                scheduler_job_id=job_id,
                state=cast("str", record.state),
                exit_code=cast("str", record.exit_code),
                source="sacct",
                restarts=(
                    record.restarts if record.state == "COMPLETED" and record.exit_code in {"0", "0:0"} else None
                ),
            )
        )
    return tuple(tasks)


def _result(
    authority: PhaseAuthorityValidation,
    *,
    outcome: PhaseResumeOutcome,
    recovered: list[str],
    submitted: list[str],
    terminal: list[str],
    warnings: list[str],
) -> PhaseResumeResult:
    submission = authority.submission
    if submission is None:
        raise ValueError("Phase Resume result requires a durable submission intent")
    terminal_by_action = _select_terminal_observations_by_action(
        authority.terminal_observations,
        authority.array_terminal_observations,
    )
    actions = tuple(
        PhaseResumeActionResult(
            action_id=view.action_id,
            durable_status=view.status,
            job_id=view.job_id,
            terminal=terminal_by_action.get(view.action_id),
        )
        for view in submission.actions
    )
    return PhaseResumeResult(
        phase_run_id=authority.phase_run.phase_run_id,
        attempt_id=authority.phase_runspec.attempt_id,
        phase_runspec_digest=authority.phase_runspec.digest,
        submission_id=submission.submission_id,
        outcome=outcome,
        recovered_action_ids=tuple(recovered),
        submitted_action_ids=tuple(submitted),
        terminal_action_ids=tuple(terminal),
        actions=actions,
        warnings=tuple(warnings),
    )


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _format_timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Phase Resume clock must be timezone-aware")
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


__all__ = [
    "PHASE_RESUME_NEW_ACTION_LIMIT",
    "PhaseResumeActionResult",
    "PhaseResumeOutcome",
    "PhaseResumeResult",
    "resume_phase",
]
