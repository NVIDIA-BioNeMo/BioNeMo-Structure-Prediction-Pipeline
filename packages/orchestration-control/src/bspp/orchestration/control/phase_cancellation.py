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

"""Bounded, replay-driven cancellation for one current Phase Attempt."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Literal, cast

from bspp.orchestration.contract.phase_cancellation import (
    PhaseCancellationActionTarget,
    PhaseCancellationCompletedEvent,
    PhaseCancellationCompletedPayload,
    PhaseCancellationIntendedEvent,
    PhaseCancellationIntendedPayload,
    PhaseCancellationTerminalReference,
    PhaseJobCancellationRequestIntendedEvent,
    PhaseJobCancellationRequestIntendedPayload,
    PhaseJobCancellationRequestResultEvent,
    PhaseJobCancellationRequestResultPayload,
    phase_cancellation_id,
)
from bspp.orchestration.contract.phase_reconciliation import (
    PhaseActionTerminalObservedEvent,
    PhaseActionTerminalObservedPayload,
    phase_action_terminal_outcome,
)
from bspp.orchestration.control.monitoring import SlurmObservation, selected_records_by_job_id
from bspp.orchestration.control.phase_adapters import phase_authority_family
from bspp.orchestration.control.phase_authority import (
    PhaseAuthorityStore,
    PhaseAuthorityValidation,
    require_complete_current_runspec,
    validate_phase_run_id,
)
from bspp.orchestration.control.phase_lifecycle import require_cancellable_phase_attempt
from bspp.orchestration.control.phase_submission import (
    PhaseSubmissionCorrelationUnresolvedError,
    _recover_dispatched_action,
    _submission_intended_at,
)
from bspp.orchestration.control.postprocessing_authority_reader import (
    reject_historical_postprocessing_mutation,
)
from bspp.orchestration.control.postprocessing_phase_adapter import (
    PostprocessingLifecycleResult,
    cancel_postprocessing_phase,
)
from bspp.orchestration.control.transport import (
    CommandRunner,
    RemoteSlurmTransport,
    command_argv,
    default_command_runner,
)

Clock = Callable[[], datetime]
SchedulerSourceAvailability = Literal["skipped", "available", "unavailable"]


@dataclass(frozen=True)
class PhaseCancellationSchedulerSource:
    source: Literal["squeue", "sacct"]
    availability: SchedulerSourceAvailability

    def to_mapping(self) -> dict[str, object]:
        return {"source": self.source, "availability": self.availability}


@dataclass(frozen=True)
class PhaseCancellationResult:
    """Stable projection of durable cancellation progress."""

    phase_run_id: str
    attempt_id: str
    phase_runspec_digest: str
    cancellation_id: str
    status: Literal["cancelling", "cancelled"]
    target_job_ids: tuple[str, ...]
    terminal_job_ids: tuple[str, ...]
    cancellation_requested_job_ids: tuple[str, ...]
    pending_job_ids: tuple[str, ...]
    unresolved_action_ids: tuple[str, ...]
    scheduler_sources: tuple[PhaseCancellationSchedulerSource, PhaseCancellationSchedulerSource]
    warnings: tuple[str, ...] = ()

    def to_mapping(self) -> dict[str, object]:
        return {
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "phase_runspec_digest": self.phase_runspec_digest,
            "cancellation_id": self.cancellation_id,
            "status": self.status,
            "target_job_ids": list(self.target_job_ids),
            "terminal_job_ids": list(self.terminal_job_ids),
            "cancellation_requested_job_ids": list(self.cancellation_requested_job_ids),
            "pending_job_ids": list(self.pending_job_ids),
            "unresolved_action_ids": list(self.unresolved_action_ids),
            "scheduler_sources": {
                source.source: {"availability": source.availability} for source in self.scheduler_sources
            },
            "warnings": list(self.warnings),
        }

    def render_json(self) -> str:
        return json.dumps(self.to_mapping(), indent=2, sort_keys=True) + "\n"


def cancel_phase(
    phase_run_id: str,
    *,
    authority_root: Path,
    clock: Clock | None = None,
    authority_store: PhaseAuthorityStore | None = None,
    runner: CommandRunner = default_command_runner,
) -> PhaseCancellationResult | PostprocessingLifecycleResult:
    """Advance one cancellation by at most two observations and one scancel per job."""
    family = phase_authority_family(authority_root, phase_run_id)
    if family == "postprocessing":
        reject_historical_postprocessing_mutation(authority_root, phase_run_id)
        if authority_store is not None:
            raise ValueError("postprocessing Phase Cancellation does not accept a preprocessing authority store")
        return cancel_postprocessing_phase(
            phase_run_id,
            authority_root=authority_root,
            clock=clock,
            runner=runner,
        )
    if family == "folding":
        # Folding cancellation flows through the generic append-only body below;
        # the target/observation/scancel machinery is already action-graph generic.
        pass
    validate_phase_run_id(phase_run_id)
    store = authority_store or PhaseAuthorityStore(authority_root)
    with store.phase_operation_lock(phase_run_id):
        return _cancel_phase_under_lock(
            phase_run_id,
            store=store,
            now=clock or _utc_now,
            runner=runner,
        )


def _cancel_phase_under_lock(
    phase_run_id: str,
    *,
    store: PhaseAuthorityStore,
    now: Clock,
    runner: CommandRunner,
) -> PhaseCancellationResult:
    authority = store.validate(phase_run_id)
    require_complete_current_runspec(authority, operation="Phase Cancellation")
    lifecycle = require_cancellable_phase_attempt(authority.lifecycle)
    if lifecycle == "cancelled":
        return _result(authority, observation=None, warnings=())
    if lifecycle == "active":
        authority = _append_cancellation_intent(store, authority, now=now)
    assert authority.cancellation is not None

    transport = RemoteSlurmTransport(
        kind=authority.phase_runspec.cluster.transport,
        ssh_target=authority.phase_runspec.cluster.ssh_target,
        runner=runner,
    )
    warnings: list[str] = []
    authority = _recover_correlating_targets(store, authority, transport=transport, now=now, warnings=warnings)

    observation = _observe_bound_jobs(transport, authority)
    if observation is not None:
        warnings.extend(observation.warnings)
        authority = _append_terminal_observations(store, authority, observation=observation, now=now)

    invoked_scancel = False
    assert authority.cancellation is not None
    action_ids = tuple(action.action_id for action in authority.cancellation.actions)
    for action_id in action_ids:
        authority = store.validate(phase_run_id)
        assert authority.cancellation is not None
        action = next(item for item in authority.cancellation.actions if item.action_id == action_id)
        if action.disposition not in {"cancel-pending", "cancel-requesting"} or action.bound_job_id is None:
            continue
        intended_argv: tuple[str, ...]
        request_ordinal: int
        if action.disposition == "cancel-pending":
            request_ordinal = len(action.requests) + 1
            intended_argv = command_argv(
                ("scancel", action.bound_job_id),
                transport=transport.kind,
                ssh_target=transport.ssh_target,
            )
            cancellation_id = authority.cancellation.cancellation_id
            payload = PhaseJobCancellationRequestIntendedPayload(
                cancellation_id=cancellation_id,
                action_id=action.action_id,
                job_id=action.bound_job_id,
                request_ordinal=request_ordinal,
                scancel_argv=intended_argv,
            )
            occurred_at = _format_timestamp(now())
            authority = store.append_event(
                phase_run_id,
                partial(
                    PhaseJobCancellationRequestIntendedEvent,
                    phase_run_id=phase_run_id,
                    attempt_id=authority.phase_runspec.attempt_id,
                    occurred_at=occurred_at,
                    payload=payload,
                ),
            )
        else:
            request = action.requests[-1]
            request_ordinal = request.request_ordinal
            intended_argv = request.scancel_argv
        result = transport.request_job_cancellation(action.bound_job_id)
        invoked_scancel = True
        if result.argv != intended_argv:
            raise ValueError("scancel transport result argv differs from durable request intent")
        assert authority.cancellation is not None
        result_payload = PhaseJobCancellationRequestResultPayload(
            cancellation_id=authority.cancellation.cancellation_id,
            action_id=action.action_id,
            job_id=action.bound_job_id,
            request_ordinal=request_ordinal,
            scancel_argv=intended_argv,
            return_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
        occurred_at = _format_timestamp(now())
        authority = store.append_event(
            phase_run_id,
            partial(
                PhaseJobCancellationRequestResultEvent,
                phase_run_id=phase_run_id,
                attempt_id=authority.phase_runspec.attempt_id,
                occurred_at=occurred_at,
                payload=result_payload,
            ),
        )

    if invoked_scancel:
        second = _observe_bound_jobs(transport, authority)
        if second is not None:
            observation = second
            warnings.extend(second.warnings)
            authority = _append_terminal_observations(store, authority, observation=second, now=now)

    assert authority.cancellation is not None
    if all(action.disposition in {"no-job", "terminal-confirmed"} for action in authority.cancellation.actions):
        authority = _append_completion(store, authority, now=now)
    return _result(authority, observation=observation, warnings=tuple(warnings))


def _append_cancellation_intent(
    store: PhaseAuthorityStore,
    authority: PhaseAuthorityValidation,
    *,
    now: Clock,
) -> PhaseAuthorityValidation:
    submission = authority.submission
    if submission is None:
        targets = tuple(
            PhaseCancellationActionTarget(
                action_id=action.action_id,
                initial_submission_status="not-submitted",
            )
            for action in authority.phase_runspec.payload.actions
        )
        submission_id = None
    else:
        if any(action.status == "rejected" for action in submission.actions):
            raise ValueError("active Phase Cancellation cannot target a rejected submission")
        targets = tuple(
            PhaseCancellationActionTarget(
                action_id=action.action_id,
                initial_submission_status=cast("Literal['planned', 'dispatching', 'submitted']", action.status),
                scheduler_correlation_token=action.scheduler_correlation_token,
                job_id=action.job_id,
            )
            for action in submission.actions
        )
        submission_id = submission.submission_id
    cancellation_id = phase_cancellation_id(
        phase_run_id=authority.phase_run.phase_run_id,
        attempt_id=authority.phase_runspec.attempt_id,
        submission_id=submission_id,
        phase_runspec_digest=authority.phase_runspec.digest,
        targets=targets,
    )
    payload = PhaseCancellationIntendedPayload(
        cancellation_id=cancellation_id,
        submission_id=submission_id,
        phase_runspec_digest=authority.phase_runspec.digest,
        targets=targets,
    )
    occurred_at = _format_timestamp(now())
    return store.append_event(
        authority.phase_run.phase_run_id,
        lambda sequence: PhaseCancellationIntendedEvent(
            sequence=sequence,
            phase_run_id=authority.phase_run.phase_run_id,
            attempt_id=authority.phase_runspec.attempt_id,
            occurred_at=occurred_at,
            payload=payload,
        ),
    )


def _recover_correlating_targets(
    store: PhaseAuthorityStore,
    authority: PhaseAuthorityValidation,
    *,
    transport: RemoteSlurmTransport,
    now: Clock,
    warnings: list[str],
) -> PhaseAuthorityValidation:
    assert authority.cancellation is not None
    action_ids = tuple(
        action.action_id for action in authority.cancellation.actions if action.disposition == "correlating"
    )
    for action_id in action_ids:
        assert authority.submission is not None
        plan = next(action.plan for action in authority.submission.actions if action.action_id == action_id)
        try:
            authority = _recover_dispatched_action(
                store,
                authority,
                plan,
                transport=transport,
                intended_at=_submission_intended_at(authority),
                now=now,
            )
        except PhaseSubmissionCorrelationUnresolvedError as exc:
            warnings.append(str(exc))
    return authority


def _observe_bound_jobs(
    transport: RemoteSlurmTransport,
    authority: PhaseAuthorityValidation,
) -> SlurmObservation | None:
    assert authority.cancellation is not None
    job_ids = tuple(
        action.bound_job_id
        for action in authority.cancellation.actions
        if action.bound_job_id is not None and action.disposition != "terminal-confirmed"
    )
    if not job_ids:
        return None
    return transport.query_observation_best_effort(job_ids, require_exact_terminal_exit=True)


def _append_terminal_observations(
    store: PhaseAuthorityStore,
    authority: PhaseAuthorityValidation,
    *,
    observation: SlurmObservation,
    now: Clock,
) -> PhaseAuthorityValidation:
    by_job_id = selected_records_by_job_id(observation.sacct_jobs)
    assert authority.cancellation is not None
    action_ids = tuple(
        action.action_id
        for action in authority.cancellation.actions
        if action.bound_job_id is not None and action.disposition != "terminal-confirmed"
    )
    for action_id in action_ids:
        assert authority.cancellation is not None
        cancel_action = next(item for item in authority.cancellation.actions if item.action_id == action_id)
        assert cancel_action.bound_job_id is not None
        record = by_job_id.get(cancel_action.bound_job_id)
        if record is None or record.state is None:
            continue
        outcome = phase_action_terminal_outcome(record.state, record.exit_code)
        if outcome is None:
            continue
        if authority.submission is None:
            raise ValueError("terminal cancellation evidence requires durable submission authority")
        submission_action = next(item for item in authority.submission.actions if item.action_id == action_id)
        if submission_action.job_id != cancel_action.bound_job_id:
            raise ValueError("terminal cancellation evidence does not match durable assignment")
        payload = PhaseActionTerminalObservedPayload(
            submission_id=authority.submission.submission_id,
            phase_runspec_digest=authority.phase_runspec.digest,
            action_id=action_id,
            runtime_action_digest=submission_action.plan.runtime_action_digest,
            scheduler_correlation_token=submission_action.scheduler_correlation_token,
            job_id=cancel_action.bound_job_id,
            state=record.state,
            exit_code=record.exit_code,
            outcome=outcome,
        )
        occurred_at = _format_timestamp(now())
        authority = store.append_event(
            authority.phase_run.phase_run_id,
            partial(
                PhaseActionTerminalObservedEvent,
                phase_run_id=authority.phase_run.phase_run_id,
                attempt_id=authority.phase_runspec.attempt_id,
                occurred_at=occurred_at,
                payload=payload,
            ),
        )
    return authority


def _append_completion(
    store: PhaseAuthorityStore,
    authority: PhaseAuthorityValidation,
    *,
    now: Clock,
) -> PhaseAuthorityValidation:
    assert authority.cancellation is not None
    references = tuple(
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
    payload = PhaseCancellationCompletedPayload(
        cancellation_id=authority.cancellation.cancellation_id,
        terminal_references=references,
    )
    occurred_at = _format_timestamp(now())
    return store.append_event(
        authority.phase_run.phase_run_id,
        lambda sequence: PhaseCancellationCompletedEvent(
            sequence=sequence,
            phase_run_id=authority.phase_run.phase_run_id,
            attempt_id=authority.phase_runspec.attempt_id,
            occurred_at=occurred_at,
            payload=payload,
        ),
    )


def _result(
    authority: PhaseAuthorityValidation,
    *,
    observation: SlurmObservation | None,
    warnings: tuple[str, ...],
) -> PhaseCancellationResult:
    cancellation = authority.cancellation
    if cancellation is None:
        raise ValueError("Phase Cancellation result requires durable cancellation authority")
    target_jobs = tuple(action.bound_job_id for action in cancellation.actions if action.bound_job_id is not None)
    terminal_jobs = tuple(
        action.bound_job_id
        for action in cancellation.actions
        if action.bound_job_id is not None and action.disposition == "terminal-confirmed"
    )
    requested_jobs = tuple(
        action.bound_job_id
        for action in cancellation.actions
        if action.bound_job_id is not None and any(request.return_code == 0 for request in action.requests)
    )
    pending_jobs = tuple(
        action.bound_job_id
        for action in cancellation.actions
        if action.bound_job_id is not None and action.disposition != "terminal-confirmed"
    )
    unresolved = tuple(action.action_id for action in cancellation.actions if action.disposition == "correlating")
    sources = _scheduler_sources(observation)
    return PhaseCancellationResult(
        phase_run_id=authority.phase_run.phase_run_id,
        attempt_id=authority.phase_runspec.attempt_id,
        phase_runspec_digest=authority.phase_runspec.digest,
        cancellation_id=cancellation.cancellation_id,
        status=cancellation.status,
        target_job_ids=target_jobs,
        terminal_job_ids=terminal_jobs,
        cancellation_requested_job_ids=requested_jobs,
        pending_job_ids=pending_jobs,
        unresolved_action_ids=unresolved,
        scheduler_sources=sources,
        warnings=warnings,
    )


def _scheduler_sources(
    observation: SlurmObservation | None,
) -> tuple[PhaseCancellationSchedulerSource, PhaseCancellationSchedulerSource]:
    if observation is None:
        return (
            PhaseCancellationSchedulerSource("squeue", "skipped"),
            PhaseCancellationSchedulerSource("sacct", "skipped"),
        )
    return (
        PhaseCancellationSchedulerSource(
            "squeue", "unavailable" if observation.squeue.parser == "unavailable" else "available"
        ),
        PhaseCancellationSchedulerSource(
            "sacct", "unavailable" if observation.sacct.parser == "unavailable" else "available"
        ),
    )


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _format_timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Phase Cancellation clock must be timezone-aware")
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


__all__ = [
    "PhaseCancellationResult",
    "PhaseCancellationSchedulerSource",
    "cancel_phase",
]
