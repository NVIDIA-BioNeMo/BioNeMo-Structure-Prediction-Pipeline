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

"""Durable, accounting-confirmed cancellation for postprocessing."""

from __future__ import annotations

from pathlib import Path

from bspp.orchestration.contract.postprocessing_cancellation_events import (
    PostprocessingCancellationIntendedPayload,
    PostprocessingCancelledPayload,
    PostprocessingJobCancellationRequestIntendedPayload,
    PostprocessingJobCancellationRequestResultPayload,
)
from bspp.orchestration.control.postprocessing_authority_reader import reject_historical_postprocessing_mutation
from bspp.orchestration.control.postprocessing_authority_store import (
    append_event,
    format_timestamp,
    postprocessing_operation_lock,
    utc_now,
)
from bspp.orchestration.control.postprocessing_phase_lifecycle import (
    Clock,
    _append_array_parent_cancelled_actions,
    _append_complete_terminal_actions,
    _cancel_requested_jobs,
    _cancellation_request_history,
    _cancellation_result,
    _expected_terminal_job_ids,
    _recover_dispatching_actions,
    _required_submission_view,
    _submission_view,
    _terminal_task_evidence_digest,
    _terminal_views,
    _validate_postprocessing_authority,
    postprocessing_transport,
)
from bspp.orchestration.control.postprocessing_phase_types import PostprocessingLifecycleResult
from bspp.orchestration.control.transport import (
    TERMINAL_SLURM_STATES,
    CommandRunner,
    RemoteSlurmTransport,
    default_command_runner,
)


def cancel_postprocessing_phase(
    phase_run_id: str,
    *,
    authority_root: Path,
    clock: Clock | None = None,
    runner: CommandRunner = default_command_runner,
) -> PostprocessingLifecycleResult:
    """Cancel assigned work through durable intent/result request pairs."""
    now = clock or utc_now
    reject_historical_postprocessing_mutation(authority_root, phase_run_id)
    with postprocessing_operation_lock(authority_root, phase_run_id):
        authority = _validate_postprocessing_authority(authority_root, phase_run_id)
        if authority.sealed:
            raise ValueError(f"postprocessing Phase Cancellation is unavailable in state {authority.status!r}")
        if not authority.current_attempt_projection_complete:
            raise ValueError("postprocessing operation requires rerunning Retry to complete Attempt projections")
        if authority.status == "cancelled":
            return _cancellation_result(authority, requested=(), pending=())
        submission = _submission_view(authority)
        if not any(
            event.attempt_id == authority.attempt_id
            and isinstance(event.payload, PostprocessingCancellationIntendedPayload)
            for event in authority.events
        ):
            authority = append_event(
                authority,
                event_type="phase-cancellation-intended",
                occurred_at=format_timestamp(now()),
                payload=PostprocessingCancellationIntendedPayload(
                    submission_id=submission.submission_id if submission else None,
                    phase_runspec_digest=authority.runspec.digest,
                ),
            )
        transport: RemoteSlurmTransport | None = None
        if submission is not None:
            transport = postprocessing_transport(authority, runner=runner)
            authority = _recover_dispatching_actions(authority, transport=transport, now=now)
            submission = _required_submission_view(authority)
        if submission is None or not submission.job_ids:
            authority = append_event(
                authority,
                event_type="phase-cancelled",
                occurred_at=format_timestamp(now()),
                payload=PostprocessingCancelledPayload(
                    terminal_parent_job_ids=(),
                    terminal_action_ids=(),
                    terminal_task_evidence_digest=_terminal_task_evidence_digest(authority, {}),
                    cancelled_parent_job_ids=(),
                ),
            )
            return _cancellation_result(authority, requested=(), pending=())
        assert transport is not None
        submission = _required_submission_view(authority)
        parent_jobs = tuple(submission.job_ids_by_action().values())
        observation = transport.query_observation_best_effort(
            parent_jobs,
            require_exact_terminal_exit=True,
            expected_terminal_job_ids=_expected_terminal_job_ids(authority, submission),
        )
        authority, _ = _append_complete_terminal_actions(
            authority,
            submission=submission,
            observation=observation,
            now=now,
        )
        authority, _ = _append_array_parent_cancelled_actions(
            authority,
            submission=submission,
            observation=observation,
            now=now,
        )
        terminal_actions = _terminal_views(authority)
        by_action = {action.action_id: action for action in authority.runspec.payload.actions}
        assigned = submission.job_ids_by_action()
        selected = observation.selected_state_by_job_id()
        cancellable = tuple(
            (action_id, assigned[action_id])
            for action_id in by_action
            if action_id in assigned
            and action_id not in terminal_actions
            and (
                assigned[action_id] not in selected or selected[assigned[action_id]].state not in TERMINAL_SLURM_STATES
            )
        )
        requested: list[str] = []
        failures: list[str] = []
        for action_id, job_id in cancellable:
            history = _cancellation_request_history(authority, action_id=action_id)
            previous_intent, previous_result = history[-1] if history else (None, None)
            if previous_result is not None and previous_result.return_code == 0:
                continue
            if previous_intent is None or previous_result is not None:
                request_ordinal = len(history) + 1
                argv: tuple[str, ...] = ("scancel", job_id)
                authority = append_event(
                    authority,
                    event_type="phase-job-cancellation-request-intended",
                    occurred_at=format_timestamp(now()),
                    payload=PostprocessingJobCancellationRequestIntendedPayload(
                        action_id=action_id,
                        parent_job_id=job_id,
                        request_ordinal=request_ordinal,
                        scancel_argv=argv,
                    ),
                )
            else:
                request_ordinal = previous_intent.request_ordinal
                argv = previous_intent.scancel_argv
            result = transport.request_job_cancellation(job_id)
            authority = append_event(
                authority,
                event_type="phase-job-cancellation-request-result",
                occurred_at=format_timestamp(now()),
                payload=PostprocessingJobCancellationRequestResultPayload(
                    action_id=action_id,
                    parent_job_id=job_id,
                    request_ordinal=request_ordinal,
                    scancel_argv=argv,
                    return_code=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                ),
            )
            if result.returncode != 0:
                failures.append(result.stderr.strip() or f"scancel failed for job {job_id}")
            else:
                requested.append(job_id)
        final_observation = transport.query_observation_best_effort(
            parent_jobs,
            require_exact_terminal_exit=True,
            expected_terminal_job_ids=_expected_terminal_job_ids(authority, submission),
        )
        authority, _ = _append_complete_terminal_actions(
            authority,
            submission=submission,
            observation=final_observation,
            now=now,
        )
        authority, _ = _append_array_parent_cancelled_actions(
            authority,
            submission=submission,
            observation=final_observation,
            now=now,
        )
        terminal_actions = _terminal_views(authority)
        remaining = tuple(
            assigned[action.action_id]
            for action in authority.runspec.payload.actions
            if action.action_id in assigned and action.action_id not in terminal_actions
        )
        terminal_action_ids = tuple(
            action.action_id for action in authority.runspec.payload.actions if action.action_id in terminal_actions
        )
        if len(terminal_action_ids) == len(assigned) and not submission.dispatching:
            terminal_parent_ids = tuple(sorted((assigned[item] for item in terminal_action_ids), key=int))
            authority = append_event(
                authority,
                event_type="phase-cancelled",
                occurred_at=format_timestamp(now()),
                payload=PostprocessingCancelledPayload(
                    terminal_parent_job_ids=terminal_parent_ids,
                    terminal_action_ids=terminal_action_ids,
                    terminal_task_evidence_digest=_terminal_task_evidence_digest(authority, terminal_actions),
                    cancelled_parent_job_ids=tuple(sorted(_cancel_requested_jobs(authority), key=int)),
                ),
            )
        if failures and authority.status != "cancelled":
            raise ValueError("; ".join(failures))
        return _cancellation_result(authority, requested=tuple(requested), pending=remaining)


__all__ = ["cancel_postprocessing_phase"]
