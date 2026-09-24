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

"""Read-only status and bounded reconciliation for postprocessing."""

from __future__ import annotations

from pathlib import Path

from bspp.orchestration.control.postprocessing_authority_reader import (
    read_postprocessing_authority,
    reject_historical_postprocessing_mutation,
)
from bspp.orchestration.control.postprocessing_authority_store import postprocessing_operation_lock, utc_now
from bspp.orchestration.control.postprocessing_phase_lifecycle import (
    Clock,
    _append_complete_terminal_actions,
    _current_task_statuses,
    _expected_terminal_job_ids,
    _recover_dispatching_actions,
    _required_submission_view,
    _terminal_views,
    _validate_postprocessing_authority,
    postprocessing_transport,
)
from bspp.orchestration.control.postprocessing_phase_types import (
    HistoricalPostprocessingAuthorityV1,
    PostprocessingLifecycleResult,
)
from bspp.orchestration.control.postprocessing_status_projection import project_postprocessing_authority_status
from bspp.orchestration.control.transport import CommandRunner, default_command_runner


def status_postprocessing_phase(
    phase_run_id: str,
    *,
    authority_root: Path,
    runner: CommandRunner = default_command_runner,
) -> PostprocessingLifecycleResult:
    """Observe durable postprocessing state and current scheduler state without mutation."""
    authority = read_postprocessing_authority(authority_root, phase_run_id)
    historical_v1 = isinstance(authority, HistoricalPostprocessingAuthorityV1)
    submission = authority.submission_state
    job_ids = tuple(submission.job_ids_by_action().values()) if submission else ()
    observation = None
    warnings: list[str] = []
    if job_ids and not historical_v1:
        assert submission is not None
        observation = postprocessing_transport(authority, runner=runner).query_observation_best_effort(
            job_ids,
            require_exact_terminal_exit=True,
            expected_terminal_job_ids=_expected_terminal_job_ids(authority, submission),
        )
        warnings.extend(observation.warnings)
    terminals = _terminal_views(authority)
    authority_projection = project_postprocessing_authority_status(authority)
    action_rows: list[dict[str, object]] = []
    durable_jobs = submission.job_ids_by_action() if submission else {}
    rejected = submission.rejected if submission else frozenset()
    dispatching = submission.dispatching if submission else frozenset()
    selected = observation.selected_state_by_job_id() if observation is not None else {}
    for action in authority.runspec.payload.actions:
        job_id = durable_jobs.get(action.action_id)
        durable_status = (
            "submitted"
            if job_id is not None
            else "rejected"
            if action.action_id in rejected
            else "dispatching"
            if action.action_id in dispatching
            else "planned"
            if submission is not None
            else "not-submitted"
        )
        row: dict[str, object] = {
            "action_id": action.action_id,
            "durable_status": durable_status,
            "job_id": job_id,
            "expected_task_indexes": list(action.expected_task_indexes),
            "terminal": (terminals[action.action_id].to_mapping() if action.action_id in terminals else None),
            "tasks": (
                _historical_task_statuses(action, job_id, terminals.get(action.action_id))
                if historical_v1
                else _current_task_statuses(
                    action,
                    job_id,
                    observation.sacct_jobs if observation is not None else (),
                    terminals.get(action.action_id),
                )
            ),
        }
        scheduler = selected.get(job_id) if job_id is not None else None
        row["scheduler"] = scheduler.to_mapping() if scheduler is not None else None
        action_rows.append(row)
    return PostprocessingLifecycleResult(
        operation="status",
        phase_run_id=phase_run_id,
        attempt_id=authority.attempt_id,
        status=authority.status,
        details={
            **authority_projection.to_mapping(),
            "actions": action_rows,
            "scheduler": observation.to_mapping() if observation is not None else None,
            "warnings": warnings,
        },
    )


def _historical_task_statuses(
    action: object,
    parent_job_id: str | None,
    terminal: object | None,
) -> list[dict[str, object]]:
    from bspp.orchestration.contract.postprocessing_action_contract import PostprocessingRuntimeAction
    from bspp.orchestration.contract.postprocessing_terminal_events import (
        PostprocessingActionTerminalObservedPayload,
    )

    if not isinstance(action, PostprocessingRuntimeAction):
        raise TypeError("historical status requires a typed Runtime Action")
    if isinstance(terminal, PostprocessingActionTerminalObservedPayload):
        return [
            {
                "task_index": task.task_index,
                "scheduler_job_id": task.scheduler_job_id,
                "observation_status": "observed",
                "state": task.state,
                "exit_code": task.exit_code,
                "source": task.source,
            }
            for task in terminal.tasks
        ]
    expected: tuple[int | None, ...] = action.expected_task_indexes or (None,)
    return [
        {
            "task_index": task_index,
            "scheduler_job_id": (
                None
                if parent_job_id is None
                else parent_job_id
                if task_index is None
                else f"{parent_job_id}_{task_index}"
            ),
            "observation_status": "not-applicable" if parent_job_id is None else "missing",
            "state": None,
            "exit_code": None,
            "source": None,
        }
        for task_index in expected
    ]


def resume_postprocessing_phase(
    phase_run_id: str,
    *,
    authority_root: Path,
    clock: Clock | None = None,
    runner: CommandRunner = default_command_runner,
) -> PostprocessingLifecycleResult:
    """Run one bounded correlation and exact task-set accounting cycle."""
    now = clock or utc_now
    reject_historical_postprocessing_mutation(authority_root, phase_run_id)
    with postprocessing_operation_lock(authority_root, phase_run_id):
        authority = _validate_postprocessing_authority(authority_root, phase_run_id)
        if authority.sealed or authority.status in {"cancelled", "cancelling"}:
            raise ValueError(f"postprocessing Phase Resume is unavailable in state {authority.status!r}")
        if not authority.current_attempt_projection_complete:
            raise ValueError("postprocessing operation requires rerunning Retry to complete Attempt projections")
        submission = _required_submission_view(authority)
        transport = postprocessing_transport(authority, runner=runner)
        authority = _recover_dispatching_actions(authority, transport=transport, now=now)
        submission = _required_submission_view(authority)
        job_ids = submission.job_ids_by_action()
        if len(job_ids) != len(authority.runspec.payload.actions):
            raise ValueError("postprocessing Phase Resume cannot invent a missing initial submission")
        observation = transport.query_observation_best_effort(
            tuple(job_ids.values()),
            require_exact_terminal_exit=True,
            expected_terminal_job_ids=_expected_terminal_job_ids(authority, submission),
        )
        authority, newly_terminal = _append_complete_terminal_actions(
            authority,
            submission=submission,
            observation=observation,
            now=now,
        )
        terminal_by_action = _terminal_views(authority)
        outcome = "failed" if authority.status == "failed" else "reconciled" if newly_terminal else "no-op"
        return PostprocessingLifecycleResult(
            operation="resume",
            phase_run_id=phase_run_id,
            attempt_id=authority.attempt_id,
            status=authority.status,
            details={
                "outcome": outcome,
                "submission_id": submission.submission_id,
                "newly_terminal_action_ids": list(newly_terminal),
                "terminal_actions": [item.to_mapping() for item in terminal_by_action.values()],
                "warnings": list(observation.warnings),
            },
        )


__all__ = ["resume_postprocessing_phase", "status_postprocessing_phase"]
