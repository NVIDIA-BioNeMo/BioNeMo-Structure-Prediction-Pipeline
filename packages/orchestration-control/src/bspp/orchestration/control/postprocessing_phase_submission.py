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

"""Submission and dispatch recovery for postprocessing Runtime Actions."""

from __future__ import annotations

import hashlib
from pathlib import Path

from bspp.orchestration.contract.postprocessing_runspec_v3 import PostprocessingPhaseRunSpecV3
from bspp.orchestration.contract.postprocessing_submission_events import (
    PostprocessingActionDispatchIntendedPayload,
    PostprocessingActionDispatchRejectedPayload,
    PostprocessingSubmissionActionPlan,
    PostprocessingSubmissionIntendedPayload,
)
from bspp.orchestration.control.postprocessing_attempt_projection import (
    validate_v3_attempt_execution_projection,
)
from bspp.orchestration.control.postprocessing_authority_reader import reject_historical_postprocessing_mutation
from bspp.orchestration.control.postprocessing_authority_store import (
    append_event,
    format_timestamp,
    mapping_digest,
    postprocessing_operation_lock,
    utc_now,
)
from bspp.orchestration.control.postprocessing_phase_lifecycle import (
    Clock,
    _all_actions_submitted,
    _append_action_assignment,
    _postprocessing_submission_id,
    _recover_dispatching_actions,
    _rendered_action_scripts,
    _renderer_contract_for_authority,
    _require_fresh_qualification,
    _required_submission_view,
    _stage_postprocessing_attempt,
    _submission_result,
    _submission_view,
    _validate_postprocessing_authority,
    postprocessing_transport,
)
from bspp.orchestration.control.postprocessing_phase_types import PostprocessingLifecycleResult
from bspp.orchestration.control.postprocessing_renderer3_compatibility import (
    render_authenticated_renderer3_scripts,
)
from bspp.orchestration.control.postprocessing_scheduler_identity import (
    postprocessing_cluster_action_script,
    postprocessing_scheduler_correlation_token,
)
from bspp.orchestration.control.postprocessing_v2_projection_safety import require_safe_v2_projection
from bspp.orchestration.control.transport import (
    CommandRunner,
    SlurmAction,
    SlurmSubmissionRejected,
    SlurmSubmissionUncertain,
    default_command_runner,
)


def submit_postprocessing_phase(
    phase_run_id: str,
    *,
    authority_root: Path,
    clock: Clock | None = None,
    runner: CommandRunner = default_command_runner,
) -> PostprocessingLifecycleResult:
    """Submit one immutable parent/array job for every frozen Runtime Action."""
    now = clock or utc_now
    reject_historical_postprocessing_mutation(authority_root, phase_run_id)
    with postprocessing_operation_lock(authority_root, phase_run_id):
        authority = _validate_postprocessing_authority(authority_root, phase_run_id)
        if authority.sealed:
            raise ValueError(f"Phase Run is already sealed: {phase_run_id}")
        if not authority.current_attempt_projection_complete:
            raise ValueError("postprocessing operation requires rerunning Retry to complete Attempt projections")
        if isinstance(authority.runspec, PostprocessingPhaseRunSpecV3):
            validate_v3_attempt_execution_projection(
                authority.runspec,
                authority.legacy_runspec,
                phase_plan_output_namespace=authority.phase_plan.output_namespace,
            )
        if authority.status in {"failed", "cancelling", "cancelled"}:
            raise ValueError(f"postprocessing Phase Submission is unavailable in state {authority.status!r}")
        submission = _submission_view(authority)
        if submission is None and not isinstance(authority.runspec, PostprocessingPhaseRunSpecV3):
            require_safe_v2_projection(authority)
        renderer_contract_version = (
            submission.actions[0].renderer_contract_version
            if submission is not None
            else _renderer_contract_for_authority(authority)
        )
        scripts = (
            render_authenticated_renderer3_scripts(
                authority,
                PostprocessingSubmissionIntendedPayload(
                    submission_id=submission.submission_id,
                    phase_runspec_digest=authority.runspec.digest,
                    actions=submission.actions,
                ),
            )
            if submission is not None and renderer_contract_version == 3
            else _rendered_action_scripts(authority, renderer_contract_version=renderer_contract_version)
        )
        if submission is not None and _all_actions_submitted(authority, submission):
            return _submission_result(authority, submission)
        if submission is None:
            _require_fresh_qualification(authority, now())
            intended_at = format_timestamp(now())
            submission_id = _postprocessing_submission_id(
                authority, scripts, renderer_contract_version=renderer_contract_version
            )
            authority = append_event(
                authority,
                event_type="phase-submission-intended",
                occurred_at=intended_at,
                payload=PostprocessingSubmissionIntendedPayload(
                    submission_id=submission_id,
                    phase_runspec_digest=authority.runspec.digest,
                    actions=tuple(
                        PostprocessingSubmissionActionPlan(
                            action_id=action.action_id,
                            runtime_action_digest=mapping_digest(action.to_mapping()),
                            dependencies=action.dependencies,
                            cluster_script_path=str(postprocessing_cluster_action_script(authority.runspec, action)),
                            script_sha256=hashlib.sha256(scripts[action.action_id].encode()).hexdigest(),
                            scheduler_correlation_token=postprocessing_scheduler_correlation_token(
                                authority.runspec, action
                            ),
                            renderer_contract_version=renderer_contract_version,
                        )
                        for action in authority.runspec.payload.actions
                    ),
                ),
            )
            submission = _submission_view(authority)
        assert submission is not None
        transport = postprocessing_transport(authority, runner=runner)
        _stage_postprocessing_attempt(authority, scripts=scripts, transport=transport, submission=submission)
        authority = _recover_dispatching_actions(authority, transport=transport, now=now)
        submission = _required_submission_view(authority)
        job_ids = submission.job_ids_by_action()
        for action in authority.runspec.payload.actions:
            if action.action_id in job_ids:
                continue
            if action.action_id in submission.rejected:
                raise ValueError(f"postprocessing Runtime Action submission was rejected: {action.action_id}")
            dependency_job_ids = tuple(job_ids[item] for item in action.dependencies)
            dispatch_at = format_timestamp(now())
            authority = append_event(
                authority,
                event_type="phase-action-dispatch-intended",
                occurred_at=dispatch_at,
                payload=PostprocessingActionDispatchIntendedPayload(
                    submission_id=submission.submission_id,
                    action_id=action.action_id,
                    scheduler_correlation_token=postprocessing_scheduler_correlation_token(authority.runspec, action),
                    dependency_job_ids=dependency_job_ids,
                ),
            )
            try:
                assigned = transport.submit_action(
                    SlurmAction(
                        action_id=action.action_id,
                        script_path=Path(postprocessing_cluster_action_script(authority.runspec, action)),
                        dependency_job_ids=dependency_job_ids,
                    )
                )
            except SlurmSubmissionRejected as exc:
                authority = append_event(
                    authority,
                    event_type="phase-action-dispatch-rejected",
                    occurred_at=format_timestamp(now()),
                    payload=PostprocessingActionDispatchRejectedPayload(
                        submission_id=submission.submission_id,
                        action_id=action.action_id,
                        return_code=exc.result.returncode,
                        stdout=exc.result.stdout,
                        stderr=exc.result.stderr,
                    ),
                )
                raise
            except SlurmSubmissionUncertain:
                authority = _recover_dispatching_actions(authority, transport=transport, now=now)
                recovered = _required_submission_view(authority)
                if recovered.job_id_for(action.action_id) is None:
                    raise
            else:
                authority = _append_action_assignment(authority, action, assigned.job_id, now=now)
            submission = _required_submission_view(authority)
            job_ids = submission.job_ids_by_action()
        return _submission_result(authority, _required_submission_view(authority))


__all__ = ["submit_postprocessing_phase"]
