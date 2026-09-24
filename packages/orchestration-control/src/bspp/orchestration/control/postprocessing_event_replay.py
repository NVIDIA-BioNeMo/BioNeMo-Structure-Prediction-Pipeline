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

"""Typed replay of the durable postprocessing event stream."""

from __future__ import annotations

from dataclasses import dataclass

from bspp.orchestration.contract.phase import canonical_mapping_digest
from bspp.orchestration.contract.postprocessing_cancellation_events import (
    PostprocessingCancellationIntendedPayload,
    PostprocessingCancelledPayload,
    PostprocessingJobCancellationRequestIntendedPayload,
    PostprocessingJobCancellationRequestResultPayload,
)
from bspp.orchestration.contract.postprocessing_event import PostprocessingPhaseEvent
from bspp.orchestration.contract.postprocessing_phase_receipt import PostprocessingFinalizedPayload
from bspp.orchestration.contract.postprocessing_retry_events import PostprocessingAttemptRetriedPayload
from bspp.orchestration.contract.postprocessing_runspec import ReadablePostprocessingPhaseRunSpec
from bspp.orchestration.contract.postprocessing_runspec_v3 import PostprocessingPhaseRunSpecV3
from bspp.orchestration.contract.postprocessing_submission_events import (
    PostprocessingActionDispatchIntendedPayload,
    PostprocessingActionDispatchRejectedPayload,
    PostprocessingActionSubmittedPayload,
    PostprocessingMaterializedPayload,
    PostprocessingSubmissionActionPlan,
    PostprocessingSubmissionIntendedPayload,
)
from bspp.orchestration.contract.postprocessing_terminal_events import (
    PostprocessingActionTerminalObservedPayload,
    PostprocessingArrayParentCancelledObservedPayload,
    PostprocessingTerminalObservation,
)
from bspp.orchestration.control.postprocessing_phase_types import (
    PostprocessingLifecycleStatus,
    PostprocessingReplayState,
    PostprocessingSubmissionState,
)
from bspp.orchestration.control.postprocessing_scheduler_identity import (
    postprocessing_cluster_action_script,
    postprocessing_scheduler_correlation_token,
)


@dataclass
class _CancellationRequest:
    intent: PostprocessingJobCancellationRequestIntendedPayload
    result: PostprocessingJobCancellationRequestResultPayload | None = None


def replay_postprocessing_events(
    events: tuple[PostprocessingPhaseEvent, ...],
    *,
    initial_runspec: ReadablePostprocessingPhaseRunSpec,
) -> PostprocessingReplayState:
    """Validate and replay concrete event payloads without map-shaped intermediates."""
    status: PostprocessingLifecycleStatus = "materialized"
    sealed = False
    runspec = initial_runspec
    embedded_legacy_bytes: bytes | None = None
    embedded_qualification_bytes: bytes | None = None
    submitted_jobs: dict[str, str] = {}
    terminal_payloads: dict[str, PostprocessingTerminalObservation] = {}
    submission_payload: PostprocessingSubmissionIntendedPayload | None = None
    submission_intended_at: str | None = None
    submission_plans: dict[str, PostprocessingSubmissionActionPlan] = {}
    dispatch_payloads: dict[str, PostprocessingActionDispatchIntendedPayload] = {}
    rejected_actions: set[str] = set()
    cancellation_intended = False
    cancellation_requested: set[str] = set()
    cancellation_requests: dict[str, list[_CancellationRequest]] = {}
    materialized_seen = False

    for event in events:
        if sealed:
            raise ValueError("postprocessing authority contains an event after terminal finalization")
        if event.phase_run_id != runspec.phase_run_id:
            raise ValueError("postprocessing event Phase Run id differs from authority")
        payload = event.payload

        if isinstance(payload, PostprocessingMaterializedPayload):
            if materialized_seen or event.sequence != 1 or event.attempt_id != runspec.attempt_id:
                raise ValueError("postprocessing materialization event must be the unique first event")
            if (
                payload.phase_plan_digest != runspec.phase_plan_digest
                or payload.phase_runspec_digest != runspec.digest
                or payload.action_graph_digest != runspec.payload.action_graph_digest
            ):
                raise ValueError("postprocessing materialization event differs from initial authority")
            materialized_seen = True
            continue

        if isinstance(payload, PostprocessingAttemptRetriedPayload):
            if (
                isinstance(runspec, PostprocessingPhaseRunSpecV3)
                != isinstance(payload.successor_phase_runspec, PostprocessingPhaseRunSpecV3)
                or status not in {"failed", "cancelled"}
                or payload.predecessor_outcome != status
                or payload.predecessor_attempt_id != runspec.attempt_id
                or payload.predecessor_phase_runspec_digest != runspec.digest
                or payload.phase_plan_digest != runspec.phase_plan_digest
                or payload.logical_input_manifest_digest != runspec.payload.logical_inputs.digest
                or payload.scientific_identity_digest != runspec.payload.scientific_identity.digest
                or payload.acceptance_policy_semantic_digest != runspec.payload.acceptance_policy.semantic_digest
                or payload.action_graph_digest != runspec.payload.action_graph_digest
                or payload.action_semantics_digest != runspec.payload.action_semantics_digest
                or event.attempt_id != payload.successor_phase_runspec.attempt_id
            ):
                raise ValueError("postprocessing Retry event does not preserve terminal predecessor invariants")
            runspec = payload.successor_phase_runspec
            embedded_legacy_bytes = payload.successor_legacy_runspec_yaml.encode()
            embedded_qualification_bytes = payload.successor_runtime_qualification_json.encode()
            status = "materialized"
            submitted_jobs = {}
            terminal_payloads = {}
            submission_payload = None
            submission_intended_at = None
            submission_plans = {}
            dispatch_payloads = {}
            rejected_actions = set()
            cancellation_intended = False
            cancellation_requested = set()
            cancellation_requests = {}
            continue

        if event.attempt_id != runspec.attempt_id:
            raise ValueError("postprocessing event Attempt id differs from current replay Attempt")
        actions_by_id = {item.action_id: item for item in runspec.payload.actions}

        if isinstance(payload, PostprocessingSubmissionIntendedPayload):
            if submission_payload is not None or status != "materialized":
                raise ValueError("postprocessing Attempt has an invalid duplicate or late submission intent")
            if payload.phase_runspec_digest != runspec.digest or tuple(
                item.action_id for item in payload.actions
            ) != tuple(actions_by_id):
                raise ValueError("postprocessing submission action plans differ from RunSpec order")
            versions = {plan.renderer_contract_version for plan in payload.actions}
            if len(versions) != 1:
                raise ValueError("postprocessing submission action plans mix renderer contract versions")
            renderer_contract_version = next(iter(versions))
            for planned_action, plan in zip(runspec.payload.actions, payload.actions, strict=True):
                expected_token = postprocessing_scheduler_correlation_token(runspec, planned_action)
                legacy_expected_token = postprocessing_scheduler_correlation_token(runspec, planned_action, legacy=True)
                expected_script = str(postprocessing_cluster_action_script(runspec, planned_action))
                legacy_expected_script = str(postprocessing_cluster_action_script(runspec, planned_action, legacy=True))
                if (
                    plan.runtime_action_digest != planned_action.digest
                    or plan.dependencies != planned_action.dependencies
                    or plan.cluster_script_path not in (expected_script, legacy_expected_script)
                    or plan.scheduler_correlation_token not in (expected_token, legacy_expected_token)
                    or plan.renderer_contract_version != renderer_contract_version
                ):
                    raise ValueError("postprocessing submission action plan differs from frozen action authority")
            submission_payload = payload
            submission_intended_at = event.occurred_at
            submission_plans = {item.action_id: item for item in payload.actions}
            status = "submitted"
        elif isinstance(payload, PostprocessingActionDispatchIntendedPayload):
            replay_action = actions_by_id.get(payload.action_id)
            if submission_payload is None or replay_action is None or payload.action_id in dispatch_payloads:
                raise ValueError("postprocessing action dispatch has no unique frozen submission plan")
            plan = submission_plans[payload.action_id]
            expected_dependency_jobs = tuple(submitted_jobs[item] for item in replay_action.dependencies)
            if (
                payload.submission_id != submission_payload.submission_id
                or payload.scheduler_correlation_token != plan.scheduler_correlation_token
                or payload.dependency_job_ids != expected_dependency_jobs
            ):
                raise ValueError("postprocessing action dispatch differs from submission dependencies")
            dispatch_payloads[payload.action_id] = payload
        elif isinstance(payload, PostprocessingActionSubmittedPayload):
            replay_action = actions_by_id.get(payload.action_id)
            dispatch = dispatch_payloads.get(payload.action_id)
            if (
                replay_action is None
                or dispatch is None
                or payload.action_id in submitted_jobs
                or payload.action_id in rejected_actions
            ):
                raise ValueError("postprocessing action has an invalid or duplicate Slurm assignment")
            if (
                payload.submission_id != dispatch.submission_id
                or payload.scheduler_correlation_token != dispatch.scheduler_correlation_token
                or payload.expected_task_indexes != replay_action.expected_task_indexes
            ):
                raise ValueError("postprocessing Slurm assignment differs from its dispatch and frozen action")
            submitted_jobs[payload.action_id] = payload.parent_job_id
        elif isinstance(payload, PostprocessingActionTerminalObservedPayload):
            replay_action = actions_by_id.get(payload.action_id)
            if (
                replay_action is None
                or payload.action_id not in submitted_jobs
                or payload.action_id in terminal_payloads
                or submission_payload is None
                or payload.submission_id != submission_payload.submission_id
                or payload.phase_runspec_digest != runspec.digest
                or payload.runtime_action_digest != replay_action.digest
                or payload.parent_job_id != submitted_jobs[payload.action_id]
                or payload.expected_task_indexes != replay_action.expected_task_indexes
            ):
                raise ValueError("postprocessing terminal evidence differs from durable assignment authority")
            terminal_payloads[payload.action_id] = payload
            if payload.outcome == "failed" and status != "cancelling":
                status = "failed"
        elif isinstance(payload, PostprocessingArrayParentCancelledObservedPayload):
            replay_action = actions_by_id.get(payload.action_id)
            if (
                not cancellation_intended
                or replay_action is None
                or payload.action_id not in submitted_jobs
                or payload.action_id in terminal_payloads
                or submission_payload is None
                or payload.submission_id != submission_payload.submission_id
                or payload.phase_runspec_digest != runspec.digest
                or payload.runtime_action_digest != replay_action.digest
                or payload.parent_job_id != submitted_jobs[payload.action_id]
                or payload.expected_task_indexes != replay_action.expected_task_indexes
                or not replay_action.expected_task_indexes
            ):
                raise ValueError("postprocessing array parent cancellation differs from durable assignment authority")
            terminal_payloads[payload.action_id] = payload
        elif isinstance(payload, PostprocessingActionDispatchRejectedPayload):
            dispatch = dispatch_payloads.get(payload.action_id)
            if (
                dispatch is None
                or payload.action_id in submitted_jobs
                or payload.action_id in rejected_actions
                or payload.submission_id != dispatch.submission_id
            ):
                raise ValueError("postprocessing dispatch rejection differs from a unique pending dispatch")
            rejected_actions.add(payload.action_id)
            status = "failed"
        elif isinstance(payload, PostprocessingCancellationIntendedPayload):
            expected_submission = submission_payload.submission_id if submission_payload is not None else None
            if (
                cancellation_intended
                or status in {"accepted", "cancelled", "cancelling"}
                or payload.phase_runspec_digest != runspec.digest
                or payload.submission_id != expected_submission
            ):
                raise ValueError("postprocessing cancellation intent differs from current Attempt authority")
            cancellation_intended = True
            status = "cancelling"
        elif isinstance(payload, PostprocessingJobCancellationRequestIntendedPayload):
            history = cancellation_requests.setdefault(payload.action_id, [])
            if (
                not cancellation_intended
                or submitted_jobs.get(payload.action_id) != payload.parent_job_id
                or (history and history[-1].result is None)
                or payload.request_ordinal != len(history) + 1
                or payload.scancel_argv != ("scancel", payload.parent_job_id)
            ):
                raise ValueError("postprocessing cancellation request intent differs from assigned authority")
            history.append(_CancellationRequest(intent=payload))
        elif isinstance(payload, PostprocessingJobCancellationRequestResultPayload):
            result_history = cancellation_requests.get(payload.action_id)
            current = result_history[-1] if result_history else None
            if (
                current is None
                or current.result is not None
                or submitted_jobs.get(payload.action_id) != payload.parent_job_id
                or payload.request_ordinal != current.intent.request_ordinal
                or payload.scancel_argv != current.intent.scancel_argv
            ):
                raise ValueError("postprocessing cancellation request result differs from its durable intent")
            current.result = payload
            if payload.return_code == 0:
                cancellation_requested.add(payload.parent_job_id)
        elif isinstance(payload, PostprocessingCancelledPayload):
            if (
                not cancellation_intended
                or set(payload.terminal_parent_job_ids) != set(submitted_jobs.values())
                or payload.terminal_action_ids
                != tuple(action.action_id for action in runspec.payload.actions if action.action_id in submitted_jobs)
                or set(terminal_payloads) != set(submitted_jobs)
                or payload.terminal_task_evidence_digest
                != canonical_mapping_digest(
                    {
                        "schema_version": 1,
                        "phase_runspec_digest": runspec.digest,
                        "terminal_actions": [
                            terminal_payloads[action.action_id].to_mapping()
                            for action in runspec.payload.actions
                            if action.action_id in terminal_payloads
                        ],
                    }
                )
                or set(payload.cancelled_parent_job_ids) != cancellation_requested
            ):
                raise ValueError("postprocessing cancellation completion differs from durable task authority")
            status = "cancelled"
        elif isinstance(payload, PostprocessingFinalizedPayload):
            expected_actions = tuple(item.action_id for item in runspec.payload.actions)
            if status != "submitted" or set(terminal_payloads) != set(expected_actions):
                raise ValueError("postprocessing finalization requires every action terminal")
            if set(submitted_jobs) != set(expected_actions):
                raise ValueError("postprocessing finalization requires every durable Slurm assignment")
            for action, receipt_action in zip(runspec.payload.actions, payload.terminal_actions, strict=True):
                terminal = terminal_payloads[action.action_id]
                if (
                    not isinstance(terminal, PostprocessingActionTerminalObservedPayload)
                    or receipt_action.action_id != action.action_id
                    or receipt_action.runtime_action_digest != action.digest
                    or receipt_action.parent_job_id != submitted_jobs[action.action_id]
                    or receipt_action.parent_job_id != terminal.parent_job_id
                    or receipt_action.expected_task_indexes != action.expected_task_indexes
                    or terminal.outcome != "succeeded"
                    or tuple(
                        (item.scheduler_job_id, item.task_index, item.state, item.exit_code, item.source)
                        for item in receipt_action.tasks
                    )
                    != tuple(
                        (item.scheduler_job_id, item.task_index, item.state, item.exit_code, item.source)
                        for item in terminal.tasks
                    )
                ):
                    raise ValueError("postprocessing finalization action evidence differs from durable authority")
            receipt = payload.receipt
            if (
                receipt.phase_run_id != runspec.phase_run_id
                or receipt.attempt_id != runspec.attempt_id
                or receipt.phase_plan_digest != runspec.phase_plan_digest
                or receipt.phase_runspec_digest != runspec.digest
                or receipt.logical_input_manifest_digest != runspec.payload.logical_inputs.digest
                or receipt.scientific_identity_digest != runspec.payload.scientific_identity.digest
                or receipt.action_semantics_digest != runspec.payload.action_semantics_digest
                or receipt.execution_projection_sha256 != runspec.payload.execution_projection.document_sha256
                or receipt.qualified_runtime_digest != runspec.payload.qualified_runtime.digest
                or receipt.acceptance_policy_id != runspec.payload.acceptance_policy.policy_id
                or receipt.acceptance_policy_sha256 != runspec.payload.acceptance_policy.sha256
                or receipt.acceptance_policy_size_bytes != runspec.payload.acceptance_policy.size_bytes
                or receipt.acceptance_policy_semantic_digest != runspec.payload.acceptance_policy.semantic_digest
            ):
                raise ValueError("postprocessing finalization receipt differs from the current RunSpec authority")
            status = "accepted"
            sealed = True
        else:  # pragma: no cover - the contract union is exhaustive
            raise TypeError(f"unsupported postprocessing payload type: {type(payload).__name__}")

    submission_state = None
    if submission_payload is not None and submission_intended_at is not None:
        submission_state = PostprocessingSubmissionState(
            submission_id=submission_payload.submission_id,
            intended_at=submission_intended_at,
            actions=submission_payload.actions,
            job_ids=tuple(
                (action.action_id, submitted_jobs[action.action_id])
                for action in runspec.payload.actions
                if action.action_id in submitted_jobs
            ),
            dispatching=frozenset(set(dispatch_payloads) - set(submitted_jobs) - rejected_actions),
            rejected=frozenset(rejected_actions),
        )
    return PostprocessingReplayState(
        status=status,
        sealed=sealed,
        runspec=runspec,
        embedded_legacy_runspec_bytes=embedded_legacy_bytes,
        embedded_runtime_qualification_bytes=embedded_qualification_bytes,
        submission_state=submission_state,
        terminal_payloads=tuple(
            terminal_payloads[action.action_id]
            for action in runspec.payload.actions
            if action.action_id in terminal_payloads
        ),
    )


__all__ = ["replay_postprocessing_events"]
