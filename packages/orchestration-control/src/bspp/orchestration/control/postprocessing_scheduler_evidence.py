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

"""Create-once export of durable postprocessing scheduler evidence."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from bspp.orchestration.contract.phase import canonical_mapping_digest
from bspp.orchestration.contract.postprocessing_handoff import (
    PostprocessingTaskReceiptEvidence,
)
from bspp.orchestration.contract.postprocessing_scheduler_evidence import (
    PostprocessingSchedulerActionEvidence,
    PostprocessingSchedulerEvidence,
    postprocessing_scheduler_evidence_from_mapping,
)
from bspp.orchestration.contract.postprocessing_submission_events import (
    PostprocessingActionDispatchIntendedPayload,
    PostprocessingActionSubmittedPayload,
    PostprocessingSubmissionIntendedPayload,
)
from bspp.orchestration.contract.postprocessing_terminal_events import (
    PostprocessingActionTerminalObservedPayload,
    PostprocessingTaskTerminalEvidence,
)
from bspp.orchestration.control.postprocessing_authority_reader import (
    reject_historical_postprocessing_mutation,
    require_postprocessing_v2_authority,
)
from bspp.orchestration.control.postprocessing_authority_store import (
    postprocessing_operation_lock as _postprocessing_operation_lock,
)
from bspp.orchestration.control.postprocessing_phase_types import PostprocessingAuthority


@dataclass(frozen=True)
class PostprocessingSchedulerEvidenceExportResult:
    phase_run_id: str
    attempt_id: str
    scheduler_evidence_id: str
    output: Path

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "phase_kind": "postprocessing",
            "operation": "evidence-export-scheduler",
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "scheduler_evidence_id": self.scheduler_evidence_id,
            "output": str(self.output),
            "status": "exported",
        }

    def render_json(self) -> str:
        return json.dumps(self.to_mapping(), indent=2, sort_keys=True) + "\n"


def export_postprocessing_scheduler_evidence(
    phase_run_id: str,
    *,
    authority_root: Path,
    output: Path,
) -> PostprocessingSchedulerEvidenceExportResult:
    """Export the exact successful current-Attempt scheduler event projection."""
    reject_historical_postprocessing_mutation(authority_root, phase_run_id)
    with _postprocessing_operation_lock(authority_root, phase_run_id):
        authority = require_postprocessing_v2_authority(authority_root, phase_run_id)
        evidence = postprocessing_scheduler_evidence_from_authority(authority)
        document = _canonical_bytes(evidence.to_mapping())
        _publish_create_once(output, document)
    return PostprocessingSchedulerEvidenceExportResult(
        phase_run_id=evidence.phase_run_id,
        attempt_id=evidence.attempt_id,
        scheduler_evidence_id=evidence.scheduler_evidence_id,
        output=output,
    )


def postprocessing_scheduler_evidence_from_authority(
    authority: PostprocessingAuthority,
) -> PostprocessingSchedulerEvidence:
    """Project scheduler evidence while enforcing every RunSpec/event cross-binding."""
    current = tuple(event for event in authority.events if event.attempt_id == authority.attempt_id)
    intents = tuple(
        event.payload for event in current if isinstance(event.payload, PostprocessingSubmissionIntendedPayload)
    )
    if len(intents) != 1:
        raise ValueError("postprocessing scheduler export requires exactly one submission intent")
    intent = intents[0]
    if intent.phase_runspec_digest != authority.runspec.digest:
        raise ValueError("postprocessing scheduler submission intent differs from the current RunSpec")
    submission_id = intent.submission_id
    plans = intent.actions
    if tuple(item.action_id for item in plans) != tuple(item.action_id for item in authority.runspec.payload.actions):
        raise ValueError("postprocessing scheduler submission action order differs from the RunSpec")

    dispatches: dict[str, PostprocessingActionDispatchIntendedPayload] = {}
    assignments: dict[str, PostprocessingActionSubmittedPayload] = {}
    terminals: dict[str, PostprocessingActionTerminalObservedPayload] = {}
    for event in current:
        payload = event.payload
        if isinstance(payload, PostprocessingActionDispatchIntendedPayload):
            if payload.action_id in dispatches:
                raise ValueError(
                    f"postprocessing scheduler authority duplicates {event.event_type} for {payload.action_id!r}"
                )
            dispatches[payload.action_id] = payload
        elif isinstance(payload, PostprocessingActionSubmittedPayload):
            if payload.action_id in assignments:
                raise ValueError(
                    f"postprocessing scheduler authority duplicates {event.event_type} for {payload.action_id!r}"
                )
            assignments[payload.action_id] = payload
        elif isinstance(payload, PostprocessingActionTerminalObservedPayload):
            if payload.action_id in terminals:
                raise ValueError(
                    f"postprocessing scheduler authority duplicates {event.event_type} for {payload.action_id!r}"
                )
            terminals[payload.action_id] = payload
    expected_ids = tuple(item.action_id for item in authority.runspec.payload.actions)
    expected_set = set(expected_ids)
    if set(dispatches) != expected_set or set(assignments) != expected_set or set(terminals) != expected_set:
        raise ValueError("postprocessing scheduler export requires one complete event chain per frozen action")

    actions: list[PostprocessingSchedulerActionEvidence] = []
    assigned_jobs: dict[str, str] = {}
    for action, plan in zip(authority.runspec.payload.actions, plans, strict=True):
        dispatch = dispatches[action.action_id]
        assignment = assignments[action.action_id]
        terminal = terminals[action.action_id]
        action_digest = action.digest
        correlation = plan.scheduler_correlation_token
        dependency_jobs = tuple(assigned_jobs[item] for item in action.dependencies)
        if (
            plan.runtime_action_digest != action_digest
            or plan.dependencies != action.dependencies
            or dispatch.submission_id != submission_id
            or dispatch.scheduler_correlation_token != correlation
            or dispatch.dependency_job_ids != dependency_jobs
            or assignment.submission_id != submission_id
            or assignment.scheduler_correlation_token != correlation
            or assignment.expected_task_indexes != action.expected_task_indexes
            or terminal.submission_id != submission_id
            or terminal.phase_runspec_digest != authority.runspec.digest
            or terminal.runtime_action_digest != action_digest
            or terminal.parent_job_id != assignment.parent_job_id
            or terminal.expected_task_indexes != action.expected_task_indexes
            or terminal.outcome != "succeeded"
        ):
            raise ValueError(f"postprocessing scheduler events differ from frozen action {action.action_id!r}")
        parent_job_id = assignment.parent_job_id
        assigned_jobs[action.action_id] = parent_job_id
        tasks = tuple(_successful_task(item) for item in terminal.tasks)
        actions.append(
            PostprocessingSchedulerActionEvidence(
                action_id=action.action_id,
                runtime_action_digest=action_digest,
                dependencies=action.dependencies,
                cluster_script_path=plan.cluster_script_path,
                script_sha256=plan.script_sha256,
                scheduler_correlation_token=correlation,
                dependency_job_ids=dependency_jobs,
                parent_job_id=parent_job_id,
                expected_task_indexes=action.expected_task_indexes,
                tasks=tasks,
            )
        )
    identity: dict[str, object] = {
        "schema_version": 1,
        "evidence_kind": "postprocessing-control-scheduler-evidence-v1",
        "phase_run_id": authority.phase_run_id,
        "attempt_id": authority.attempt_id,
        "phase_runspec_digest": authority.runspec.digest,
        "action_graph_digest": authority.runspec.payload.action_graph_digest,
        "submission_id": submission_id,
        "actions": [item.to_mapping() for item in actions],
    }
    return PostprocessingSchedulerEvidence(
        phase_run_id=authority.phase_run_id,
        attempt_id=authority.attempt_id,
        phase_runspec_digest=authority.runspec.digest,
        action_graph_digest=authority.runspec.payload.action_graph_digest,
        submission_id=submission_id,
        actions=tuple(actions),
        scheduler_evidence_id=canonical_mapping_digest(identity),
    )


def _successful_task(payload: PostprocessingTaskTerminalEvidence) -> PostprocessingTaskReceiptEvidence:
    if payload.state != "COMPLETED" or payload.exit_code not in {"0", "0:0"}:
        raise ValueError("postprocessing scheduler terminal task is not successful sacct evidence")
    if payload.restarts is not None and (
        not isinstance(payload.restarts, int) or isinstance(payload.restarts, bool) or payload.restarts < 0
    ):
        raise ValueError("postprocessing scheduler terminal task restarts must be a non-negative integer")
    return PostprocessingTaskReceiptEvidence(
        task_index=payload.task_index,
        scheduler_job_id=payload.scheduler_job_id,
        state="COMPLETED",
        exit_code=cast("Literal['0', '0:0']", payload.exit_code),
        source="sacct",
        restarts=payload.restarts,
    )


def _publish_create_once(path: Path, document: bytes) -> None:
    if os.path.lexists(path):
        if path.is_symlink() or not path.is_file() or path.read_bytes() != document:
            raise ValueError("existing postprocessing scheduler evidence differs from durable authority")
        payload = _canonical_mapping(document)
        postprocessing_scheduler_evidence_from_mapping(payload)
        return
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError("postprocessing scheduler evidence parent must be an existing non-symlink directory")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(document)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.link(temporary, path, follow_symlinks=False)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != document:
            raise ValueError("existing postprocessing scheduler evidence differs from durable authority") from None
    finally:
        temporary.unlink(missing_ok=True)


def _canonical_bytes(payload: Mapping[str, object]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode()


def _canonical_mapping(document: bytes) -> Mapping[str, object]:
    try:
        payload = json.loads(document)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("postprocessing scheduler evidence must be valid UTF-8 JSON") from exc
    if not isinstance(payload, Mapping) or _canonical_bytes(payload) != document:
        raise ValueError("postprocessing scheduler evidence must be a canonical JSON mapping")
    return cast("Mapping[str, object]", payload)


def _string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


__all__ = [
    "PostprocessingSchedulerEvidenceExportResult",
    "export_postprocessing_scheduler_evidence",
    "postprocessing_scheduler_evidence_from_authority",
]
