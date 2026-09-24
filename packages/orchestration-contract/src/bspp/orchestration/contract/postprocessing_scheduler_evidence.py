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

"""Durable Control Plane scheduler evidence for postprocessing finalization."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from bspp.orchestration.contract.phase import (
    canonical_mapping_digest,
    validate_phase_attempt_id,
    validate_phase_run_id,
)
from bspp.orchestration.contract.postprocessing_handoff import PostprocessingTaskReceiptEvidence
from bspp.orchestration.contract.postprocessing_phase_ids import POSTPROCESSING_ACTION_IDS
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SUBMISSION_ID = re.compile(r"postprocessing-submission-[0-9a-f]{64}")
_JOB_ID = re.compile(r"[0-9]+")
_CORRELATION = re.compile(r"bspp-pp-[0-9a-f]{48}")
_LEGACY_CORRELATION = re.compile(r"afcdb-pp-[0-9a-f]{48}")


@dataclass(frozen=True)
class PostprocessingSchedulerActionEvidence:
    """Exact successful submission, dispatch, assignment, and sacct observation."""

    action_id: str
    runtime_action_digest: str
    dependencies: tuple[str, ...]
    cluster_script_path: str
    script_sha256: str
    scheduler_correlation_token: str
    dependency_job_ids: tuple[str, ...]
    parent_job_id: str
    expected_task_indexes: tuple[int, ...]
    tasks: tuple[PostprocessingTaskReceiptEvidence, ...]
    outcome: Literal["succeeded"] = "succeeded"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        permanent_ids = tuple(POSTPROCESSING_ACTION_IDS.values())
        if self.action_id not in permanent_ids:
            raise ValueError("postprocessing scheduler evidence action id is unknown")
        _sha(self.runtime_action_digest)
        _sha(self.script_sha256)
        if (
            len(set(self.dependencies)) != len(self.dependencies)
            or any(item not in permanent_ids for item in self.dependencies)
            or tuple(sorted(self.dependencies, key=permanent_ids.index)) != self.dependencies
            or any(permanent_ids.index(item) >= permanent_ids.index(self.action_id) for item in self.dependencies)
        ):
            raise ValueError("postprocessing scheduler evidence dependencies are invalid")
        if not self.cluster_script_path.startswith("/") or not self.cluster_script_path.endswith(".sbatch"):
            raise ValueError("postprocessing scheduler evidence script path is invalid")
        if (
            _CORRELATION.fullmatch(self.scheduler_correlation_token) is None
            and _LEGACY_CORRELATION.fullmatch(self.scheduler_correlation_token) is None
        ):
            raise ValueError("postprocessing scheduler evidence correlation token is invalid")
        if (
            len(self.dependency_job_ids) != len(self.dependencies)
            or any(_JOB_ID.fullmatch(item) is None for item in self.dependency_job_ids)
            or _JOB_ID.fullmatch(self.parent_job_id) is None
        ):
            raise ValueError("postprocessing scheduler evidence job identities are invalid")
        if tuple(sorted(set(self.expected_task_indexes))) != self.expected_task_indexes or any(
            item < 0 for item in self.expected_task_indexes
        ):
            raise ValueError("postprocessing scheduler evidence task indexes are invalid")
        if not self.tasks or self.outcome != "succeeded":
            raise ValueError("postprocessing scheduler evidence requires successful tasks")
        if self.expected_task_indexes:
            if tuple(item.task_index for item in self.tasks) != self.expected_task_indexes or any(
                item.scheduler_job_id != f"{self.parent_job_id}_{item.task_index}" for item in self.tasks
            ):
                raise ValueError("postprocessing scheduler array evidence differs from the expected tasks")
        elif (
            len(self.tasks) != 1
            or self.tasks[0].task_index is not None
            or self.tasks[0].scheduler_job_id != self.parent_job_id
        ):
            raise ValueError("postprocessing scheduler non-array evidence must bind its parent job")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "action_id": self.action_id,
            "runtime_action_digest": self.runtime_action_digest,
            "dependencies": list(self.dependencies),
            "cluster_script_path": self.cluster_script_path,
            "script_sha256": self.script_sha256,
            "scheduler_correlation_token": self.scheduler_correlation_token,
            "dependency_job_ids": list(self.dependency_job_ids),
            "parent_job_id": self.parent_job_id,
            "expected_task_indexes": list(self.expected_task_indexes),
            "tasks": [item.to_mapping() for item in self.tasks],
            "outcome": self.outcome,
        }


@dataclass(frozen=True)
class PostprocessingSchedulerEvidence:
    phase_run_id: str
    attempt_id: str
    phase_runspec_digest: str
    action_graph_digest: str
    submission_id: str
    actions: tuple[PostprocessingSchedulerActionEvidence, ...]
    scheduler_evidence_id: str
    evidence_kind: Literal["postprocessing-control-scheduler-evidence-v1"] = (
        "postprocessing-control-scheduler-evidence-v1"
    )
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        validate_phase_run_id(self.phase_run_id)
        validate_phase_attempt_id(self.attempt_id)
        _sha(self.phase_runspec_digest)
        _sha(self.action_graph_digest)
        if _SUBMISSION_ID.fullmatch(self.submission_id) is None:
            raise ValueError("postprocessing scheduler evidence submission id is invalid")
        if self.evidence_kind != "postprocessing-control-scheduler-evidence-v1":
            raise ValueError("postprocessing scheduler evidence kind is unsupported")
        permanent_ids = tuple(POSTPROCESSING_ACTION_IDS.values())
        action_ids = tuple(item.action_id for item in self.actions)
        required = {
            POSTPROCESSING_ACTION_IDS[name]
            for name in (
                "preflight",
                "analysis-finalize",
                "acceptance-tar-payload-parity",
                "acceptance-semantic",
                "acceptance-verify-evidence",
                "acceptance-adjudication",
            )
        }
        if (
            not self.actions
            or action_ids != tuple(sorted(action_ids, key=permanent_ids.index))
            or len(set(action_ids)) != len(action_ids)
            or not required <= set(action_ids)
        ):
            raise ValueError("postprocessing scheduler evidence action set or order is invalid")
        jobs = {item.action_id: item.parent_job_id for item in self.actions}
        if any(
            item.dependency_job_ids != tuple(jobs[dependency] for dependency in item.dependencies)
            for item in self.actions
        ):
            raise ValueError("postprocessing scheduler dependency job ids differ from action assignments")
        _sha(self.scheduler_evidence_id)
        if self.scheduler_evidence_id != canonical_mapping_digest(self.identity_mapping()):
            raise ValueError("postprocessing scheduler evidence id differs from its identity preimage")

    def identity_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "evidence_kind": self.evidence_kind,
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "phase_runspec_digest": self.phase_runspec_digest,
            "action_graph_digest": self.action_graph_digest,
            "submission_id": self.submission_id,
            "actions": [item.to_mapping() for item in self.actions],
        }

    def to_mapping(self) -> dict[str, object]:
        return {**self.identity_mapping(), "scheduler_evidence_id": self.scheduler_evidence_id}


def postprocessing_scheduler_evidence_from_mapping(
    payload: Mapping[str, object],
) -> PostprocessingSchedulerEvidence:
    _fields(
        payload,
        {
            "schema_version",
            "evidence_kind",
            "phase_run_id",
            "attempt_id",
            "phase_runspec_digest",
            "action_graph_digest",
            "submission_id",
            "actions",
            "scheduler_evidence_id",
        },
        "PostprocessingSchedulerEvidence",
    )
    return PostprocessingSchedulerEvidence(
        schema_version=_schema(payload, "PostprocessingSchedulerEvidence"),
        evidence_kind=cast(
            "Literal['postprocessing-control-scheduler-evidence-v1']", _string(payload, "evidence_kind")
        ),
        phase_run_id=_string(payload, "phase_run_id"),
        attempt_id=_string(payload, "attempt_id"),
        phase_runspec_digest=_string(payload, "phase_runspec_digest"),
        action_graph_digest=_string(payload, "action_graph_digest"),
        submission_id=_string(payload, "submission_id"),
        actions=tuple(_action(item) for item in _mapping_list(payload, "actions")),
        scheduler_evidence_id=_string(payload, "scheduler_evidence_id"),
    )


def _action(payload: Mapping[str, object]) -> PostprocessingSchedulerActionEvidence:
    _fields(
        payload,
        {
            "schema_version",
            "action_id",
            "runtime_action_digest",
            "dependencies",
            "cluster_script_path",
            "script_sha256",
            "scheduler_correlation_token",
            "dependency_job_ids",
            "parent_job_id",
            "expected_task_indexes",
            "tasks",
            "outcome",
        },
        "PostprocessingSchedulerActionEvidence",
    )
    if payload.get("outcome") != "succeeded":
        raise ValueError("postprocessing scheduler action must have succeeded")
    return PostprocessingSchedulerActionEvidence(
        schema_version=_schema(payload, "PostprocessingSchedulerActionEvidence"),
        action_id=_string(payload, "action_id"),
        runtime_action_digest=_string(payload, "runtime_action_digest"),
        dependencies=_strings(payload, "dependencies"),
        cluster_script_path=_string(payload, "cluster_script_path"),
        script_sha256=_string(payload, "script_sha256"),
        scheduler_correlation_token=_string(payload, "scheduler_correlation_token"),
        dependency_job_ids=_strings(payload, "dependency_job_ids"),
        parent_job_id=_string(payload, "parent_job_id"),
        expected_task_indexes=_integers(payload, "expected_task_indexes"),
        tasks=tuple(_task(item) for item in _mapping_list(payload, "tasks")),
        outcome="succeeded",
    )


def _task(payload: Mapping[str, object]) -> PostprocessingTaskReceiptEvidence:
    allowed = {"schema_version", "task_index", "scheduler_job_id", "state", "exit_code", "source", "restarts"}
    if set(payload) != allowed and set(payload) != allowed - {"restarts"}:
        raise ValueError("PostprocessingTaskReceiptEvidence has missing or extra fields")
    task_index = payload.get("task_index")
    if task_index is not None and (not isinstance(task_index, int) or isinstance(task_index, bool)):
        raise ValueError("postprocessing scheduler task index must be an integer or null")
    restarts = payload.get("restarts")
    if restarts is not None and (not isinstance(restarts, int) or isinstance(restarts, bool) or restarts < 0):
        raise ValueError("postprocessing scheduler task restarts must be a non-negative integer")
    state = payload.get("state")
    exit_code = payload.get("exit_code")
    source = payload.get("source")
    if state != "COMPLETED" or exit_code not in {"0", "0:0"} or source != "sacct":
        raise ValueError("postprocessing scheduler task is not successful sacct evidence")
    return PostprocessingTaskReceiptEvidence(
        schema_version=_schema(payload, "PostprocessingTaskReceiptEvidence"),
        task_index=task_index,
        scheduler_job_id=_string(payload, "scheduler_job_id"),
        state="COMPLETED",
        exit_code=cast("Literal['0', '0:0']", exit_code),
        source="sacct",
        restarts=restarts,
    )


def _schema(value: int | Mapping[str, object], record: str) -> int:
    raw = value.get("schema_version") if isinstance(value, Mapping) else value
    return validate_schema_version(raw, record_name=record)


def _sha(value: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise ValueError("postprocessing scheduler digest must be lowercase SHA-256")


def _fields(payload: Mapping[str, object], expected: set[str], record: str) -> None:
    if set(payload) != expected:
        raise ValueError(f"{record} has missing or extra fields")


def _string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _strings(payload: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be a list of strings")
    return tuple(value)


def _integers(payload: Mapping[str, object], key: str) -> tuple[int, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        raise ValueError(f"{key} must be a list of integers")
    return tuple(value)


def _mapping_list(payload: Mapping[str, object], key: str) -> tuple[Mapping[str, object], ...]:
    value = payload.get(key)
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"{key} must be a list of mappings")
    return tuple(cast("Mapping[str, object]", item) for item in value)


__all__ = [
    "PostprocessingSchedulerActionEvidence",
    "PostprocessingSchedulerEvidence",
    "postprocessing_scheduler_evidence_from_mapping",
]
