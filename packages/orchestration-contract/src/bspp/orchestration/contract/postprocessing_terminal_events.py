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

"""Focused postprocessing contracts extracted from postprocessing_lifecycle.py."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from bspp.orchestration.contract.postprocessing_submission_events import _common_submission_action
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

_SHA256 = re.compile(r"[0-9a-f]{64}")


_JOB_ID = re.compile(r"[0-9]+")


@dataclass(frozen=True)
class PostprocessingTaskTerminalEvidence:
    scheduler_job_id: str
    state: str
    exit_code: str
    source: Literal["sacct"]
    restarts: int | None = None
    task_index: int | None = None
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _version(self.schema_version, type(self).__name__)
        if re.fullmatch(r"[0-9]+(?:_[0-9]+)?", self.scheduler_job_id) is None:
            raise ValueError("postprocessing task scheduler id is invalid")
        if not self.state or not self.exit_code or self.source != "sacct":
            raise ValueError("postprocessing task evidence requires sacct state and exit")
        if self.restarts is not None and (
            not isinstance(self.restarts, int) or isinstance(self.restarts, bool) or self.restarts < 0
        ):
            raise ValueError("postprocessing task restarts must be a non-negative integer or null")
        if self.task_index is not None and (not isinstance(self.task_index, int) or self.task_index < 0):
            raise ValueError("postprocessing task index must be non-negative")

    def to_mapping(self) -> dict[str, object]:
        mapping: dict[str, object] = {
            "schema_version": self.schema_version,
            "task_index": self.task_index,
            "scheduler_job_id": self.scheduler_job_id,
            "state": self.state,
            "exit_code": self.exit_code,
            "source": self.source,
        }
        if self.restarts is not None:
            mapping["restarts"] = self.restarts
        return mapping


@dataclass(frozen=True)
class PostprocessingActionTerminalObservedPayload:
    submission_id: str
    phase_runspec_digest: str
    action_id: str
    runtime_action_digest: str
    parent_job_id: str
    expected_task_indexes: tuple[int, ...]
    tasks: tuple[PostprocessingTaskTerminalEvidence, ...]
    outcome: Literal["succeeded", "failed"]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _common_submission_action(self.schema_version, self.submission_id, self.action_id)
        _sha(self.phase_runspec_digest)
        _sha(self.runtime_action_digest)
        if _JOB_ID.fullmatch(self.parent_job_id) is None or not self.tasks:
            raise ValueError("postprocessing terminal evidence requires parent id and tasks")
        _task_indexes(self.expected_task_indexes)
        indexes = tuple(item.task_index for item in self.tasks if item.task_index is not None)
        if self.expected_task_indexes:
            if indexes != self.expected_task_indexes or len(indexes) != len(self.tasks):
                raise ValueError("postprocessing terminal tasks differ from the exact expected indexes")
            if any(item.scheduler_job_id != f"{self.parent_job_id}_{item.task_index}" for item in self.tasks):
                raise ValueError("postprocessing array task scheduler IDs do not bind parent/index")
        elif len(self.tasks) != 1 or self.tasks[0].task_index is not None:
            raise ValueError("non-array postprocessing action requires one parent task")
        elif self.tasks[0].scheduler_job_id != self.parent_job_id:
            raise ValueError("postprocessing non-array task scheduler ID does not bind its parent")
        succeeded = all(item.state == "COMPLETED" and item.exit_code in {"0", "0:0"} for item in self.tasks)
        if (self.outcome == "succeeded") != succeeded:
            raise ValueError("postprocessing terminal outcome differs from task evidence")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "submission_id": self.submission_id,
            "phase_runspec_digest": self.phase_runspec_digest,
            "action_id": self.action_id,
            "runtime_action_digest": self.runtime_action_digest,
            "parent_job_id": self.parent_job_id,
            "expected_task_indexes": list(self.expected_task_indexes),
            "tasks": [item.to_mapping() for item in self.tasks],
            "outcome": self.outcome,
        }


@dataclass(frozen=True)
class PostprocessingArrayParentCancelledObservedPayload:
    """Cancellation-only evidence for an array with no instantiated task records."""

    submission_id: str
    phase_runspec_digest: str
    action_id: str
    runtime_action_digest: str
    parent_job_id: str
    expected_task_indexes: tuple[int, ...]
    parent: PostprocessingTaskTerminalEvidence
    observation_kind: Literal["array-parent-cancelled-before-task-instantiation-v1"] = (
        "array-parent-cancelled-before-task-instantiation-v1"
    )
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _common_submission_action(self.schema_version, self.submission_id, self.action_id)
        _sha(self.phase_runspec_digest)
        _sha(self.runtime_action_digest)
        _task_indexes(self.expected_task_indexes)
        if not self.expected_task_indexes:
            raise ValueError("array parent cancellation evidence requires expected task indexes")
        if (
            _JOB_ID.fullmatch(self.parent_job_id) is None
            or self.parent.task_index is not None
            or self.parent.scheduler_job_id != self.parent_job_id
            or self.parent.source != "sacct"
            or self.parent.state != "CANCELLED"
            or self.parent.exit_code != "0:0"
            or self.observation_kind != "array-parent-cancelled-before-task-instantiation-v1"
        ):
            raise ValueError("array parent cancellation evidence requires one exact CANCELLED 0:0 parent")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "submission_id": self.submission_id,
            "phase_runspec_digest": self.phase_runspec_digest,
            "action_id": self.action_id,
            "runtime_action_digest": self.runtime_action_digest,
            "parent_job_id": self.parent_job_id,
            "expected_task_indexes": list(self.expected_task_indexes),
            "parent": self.parent.to_mapping(),
            "observation_kind": self.observation_kind,
        }


def _terminal(payload: Mapping[str, object]) -> PostprocessingActionTerminalObservedPayload:
    _fields(
        payload,
        {
            "schema_version",
            "submission_id",
            "phase_runspec_digest",
            "action_id",
            "runtime_action_digest",
            "parent_job_id",
            "expected_task_indexes",
            "tasks",
            "outcome",
        },
        "PostprocessingActionTerminalObservedPayload",
    )
    outcome = _string(payload, "outcome")
    if outcome not in {"succeeded", "failed"}:
        raise ValueError("postprocessing terminal outcome is invalid")
    return PostprocessingActionTerminalObservedPayload(
        schema_version=_schema(payload, "PostprocessingActionTerminalObservedPayload"),
        submission_id=_string(payload, "submission_id"),
        phase_runspec_digest=_string(payload, "phase_runspec_digest"),
        action_id=_string(payload, "action_id"),
        runtime_action_digest=_string(payload, "runtime_action_digest"),
        parent_job_id=_string(payload, "parent_job_id"),
        expected_task_indexes=_integers(payload, "expected_task_indexes"),
        tasks=tuple(_task(item) for item in _mapping_list(payload, "tasks")),
        outcome=cast("Literal['succeeded', 'failed']", outcome),
    )


def _array_parent_cancelled(payload: Mapping[str, object]) -> PostprocessingArrayParentCancelledObservedPayload:
    _fields(
        payload,
        {
            "schema_version",
            "submission_id",
            "phase_runspec_digest",
            "action_id",
            "runtime_action_digest",
            "parent_job_id",
            "expected_task_indexes",
            "parent",
            "observation_kind",
        },
        "PostprocessingArrayParentCancelledObservedPayload",
    )
    parent = payload.get("parent")
    if not isinstance(parent, Mapping):
        raise ValueError("array parent cancellation evidence parent must be a mapping")
    return PostprocessingArrayParentCancelledObservedPayload(
        schema_version=_schema(payload, "PostprocessingArrayParentCancelledObservedPayload"),
        submission_id=_string(payload, "submission_id"),
        phase_runspec_digest=_string(payload, "phase_runspec_digest"),
        action_id=_string(payload, "action_id"),
        runtime_action_digest=_string(payload, "runtime_action_digest"),
        parent_job_id=_string(payload, "parent_job_id"),
        expected_task_indexes=_integers(payload, "expected_task_indexes"),
        parent=_task(cast("Mapping[str, object]", parent)),
        observation_kind=cast(
            "Literal['array-parent-cancelled-before-task-instantiation-v1']",
            _string(payload, "observation_kind"),
        ),
    )


def _task(payload: Mapping[str, object]) -> PostprocessingTaskTerminalEvidence:
    required = {"schema_version", "task_index", "scheduler_job_id", "state", "exit_code", "source"}
    optional = {"restarts"}
    if not required <= set(payload) or not set(payload) <= (required | optional):
        raise ValueError("PostprocessingTaskTerminalEvidence has missing or extra fields")
    task_index = payload.get("task_index")
    if task_index is not None and (not isinstance(task_index, int) or isinstance(task_index, bool)):
        raise ValueError("postprocessing terminal task index must be integer or null")
    restarts = payload.get("restarts")
    if restarts is not None and (not isinstance(restarts, int) or isinstance(restarts, bool) or restarts < 0):
        raise ValueError("postprocessing terminal task restarts must be a non-negative integer or null")
    return PostprocessingTaskTerminalEvidence(
        schema_version=_schema(payload, "PostprocessingTaskTerminalEvidence"),
        task_index=task_index,
        scheduler_job_id=_string(payload, "scheduler_job_id"),
        state=_string(payload, "state"),
        exit_code=_string(payload, "exit_code"),
        source=cast("Literal['sacct']", _string(payload, "source")),
        restarts=restarts,
    )


def _version(value: int, name: str) -> None:
    validate_schema_version(value, record_name=name)


def _sha(value: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise ValueError("postprocessing lifecycle digest must be lowercase SHA-256")


def _task_indexes(values: tuple[int, ...]) -> None:
    if tuple(sorted(set(values))) != values or any(item < 0 for item in values):
        raise ValueError("postprocessing task indexes must be sorted, unique, and non-negative")


def _fields(payload: Mapping[str, object], expected: set[str], name: str) -> None:
    if set(payload) != expected:
        raise ValueError(f"{name} has missing or extra fields")


def _schema(payload: Mapping[str, object], name: str) -> int:
    return validate_schema_version(payload.get("schema_version"), record_name=name)


def _string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


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


PostprocessingTerminalObservation = (
    PostprocessingActionTerminalObservedPayload | PostprocessingArrayParentCancelledObservedPayload
)


__all__ = [
    "PostprocessingActionTerminalObservedPayload",
    "PostprocessingArrayParentCancelledObservedPayload",
    "PostprocessingTaskTerminalEvidence",
    "PostprocessingTerminalObservation",
]
