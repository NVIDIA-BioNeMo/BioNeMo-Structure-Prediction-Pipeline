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

"""Focused postprocessing contracts extracted from postprocessing_finalization_bundle.py."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from bspp.orchestration.contract.phase import validate_phase_attempt_id, validate_phase_run_id
from bspp.orchestration.contract.postprocessing_action09_bundle import (
    PostprocessingAction09AssemblyWitness,
    postprocessing_action09_assembly_witness_from_mapping,
)
from bspp.orchestration.contract.postprocessing_phase_ids import POSTPROCESSING_ACTION_IDS
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class PostprocessingRuntimeTaskEvidence:
    phase_run_id: str
    attempt_id: str
    phase_runspec_digest: str
    action_graph_digest: str
    action_id: str
    runtime_action_digest: str
    command_digest: str
    task_index: int | None
    scheduler_job_id: str
    completed_at: str
    outcome: Literal["succeeded"] = "succeeded"
    evidence_kind: Literal["postprocessing-runtime-task-evidence-v1"] = "postprocessing-runtime-task-evidence-v1"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        validate_phase_run_id(self.phase_run_id)
        validate_phase_attempt_id(self.attempt_id)
        for value in (
            self.phase_runspec_digest,
            self.action_graph_digest,
            self.runtime_action_digest,
            self.command_digest,
        ):
            _sha(value, "postprocessing Runtime task digest")
        if (
            self.evidence_kind != "postprocessing-runtime-task-evidence-v1"
            or self.action_id not in tuple(POSTPROCESSING_ACTION_IDS.values())[:-1]
            or re.fullmatch(r"[0-9]+(?:_[0-9]+)?", self.scheduler_job_id) is None
            or not self.completed_at
            or self.outcome != "succeeded"
        ):
            raise ValueError("postprocessing Runtime task evidence identity or outcome is invalid")
        if self.task_index is not None and (
            not isinstance(self.task_index, int) or isinstance(self.task_index, bool) or self.task_index < 0
        ):
            raise ValueError("postprocessing Runtime task index must be non-negative or null")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "evidence_kind": self.evidence_kind,
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "phase_runspec_digest": self.phase_runspec_digest,
            "action_graph_digest": self.action_graph_digest,
            "action_id": self.action_id,
            "runtime_action_digest": self.runtime_action_digest,
            "command_digest": self.command_digest,
            "task_index": self.task_index,
            "scheduler_job_id": self.scheduler_job_id,
            "completed_at": self.completed_at,
            "outcome": self.outcome,
        }


@dataclass(frozen=True)
class PostprocessingCompletedRuntimeActionEvidence:
    action_id: str
    runtime_action_digest: str
    expected_task_indexes: tuple[int, ...]
    tasks: tuple[PostprocessingRuntimeTaskEvidence, ...]

    def __post_init__(self) -> None:
        _sha(self.runtime_action_digest, "postprocessing completed Runtime action digest")
        if self.action_id not in tuple(POSTPROCESSING_ACTION_IDS.values())[:-1]:
            raise ValueError("completed Runtime action evidence may only describe Actions 01--08")
        if tuple(sorted(set(self.expected_task_indexes))) != self.expected_task_indexes:
            raise ValueError("completed Runtime action expected task indexes are not sorted and unique")
        observed_indexes = tuple(item.task_index for item in self.tasks)
        expected_indexes: tuple[int | None, ...] = self.expected_task_indexes or (None,)
        if (
            not self.tasks
            or observed_indexes != expected_indexes
            or any(
                item.action_id != self.action_id or item.runtime_action_digest != self.runtime_action_digest
                for item in self.tasks
            )
            or len({item.command_digest for item in self.tasks}) != 1
        ):
            raise ValueError("completed Runtime action tasks differ from its exact action/task closure")

    def to_mapping(self) -> dict[str, object]:
        return {
            "action_id": self.action_id,
            "runtime_action_digest": self.runtime_action_digest,
            "expected_task_indexes": list(self.expected_task_indexes),
            "tasks": [item.to_mapping() for item in self.tasks],
        }


@dataclass(frozen=True)
class PostprocessingRuntimeActionEvidenceAggregate:
    phase_run_id: str
    attempt_id: str
    phase_runspec_digest: str
    action_graph_digest: str
    completed_actions: tuple[PostprocessingCompletedRuntimeActionEvidence, ...]
    action09_prepublication_witness: PostprocessingAction09AssemblyWitness
    aggregate_kind: Literal["postprocessing-runtime-action-evidence-aggregate-v1"] = (
        "postprocessing-runtime-action-evidence-aggregate-v1"
    )
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        validate_phase_run_id(self.phase_run_id)
        validate_phase_attempt_id(self.attempt_id)
        _sha(self.phase_runspec_digest, "postprocessing aggregate RunSpec digest")
        _sha(self.action_graph_digest, "postprocessing aggregate action graph digest")
        permanent_ids = tuple(POSTPROCESSING_ACTION_IDS.values())
        action_ids = tuple(item.action_id for item in self.completed_actions)
        mandatory_ids = {
            POSTPROCESSING_ACTION_IDS[step]
            for step in (
                "preflight",
                "analysis-finalize",
                "acceptance-tar-payload-parity",
                "acceptance-semantic",
                "acceptance-verify-evidence",
            )
        }
        if (
            self.aggregate_kind != "postprocessing-runtime-action-evidence-aggregate-v1"
            or not mandatory_ids.issubset(action_ids)
            or POSTPROCESSING_ACTION_IDS["acceptance-adjudication"] in action_ids
            or action_ids != tuple(sorted(action_ids, key=permanent_ids.index))
            or len(set(action_ids)) != len(action_ids)
            or any(
                (
                    task.phase_run_id,
                    task.attempt_id,
                    task.phase_runspec_digest,
                    task.action_graph_digest,
                )
                != (self.phase_run_id, self.attempt_id, self.phase_runspec_digest, self.action_graph_digest)
                for action in self.completed_actions
                for task in action.tasks
            )
        ):
            raise ValueError("postprocessing aggregate does not contain a valid ordered Action 01--08 subset")
        witness = self.action09_prepublication_witness
        if (
            witness.phase_run_id,
            witness.attempt_id,
            witness.phase_runspec_digest,
            witness.action_graph_digest,
            witness.action_id,
            witness.publication_claim,
        ) != (
            self.phase_run_id,
            self.attempt_id,
            self.phase_runspec_digest,
            self.action_graph_digest,
            POSTPROCESSING_ACTION_IDS["acceptance-adjudication"],
            "none",
        ):
            raise ValueError("postprocessing aggregate Action 09 witness differs from its authority")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "aggregate_kind": self.aggregate_kind,
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "phase_runspec_digest": self.phase_runspec_digest,
            "action_graph_digest": self.action_graph_digest,
            "completed_actions": [item.to_mapping() for item in self.completed_actions],
            "action09_prepublication_witness": self.action09_prepublication_witness.to_mapping(),
        }


def postprocessing_runtime_task_evidence_from_mapping(
    payload: Mapping[str, object],
) -> PostprocessingRuntimeTaskEvidence:
    _fields(
        payload,
        {
            "schema_version",
            "evidence_kind",
            "phase_run_id",
            "attempt_id",
            "phase_runspec_digest",
            "action_graph_digest",
            "action_id",
            "runtime_action_digest",
            "command_digest",
            "task_index",
            "scheduler_job_id",
            "completed_at",
            "outcome",
        },
        "PostprocessingRuntimeTaskEvidence",
    )
    if (
        _string(payload, "evidence_kind") != "postprocessing-runtime-task-evidence-v1"
        or _string(payload, "outcome") != "succeeded"
    ):
        raise ValueError("unsupported postprocessing Runtime task evidence discriminator")
    task_index = payload.get("task_index")
    if task_index is not None and (not isinstance(task_index, int) or isinstance(task_index, bool)):
        raise ValueError("postprocessing Runtime task_index must be an integer or null")
    return PostprocessingRuntimeTaskEvidence(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="PostprocessingRuntimeTaskEvidence"
        ),
        evidence_kind="postprocessing-runtime-task-evidence-v1",
        phase_run_id=_string(payload, "phase_run_id"),
        attempt_id=_string(payload, "attempt_id"),
        phase_runspec_digest=_string(payload, "phase_runspec_digest"),
        action_graph_digest=_string(payload, "action_graph_digest"),
        action_id=_string(payload, "action_id"),
        runtime_action_digest=_string(payload, "runtime_action_digest"),
        command_digest=_string(payload, "command_digest"),
        task_index=task_index,
        scheduler_job_id=_string(payload, "scheduler_job_id"),
        completed_at=_string(payload, "completed_at"),
        outcome="succeeded",
    )


def postprocessing_runtime_action_evidence_aggregate_from_mapping(
    payload: Mapping[str, object],
) -> PostprocessingRuntimeActionEvidenceAggregate:
    _fields(
        payload,
        {
            "schema_version",
            "aggregate_kind",
            "phase_run_id",
            "attempt_id",
            "phase_runspec_digest",
            "action_graph_digest",
            "completed_actions",
            "action09_prepublication_witness",
        },
        "PostprocessingRuntimeActionEvidenceAggregate",
    )
    if _string(payload, "aggregate_kind") != "postprocessing-runtime-action-evidence-aggregate-v1":
        raise ValueError("unsupported postprocessing Runtime aggregate discriminator")
    completed: list[PostprocessingCompletedRuntimeActionEvidence] = []
    for item in _mapping_list(payload, "completed_actions"):
        _fields(
            item,
            {"action_id", "runtime_action_digest", "expected_task_indexes", "tasks"},
            "PostprocessingCompletedRuntimeActionEvidence",
        )
        expected = item.get("expected_task_indexes")
        if not isinstance(expected, list) or any(
            not isinstance(value, int) or isinstance(value, bool) for value in expected
        ):
            raise ValueError("expected_task_indexes must be a list of integers")
        completed.append(
            PostprocessingCompletedRuntimeActionEvidence(
                action_id=_string(item, "action_id"),
                runtime_action_digest=_string(item, "runtime_action_digest"),
                expected_task_indexes=tuple(expected),
                tasks=tuple(
                    postprocessing_runtime_task_evidence_from_mapping(task) for task in _mapping_list(item, "tasks")
                ),
            )
        )
    witness = _mapping(payload, "action09_prepublication_witness")
    return PostprocessingRuntimeActionEvidenceAggregate(
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="PostprocessingRuntimeActionEvidenceAggregate"
        ),
        aggregate_kind="postprocessing-runtime-action-evidence-aggregate-v1",
        phase_run_id=_string(payload, "phase_run_id"),
        attempt_id=_string(payload, "attempt_id"),
        phase_runspec_digest=_string(payload, "phase_runspec_digest"),
        action_graph_digest=_string(payload, "action_graph_digest"),
        completed_actions=tuple(completed),
        action09_prepublication_witness=postprocessing_action09_assembly_witness_from_mapping(witness),
    )


def _sha(value: str, label: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase SHA-256")


def _schema(value: int, record: str) -> None:
    validate_schema_version(value, record_name=record)


def _fields(payload: Mapping[str, object], allowed: set[str], record: str) -> None:
    unknown = sorted(set(payload) - allowed)
    missing = sorted(allowed - set(payload))
    if unknown or missing:
        raise ValueError(f"{record} fields differ; missing={missing!r}, unknown={unknown!r}")


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _mapping_list(payload: Mapping[str, object], key: str) -> tuple[Mapping[str, object], ...]:
    value = payload.get(key)
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"{key} must be a list of mappings")
    return tuple(cast("Mapping[str, object]", item) for item in value)


def _string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


__all__ = [
    "PostprocessingCompletedRuntimeActionEvidence",
    "PostprocessingRuntimeActionEvidenceAggregate",
    "PostprocessingRuntimeTaskEvidence",
    "postprocessing_runtime_action_evidence_aggregate_from_mapping",
    "postprocessing_runtime_task_evidence_from_mapping",
]
