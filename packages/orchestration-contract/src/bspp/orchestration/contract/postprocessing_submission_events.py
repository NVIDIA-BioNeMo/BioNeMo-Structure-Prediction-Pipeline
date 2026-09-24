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
from typing import cast

from bspp.orchestration.contract.postprocessing_phase_ids import (
    POSTPROCESSING_ACTION_IDS,
    validate_postprocessing_action_id,
)
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

_SHA256 = re.compile(r"[0-9a-f]{64}")


_SUBMISSION_ID = re.compile(r"postprocessing-submission-[0-9a-f]{64}")


_JOB_ID = re.compile(r"[0-9]+")


_CORRELATION = re.compile(r"bspp-pp-[0-9a-f]{48}")
# Historical V1/V2 submission events were authored with the pre-rename token
# prefix; the loader must keep accepting them byte-identically.
_LEGACY_CORRELATION = re.compile(r"afcdb-pp-[0-9a-f]{48}")

POSTPROCESSING_RENDERER_CONTRACT_CURRENT = 5
POSTPROCESSING_RENDERER_CONTRACT_SUPPORTED = frozenset({1, 2, 3, 4, POSTPROCESSING_RENDERER_CONTRACT_CURRENT})
POSTPROCESSING_V2_RENDERER_CONTRACT_SUPPORTED = frozenset({1, 2})
POSTPROCESSING_V3_RENDERER_CONTRACT_SUPPORTED = frozenset({3, 4, POSTPROCESSING_RENDERER_CONTRACT_CURRENT})


@dataclass(frozen=True)
class PostprocessingMaterializedPayload:
    phase_plan_digest: str
    phase_runspec_digest: str
    action_graph_digest: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _version(self.schema_version, type(self).__name__)
        for value in (self.phase_plan_digest, self.phase_runspec_digest, self.action_graph_digest):
            _sha(value)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "phase_plan_digest": self.phase_plan_digest,
            "phase_runspec_digest": self.phase_runspec_digest,
            "action_graph_digest": self.action_graph_digest,
        }


@dataclass(frozen=True)
class PostprocessingSubmissionActionPlan:
    action_id: str
    runtime_action_digest: str
    dependencies: tuple[str, ...]
    cluster_script_path: str
    script_sha256: str
    scheduler_correlation_token: str
    renderer_contract_version: int = 1
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _version(self.schema_version, type(self).__name__)
        _action(self.action_id)
        _sha(self.runtime_action_digest)
        _sha(self.script_sha256)
        if self.renderer_contract_version not in POSTPROCESSING_RENDERER_CONTRACT_SUPPORTED:
            raise ValueError("unsupported postprocessing renderer contract version")
        permanent_ids = tuple(POSTPROCESSING_ACTION_IDS.values())
        if (
            len(set(self.dependencies)) != len(self.dependencies)
            or any(item not in permanent_ids for item in self.dependencies)
            or tuple(sorted(self.dependencies, key=permanent_ids.index)) != self.dependencies
            or any(permanent_ids.index(item) >= permanent_ids.index(self.action_id) for item in self.dependencies)
        ):
            raise ValueError("postprocessing submission dependencies contain an unknown action")
        if not self.cluster_script_path.startswith("/") or not self.cluster_script_path.endswith(".sbatch"):
            raise ValueError("postprocessing cluster script path must be absolute and end in .sbatch")
        if (
            _CORRELATION.fullmatch(self.scheduler_correlation_token) is None
            and _LEGACY_CORRELATION.fullmatch(self.scheduler_correlation_token) is None
        ):
            raise ValueError("postprocessing scheduler correlation token is invalid")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "action_id": self.action_id,
            "runtime_action_digest": self.runtime_action_digest,
            "dependencies": list(self.dependencies),
            "cluster_script_path": self.cluster_script_path,
            "script_sha256": self.script_sha256,
            "scheduler_correlation_token": self.scheduler_correlation_token,
            "renderer_contract_version": self.renderer_contract_version,
        }


@dataclass(frozen=True)
class PostprocessingSubmissionIntendedPayload:
    submission_id: str
    phase_runspec_digest: str
    actions: tuple[PostprocessingSubmissionActionPlan, ...]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _version(self.schema_version, type(self).__name__)
        _submission(self.submission_id)
        _sha(self.phase_runspec_digest)
        if not self.actions or len({item.action_id for item in self.actions}) != len(self.actions):
            raise ValueError("postprocessing submission actions must be non-empty and unique")
        if len({item.renderer_contract_version for item in self.actions}) != 1:
            raise ValueError("postprocessing submission actions must use one renderer contract version")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "submission_id": self.submission_id,
            "phase_runspec_digest": self.phase_runspec_digest,
            "actions": [item.to_mapping() for item in self.actions],
        }


@dataclass(frozen=True)
class PostprocessingActionDispatchIntendedPayload:
    submission_id: str
    action_id: str
    scheduler_correlation_token: str
    dependency_job_ids: tuple[str, ...]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _common_submission_action(self.schema_version, self.submission_id, self.action_id)
        if (
            _CORRELATION.fullmatch(self.scheduler_correlation_token) is None
            and _LEGACY_CORRELATION.fullmatch(self.scheduler_correlation_token) is None
        ):
            raise ValueError("postprocessing dispatch correlation token is invalid")
        if any(_JOB_ID.fullmatch(item) is None for item in self.dependency_job_ids):
            raise ValueError("postprocessing dispatch dependency job IDs must be numeric")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "submission_id": self.submission_id,
            "action_id": self.action_id,
            "scheduler_correlation_token": self.scheduler_correlation_token,
            "dependency_job_ids": list(self.dependency_job_ids),
        }


@dataclass(frozen=True)
class PostprocessingActionSubmittedPayload:
    submission_id: str
    action_id: str
    parent_job_id: str
    scheduler_correlation_token: str
    expected_task_indexes: tuple[int, ...]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _common_submission_action(self.schema_version, self.submission_id, self.action_id)
        if _JOB_ID.fullmatch(self.parent_job_id) is None:
            raise ValueError("postprocessing parent job id must be numeric")
        if (
            _CORRELATION.fullmatch(self.scheduler_correlation_token) is None
            and _LEGACY_CORRELATION.fullmatch(self.scheduler_correlation_token) is None
        ):
            raise ValueError("postprocessing assignment correlation token is invalid")
        _task_indexes(self.expected_task_indexes)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "submission_id": self.submission_id,
            "action_id": self.action_id,
            "parent_job_id": self.parent_job_id,
            "scheduler_correlation_token": self.scheduler_correlation_token,
            "expected_task_indexes": list(self.expected_task_indexes),
        }


@dataclass(frozen=True)
class PostprocessingActionDispatchRejectedPayload:
    submission_id: str
    action_id: str
    return_code: int
    stdout: str
    stderr: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _common_submission_action(self.schema_version, self.submission_id, self.action_id)
        if not isinstance(self.return_code, int) or isinstance(self.return_code, bool):
            raise ValueError("postprocessing rejected return code must be an integer")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "submission_id": self.submission_id,
            "action_id": self.action_id,
            "return_code": self.return_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


def _materialized(payload: Mapping[str, object]) -> PostprocessingMaterializedPayload:
    _fields(
        payload,
        {"schema_version", "phase_plan_digest", "phase_runspec_digest", "action_graph_digest"},
        "PostprocessingMaterializedPayload",
    )
    return PostprocessingMaterializedPayload(
        schema_version=_schema(payload, "PostprocessingMaterializedPayload"),
        phase_plan_digest=_string(payload, "phase_plan_digest"),
        phase_runspec_digest=_string(payload, "phase_runspec_digest"),
        action_graph_digest=_string(payload, "action_graph_digest"),
    )


def _submission_intended(payload: Mapping[str, object]) -> PostprocessingSubmissionIntendedPayload:
    _fields(
        payload,
        {"schema_version", "submission_id", "phase_runspec_digest", "actions"},
        "PostprocessingSubmissionIntendedPayload",
    )
    return PostprocessingSubmissionIntendedPayload(
        schema_version=_schema(payload, "PostprocessingSubmissionIntendedPayload"),
        submission_id=_string(payload, "submission_id"),
        phase_runspec_digest=_string(payload, "phase_runspec_digest"),
        actions=tuple(_submission_action(item) for item in _mapping_list(payload, "actions")),
    )


def _submission_action(payload: Mapping[str, object]) -> PostprocessingSubmissionActionPlan:
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
            "renderer_contract_version",
        },
        "PostprocessingSubmissionActionPlan",
    )
    return PostprocessingSubmissionActionPlan(
        schema_version=_schema(payload, "PostprocessingSubmissionActionPlan"),
        action_id=_string(payload, "action_id"),
        runtime_action_digest=_string(payload, "runtime_action_digest"),
        dependencies=_strings(payload, "dependencies"),
        cluster_script_path=_string(payload, "cluster_script_path"),
        script_sha256=_string(payload, "script_sha256"),
        scheduler_correlation_token=_string(payload, "scheduler_correlation_token"),
        renderer_contract_version=_integer(payload, "renderer_contract_version"),
    )


def _dispatch(payload: Mapping[str, object]) -> PostprocessingActionDispatchIntendedPayload:
    _fields(
        payload,
        {"schema_version", "submission_id", "action_id", "scheduler_correlation_token", "dependency_job_ids"},
        "PostprocessingActionDispatchIntendedPayload",
    )
    return PostprocessingActionDispatchIntendedPayload(
        schema_version=_schema(payload, "PostprocessingActionDispatchIntendedPayload"),
        submission_id=_string(payload, "submission_id"),
        action_id=_string(payload, "action_id"),
        scheduler_correlation_token=_string(payload, "scheduler_correlation_token"),
        dependency_job_ids=_strings(payload, "dependency_job_ids"),
    )


def _submitted(payload: Mapping[str, object]) -> PostprocessingActionSubmittedPayload:
    _fields(
        payload,
        {
            "schema_version",
            "submission_id",
            "action_id",
            "parent_job_id",
            "scheduler_correlation_token",
            "expected_task_indexes",
        },
        "PostprocessingActionSubmittedPayload",
    )
    return PostprocessingActionSubmittedPayload(
        schema_version=_schema(payload, "PostprocessingActionSubmittedPayload"),
        submission_id=_string(payload, "submission_id"),
        action_id=_string(payload, "action_id"),
        parent_job_id=_string(payload, "parent_job_id"),
        scheduler_correlation_token=_string(payload, "scheduler_correlation_token"),
        expected_task_indexes=_integers(payload, "expected_task_indexes"),
    )


def _rejected(payload: Mapping[str, object]) -> PostprocessingActionDispatchRejectedPayload:
    _fields(
        payload,
        {"schema_version", "submission_id", "action_id", "return_code", "stdout", "stderr"},
        "PostprocessingActionDispatchRejectedPayload",
    )
    return PostprocessingActionDispatchRejectedPayload(
        schema_version=_schema(payload, "PostprocessingActionDispatchRejectedPayload"),
        submission_id=_string(payload, "submission_id"),
        action_id=_string(payload, "action_id"),
        return_code=_integer(payload, "return_code"),
        stdout=_text(payload, "stdout"),
        stderr=_text(payload, "stderr"),
    )


def _common_submission_action(version: int, submission_id: str, action_id: str) -> None:
    _version(version, "postprocessing submission action payload")
    _submission(submission_id)
    _action(action_id)


def _version(value: int, name: str) -> None:
    validate_schema_version(value, record_name=name)


def _sha(value: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise ValueError("postprocessing lifecycle digest must be lowercase SHA-256")


def _submission(value: str) -> None:
    if _SUBMISSION_ID.fullmatch(value) is None:
        raise ValueError("postprocessing submission id is invalid")


def _action(value: str) -> None:
    validate_postprocessing_action_id(value)


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


def _text(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _integer(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
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
    "POSTPROCESSING_RENDERER_CONTRACT_CURRENT",
    "POSTPROCESSING_RENDERER_CONTRACT_SUPPORTED",
    "POSTPROCESSING_V2_RENDERER_CONTRACT_SUPPORTED",
    "POSTPROCESSING_V3_RENDERER_CONTRACT_SUPPORTED",
    "PostprocessingActionDispatchIntendedPayload",
    "PostprocessingActionDispatchRejectedPayload",
    "PostprocessingActionSubmittedPayload",
    "PostprocessingMaterializedPayload",
    "PostprocessingSubmissionActionPlan",
    "PostprocessingSubmissionIntendedPayload",
]
