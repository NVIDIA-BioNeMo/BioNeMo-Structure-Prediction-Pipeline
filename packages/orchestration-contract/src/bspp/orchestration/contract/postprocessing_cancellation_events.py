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

from bspp.orchestration.contract.postprocessing_phase_ids import (
    POSTPROCESSING_ACTION_IDS,
    validate_postprocessing_action_id,
)
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

_SHA256 = re.compile(r"[0-9a-f]{64}")


_SUBMISSION_ID = re.compile(r"postprocessing-submission-[0-9a-f]{64}")


_JOB_ID = re.compile(r"[0-9]+")


@dataclass(frozen=True)
class PostprocessingCancellationIntendedPayload:
    phase_runspec_digest: str
    submission_id: str | None
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _version(self.schema_version, type(self).__name__)
        _sha(self.phase_runspec_digest)
        if self.submission_id is not None:
            _submission(self.submission_id)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "submission_id": self.submission_id,
            "phase_runspec_digest": self.phase_runspec_digest,
        }


@dataclass(frozen=True)
class PostprocessingJobCancellationRequestIntendedPayload:
    action_id: str
    parent_job_id: str
    request_ordinal: int
    scancel_argv: tuple[str, ...]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _version(self.schema_version, type(self).__name__)
        _action(self.action_id)
        if _JOB_ID.fullmatch(self.parent_job_id) is None:
            raise ValueError("postprocessing cancelled parent job id must be numeric")
        if self.request_ordinal <= 0 or self.scancel_argv != ("scancel", self.parent_job_id):
            raise ValueError("postprocessing cancellation request intent is invalid")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "action_id": self.action_id,
            "parent_job_id": self.parent_job_id,
            "request_ordinal": self.request_ordinal,
            "scancel_argv": list(self.scancel_argv),
        }


@dataclass(frozen=True)
class PostprocessingJobCancellationRequestResultPayload:
    action_id: str
    parent_job_id: str
    request_ordinal: int
    scancel_argv: tuple[str, ...]
    return_code: int
    stdout: str
    stderr: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _version(self.schema_version, type(self).__name__)
        _action(self.action_id)
        if _JOB_ID.fullmatch(self.parent_job_id) is None:
            raise ValueError("postprocessing cancelled parent job id must be numeric")
        if (
            self.request_ordinal <= 0
            or self.scancel_argv != ("scancel", self.parent_job_id)
            or not isinstance(self.return_code, int)
            or isinstance(self.return_code, bool)
        ):
            raise ValueError("postprocessing cancellation return code must be an integer")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "action_id": self.action_id,
            "parent_job_id": self.parent_job_id,
            "request_ordinal": self.request_ordinal,
            "scancel_argv": list(self.scancel_argv),
            "return_code": self.return_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


@dataclass(frozen=True)
class PostprocessingCancelledPayload:
    terminal_parent_job_ids: tuple[str, ...]
    terminal_action_ids: tuple[str, ...]
    terminal_task_evidence_digest: str
    cancelled_parent_job_ids: tuple[str, ...]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _version(self.schema_version, type(self).__name__)
        for values in (self.terminal_parent_job_ids, self.cancelled_parent_job_ids):
            if tuple(sorted(set(values), key=int)) != values or any(_JOB_ID.fullmatch(item) is None for item in values):
                raise ValueError("postprocessing cancelled job ids must be numeric, unique, and sorted")
        if not set(self.cancelled_parent_job_ids) <= set(self.terminal_parent_job_ids):
            raise ValueError("cancelled postprocessing jobs must be accounting-terminal")
        permanent_ids = tuple(POSTPROCESSING_ACTION_IDS.values())
        if (
            len(set(self.terminal_action_ids)) != len(self.terminal_action_ids)
            or any(item not in permanent_ids for item in self.terminal_action_ids)
            or tuple(sorted(self.terminal_action_ids, key=permanent_ids.index)) != self.terminal_action_ids
        ):
            raise ValueError("postprocessing cancelled terminal actions must use canonical graph order")
        _sha(self.terminal_task_evidence_digest)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "terminal_parent_job_ids": list(self.terminal_parent_job_ids),
            "terminal_action_ids": list(self.terminal_action_ids),
            "terminal_task_evidence_digest": self.terminal_task_evidence_digest,
            "cancelled_parent_job_ids": list(self.cancelled_parent_job_ids),
        }


def _cancellation_intended(payload: Mapping[str, object]) -> PostprocessingCancellationIntendedPayload:
    _fields(
        payload,
        {"schema_version", "submission_id", "phase_runspec_digest"},
        "PostprocessingCancellationIntendedPayload",
    )
    submission_id = payload.get("submission_id")
    if submission_id is not None and not isinstance(submission_id, str):
        raise ValueError("postprocessing cancellation submission id must be string or null")
    return PostprocessingCancellationIntendedPayload(
        schema_version=_schema(payload, "PostprocessingCancellationIntendedPayload"),
        submission_id=submission_id,
        phase_runspec_digest=_string(payload, "phase_runspec_digest"),
    )


def _cancellation_request_intended(
    payload: Mapping[str, object],
) -> PostprocessingJobCancellationRequestIntendedPayload:
    _fields(
        payload,
        {"schema_version", "action_id", "parent_job_id", "request_ordinal", "scancel_argv"},
        "PostprocessingJobCancellationRequestIntendedPayload",
    )
    return PostprocessingJobCancellationRequestIntendedPayload(
        schema_version=_schema(payload, "PostprocessingJobCancellationRequestIntendedPayload"),
        action_id=_string(payload, "action_id"),
        parent_job_id=_string(payload, "parent_job_id"),
        request_ordinal=_integer(payload, "request_ordinal"),
        scancel_argv=_strings(payload, "scancel_argv"),
    )


def _cancellation_request_result(
    payload: Mapping[str, object],
) -> PostprocessingJobCancellationRequestResultPayload:
    _fields(
        payload,
        {
            "schema_version",
            "action_id",
            "parent_job_id",
            "request_ordinal",
            "scancel_argv",
            "return_code",
            "stdout",
            "stderr",
        },
        "PostprocessingJobCancellationRequestResultPayload",
    )
    return PostprocessingJobCancellationRequestResultPayload(
        schema_version=_schema(payload, "PostprocessingJobCancellationRequestResultPayload"),
        action_id=_string(payload, "action_id"),
        parent_job_id=_string(payload, "parent_job_id"),
        request_ordinal=_integer(payload, "request_ordinal"),
        scancel_argv=_strings(payload, "scancel_argv"),
        return_code=_integer(payload, "return_code"),
        stdout=_text(payload, "stdout"),
        stderr=_text(payload, "stderr"),
    )


def _cancelled(payload: Mapping[str, object]) -> PostprocessingCancelledPayload:
    _fields(
        payload,
        {
            "schema_version",
            "terminal_parent_job_ids",
            "terminal_action_ids",
            "terminal_task_evidence_digest",
            "cancelled_parent_job_ids",
        },
        "PostprocessingCancelledPayload",
    )
    return PostprocessingCancelledPayload(
        schema_version=_schema(payload, "PostprocessingCancelledPayload"),
        terminal_parent_job_ids=_strings(payload, "terminal_parent_job_ids"),
        terminal_action_ids=_strings(payload, "terminal_action_ids"),
        terminal_task_evidence_digest=_string(payload, "terminal_task_evidence_digest"),
        cancelled_parent_job_ids=_strings(payload, "cancelled_parent_job_ids"),
    )


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


__all__ = [
    "PostprocessingCancellationIntendedPayload",
    "PostprocessingCancelledPayload",
    "PostprocessingJobCancellationRequestIntendedPayload",
    "PostprocessingJobCancellationRequestResultPayload",
]
