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

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, cast

from bspp.orchestration.contract.phase import validate_phase_attempt_id, validate_phase_run_id
from bspp.orchestration.contract.postprocessing_cancellation_events import (
    PostprocessingCancellationIntendedPayload,
    PostprocessingCancelledPayload,
    PostprocessingJobCancellationRequestIntendedPayload,
    PostprocessingJobCancellationRequestResultPayload,
    _cancellation_intended,
    _cancellation_request_intended,
    _cancellation_request_result,
    _cancelled,
)
from bspp.orchestration.contract.postprocessing_phase_receipt import (
    PostprocessingFinalizedPayload,
    postprocessing_finalized_payload_from_mapping,
)
from bspp.orchestration.contract.postprocessing_retry_events import PostprocessingAttemptRetriedPayload, _retried
from bspp.orchestration.contract.postprocessing_submission_events import (
    PostprocessingActionDispatchIntendedPayload,
    PostprocessingActionDispatchRejectedPayload,
    PostprocessingActionSubmittedPayload,
    PostprocessingMaterializedPayload,
    PostprocessingSubmissionIntendedPayload,
    _dispatch,
    _materialized,
    _rejected,
    _submission_intended,
    _submitted,
)
from bspp.orchestration.contract.postprocessing_terminal_events import (
    PostprocessingActionTerminalObservedPayload,
    PostprocessingArrayParentCancelledObservedPayload,
    _array_parent_cancelled,
    _terminal,
)
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

PostprocessingEventType = Literal[
    "phase-materialized",
    "phase-submission-intended",
    "phase-action-dispatch-intended",
    "phase-action-submitted",
    "phase-action-dispatch-rejected",
    "phase-action-terminal-observed",
    "phase-array-parent-cancelled-observed",
    "phase-cancellation-intended",
    "phase-job-cancellation-request-intended",
    "phase-job-cancellation-request-result",
    "phase-cancelled",
    "phase-attempt-retried",
    "postprocessing-phase-finalized",
]


PostprocessingEventPayload = (
    PostprocessingMaterializedPayload
    | PostprocessingSubmissionIntendedPayload
    | PostprocessingActionDispatchIntendedPayload
    | PostprocessingActionSubmittedPayload
    | PostprocessingActionDispatchRejectedPayload
    | PostprocessingActionTerminalObservedPayload
    | PostprocessingArrayParentCancelledObservedPayload
    | PostprocessingCancellationIntendedPayload
    | PostprocessingJobCancellationRequestIntendedPayload
    | PostprocessingJobCancellationRequestResultPayload
    | PostprocessingCancelledPayload
    | PostprocessingAttemptRetriedPayload
    | PostprocessingFinalizedPayload
)


@dataclass(frozen=True)
class PostprocessingPhaseEvent:
    sequence: int
    event_type: PostprocessingEventType
    phase_run_id: str
    attempt_id: str
    occurred_at: str
    payload: PostprocessingEventPayload
    phase_kind: Literal["postprocessing"] = "postprocessing"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _version(self.schema_version, type(self).__name__)
        if self.phase_kind != "postprocessing" or self.sequence <= 0 or not self.occurred_at:
            raise ValueError("postprocessing event envelope is invalid")
        validate_phase_run_id(self.phase_run_id)
        validate_phase_attempt_id(self.attempt_id)
        expected = _EVENT_PAYLOAD_TYPES.get(self.event_type)
        if expected is None or not isinstance(self.payload, expected):
            raise ValueError("postprocessing event type and payload family differ")
        if isinstance(self.payload, PostprocessingAttemptRetriedPayload) and (
            self.attempt_id != self.payload.successor_phase_runspec.attempt_id
            or self.phase_run_id != self.payload.successor_phase_runspec.phase_run_id
        ):
            raise ValueError("postprocessing Retry event does not bind its successor RunSpec")
        if isinstance(self.payload, PostprocessingFinalizedPayload) and (
            self.attempt_id != self.payload.receipt.attempt_id or self.phase_run_id != self.payload.receipt.phase_run_id
        ):
            raise ValueError("postprocessing finalized event does not bind its Phase Receipt")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "event_type": self.event_type,
            "phase_kind": self.phase_kind,
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "occurred_at": self.occurred_at,
            "payload": self.payload.to_mapping(),
        }


_EVENT_PAYLOAD_TYPES: dict[str, type[object]] = {
    "phase-materialized": PostprocessingMaterializedPayload,
    "phase-submission-intended": PostprocessingSubmissionIntendedPayload,
    "phase-action-dispatch-intended": PostprocessingActionDispatchIntendedPayload,
    "phase-action-submitted": PostprocessingActionSubmittedPayload,
    "phase-action-dispatch-rejected": PostprocessingActionDispatchRejectedPayload,
    "phase-action-terminal-observed": PostprocessingActionTerminalObservedPayload,
    "phase-array-parent-cancelled-observed": PostprocessingArrayParentCancelledObservedPayload,
    "phase-cancellation-intended": PostprocessingCancellationIntendedPayload,
    "phase-job-cancellation-request-intended": PostprocessingJobCancellationRequestIntendedPayload,
    "phase-job-cancellation-request-result": PostprocessingJobCancellationRequestResultPayload,
    "phase-cancelled": PostprocessingCancelledPayload,
    "phase-attempt-retried": PostprocessingAttemptRetriedPayload,
    "postprocessing-phase-finalized": PostprocessingFinalizedPayload,
}


def postprocessing_phase_event_from_mapping(payload: Mapping[str, object]) -> PostprocessingPhaseEvent:
    _fields(
        payload,
        {
            "schema_version",
            "sequence",
            "event_type",
            "phase_kind",
            "phase_run_id",
            "attempt_id",
            "occurred_at",
            "payload",
        },
        "PostprocessingPhaseEvent",
    )
    event_type = _string(payload, "event_type")
    payload_mapping = payload.get("payload")
    if not isinstance(payload_mapping, Mapping):
        raise ValueError("postprocessing event payload must be a mapping")
    parser = _PAYLOAD_PARSERS.get(event_type)
    if parser is None:
        raise ValueError(f"unsupported postprocessing event type: {event_type!r}")
    return PostprocessingPhaseEvent(
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="PostprocessingPhaseEvent"),
        sequence=_integer(payload, "sequence"),
        event_type=cast("PostprocessingEventType", event_type),
        phase_kind=cast("Literal['postprocessing']", _string(payload, "phase_kind")),
        phase_run_id=_string(payload, "phase_run_id"),
        attempt_id=_string(payload, "attempt_id"),
        occurred_at=_string(payload, "occurred_at"),
        payload=parser(cast("Mapping[str, object]", payload_mapping)),
    )


def _finalized(payload: Mapping[str, object]) -> PostprocessingFinalizedPayload:
    return postprocessing_finalized_payload_from_mapping(payload)


_PAYLOAD_PARSERS: dict[
    str,
    Callable[[Mapping[str, object]], PostprocessingEventPayload],
] = {
    "phase-materialized": _materialized,
    "phase-submission-intended": _submission_intended,
    "phase-action-dispatch-intended": _dispatch,
    "phase-action-submitted": _submitted,
    "phase-action-dispatch-rejected": _rejected,
    "phase-action-terminal-observed": _terminal,
    "phase-array-parent-cancelled-observed": _array_parent_cancelled,
    "phase-cancellation-intended": _cancellation_intended,
    "phase-job-cancellation-request-intended": _cancellation_request_intended,
    "phase-job-cancellation-request-result": _cancellation_request_result,
    "phase-cancelled": _cancelled,
    "phase-attempt-retried": _retried,
    "postprocessing-phase-finalized": _finalized,
}


def _version(value: int, name: str) -> None:
    validate_schema_version(value, record_name=name)


def _fields(payload: Mapping[str, object], expected: set[str], name: str) -> None:
    if set(payload) != expected:
        raise ValueError(f"{name} has missing or extra fields")


def _string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _integer(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


__all__ = [
    "PostprocessingEventPayload",
    "PostprocessingEventType",
    "PostprocessingPhaseEvent",
    "postprocessing_phase_event_from_mapping",
]
