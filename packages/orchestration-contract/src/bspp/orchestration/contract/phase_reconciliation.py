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

"""Durable terminal scheduler observations for one Phase Attempt."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, cast

from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

PhaseActionTerminalOutcome = Literal["succeeded", "failed"]
PhaseActionTerminalSource = Literal["sacct"]
PhaseActionTerminalObservedEventType = Literal["phase-action-terminal-observed"]
FoldingActionTerminalObservedEventType = Literal["phase-action-array-terminal-observed"]

TERMINAL_PHASE_SLURM_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "TIMEOUT",
    }
)

_PHASE_RUN_ID = re.compile(r"phase-run-[0-9a-f]{32}")
_ATTEMPT_ID = re.compile(r"attempt-[0-9]{4}")
_SUBMISSION_ID = re.compile(r"phase-submission-[0-9a-f]{64}")
_ACTION_ID = re.compile(r"(?:preprocessing-chunk|msa-flatten|split|preprocess|fold|canonical-pair)-[0-9]{6}")
_CORRELATION_TOKEN = re.compile(r"bspp-phase-[0-9a-f]{64}")
_LEGACY_CORRELATION_TOKEN = re.compile(r"afcdb-phase-[0-9a-f]{64}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_JOB_ID = re.compile(r"[0-9]+")
_JOB_ID_WITH_TASK = re.compile(r"[0-9]+(?:_[0-9]+)?")
_EXIT_CODE = re.compile(r"[0-9]+:[0-9]+")


def phase_action_terminal_outcome(state: str, exit_code: str | None) -> PhaseActionTerminalOutcome | None:
    """Classify conclusive normalized terminal accounting, or return ``None``."""
    if state not in TERMINAL_PHASE_SLURM_STATES:
        return None
    if exit_code is not None and _EXIT_CODE.fullmatch(exit_code) is None:
        return None
    if state == "COMPLETED":
        if exit_code is None:
            return None
        return "succeeded" if exit_code == "0:0" else "failed"
    return "failed"


@dataclass(frozen=True)
class PhaseActionTerminalObservedPayload:
    """One conclusive terminal ``sacct`` fact bound to a durable assignment."""

    submission_id: str
    phase_runspec_digest: str
    action_id: str
    runtime_action_digest: str
    scheduler_correlation_token: str
    job_id: str
    state: str
    exit_code: str | None
    outcome: PhaseActionTerminalOutcome
    source: PhaseActionTerminalSource = "sacct"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        _match(self.submission_id, _SUBMISSION_ID, "terminal observation submission id")
        _match(self.phase_runspec_digest, _SHA256, "terminal observation RunSpec digest")
        _match(self.action_id, _ACTION_ID, "terminal observation action id")
        _match(self.runtime_action_digest, _SHA256, "terminal observation Runtime Action digest")
        _match(self.scheduler_correlation_token, _CORRELATION_TOKEN, "terminal observation correlation token")
        _match(self.job_id, _JOB_ID, "terminal observation job id")
        if self.source != "sacct":
            raise ValueError("terminal observation source must be 'sacct'")
        if self.exit_code is not None and _EXIT_CODE.fullmatch(self.exit_code) is None:
            raise ValueError("terminal observation exit code must use Slurm status:signal grammar")
        expected = phase_action_terminal_outcome(self.state, self.exit_code)
        if expected is None:
            raise ValueError("terminal observation requires conclusive terminal sacct state")
        if self.outcome != expected:
            raise ValueError("terminal observation outcome does not match its state and exit code")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "submission_id": self.submission_id,
            "phase_runspec_digest": self.phase_runspec_digest,
            "action_id": self.action_id,
            "runtime_action_digest": self.runtime_action_digest,
            "scheduler_correlation_token": self.scheduler_correlation_token,
            "job_id": self.job_id,
            "state": self.state,
            "exit_code": self.exit_code,
            "outcome": self.outcome,
            "source": self.source,
        }


@dataclass(frozen=True)
class PhaseActionTerminalObservedEvent:
    """Append-only terminal accounting event for one submitted action."""

    sequence: int
    phase_run_id: str
    attempt_id: str
    occurred_at: str
    payload: PhaseActionTerminalObservedPayload
    event_type: PhaseActionTerminalObservedEventType = "phase-action-terminal-observed"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        if not isinstance(self.sequence, int) or isinstance(self.sequence, bool) or self.sequence <= 1:
            raise ValueError("terminal observation event sequence must follow materialization")
        if self.event_type != "phase-action-terminal-observed":
            raise ValueError("terminal observation event has an invalid discriminator")
        _match(self.phase_run_id, _PHASE_RUN_ID, "terminal observation Phase Run id")
        _match(self.attempt_id, _ATTEMPT_ID, "terminal observation Attempt id")
        _timestamp(self.occurred_at)
        if not isinstance(self.payload, PhaseActionTerminalObservedPayload):
            raise ValueError("terminal observation event requires its strict payload")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "event_type": self.event_type,
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "occurred_at": self.occurred_at,
            "payload": self.payload.to_mapping(),
        }


@dataclass(frozen=True)
class PhaseActionTerminalObservationView:
    """Replay-only terminal scheduler fact exposed to Control services."""

    action_id: str
    job_id: str
    state: str
    exit_code: str | None
    outcome: PhaseActionTerminalOutcome
    source: PhaseActionTerminalSource
    observed_at: str

    def __post_init__(self) -> None:
        _match(self.action_id, _ACTION_ID, "terminal view action id")
        _match(self.job_id, _JOB_ID, "terminal view job id")
        if self.source != "sacct":
            raise ValueError("terminal view source must be 'sacct'")
        expected = phase_action_terminal_outcome(self.state, self.exit_code)
        if expected is None or expected != self.outcome:
            raise ValueError("terminal view requires a coherent terminal outcome")
        _timestamp(self.observed_at)

    @classmethod
    def from_event(cls, event: PhaseActionTerminalObservedEvent) -> PhaseActionTerminalObservationView:
        return cls(
            action_id=event.payload.action_id,
            job_id=event.payload.job_id,
            state=event.payload.state,
            exit_code=event.payload.exit_code,
            outcome=event.payload.outcome,
            source=event.payload.source,
            observed_at=event.occurred_at,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "action_id": self.action_id,
            "job_id": self.job_id,
            "state": self.state,
            "exit_code": self.exit_code,
            "outcome": self.outcome,
            "source": self.source,
            "observed_at": self.observed_at,
        }


@dataclass(frozen=True)
class FoldingTaskTerminalEvidence:
    """One conclusive terminal ``sacct`` fact for a packed fold array task."""

    scheduler_job_id: str
    state: str
    exit_code: str
    source: PhaseActionTerminalSource
    restarts: int | None = None
    task_index: int | None = None
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        if _JOB_ID_WITH_TASK.fullmatch(self.scheduler_job_id) is None:
            raise ValueError("folding task scheduler id is invalid")
        if not self.state or not self.exit_code or self.source != "sacct":
            raise ValueError("folding task evidence requires sacct state and exit")
        if self.restarts is not None and (
            not isinstance(self.restarts, int) or isinstance(self.restarts, bool) or self.restarts < 0
        ):
            raise ValueError("folding task restarts must be a non-negative integer or null")
        if self.task_index is not None and (
            not isinstance(self.task_index, int) or isinstance(self.task_index, bool) or self.task_index < 0
        ):
            raise ValueError("folding task index must be non-negative")

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
class FoldingActionTerminalObservedPayload:
    """One conclusive terminal ``sacct`` fact set bound to a packed fold array."""

    submission_id: str
    phase_runspec_digest: str
    action_id: str
    runtime_action_digest: str
    parent_job_id: str
    expected_task_indexes: tuple[int, ...]
    tasks: tuple[FoldingTaskTerminalEvidence, ...]
    outcome: PhaseActionTerminalOutcome
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        _match(self.submission_id, _SUBMISSION_ID, "folding terminal observation submission id")
        _match(self.phase_runspec_digest, _SHA256, "folding terminal observation RunSpec digest")
        _match(self.action_id, _ACTION_ID, "folding terminal observation action id")
        _match(self.runtime_action_digest, _SHA256, "folding terminal observation Runtime Action digest")
        _match(self.parent_job_id, _JOB_ID, "folding terminal observation parent job id")
        if not self.tasks:
            raise ValueError("folding terminal observation requires at least one task")
        _task_indexes(self.expected_task_indexes)
        indexes = tuple(item.task_index for item in self.tasks if item.task_index is not None)
        if self.expected_task_indexes:
            if indexes != self.expected_task_indexes or len(indexes) != len(self.tasks):
                raise ValueError("folding terminal tasks differ from the exact expected indexes")
            if any(item.scheduler_job_id != f"{self.parent_job_id}_{item.task_index}" for item in self.tasks):
                raise ValueError("folding array task scheduler IDs do not bind parent/index")
        elif len(self.tasks) != 1 or self.tasks[0].task_index is not None:
            raise ValueError("non-array folding action requires one parent task")
        elif self.tasks[0].scheduler_job_id != self.parent_job_id:
            raise ValueError("folding non-array task scheduler ID does not bind its parent")
        succeeded = all(item.state == "COMPLETED" and item.exit_code in {"0", "0:0"} for item in self.tasks)
        if (self.outcome == "succeeded") != succeeded:
            raise ValueError("folding terminal outcome differs from task evidence")

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
class FoldingActionTerminalObservedEvent:
    """Append-only terminal accounting event for one packed fold array action."""

    sequence: int
    phase_run_id: str
    attempt_id: str
    occurred_at: str
    payload: FoldingActionTerminalObservedPayload
    event_type: FoldingActionTerminalObservedEventType = "phase-action-array-terminal-observed"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        if not isinstance(self.sequence, int) or isinstance(self.sequence, bool) or self.sequence <= 1:
            raise ValueError("folding terminal observation event sequence must follow materialization")
        if self.event_type != "phase-action-array-terminal-observed":
            raise ValueError("folding terminal observation event has an invalid discriminator")
        _match(self.phase_run_id, _PHASE_RUN_ID, "folding terminal observation Phase Run id")
        _match(self.attempt_id, _ATTEMPT_ID, "folding terminal observation Attempt id")
        _timestamp(self.occurred_at)
        if not isinstance(self.payload, FoldingActionTerminalObservedPayload):
            raise ValueError("folding terminal observation event requires its strict payload")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "event_type": self.event_type,
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "occurred_at": self.occurred_at,
            "payload": self.payload.to_mapping(),
        }


@dataclass(frozen=True)
class FoldingActionTerminalObservationView:
    """Replay-only packed fold array terminal fact exposed to Control services."""

    action_id: str
    parent_job_id: str
    expected_task_indexes: tuple[int, ...]
    tasks: tuple[FoldingTaskTerminalEvidence, ...]
    outcome: PhaseActionTerminalOutcome
    observed_at: str

    def __post_init__(self) -> None:
        _match(self.action_id, _ACTION_ID, "folding terminal view action id")
        _match(self.parent_job_id, _JOB_ID, "folding terminal view parent job id")
        _task_indexes(self.expected_task_indexes)
        if not self.tasks:
            raise ValueError("folding terminal view requires at least one task")
        succeeded = all(item.state == "COMPLETED" and item.exit_code in {"0", "0:0"} for item in self.tasks)
        if (self.outcome == "succeeded") != succeeded:
            raise ValueError("folding terminal view requires a coherent terminal outcome")
        _timestamp(self.observed_at)

    @classmethod
    def from_event(cls, event: FoldingActionTerminalObservedEvent) -> FoldingActionTerminalObservationView:
        return cls(
            action_id=event.payload.action_id,
            parent_job_id=event.payload.parent_job_id,
            expected_task_indexes=event.payload.expected_task_indexes,
            tasks=event.payload.tasks,
            outcome=event.payload.outcome,
            observed_at=event.occurred_at,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "action_id": self.action_id,
            "parent_job_id": self.parent_job_id,
            "expected_task_indexes": list(self.expected_task_indexes),
            "tasks": [item.to_mapping() for item in self.tasks],
            "outcome": self.outcome,
            "observed_at": self.observed_at,
        }


def phase_action_terminal_observed_payload_from_mapping(
    payload: Mapping[str, object],
) -> PhaseActionTerminalObservedPayload:
    _strict(
        payload,
        {
            "schema_version",
            "submission_id",
            "phase_runspec_digest",
            "action_id",
            "runtime_action_digest",
            "scheduler_correlation_token",
            "job_id",
            "state",
            "exit_code",
            "outcome",
            "source",
        },
        "PhaseActionTerminalObservedPayload",
    )
    outcome = _string(payload, "outcome")
    source = _string(payload, "source")
    if outcome not in {"succeeded", "failed"}:
        raise ValueError(f"unsupported terminal observation outcome: {outcome!r}")
    if source != "sacct":
        raise ValueError(f"unsupported terminal observation source: {source!r}")
    return PhaseActionTerminalObservedPayload(
        schema_version=_version(payload, "PhaseActionTerminalObservedPayload"),
        submission_id=_string(payload, "submission_id"),
        phase_runspec_digest=_string(payload, "phase_runspec_digest"),
        action_id=_string(payload, "action_id"),
        runtime_action_digest=_string(payload, "runtime_action_digest"),
        scheduler_correlation_token=_string(payload, "scheduler_correlation_token"),
        job_id=_string(payload, "job_id"),
        state=_string(payload, "state"),
        exit_code=_optional_string(payload, "exit_code"),
        outcome=cast("PhaseActionTerminalOutcome", outcome),
        source=cast("PhaseActionTerminalSource", source),
    )


def phase_action_terminal_observed_event_from_mapping(
    payload: Mapping[str, object],
) -> PhaseActionTerminalObservedEvent:
    _strict(
        payload,
        {"schema_version", "sequence", "event_type", "phase_run_id", "attempt_id", "occurred_at", "payload"},
        "PhaseActionTerminalObservedEvent",
    )
    event_type = _string(payload, "event_type")
    if event_type != "phase-action-terminal-observed":
        raise ValueError(f"unsupported terminal observation event type: {event_type!r}")
    nested = payload.get("payload")
    if not isinstance(nested, Mapping):
        raise ValueError("terminal observation payload must be a mapping")
    sequence = payload.get("sequence")
    if not isinstance(sequence, int) or isinstance(sequence, bool):
        raise ValueError("terminal observation event sequence must be an integer")
    return PhaseActionTerminalObservedEvent(
        schema_version=_version(payload, "PhaseActionTerminalObservedEvent"),
        sequence=sequence,
        event_type=cast("PhaseActionTerminalObservedEventType", event_type),
        phase_run_id=_string(payload, "phase_run_id"),
        attempt_id=_string(payload, "attempt_id"),
        occurred_at=_string(payload, "occurred_at"),
        payload=phase_action_terminal_observed_payload_from_mapping(nested),
    )


def folding_task_terminal_evidence_from_mapping(
    payload: Mapping[str, object],
) -> FoldingTaskTerminalEvidence:
    required = {"schema_version", "task_index", "scheduler_job_id", "state", "exit_code", "source"}
    optional = {"restarts"}
    if not required <= set(payload) or not set(payload) <= (required | optional):
        raise ValueError("FoldingTaskTerminalEvidence has missing or extra fields")
    task_index = payload.get("task_index")
    if task_index is not None and (not isinstance(task_index, int) or isinstance(task_index, bool)):
        raise ValueError("folding terminal task index must be integer or null")
    restarts = payload.get("restarts")
    if restarts is not None and (not isinstance(restarts, int) or isinstance(restarts, bool) or restarts < 0):
        raise ValueError("folding terminal task restarts must be a non-negative integer or null")
    return FoldingTaskTerminalEvidence(
        schema_version=_version(payload, "FoldingTaskTerminalEvidence"),
        task_index=task_index,
        scheduler_job_id=_string(payload, "scheduler_job_id"),
        state=_string(payload, "state"),
        exit_code=_string(payload, "exit_code"),
        source=cast("PhaseActionTerminalSource", _string(payload, "source")),
        restarts=restarts,
    )


def folding_action_terminal_observed_payload_from_mapping(
    payload: Mapping[str, object],
) -> FoldingActionTerminalObservedPayload:
    _strict(
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
        "FoldingActionTerminalObservedPayload",
    )
    outcome = _string(payload, "outcome")
    if outcome not in {"succeeded", "failed"}:
        raise ValueError(f"unsupported folding terminal observation outcome: {outcome!r}")
    return FoldingActionTerminalObservedPayload(
        schema_version=_version(payload, "FoldingActionTerminalObservedPayload"),
        submission_id=_string(payload, "submission_id"),
        phase_runspec_digest=_string(payload, "phase_runspec_digest"),
        action_id=_string(payload, "action_id"),
        runtime_action_digest=_string(payload, "runtime_action_digest"),
        parent_job_id=_string(payload, "parent_job_id"),
        expected_task_indexes=_integers(payload, "expected_task_indexes"),
        tasks=tuple(folding_task_terminal_evidence_from_mapping(item) for item in _mapping_list(payload, "tasks")),
        outcome=cast("PhaseActionTerminalOutcome", outcome),
    )


def folding_action_terminal_observed_event_from_mapping(
    payload: Mapping[str, object],
) -> FoldingActionTerminalObservedEvent:
    _strict(
        payload,
        {"schema_version", "sequence", "event_type", "phase_run_id", "attempt_id", "occurred_at", "payload"},
        "FoldingActionTerminalObservedEvent",
    )
    event_type = _string(payload, "event_type")
    if event_type != "phase-action-array-terminal-observed":
        raise ValueError(f"unsupported folding terminal observation event type: {event_type!r}")
    nested = payload.get("payload")
    if not isinstance(nested, Mapping):
        raise ValueError("folding terminal observation payload must be a mapping")
    sequence = payload.get("sequence")
    if not isinstance(sequence, int) or isinstance(sequence, bool):
        raise ValueError("folding terminal observation event sequence must be an integer")
    return FoldingActionTerminalObservedEvent(
        schema_version=_version(payload, "FoldingActionTerminalObservedEvent"),
        sequence=sequence,
        event_type=cast("FoldingActionTerminalObservedEventType", event_type),
        phase_run_id=_string(payload, "phase_run_id"),
        attempt_id=_string(payload, "attempt_id"),
        occurred_at=_string(payload, "occurred_at"),
        payload=folding_action_terminal_observed_payload_from_mapping(nested),
    )


def _strict(payload: Mapping[str, object], allowed: set[str], name: str) -> None:
    unknown = sorted(set(payload) - allowed)
    missing = sorted(allowed - set(payload))
    if unknown or missing:
        raise ValueError(f"{name} fields mismatch; missing={missing}, unknown={unknown}")


def _version(payload: Mapping[str, object], name: str) -> int:
    value = payload.get("schema_version")
    return validate_schema_version(value, record_name=name)


def _schema(value: int, name: str) -> None:
    if validate_schema_version(value, record_name=name) != value:
        raise ValueError(f"{name} schema_version must be explicit")


def _string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_string(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError(f"{key} must be null or a non-empty string")
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


def _task_indexes(values: tuple[int, ...]) -> None:
    if tuple(sorted(set(values))) != values or any(item < 0 for item in values):
        raise ValueError("folding task indexes must be sorted, unique, and non-negative")


def _match(value: str, pattern: re.Pattern[str], name: str) -> None:
    if pattern.fullmatch(value) is None:
        raise ValueError(f"{name} has an invalid format")


def _timestamp(value: str) -> None:
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z", value) is None:
        raise ValueError("terminal observation time must be an explicit UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("terminal observation time must be a valid UTC timestamp") from exc
    if parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError("terminal observation time must be an explicit UTC timestamp")


__all__ = [
    "TERMINAL_PHASE_SLURM_STATES",
    "FoldingActionTerminalObservationView",
    "FoldingActionTerminalObservedEvent",
    "FoldingActionTerminalObservedEventType",
    "FoldingActionTerminalObservedPayload",
    "FoldingTaskTerminalEvidence",
    "PhaseActionTerminalObservationView",
    "PhaseActionTerminalObservedEvent",
    "PhaseActionTerminalObservedPayload",
    "PhaseActionTerminalOutcome",
    "PhaseActionTerminalSource",
    "folding_action_terminal_observed_event_from_mapping",
    "folding_action_terminal_observed_payload_from_mapping",
    "folding_task_terminal_evidence_from_mapping",
    "phase_action_terminal_observed_event_from_mapping",
    "phase_action_terminal_observed_payload_from_mapping",
    "phase_action_terminal_outcome",
]
