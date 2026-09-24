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

"""Durable intent, scheduler effects, and completion for Phase cancellation."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, Protocol, cast

from bspp.orchestration.contract.phase import canonical_mapping_digest
from bspp.orchestration.contract.phase_submission import phase_action_scheduler_correlation_token
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

PhaseCancellationInitialSubmissionStatus = Literal["not-submitted", "planned", "dispatching", "submitted", "satisfied"]
PhaseCancellationActionDisposition = Literal[
    "no-job",
    "correlating",
    "cancel-pending",
    "cancel-requesting",
    "cancel-requested",
    "terminal-confirmed",
]
PhaseCancellationStatus = Literal["cancelling", "cancelled"]

_PHASE_RUN_ID = re.compile(r"phase-run-[0-9a-f]{32}")
_ATTEMPT_ID = re.compile(r"attempt-[0-9]{4}")
_SUBMISSION_ID = re.compile(r"phase-submission-[0-9a-f]{64}")
_CANCELLATION_ID = re.compile(r"phase-cancellation-[0-9a-f]{64}")
_ACTION_ID = re.compile(r"(?:preprocessing-chunk|msa-flatten|split|preprocess|fold|canonical-pair)-[0-9]{6}")
_CORRELATION_TOKEN = re.compile(r"bspp-phase-[0-9a-f]{64}")
_LEGACY_CORRELATION_TOKEN = re.compile(r"afcdb-phase-[0-9a-f]{64}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_JOB_ID = re.compile(r"[0-9]+")


@dataclass(frozen=True)
class PhaseCancellationActionTarget:
    """The frozen submission state of one Runtime Action at cancellation intent."""

    action_id: str
    initial_submission_status: PhaseCancellationInitialSubmissionStatus
    scheduler_correlation_token: str | None = None
    job_id: str | None = None
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        _match(self.action_id, _ACTION_ID, "cancellation target action id")
        if self.initial_submission_status not in {"not-submitted", "planned", "dispatching", "submitted", "satisfied"}:
            raise ValueError(f"unsupported cancellation target submission status: {self.initial_submission_status!r}")
        if self.initial_submission_status == "not-submitted":
            if self.scheduler_correlation_token is not None or self.job_id is not None:
                raise ValueError("not-submitted cancellation target cannot have scheduler identity")
        else:
            if self.scheduler_correlation_token is None:
                raise ValueError("submitted cancellation target state requires a correlation token")
            _match(self.scheduler_correlation_token, _CORRELATION_TOKEN, "cancellation target correlation token")
            if (self.initial_submission_status == "submitted") != (self.job_id is not None):
                raise ValueError("only a submitted cancellation target may have a job id")
        if self.job_id is not None:
            _match(self.job_id, _JOB_ID, "cancellation target job id")

    def identity_mapping(self) -> dict[str, object]:
        return {
            "action_id": self.action_id,
            "initial_submission_status": self.initial_submission_status,
            "scheduler_correlation_token": self.scheduler_correlation_token,
            "job_id": self.job_id,
        }

    def to_mapping(self) -> dict[str, object]:
        return {"schema_version": self.schema_version, **self.identity_mapping()}


def phase_cancellation_id(
    *,
    phase_run_id: str,
    attempt_id: str,
    submission_id: str | None,
    phase_runspec_digest: str,
    targets: tuple[PhaseCancellationActionTarget, ...],
) -> str:
    """Derive the stable identity of an exact cancellation target set."""
    digest = canonical_mapping_digest(
        {
            "phase_run_id": phase_run_id,
            "attempt_id": attempt_id,
            "submission_id": submission_id,
            "phase_runspec_digest": phase_runspec_digest,
            "targets": [target.identity_mapping() for target in targets],
        }
    )
    return f"phase-cancellation-{digest}"


@dataclass(frozen=True)
class PhaseCancellationIntendedPayload:
    cancellation_id: str
    submission_id: str | None
    phase_runspec_digest: str
    targets: tuple[PhaseCancellationActionTarget, ...]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        _match(self.cancellation_id, _CANCELLATION_ID, "Phase Cancellation id")
        if self.submission_id is not None:
            _match(self.submission_id, _SUBMISSION_ID, "Phase Cancellation submission id")
        _match(self.phase_runspec_digest, _SHA256, "Phase Cancellation RunSpec digest")
        if not isinstance(self.targets, tuple) or not self.targets:
            raise ValueError("Phase Cancellation intent requires a non-empty target tuple")
        if any(not isinstance(target, PhaseCancellationActionTarget) for target in self.targets):
            raise ValueError("Phase Cancellation targets must be strict target records")
        action_ids = tuple(target.action_id for target in self.targets)
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("Phase Cancellation target action ids must be unique")
        job_ids = tuple(target.job_id for target in self.targets if target.job_id is not None)
        if len(job_ids) != len(set(job_ids)):
            raise ValueError("Phase Cancellation target job ids must be unique")
        if self.submission_id is None and any(
            target.initial_submission_status != "not-submitted" for target in self.targets
        ):
            raise ValueError("Phase Cancellation without submission authority may target only not-submitted actions")
        if self.submission_id is not None and any(
            target.initial_submission_status == "not-submitted" for target in self.targets
        ):
            raise ValueError("Phase Cancellation with submission authority must freeze every submission action")
        if self.submission_id is not None:
            for target in self.targets:
                expected = phase_action_scheduler_correlation_token(self.submission_id, target.action_id)
                if target.scheduler_correlation_token != expected:
                    raise ValueError("Phase Cancellation target correlation token does not match its submission")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "cancellation_id": self.cancellation_id,
            "submission_id": self.submission_id,
            "phase_runspec_digest": self.phase_runspec_digest,
            "targets": [target.to_mapping() for target in self.targets],
        }


@dataclass(frozen=True)
class PhaseCancellationIntendedEvent:
    sequence: int
    phase_run_id: str
    attempt_id: str
    occurred_at: str
    payload: PhaseCancellationIntendedPayload
    event_type: Literal["phase-cancellation-intended"] = "phase-cancellation-intended"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _event(self, "phase-cancellation-intended")
        if not isinstance(self.payload, PhaseCancellationIntendedPayload):
            raise ValueError("Phase Cancellation intent event requires its strict payload")
        expected = phase_cancellation_id(
            phase_run_id=self.phase_run_id,
            attempt_id=self.attempt_id,
            submission_id=self.payload.submission_id,
            phase_runspec_digest=self.payload.phase_runspec_digest,
            targets=self.payload.targets,
        )
        if self.payload.cancellation_id != expected:
            raise ValueError("Phase Cancellation id does not match its exact frozen targets")

    def to_mapping(self) -> dict[str, object]:
        return _event_mapping(self, self.payload.to_mapping())


@dataclass(frozen=True)
class PhaseJobCancellationRequestIntendedPayload:
    cancellation_id: str
    action_id: str
    job_id: str
    request_ordinal: int
    scancel_argv: tuple[str, ...]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _request_identity(self)

    def to_mapping(self) -> dict[str, object]:
        return _request_mapping(self)


@dataclass(frozen=True)
class PhaseJobCancellationRequestIntendedEvent:
    sequence: int
    phase_run_id: str
    attempt_id: str
    occurred_at: str
    payload: PhaseJobCancellationRequestIntendedPayload
    event_type: Literal["phase-job-cancellation-request-intended"] = "phase-job-cancellation-request-intended"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _event(self, "phase-job-cancellation-request-intended")
        if not isinstance(self.payload, PhaseJobCancellationRequestIntendedPayload):
            raise ValueError("cancellation request intent event requires its strict payload")

    def to_mapping(self) -> dict[str, object]:
        return _event_mapping(self, self.payload.to_mapping())


@dataclass(frozen=True)
class PhaseJobCancellationRequestResultPayload:
    cancellation_id: str
    action_id: str
    job_id: str
    request_ordinal: int
    scancel_argv: tuple[str, ...]
    return_code: int
    stdout: str
    stderr: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _request_identity(self)
        if not isinstance(self.return_code, int) or isinstance(self.return_code, bool):
            raise ValueError("scancel result return code must be an integer")
        if not isinstance(self.stdout, str) or not isinstance(self.stderr, str):
            raise ValueError("scancel result stdout and stderr must be strings")

    def to_mapping(self) -> dict[str, object]:
        return {
            **_request_mapping(self),
            "return_code": self.return_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


@dataclass(frozen=True)
class PhaseJobCancellationRequestResultEvent:
    sequence: int
    phase_run_id: str
    attempt_id: str
    occurred_at: str
    payload: PhaseJobCancellationRequestResultPayload
    event_type: Literal["phase-job-cancellation-request-result"] = "phase-job-cancellation-request-result"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _event(self, "phase-job-cancellation-request-result")
        if not isinstance(self.payload, PhaseJobCancellationRequestResultPayload):
            raise ValueError("cancellation request result event requires its strict payload")

    def to_mapping(self) -> dict[str, object]:
        return _event_mapping(self, self.payload.to_mapping())


@dataclass(frozen=True)
class PhaseCancellationTerminalReference:
    action_id: str
    job_id: str
    terminal_observation_digest: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        _match(self.action_id, _ACTION_ID, "cancellation completion action id")
        _match(self.job_id, _JOB_ID, "cancellation completion job id")
        _match(self.terminal_observation_digest, _SHA256, "cancellation terminal observation digest")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "action_id": self.action_id,
            "job_id": self.job_id,
            "terminal_observation_digest": self.terminal_observation_digest,
        }


@dataclass(frozen=True)
class PhaseCancellationCompletedPayload:
    cancellation_id: str
    terminal_references: tuple[PhaseCancellationTerminalReference, ...]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        _match(self.cancellation_id, _CANCELLATION_ID, "Phase Cancellation id")
        if not isinstance(self.terminal_references, tuple) or any(
            not isinstance(item, PhaseCancellationTerminalReference) for item in self.terminal_references
        ):
            raise ValueError("Phase Cancellation completion requires strict terminal references")
        action_ids = tuple(item.action_id for item in self.terminal_references)
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("Phase Cancellation completion action references must be unique")
        job_ids = tuple(item.job_id for item in self.terminal_references)
        if len(job_ids) != len(set(job_ids)):
            raise ValueError("Phase Cancellation completion job references must be unique")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "cancellation_id": self.cancellation_id,
            "terminal_references": [item.to_mapping() for item in self.terminal_references],
        }


@dataclass(frozen=True)
class PhaseCancellationCompletedEvent:
    sequence: int
    phase_run_id: str
    attempt_id: str
    occurred_at: str
    payload: PhaseCancellationCompletedPayload
    event_type: Literal["phase-cancellation-completed"] = "phase-cancellation-completed"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _event(self, "phase-cancellation-completed")
        if not isinstance(self.payload, PhaseCancellationCompletedPayload):
            raise ValueError("Phase Cancellation completion event requires its strict payload")

    def to_mapping(self) -> dict[str, object]:
        return _event_mapping(self, self.payload.to_mapping())


@dataclass(frozen=True)
class PhaseJobCancellationRequestView:
    request_ordinal: int
    scancel_argv: tuple[str, ...]
    return_code: int | None = None
    stdout: str | None = None
    stderr: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.request_ordinal, int)
            or isinstance(self.request_ordinal, bool)
            or self.request_ordinal <= 0
        ):
            raise ValueError("cancellation request view ordinal must be positive")
        _argv(self.scancel_argv)
        absent = self.return_code is None and self.stdout is None and self.stderr is None
        complete = (
            isinstance(self.return_code, int)
            and not isinstance(self.return_code, bool)
            and self.stdout is not None
            and self.stderr is not None
        )
        if not (absent or complete):
            raise ValueError("cancellation request view result must be entirely absent or complete")

    def to_mapping(self) -> dict[str, object]:
        return {
            "request_ordinal": self.request_ordinal,
            "scancel_argv": list(self.scancel_argv),
            "return_code": self.return_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


@dataclass(frozen=True)
class PhaseCancellationActionView:
    target: PhaseCancellationActionTarget
    disposition: PhaseCancellationActionDisposition
    bound_job_id: str | None = None
    requests: tuple[PhaseJobCancellationRequestView, ...] = ()
    terminal_observation_digest: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.target, PhaseCancellationActionTarget):
            raise ValueError("cancellation action view requires its strict frozen target")
        if self.disposition not in {
            "no-job",
            "correlating",
            "cancel-pending",
            "cancel-requesting",
            "cancel-requested",
            "terminal-confirmed",
        }:
            raise ValueError(f"unsupported cancellation action disposition: {self.disposition!r}")
        if self.bound_job_id is not None:
            _match(self.bound_job_id, _JOB_ID, "cancellation action bound job id")
        if self.target.initial_submission_status in {"not-submitted", "planned", "satisfied"} and (
            self.disposition != "no-job"
        ):
            raise ValueError("job-free cancellation target must retain the no-job disposition")
        if self.target.initial_submission_status == "dispatching" and (
            (self.disposition == "correlating") != (self.bound_job_id is None)
        ):
            raise ValueError("dispatching cancellation target must correlate before gaining a bound job")
        if self.target.initial_submission_status == "submitted" and (
            self.disposition in {"no-job", "correlating"} or self.bound_job_id != self.target.job_id
        ):
            raise ValueError("submitted cancellation target must retain its exact frozen job binding")
        if not isinstance(self.requests, tuple) or any(
            not isinstance(item, PhaseJobCancellationRequestView) for item in self.requests
        ):
            raise ValueError("cancellation action requests must be strict request views")
        if tuple(item.request_ordinal for item in self.requests) != tuple(range(1, len(self.requests) + 1)):
            raise ValueError("cancellation request view ordinals must be contiguous from one")
        unresolved = tuple(index for index, request in enumerate(self.requests) if request.return_code is None)
        if len(unresolved) > 1 or (unresolved and unresolved[0] != len(self.requests) - 1):
            raise ValueError("only the last cancellation request may have an unresolved result")
        if self.disposition == "no-job":
            if self.bound_job_id is not None or self.requests or self.terminal_observation_digest is not None:
                raise ValueError("no-job cancellation action cannot contain scheduler evidence")
        elif self.disposition == "correlating":
            if self.bound_job_id is not None or self.requests or self.terminal_observation_digest is not None:
                raise ValueError("correlating cancellation action cannot contain a bound job")
        else:
            if self.bound_job_id is None:
                raise ValueError("job-bound cancellation disposition requires a job id")
            if self.disposition == "cancel-pending" and self.requests and self.requests[-1].return_code in {None, 0}:
                raise ValueError("cancel-pending action must end with a nonzero request result")
            if self.disposition == "cancel-requesting" and (
                not self.requests or self.requests[-1].return_code is not None
            ):
                raise ValueError("cancel-requesting action requires an unresolved last request")
            if self.disposition == "cancel-requested" and (not self.requests or self.requests[-1].return_code != 0):
                raise ValueError("cancel-requested action requires an acknowledged last request")
            if self.disposition == "terminal-confirmed":
                if self.terminal_observation_digest is None:
                    raise ValueError("terminal-confirmed action requires an observation digest")
                _match(self.terminal_observation_digest, _SHA256, "terminal observation digest")
            elif self.terminal_observation_digest is not None:
                raise ValueError("only terminal-confirmed action may contain an observation digest")

    @property
    def action_id(self) -> str:
        return self.target.action_id

    def to_mapping(self) -> dict[str, object]:
        return {
            "target": self.target.to_mapping(),
            "disposition": self.disposition,
            "bound_job_id": self.bound_job_id,
            "requests": [item.to_mapping() for item in self.requests],
            "terminal_observation_digest": self.terminal_observation_digest,
        }


@dataclass(frozen=True)
class PhaseCancellationLifecycleView:
    phase_run_id: str
    attempt_id: str
    cancellation_id: str
    submission_id: str | None
    phase_runspec_digest: str
    actions: tuple[PhaseCancellationActionView, ...]
    status: PhaseCancellationStatus

    def __post_init__(self) -> None:
        _match(self.phase_run_id, _PHASE_RUN_ID, "cancellation lifecycle Phase Run id")
        _match(self.attempt_id, _ATTEMPT_ID, "cancellation lifecycle Attempt id")
        _match(self.cancellation_id, _CANCELLATION_ID, "cancellation lifecycle id")
        if self.submission_id is not None:
            _match(self.submission_id, _SUBMISSION_ID, "cancellation lifecycle submission id")
        _match(self.phase_runspec_digest, _SHA256, "cancellation lifecycle RunSpec digest")
        if (
            not isinstance(self.actions, tuple)
            or not self.actions
            or any(not isinstance(item, PhaseCancellationActionView) for item in self.actions)
        ):
            raise ValueError("cancellation lifecycle requires strict action views")
        action_ids = tuple(item.action_id for item in self.actions)
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("cancellation lifecycle action ids must be unique")
        job_ids = tuple(item.bound_job_id for item in self.actions if item.bound_job_id is not None)
        if len(job_ids) != len(set(job_ids)):
            raise ValueError("cancellation lifecycle bound job ids must be unique")
        targets = tuple(item.target for item in self.actions)
        if self.submission_id is None:
            if any(target.initial_submission_status != "not-submitted" for target in targets):
                raise ValueError("cancellation lifecycle without submission may contain only not-submitted targets")
        else:
            for target in targets:
                if target.initial_submission_status == "not-submitted":
                    raise ValueError("cancellation lifecycle submission cannot contain not-submitted targets")
                expected_token = phase_action_scheduler_correlation_token(self.submission_id, target.action_id)
                if target.scheduler_correlation_token != expected_token:
                    raise ValueError("cancellation lifecycle target token does not match its submission")
        expected_cancellation_id = phase_cancellation_id(
            phase_run_id=self.phase_run_id,
            attempt_id=self.attempt_id,
            submission_id=self.submission_id,
            phase_runspec_digest=self.phase_runspec_digest,
            targets=targets,
        )
        if self.cancellation_id != expected_cancellation_id:
            raise ValueError("cancellation lifecycle id does not match its exact frozen targets")
        if self.status not in {"cancelling", "cancelled"}:
            raise ValueError(f"unsupported cancellation lifecycle status: {self.status!r}")
        if self.status == "cancelled" and not all(
            item.disposition in {"no-job", "terminal-confirmed"} for item in self.actions
        ):
            raise ValueError("cancelled lifecycle requires every action to be terminal or job-free")

    def to_mapping(self) -> dict[str, object]:
        target_job_ids = tuple(item.bound_job_id for item in self.actions if item.bound_job_id is not None)
        terminal_job_ids = tuple(
            item.bound_job_id
            for item in self.actions
            if item.bound_job_id is not None and item.disposition == "terminal-confirmed"
        )
        pending_job_ids = tuple(
            item.bound_job_id
            for item in self.actions
            if item.bound_job_id is not None and item.disposition != "terminal-confirmed"
        )
        acknowledged_job_ids = tuple(
            item.bound_job_id
            for item in self.actions
            if item.bound_job_id is not None and any(request.return_code == 0 for request in item.requests)
        )
        return {
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
            "cancellation_id": self.cancellation_id,
            "submission_id": self.submission_id,
            "phase_runspec_digest": self.phase_runspec_digest,
            "status": self.status,
            "target_job_ids": list(target_job_ids),
            "terminal_job_ids": list(terminal_job_ids),
            "pending_job_ids": list(pending_job_ids),
            "request_acknowledged_job_ids": list(acknowledged_job_ids),
            "unresolved_action_ids": [item.action_id for item in self.actions if item.disposition == "correlating"],
            "actions": [item.to_mapping() for item in self.actions],
        }


def phase_cancellation_intended_event_from_mapping(payload: Mapping[str, object]) -> PhaseCancellationIntendedEvent:
    sequence, phase_run_id, attempt_id, occurred_at, schema_version = _event_values(
        payload, "PhaseCancellationIntendedEvent", "phase-cancellation-intended"
    )
    return PhaseCancellationIntendedEvent(
        sequence=sequence,
        phase_run_id=phase_run_id,
        attempt_id=attempt_id,
        occurred_at=occurred_at,
        schema_version=schema_version,
        payload=_cancellation_intended_payload(_mapping(payload, "payload")),
    )


def phase_job_cancellation_request_intended_event_from_mapping(
    payload: Mapping[str, object],
) -> PhaseJobCancellationRequestIntendedEvent:
    sequence, phase_run_id, attempt_id, occurred_at, schema_version = _event_values(
        payload,
        "PhaseJobCancellationRequestIntendedEvent",
        "phase-job-cancellation-request-intended",
    )
    return PhaseJobCancellationRequestIntendedEvent(
        sequence=sequence,
        phase_run_id=phase_run_id,
        attempt_id=attempt_id,
        occurred_at=occurred_at,
        schema_version=schema_version,
        payload=_request_intended_payload(_mapping(payload, "payload")),
    )


def phase_job_cancellation_request_result_event_from_mapping(
    payload: Mapping[str, object],
) -> PhaseJobCancellationRequestResultEvent:
    sequence, phase_run_id, attempt_id, occurred_at, schema_version = _event_values(
        payload,
        "PhaseJobCancellationRequestResultEvent",
        "phase-job-cancellation-request-result",
    )
    return PhaseJobCancellationRequestResultEvent(
        sequence=sequence,
        phase_run_id=phase_run_id,
        attempt_id=attempt_id,
        occurred_at=occurred_at,
        schema_version=schema_version,
        payload=_request_result_payload(_mapping(payload, "payload")),
    )


def phase_cancellation_completed_event_from_mapping(payload: Mapping[str, object]) -> PhaseCancellationCompletedEvent:
    sequence, phase_run_id, attempt_id, occurred_at, schema_version = _event_values(
        payload, "PhaseCancellationCompletedEvent", "phase-cancellation-completed"
    )
    return PhaseCancellationCompletedEvent(
        sequence=sequence,
        phase_run_id=phase_run_id,
        attempt_id=attempt_id,
        occurred_at=occurred_at,
        schema_version=schema_version,
        payload=_cancellation_completed_payload(_mapping(payload, "payload")),
    )


def _target(payload: Mapping[str, object]) -> PhaseCancellationActionTarget:
    _strict(
        payload,
        {"schema_version", "action_id", "initial_submission_status", "scheduler_correlation_token", "job_id"},
        "PhaseCancellationActionTarget",
    )
    status = _string(payload, "initial_submission_status")
    if status not in {"not-submitted", "planned", "dispatching", "submitted", "satisfied"}:
        raise ValueError(f"unsupported cancellation target submission status: {status!r}")
    return PhaseCancellationActionTarget(
        schema_version=_version(payload, "PhaseCancellationActionTarget"),
        action_id=_string(payload, "action_id"),
        initial_submission_status=cast("PhaseCancellationInitialSubmissionStatus", status),
        scheduler_correlation_token=_optional_string(payload, "scheduler_correlation_token"),
        job_id=_optional_string(payload, "job_id"),
    )


def _cancellation_intended_payload(payload: Mapping[str, object]) -> PhaseCancellationIntendedPayload:
    _strict(
        payload,
        {"schema_version", "cancellation_id", "submission_id", "phase_runspec_digest", "targets"},
        "PhaseCancellationIntendedPayload",
    )
    return PhaseCancellationIntendedPayload(
        schema_version=_version(payload, "PhaseCancellationIntendedPayload"),
        cancellation_id=_string(payload, "cancellation_id"),
        submission_id=_optional_string(payload, "submission_id"),
        phase_runspec_digest=_string(payload, "phase_runspec_digest"),
        targets=tuple(_target(item) for item in _mapping_sequence(payload, "targets")),
    )


def _request_intended_payload(payload: Mapping[str, object]) -> PhaseJobCancellationRequestIntendedPayload:
    _strict(
        payload,
        {"schema_version", "cancellation_id", "action_id", "job_id", "request_ordinal", "scancel_argv"},
        "PhaseJobCancellationRequestIntendedPayload",
    )
    return PhaseJobCancellationRequestIntendedPayload(
        schema_version=_version(payload, "PhaseJobCancellationRequestIntendedPayload"),
        cancellation_id=_string(payload, "cancellation_id"),
        action_id=_string(payload, "action_id"),
        job_id=_string(payload, "job_id"),
        request_ordinal=_integer(payload, "request_ordinal"),
        scancel_argv=_string_sequence(payload, "scancel_argv"),
    )


def _request_result_payload(payload: Mapping[str, object]) -> PhaseJobCancellationRequestResultPayload:
    _strict(
        payload,
        {
            "schema_version",
            "cancellation_id",
            "action_id",
            "job_id",
            "request_ordinal",
            "scancel_argv",
            "return_code",
            "stdout",
            "stderr",
        },
        "PhaseJobCancellationRequestResultPayload",
    )
    return PhaseJobCancellationRequestResultPayload(
        schema_version=_version(payload, "PhaseJobCancellationRequestResultPayload"),
        cancellation_id=_string(payload, "cancellation_id"),
        action_id=_string(payload, "action_id"),
        job_id=_string(payload, "job_id"),
        request_ordinal=_integer(payload, "request_ordinal"),
        scancel_argv=_string_sequence(payload, "scancel_argv"),
        return_code=_integer(payload, "return_code"),
        stdout=_string(payload, "stdout", allow_empty=True),
        stderr=_string(payload, "stderr", allow_empty=True),
    )


def _terminal_reference(payload: Mapping[str, object]) -> PhaseCancellationTerminalReference:
    _strict(
        payload,
        {"schema_version", "action_id", "job_id", "terminal_observation_digest"},
        "PhaseCancellationTerminalReference",
    )
    return PhaseCancellationTerminalReference(
        schema_version=_version(payload, "PhaseCancellationTerminalReference"),
        action_id=_string(payload, "action_id"),
        job_id=_string(payload, "job_id"),
        terminal_observation_digest=_string(payload, "terminal_observation_digest"),
    )


def _cancellation_completed_payload(payload: Mapping[str, object]) -> PhaseCancellationCompletedPayload:
    _strict(payload, {"schema_version", "cancellation_id", "terminal_references"}, "PhaseCancellationCompletedPayload")
    return PhaseCancellationCompletedPayload(
        schema_version=_version(payload, "PhaseCancellationCompletedPayload"),
        cancellation_id=_string(payload, "cancellation_id"),
        terminal_references=tuple(
            _terminal_reference(item) for item in _mapping_sequence(payload, "terminal_references")
        ),
    )


def _event_values(payload: Mapping[str, object], name: str, expected: str) -> tuple[int, str, str, str, int]:
    _strict(
        payload,
        {"schema_version", "sequence", "event_type", "phase_run_id", "attempt_id", "occurred_at", "payload"},
        name,
    )
    actual = _string(payload, "event_type")
    if actual != expected:
        raise ValueError(f"unsupported {name} event type: {actual!r}")
    return (
        _integer(payload, "sequence"),
        _string(payload, "phase_run_id"),
        _string(payload, "attempt_id"),
        _string(payload, "occurred_at"),
        _version(payload, name),
    )


class _EventLike(Protocol):
    @property
    def schema_version(self) -> int: ...

    @property
    def sequence(self) -> int: ...

    @property
    def event_type(self) -> str: ...

    @property
    def phase_run_id(self) -> str: ...

    @property
    def attempt_id(self) -> str: ...

    @property
    def occurred_at(self) -> str: ...


class _RequestLike(Protocol):
    @property
    def schema_version(self) -> int: ...

    @property
    def cancellation_id(self) -> str: ...

    @property
    def action_id(self) -> str: ...

    @property
    def job_id(self) -> str: ...

    @property
    def request_ordinal(self) -> int: ...

    @property
    def scancel_argv(self) -> tuple[str, ...]: ...


def _event(value: _EventLike, expected: str) -> None:
    _schema(value.schema_version, type(value).__name__)
    sequence = value.sequence
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence <= 1:
        raise ValueError("Phase Cancellation event sequence must follow materialization")
    if value.event_type != expected:
        raise ValueError(f"Phase Cancellation event must be {expected!r}")
    _match(value.phase_run_id, _PHASE_RUN_ID, "Phase Cancellation event Phase Run id")
    _match(value.attempt_id, _ATTEMPT_ID, "Phase Cancellation event Attempt id")
    _timestamp(value.occurred_at)


def _event_mapping(value: _EventLike, payload: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": value.schema_version,
        "sequence": value.sequence,
        "event_type": value.event_type,
        "phase_run_id": value.phase_run_id,
        "attempt_id": value.attempt_id,
        "occurred_at": value.occurred_at,
        "payload": dict(payload),
    }


def _request_identity(value: _RequestLike) -> None:
    _schema(value.schema_version, type(value).__name__)
    _match(value.cancellation_id, _CANCELLATION_ID, "Phase Cancellation id")
    _match(value.action_id, _ACTION_ID, "cancellation request action id")
    _match(value.job_id, _JOB_ID, "cancellation request job id")
    ordinal = value.request_ordinal
    if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal <= 0:
        raise ValueError("cancellation request ordinal must be positive")
    _argv(value.scancel_argv)


def _request_mapping(value: _RequestLike) -> dict[str, object]:
    return {
        "schema_version": value.schema_version,
        "cancellation_id": value.cancellation_id,
        "action_id": value.action_id,
        "job_id": value.job_id,
        "request_ordinal": value.request_ordinal,
        "scancel_argv": list(value.scancel_argv),
    }


def _schema(version: object, name: str) -> None:
    if not isinstance(version, int) or isinstance(version, bool):
        raise ValueError(f"{name} schema_version must be an integer")
    validate_schema_version(version, record_name=name)


def _match(value: object, pattern: re.Pattern[str], label: str) -> None:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{label} is invalid")


def _timestamp(value: object) -> None:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("Phase Cancellation timestamp must be canonical UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError("Phase Cancellation timestamp must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.astimezone(UTC).utcoffset() is None:
        raise ValueError("Phase Cancellation timestamp must be timezone-aware")


def _argv(value: object) -> None:
    if not isinstance(value, tuple) or not value or any(not isinstance(item, str) or not item for item in value):
        raise ValueError("cancellation command argv must be a non-empty string tuple")


def _strict(payload: Mapping[str, object], fields: set[str], name: str) -> None:
    missing = sorted(fields - set(payload))
    unknown = sorted(set(payload) - fields)
    if missing or unknown:
        raise ValueError(f"{name} fields mismatch; missing={missing}, unknown={unknown}")


def _version(payload: Mapping[str, object], name: str) -> int:
    value = payload.get("schema_version")
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} schema_version must be an integer")
    return value


def _string(payload: Mapping[str, object], key: str, *, allow_empty: bool = False) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValueError(f"{key} must be a string")
    return value


def _optional_string(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError(f"{key} must be null or a non-empty string")
    return value


def _integer(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _mapping_sequence(payload: Mapping[str, object], key: str) -> tuple[Mapping[str, object], ...]:
    value = payload.get(key)
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or any(not isinstance(item, Mapping) for item in value)
    ):
        raise ValueError(f"{key} must be a sequence of mappings")
    return tuple(cast("Mapping[str, object]", item) for item in value)


def _string_sequence(payload: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or any(not isinstance(item, str) for item in value)
    ):
        raise ValueError(f"{key} must be a sequence of strings")
    return tuple(cast("str", item) for item in value)


__all__ = [
    "PhaseCancellationActionDisposition",
    "PhaseCancellationActionTarget",
    "PhaseCancellationActionView",
    "PhaseCancellationCompletedEvent",
    "PhaseCancellationCompletedPayload",
    "PhaseCancellationIntendedEvent",
    "PhaseCancellationIntendedPayload",
    "PhaseCancellationLifecycleView",
    "PhaseCancellationTerminalReference",
    "PhaseJobCancellationRequestIntendedEvent",
    "PhaseJobCancellationRequestIntendedPayload",
    "PhaseJobCancellationRequestResultEvent",
    "PhaseJobCancellationRequestResultPayload",
    "PhaseJobCancellationRequestView",
    "phase_cancellation_completed_event_from_mapping",
    "phase_cancellation_id",
    "phase_cancellation_intended_event_from_mapping",
    "phase_job_cancellation_request_intended_event_from_mapping",
    "phase_job_cancellation_request_result_event_from_mapping",
]
