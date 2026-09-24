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

"""Initial durable state and replay contracts for phase materialization."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, cast

from bspp.orchestration.contract.phase import (
    FoldingPhaseRunSpec,
    PhaseKind,
    PhaseRunSpec,
    phase_runspec_family_from_mapping,
)
from bspp.orchestration.contract.preprocessing_execution import validate_scientific_schema_version
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

PhaseAttemptStatus = Literal["materialized"]
PhaseRunStatus = Literal["materialized"]
PhaseEventType = Literal["phase-materialized"]
PhaseAttemptLifecycleStatus = Literal["materialized", "failed", "cancelling", "cancelled", "succeeded"]
PhaseRunLifecycleStatus = Literal["materialized", "failed", "cancelling", "cancelled", "accepted"]

_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class PhaseAttempt:
    """One immutable, scheduler-free materialized Phase Attempt."""

    attempt_id: str
    ordinal: int
    phase_runspec_location: str
    phase_runspec_digest: str
    created_at: str
    status: PhaseAttemptStatus = "materialized"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "PhaseAttempt")
        expected_attempt_id = f"attempt-{self.ordinal:04d}"
        if not 1 <= self.ordinal <= 9999 or self.attempt_id != expected_attempt_id:
            msg = "Phase Attempt id must match its ordinal in the range 1..9999"
            raise ValueError(msg)
        expected_location = f"attempts/{self.attempt_id}/phase-runspec.json"
        if self.phase_runspec_location != expected_location:
            msg = "Phase Attempt RunSpec location must be authority-relative and attempt-bound"
            raise ValueError(msg)
        if _SHA256.fullmatch(self.phase_runspec_digest) is None:
            msg = "Phase Attempt RunSpec digest must be a lowercase SHA-256"
            raise ValueError(msg)
        _validate_timestamp(self.created_at, "Phase Attempt created_at")
        if self.status != "materialized":
            msg = "Phase Attempt status must be 'materialized'"
            raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "attempt_id": self.attempt_id,
            "ordinal": self.ordinal,
            "phase_runspec_location": self.phase_runspec_location,
            "phase_runspec_digest": self.phase_runspec_digest,
            "created_at": self.created_at,
            "status": self.status,
        }


@dataclass(frozen=True)
class PhaseRun:
    """Initial durable Phase Run state reconstructed from lifecycle events."""

    phase_run_id: str
    phase_plan_location: str
    phase_plan_digest: str
    created_at: str
    current_attempt_id: str
    attempts: tuple[PhaseAttempt, ...]
    phase_kind: PhaseKind = "preprocessing"
    status: PhaseRunStatus = "materialized"
    sealed: bool = False
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "PhaseRun")
        if re.fullmatch(r"phase-run-[0-9a-f]{32}", self.phase_run_id) is None:
            msg = "Phase Run id must be an opaque phase-run id"
            raise ValueError(msg)
        if self.phase_kind not in {"preprocessing", "folding"}:
            msg = "initial Phase Run phase_kind must be 'preprocessing' or 'folding'"
            raise ValueError(msg)
        if self.phase_plan_location != "phase-plan.json":
            msg = "Phase Run Phase Plan location must be authority-relative"
            raise ValueError(msg)
        if _SHA256.fullmatch(self.phase_plan_digest) is None:
            msg = "Phase Run Phase Plan digest must be a lowercase SHA-256"
            raise ValueError(msg)
        _validate_timestamp(self.created_at, "Phase Run created_at")
        if (
            len(self.attempts) != 1
            or self.attempts[0].attempt_id != "attempt-0001"
            or self.attempts[0].ordinal != 1
            or self.current_attempt_id != "attempt-0001"
        ):
            msg = "initial Phase Run must reference exactly its first immutable attempt"
            raise ValueError(msg)
        if self.status != "materialized" or self.sealed:
            msg = "initial Phase Run must be unsealed and materialized"
            raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "phase_kind": self.phase_kind,
            "phase_run_id": self.phase_run_id,
            "phase_plan_location": self.phase_plan_location,
            "phase_plan_digest": self.phase_plan_digest,
            "created_at": self.created_at,
            "current_attempt_id": self.current_attempt_id,
            "attempts": [attempt.to_mapping() for attempt in self.attempts],
            "status": self.status,
            "sealed": self.sealed,
        }


@dataclass(frozen=True)
class PhaseRunLifecycleView:
    """Current state derived by replay without mutating materialization records."""

    phase_run_id: str
    current_attempt_id: str
    attempt_status: PhaseAttemptLifecycleStatus
    run_status: PhaseRunLifecycleStatus
    sealed: bool
    phase_receipt_id: str | None = None

    def __post_init__(self) -> None:
        if re.fullmatch(r"phase-run-[0-9a-f]{32}", self.phase_run_id) is None:
            raise ValueError("Phase lifecycle view has an invalid Phase Run id")
        if re.fullmatch(r"attempt-[0-9]{4}", self.current_attempt_id) is None:
            raise ValueError("Phase lifecycle view has an invalid current Attempt id")
        if self.attempt_status not in {"materialized", "failed", "cancelling", "cancelled", "succeeded"}:
            raise ValueError(f"unsupported Phase Attempt lifecycle status: {self.attempt_status!r}")
        if self.run_status not in {"materialized", "failed", "cancelling", "cancelled", "accepted"}:
            raise ValueError(f"unsupported Phase Run lifecycle status: {self.run_status!r}")
        active = (self.attempt_status, self.run_status, self.sealed, self.phase_receipt_id) == (
            "materialized",
            "materialized",
            False,
            None,
        )
        failed = (self.attempt_status, self.run_status, self.sealed, self.phase_receipt_id) == (
            "failed",
            "failed",
            False,
            None,
        )
        cancelling = (self.attempt_status, self.run_status, self.sealed, self.phase_receipt_id) == (
            "cancelling",
            "cancelling",
            False,
            None,
        )
        cancelled = (self.attempt_status, self.run_status, self.sealed, self.phase_receipt_id) == (
            "cancelled",
            "cancelled",
            False,
            None,
        )
        accepted = (
            self.attempt_status == "succeeded"
            and self.run_status == "accepted"
            and self.sealed
            and self.phase_receipt_id is not None
        )
        if not (active or failed or cancelling or cancelled or accepted):
            raise ValueError("Phase lifecycle state must be exactly active, failed, cancelling, cancelled, or accepted")
        if (
            self.phase_receipt_id is not None
            and re.fullmatch(r"phase-receipt-[0-9a-f]{64}", self.phase_receipt_id) is None
        ):
            raise ValueError("Phase lifecycle view has an invalid receipt id")


@dataclass(frozen=True)
class PhaseMaterializedPayload:
    """Complete event payload needed to replay the initial run and RunSpec."""

    phase_run: PhaseRun
    phase_runspec: PhaseRunSpec | FoldingPhaseRunSpec
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "PhaseMaterializedPayload")
        attempt = self.phase_run.attempts[0]
        if self.phase_runspec.phase_run_id != self.phase_run.phase_run_id:
            msg = "materialization payload RunSpec must reference its Phase Run"
            raise ValueError(msg)
        if self.phase_runspec.attempt_id != attempt.attempt_id:
            msg = "materialization payload RunSpec must reference its Phase Attempt"
            raise ValueError(msg)
        if self.phase_runspec.phase_plan_digest != self.phase_run.phase_plan_digest:
            msg = "materialization payload plan digests must match"
            raise ValueError(msg)
        if self.phase_runspec.digest != attempt.phase_runspec_digest:
            msg = "materialization payload RunSpec digest must match the Phase Attempt"
            raise ValueError(msg)
        if (
            self.phase_runspec.materialized_at != self.phase_run.created_at
            or attempt.created_at != self.phase_run.created_at
        ):
            msg = "materialization payload timestamps must describe one atomic transition"
            raise ValueError(msg)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "phase_run": self.phase_run.to_mapping(),
            "phase_runspec": self.phase_runspec.to_mapping(),
        }


@dataclass(frozen=True)
class PhaseMaterializedEvent:
    """The first immutable event in a Phase Run authority."""

    phase_run_id: str
    attempt_id: str
    occurred_at: str
    payload: PhaseMaterializedPayload
    sequence: int = 1
    event_type: PhaseEventType = "phase-materialized"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "PhaseMaterializedEvent")
        if self.sequence != 1 or self.event_type != "phase-materialized":
            msg = "initial Phase Event must be sequence 1 named 'phase-materialized'"
            raise ValueError(msg)
        if self.phase_run_id != self.payload.phase_run.phase_run_id:
            msg = "Phase Event run id must match its replay payload"
            raise ValueError(msg)
        if self.attempt_id != self.payload.phase_runspec.attempt_id:
            msg = "Phase Event attempt id must match its replay payload"
            raise ValueError(msg)
        if self.occurred_at != self.payload.phase_run.created_at:
            msg = "Phase Event timestamp must match its replay payload"
            raise ValueError(msg)

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


def phase_attempt_from_mapping(payload: Mapping[str, object]) -> PhaseAttempt:
    _require_explicit_schema_versions(payload, path=("phase_attempt",))
    _reject_unknown_fields(
        payload,
        {
            "schema_version",
            "attempt_id",
            "ordinal",
            "phase_runspec_location",
            "phase_runspec_digest",
            "created_at",
            "status",
        },
        "PhaseAttempt",
    )
    status = _required_str(payload, "status")
    if status != "materialized":
        msg = f"unsupported Phase Attempt status: {status!r}"
        raise ValueError(msg)
    return PhaseAttempt(
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="PhaseAttempt"),
        attempt_id=_required_str(payload, "attempt_id"),
        ordinal=_required_int(payload, "ordinal"),
        phase_runspec_location=_required_str(payload, "phase_runspec_location"),
        phase_runspec_digest=_required_str(payload, "phase_runspec_digest"),
        created_at=_required_str(payload, "created_at"),
        status=cast("PhaseAttemptStatus", status),
    )


def phase_run_from_mapping(payload: Mapping[str, object]) -> PhaseRun:
    _require_explicit_schema_versions(payload, path=("phase_run",))
    _reject_unknown_fields(
        payload,
        {
            "schema_version",
            "phase_kind",
            "phase_run_id",
            "phase_plan_location",
            "phase_plan_digest",
            "created_at",
            "current_attempt_id",
            "attempts",
            "status",
            "sealed",
        },
        "PhaseRun",
    )
    phase_kind = _required_str(payload, "phase_kind")
    status = _required_str(payload, "status")
    if phase_kind not in {"preprocessing", "folding"} or status != "materialized":
        msg = "unsupported initial Phase Run discriminator or status"
        raise ValueError(msg)
    return PhaseRun(
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="PhaseRun"),
        phase_kind=cast("PhaseKind", phase_kind),
        phase_run_id=_required_str(payload, "phase_run_id"),
        phase_plan_location=_required_str(payload, "phase_plan_location"),
        phase_plan_digest=_required_str(payload, "phase_plan_digest"),
        created_at=_required_str(payload, "created_at"),
        current_attempt_id=_required_str(payload, "current_attempt_id"),
        attempts=tuple(phase_attempt_from_mapping(item) for item in _required_mapping_sequence(payload, "attempts")),
        status=cast("PhaseRunStatus", status),
        sealed=_required_bool(payload, "sealed"),
    )


def phase_materialized_payload_from_mapping(payload: Mapping[str, object]) -> PhaseMaterializedPayload:
    _reject_unknown_fields(payload, {"schema_version", "phase_run", "phase_runspec"}, "PhaseMaterializedPayload")
    runspec = phase_runspec_family_from_mapping(_required_mapping(payload, "phase_runspec"))
    if not isinstance(runspec, PhaseRunSpec | FoldingPhaseRunSpec):
        raise ValueError("materialization payload RunSpec must be a preprocessing or folding Phase RunSpec")
    return PhaseMaterializedPayload(
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="PhaseMaterializedPayload"),
        phase_run=phase_run_from_mapping(_required_mapping(payload, "phase_run")),
        phase_runspec=runspec,
    )


def phase_materialized_event_from_mapping(payload: Mapping[str, object]) -> PhaseMaterializedEvent:
    """Strict-load and fully replay-validate the initial materialization event."""
    _require_explicit_schema_versions(payload, path=("phase_event",))
    _reject_unknown_fields(
        payload,
        {
            "schema_version",
            "sequence",
            "event_type",
            "phase_run_id",
            "attempt_id",
            "occurred_at",
            "payload",
        },
        "PhaseMaterializedEvent",
    )
    event_type = _required_str(payload, "event_type")
    if event_type != "phase-materialized":
        msg = f"unsupported initial Phase Event type: {event_type!r}"
        raise ValueError(msg)
    return PhaseMaterializedEvent(
        schema_version=validate_schema_version(payload.get("schema_version"), record_name="PhaseMaterializedEvent"),
        sequence=_required_int(payload, "sequence"),
        event_type=cast("PhaseEventType", event_type),
        phase_run_id=_required_str(payload, "phase_run_id"),
        attempt_id=_required_str(payload, "attempt_id"),
        occurred_at=_required_str(payload, "occurred_at"),
        payload=phase_materialized_payload_from_mapping(_required_mapping(payload, "payload")),
    )


def _require_explicit_schema_versions(value: object, *, path: tuple[str, ...]) -> None:
    if isinstance(value, Mapping):
        if path[-1] in {"database_set", "source_manifest"}:
            return
        location = ".".join(path)
        if "schema_version" not in value:
            msg = f"missing explicit schema_version at {location}"
            raise ValueError(msg)
        if path[-1] == "scientific":
            validate_scientific_schema_version(value["schema_version"], record_name=location)
        else:
            validate_schema_version(value["schema_version"], record_name=location)
        for key, nested in value.items():
            if key != "schema_version":
                _require_explicit_schema_versions(nested, path=(*path, str(key)))
    elif isinstance(value, list | tuple):
        for index, nested in enumerate(value):
            _require_explicit_schema_versions(nested, path=(*path, str(index)))


def _validate_timestamp(value: str, record_name: str) -> None:
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z", value) is None:
        msg = f"{record_name} must be an explicit UTC timestamp"
        raise ValueError(msg)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        msg = f"{record_name} must be a valid UTC timestamp"
        raise ValueError(msg) from exc
    if parsed.utcoffset() != UTC.utcoffset(parsed):
        msg = f"{record_name} must be an explicit UTC timestamp"
        raise ValueError(msg)


def _validate_direct_schema_version(schema_version: int, record_name: str) -> None:
    validated = validate_schema_version(schema_version, record_name=record_name)
    if validated != schema_version:
        msg = f"{record_name} schema_version must be declared explicitly"
        raise ValueError(msg)


def _reject_unknown_fields(payload: Mapping[str, object], allowed: set[str], record_name: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        msg = f"Unknown {record_name} field(s): {', '.join(unknown)}"
        raise ValueError(msg)


def _required_mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        msg = f"{key} must be a mapping"
        raise ValueError(msg)
    return value


def _required_mapping_sequence(payload: Mapping[str, object], key: str) -> tuple[Mapping[str, object], ...]:
    value = payload.get(key)
    if not isinstance(value, list | tuple):
        msg = f"{key} must be a list"
        raise ValueError(msg)
    mappings: list[Mapping[str, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            msg = f"{key}[{index}] must be a mapping"
            raise ValueError(msg)
        mappings.append(item)
    return tuple(mappings)


def _required_str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        msg = f"{key} must be a non-empty string"
        raise ValueError(msg)
    return value


def _required_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"{key} must be an integer"
        raise ValueError(msg)
    return value


def _required_bool(payload: Mapping[str, object], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        msg = f"{key} must be a boolean"
        raise ValueError(msg)
    return value


__all__ = [
    "PhaseAttempt",
    "PhaseAttemptLifecycleStatus",
    "PhaseAttemptStatus",
    "PhaseEventType",
    "PhaseMaterializedEvent",
    "PhaseMaterializedPayload",
    "PhaseRun",
    "PhaseRunLifecycleStatus",
    "PhaseRunLifecycleView",
    "PhaseRunStatus",
    "phase_attempt_from_mapping",
    "phase_materialized_event_from_mapping",
    "phase_run_from_mapping",
]
