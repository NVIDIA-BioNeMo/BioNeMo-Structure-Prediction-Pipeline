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

"""Durable job-bound attestation of one Phase Runtime action evidence file."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePosixPath
from typing import Literal, cast

from bspp.orchestration.contract.preprocessing_action import (
    PreprocessingChunkActionEvidence,
    preprocessing_chunk_action_evidence_from_mapping,
)
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

PhaseActionEvidenceAttestedEventType = Literal["phase-action-evidence-attested"]

_SHA256 = re.compile(r"[0-9a-f]{64}")
_PHASE_RUN_ID = re.compile(r"phase-run-[0-9a-f]{32}")
_ATTEMPT_ID = re.compile(r"attempt-[0-9]{4}")
_ACTION_ID = re.compile(r"preprocessing-chunk-[0-9]{6}")
_SUBMISSION_ID = re.compile(r"phase-submission-[0-9a-f]{64}")
_CORRELATION = re.compile(r"bspp-phase-[0-9a-f]{64}")
_LEGACY_CORRELATION = re.compile(r"afcdb-phase-[0-9a-f]{64}")
_ATTESTATION_ID = re.compile(r"phase-action-evidence-attestation-[0-9a-f]{64}")
_JOB_ID = re.compile(r"[0-9]+")


def _digest(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def phase_action_evidence_attestation_id(identity: Mapping[str, object]) -> str:
    return f"phase-action-evidence-attestation-{_digest(identity)}"


@dataclass(frozen=True)
class PhaseActionEvidenceAttestedPayload:
    attestation_id: str
    phase_runspec_digest: str
    submission_id: str
    action_id: str
    runtime_action_digest: str
    scheduler_correlation_token: str
    job_id: str
    action_evidence_path: str
    evidence_document_sha256: str
    evidence_mapping_digest: str
    terminal_event_sequence: int
    terminal_event_digest: str
    terminal_state: str
    terminal_outcome: str
    terminal_observed_at: str
    evidence: PreprocessingChunkActionEvidence
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        _match(self.attestation_id, _ATTESTATION_ID, "attestation id")
        _sha(self.phase_runspec_digest, "attested RunSpec")
        _match(self.submission_id, _SUBMISSION_ID, "attested submission id")
        _match(self.action_id, _ACTION_ID, "attested action id")
        _sha(self.runtime_action_digest, "attested Runtime Action")
        _match(self.scheduler_correlation_token, _CORRELATION, "attested scheduler correlation")
        _match(self.job_id, _JOB_ID, "attested job id")
        path = PurePosixPath(self.action_evidence_path)
        if not path.is_absolute() or path.suffix != ".json" or ".." in path.parts:
            raise ValueError("attested action evidence path must be an absolute JSON path")
        _sha(self.evidence_document_sha256, "action evidence document")
        _sha(self.evidence_mapping_digest, "action evidence mapping")
        if self.terminal_event_sequence <= 1:
            raise ValueError("attested terminal event sequence must follow materialization")
        _sha(self.terminal_event_digest, "attested terminal event")
        if self.terminal_outcome not in {"succeeded", "failed"} or not self.terminal_state:
            raise ValueError("attested terminal outcome must be conclusive")
        _timestamp(self.terminal_observed_at, "terminal observation")
        if self.evidence.phase_runspec_digest != self.phase_runspec_digest or self.evidence.action_id != self.action_id:
            raise ValueError("attested evidence identity must match its RunSpec and action")
        if self.evidence_mapping_digest != _digest(self.evidence.to_mapping()):
            raise ValueError("attested evidence mapping digest does not match")
        if _parse(self.evidence.finished_at) > _parse(self.terminal_observed_at):
            raise ValueError("action evidence cannot finish after its terminal observation")
        if self.attestation_id != phase_action_evidence_attestation_id(self.identity_mapping()):
            raise ValueError("attestation id does not match canonical identity")

    def identity_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "phase_runspec_digest": self.phase_runspec_digest,
            "submission_id": self.submission_id,
            "action_id": self.action_id,
            "runtime_action_digest": self.runtime_action_digest,
            "scheduler_correlation_token": self.scheduler_correlation_token,
            "job_id": self.job_id,
            "action_evidence_path": self.action_evidence_path,
            "evidence_document_sha256": self.evidence_document_sha256,
            "evidence_mapping_digest": self.evidence_mapping_digest,
            "terminal_event_sequence": self.terminal_event_sequence,
            "terminal_event_digest": self.terminal_event_digest,
            "terminal_state": self.terminal_state,
            "terminal_outcome": self.terminal_outcome,
            "terminal_observed_at": self.terminal_observed_at,
            "evidence": self.evidence.to_mapping(),
        }

    @property
    def digest(self) -> str:
        return _digest(self.to_mapping())

    def to_mapping(self) -> dict[str, object]:
        return {"attestation_id": self.attestation_id, **self.identity_mapping()}


@dataclass(frozen=True)
class PhaseActionEvidenceAttestedEvent:
    sequence: int
    phase_run_id: str
    attempt_id: str
    occurred_at: str
    payload: PhaseActionEvidenceAttestedPayload
    event_type: PhaseActionEvidenceAttestedEventType = "phase-action-evidence-attested"
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        if self.sequence <= self.payload.terminal_event_sequence:
            raise ValueError("evidence attestation must follow its terminal event")
        if self.event_type != "phase-action-evidence-attested":
            raise ValueError("invalid evidence-attestation event discriminator")
        _match(self.phase_run_id, _PHASE_RUN_ID, "attestation Phase Run id")
        _match(self.attempt_id, _ATTEMPT_ID, "attestation Attempt id")
        if (
            self.payload.evidence.phase_run_id != self.phase_run_id
            or self.payload.evidence.attempt_id != self.attempt_id
        ):
            raise ValueError("attestation envelope must match embedded evidence")
        _timestamp(self.occurred_at, "attestation event")
        if _parse(self.occurred_at) < _parse(self.payload.terminal_observed_at):
            raise ValueError("attestation event cannot predate terminal evidence")

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


def phase_action_evidence_attested_payload_from_mapping(
    payload: Mapping[str, object],
) -> PhaseActionEvidenceAttestedPayload:
    allowed = set(PhaseActionEvidenceAttestedPayload.__dataclass_fields__)
    _strict(payload, allowed, "PhaseActionEvidenceAttestedPayload")
    return PhaseActionEvidenceAttestedPayload(
        schema_version=_version(payload, "PhaseActionEvidenceAttestedPayload"),
        attestation_id=_str(payload, "attestation_id"),
        phase_runspec_digest=_str(payload, "phase_runspec_digest"),
        submission_id=_str(payload, "submission_id"),
        action_id=_str(payload, "action_id"),
        runtime_action_digest=_str(payload, "runtime_action_digest"),
        scheduler_correlation_token=_str(payload, "scheduler_correlation_token"),
        job_id=_str(payload, "job_id"),
        action_evidence_path=_str(payload, "action_evidence_path"),
        evidence_document_sha256=_str(payload, "evidence_document_sha256"),
        evidence_mapping_digest=_str(payload, "evidence_mapping_digest"),
        terminal_event_sequence=_int(payload, "terminal_event_sequence"),
        terminal_event_digest=_str(payload, "terminal_event_digest"),
        terminal_state=_str(payload, "terminal_state"),
        terminal_outcome=_str(payload, "terminal_outcome"),
        terminal_observed_at=_str(payload, "terminal_observed_at"),
        evidence=preprocessing_chunk_action_evidence_from_mapping(_mapping(payload, "evidence")),
    )


def phase_action_evidence_attested_event_from_mapping(
    payload: Mapping[str, object],
) -> PhaseActionEvidenceAttestedEvent:
    _strict(
        payload,
        {"schema_version", "sequence", "event_type", "phase_run_id", "attempt_id", "occurred_at", "payload"},
        "PhaseActionEvidenceAttestedEvent",
    )
    event_type = _str(payload, "event_type")
    if event_type != "phase-action-evidence-attested":
        raise ValueError(f"unsupported evidence-attestation event type: {event_type!r}")
    return PhaseActionEvidenceAttestedEvent(
        schema_version=_version(payload, "PhaseActionEvidenceAttestedEvent"),
        sequence=_int(payload, "sequence"),
        event_type=cast("PhaseActionEvidenceAttestedEventType", event_type),
        phase_run_id=_str(payload, "phase_run_id"),
        attempt_id=_str(payload, "attempt_id"),
        occurred_at=_str(payload, "occurred_at"),
        payload=phase_action_evidence_attested_payload_from_mapping(_mapping(payload, "payload")),
    )


def _schema(value: int, name: str) -> None:
    if validate_schema_version(value, record_name=name) != value:
        raise ValueError(f"{name} schema_version must be explicit")


def _version(payload: Mapping[str, object], name: str) -> int:
    return validate_schema_version(payload.get("schema_version"), record_name=name)


def _strict(payload: Mapping[str, object], allowed: set[str], name: str) -> None:
    missing = sorted(allowed - set(payload))
    unknown = sorted(set(payload) - allowed)
    if missing or unknown:
        raise ValueError(f"invalid {name} fields; missing={missing}, unknown={unknown}")


def _str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _match(value: str, pattern: re.Pattern[str], label: str) -> None:
    if pattern.fullmatch(value) is None:
        raise ValueError(f"invalid {label}")


def _sha(value: str, label: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase SHA-256")


def _parse(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include timezone")
    return parsed


def _timestamp(value: str, label: str) -> None:
    try:
        _parse(value)
    except ValueError as exc:
        raise ValueError(f"invalid {label} timestamp") from exc


__all__ = [
    "PhaseActionEvidenceAttestedEvent",
    "PhaseActionEvidenceAttestedEventType",
    "PhaseActionEvidenceAttestedPayload",
    "phase_action_evidence_attestation_id",
    "phase_action_evidence_attested_event_from_mapping",
    "phase_action_evidence_attested_payload_from_mapping",
]
