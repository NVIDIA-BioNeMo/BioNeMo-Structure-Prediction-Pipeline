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

"""Create or verify durable predecessor action-evidence attestations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime

from bspp.orchestration.contract.phase import canonical_mapping_digest
from bspp.orchestration.contract.phase_action_evidence_attestation import (
    PhaseActionEvidenceAttestedEvent,
    PhaseActionEvidenceAttestedPayload,
    phase_action_evidence_attestation_id,
)
from bspp.orchestration.contract.phase_reconciliation import PhaseActionTerminalObservedEvent
from bspp.orchestration.contract.preprocessing_action import (
    preprocessing_chunk_action_evidence_from_mapping,
)
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION
from bspp.orchestration.control.phase_authority import (
    PhaseAuthorityStore,
    PhaseAuthorityValidation,
)
from bspp.orchestration.control.phase_lifecycle import classify_phase_lifecycle
from bspp.orchestration.control.transport import RemoteSlurmTransport

Clock = Callable[[], datetime]


def ensure_phase_action_evidence_attestation(
    authority: PhaseAuthorityValidation,
    *,
    store: PhaseAuthorityStore,
    transport: RemoteSlurmTransport,
    clock: Clock,
) -> PhaseAuthorityValidation:
    """Attest the exact failed action evidence, or prove an existing attestation unchanged."""
    if classify_phase_lifecycle(authority.lifecycle) not in {"failed", "cancelled"}:
        raise ValueError("action-evidence attestation requires a retryable Phase Attempt")
    submission = authority.submission
    if submission is None or submission.status != "submitted" or len(submission.actions) != 1:
        raise ValueError("action-evidence attestation requires one complete submitted Runtime Action")
    action = submission.actions[0]
    if action.status != "submitted" or action.job_id is None:
        raise ValueError("action-evidence attestation requires a durable Slurm assignment")
    terminal = _unique_terminal_event(authority, submission.submission_id, action.action_id)
    document = transport.read_immutable_text_artifact_no_follow(action.plan.action_evidence_path)
    mapping = _strict_canonical_mapping(document)
    evidence = preprocessing_chunk_action_evidence_from_mapping(mapping)
    document_sha256 = hashlib.sha256(document).hexdigest()
    mapping_digest = canonical_mapping_digest(evidence.to_mapping())

    existing = authority.current_action_evidence_attestation
    if existing is not None:
        if (
            existing.payload.action_evidence_path != action.plan.action_evidence_path
            or existing.payload.evidence_document_sha256 != document_sha256
            or existing.payload.evidence_mapping_digest != mapping_digest
            or existing.payload.evidence != evidence
        ):
            raise ValueError("durably attested action evidence has changed or disappeared")
        return authority

    occurred_at = _format_timestamp(clock())
    terminal_digest = canonical_mapping_digest(terminal.to_mapping())
    identity: dict[str, object] = {
        "schema_version": CURRENT_CONTRACT_SCHEMA_VERSION,
        "phase_runspec_digest": authority.phase_runspec.digest,
        "submission_id": submission.submission_id,
        "action_id": action.action_id,
        "runtime_action_digest": action.plan.runtime_action_digest,
        "scheduler_correlation_token": action.scheduler_correlation_token,
        "job_id": action.job_id,
        "action_evidence_path": action.plan.action_evidence_path,
        "evidence_document_sha256": document_sha256,
        "evidence_mapping_digest": mapping_digest,
        "terminal_event_sequence": terminal.sequence,
        "terminal_event_digest": terminal_digest,
        "terminal_state": terminal.payload.state,
        "terminal_outcome": terminal.payload.outcome,
        "terminal_observed_at": terminal.occurred_at,
        "evidence": evidence.to_mapping(),
    }
    payload = PhaseActionEvidenceAttestedPayload(
        attestation_id=phase_action_evidence_attestation_id(identity),
        phase_runspec_digest=authority.phase_runspec.digest,
        submission_id=submission.submission_id,
        action_id=action.action_id,
        runtime_action_digest=action.plan.runtime_action_digest,
        scheduler_correlation_token=action.scheduler_correlation_token,
        job_id=action.job_id,
        action_evidence_path=action.plan.action_evidence_path,
        evidence_document_sha256=document_sha256,
        evidence_mapping_digest=mapping_digest,
        terminal_event_sequence=terminal.sequence,
        terminal_event_digest=terminal_digest,
        terminal_state=terminal.payload.state,
        terminal_outcome=terminal.payload.outcome,
        terminal_observed_at=terminal.occurred_at,
        evidence=evidence,
    )
    return store.append_event(
        authority.phase_run.phase_run_id,
        lambda sequence: PhaseActionEvidenceAttestedEvent(
            sequence=sequence,
            phase_run_id=authority.phase_run.phase_run_id,
            attempt_id=authority.current_attempt.attempt_id,
            occurred_at=occurred_at,
            payload=payload,
        ),
    )


def _unique_terminal_event(
    authority: PhaseAuthorityValidation,
    submission_id: str,
    action_id: str,
) -> PhaseActionTerminalObservedEvent:
    matches = tuple(
        event
        for event in authority.events
        if isinstance(event, PhaseActionTerminalObservedEvent)
        and event.phase_run_id == authority.phase_run.phase_run_id
        and event.attempt_id == authority.current_attempt.attempt_id
        and event.payload.submission_id == submission_id
        and event.payload.action_id == action_id
    )
    if len(matches) != 1:
        raise ValueError("action-evidence attestation requires exactly one durable terminal event")
    return matches[0]


def _strict_canonical_mapping(document: bytes) -> Mapping[str, object]:
    try:
        payload = json.loads(document)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("action evidence must be canonical UTF-8 JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("action evidence must be one JSON mapping")
    canonical = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    if canonical != document:
        raise ValueError("action evidence document must use canonical JSON bytes")
    return payload


def _format_timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("evidence-attestation clock must return a timezone-aware datetime")
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


__all__ = ["ensure_phase_action_evidence_attestation"]
