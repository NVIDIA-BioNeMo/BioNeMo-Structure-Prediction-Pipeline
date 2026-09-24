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

"""Strict source action-evidence attestation contract tests."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from bspp.orchestration.contract.phase import canonical_mapping_digest
from bspp.orchestration.contract.phase_action_evidence_attestation import (
    PhaseActionEvidenceAttestedEvent,
    PhaseActionEvidenceAttestedPayload,
    phase_action_evidence_attestation_id,
    phase_action_evidence_attested_event_from_mapping,
)
from bspp.orchestration.contract.preprocessing_action import (
    preprocessing_chunk_action_evidence_from_mapping,
)
from tests.support.preprocessing_execution import (
    configure_preprocessing_fakes,
    invoke_preprocessing_execution,
    preprocessing_execution_fixture,
    skip_preprocessing_server_warmup,
)


def _event(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PhaseActionEvidenceAttestedEvent:
    fixture = preprocessing_execution_fixture(tmp_path / "work")
    configure_preprocessing_fakes(fixture, monkeypatch)
    skip_preprocessing_server_warmup(monkeypatch)
    result = invoke_preprocessing_execution(fixture)
    assert result.exit_code == 0, result.output
    evidence = preprocessing_chunk_action_evidence_from_mapping(json.loads(fixture.evidence_path.read_text()))
    finished = datetime.fromisoformat(evidence.finished_at.replace("Z", "+00:00"))
    terminal_at = (finished + timedelta(seconds=1)).astimezone(UTC).isoformat().replace("+00:00", "Z")
    fields: dict[str, object] = {
        "schema_version": 1,
        "phase_runspec_digest": evidence.phase_runspec_digest,
        "submission_id": "phase-submission-" + "1" * 64,
        "action_id": evidence.action_id,
        "runtime_action_digest": "2" * 64,
        "scheduler_correlation_token": "bspp-phase-" + "3" * 64,
        "job_id": "12345",
        "action_evidence_path": str(fixture.evidence_path),
        "evidence_document_sha256": "4" * 64,
        "evidence_mapping_digest": canonical_mapping_digest(evidence.to_mapping()),
        "terminal_event_sequence": 4,
        "terminal_event_digest": "5" * 64,
        "terminal_state": "FAILED",
        "terminal_outcome": "failed",
        "terminal_observed_at": terminal_at,
        "evidence": evidence.to_mapping(),
    }
    payload = PhaseActionEvidenceAttestedPayload(
        attestation_id=phase_action_evidence_attestation_id(fields),
        phase_runspec_digest=evidence.phase_runspec_digest,
        submission_id=str(fields["submission_id"]),
        action_id=evidence.action_id,
        runtime_action_digest="2" * 64,
        scheduler_correlation_token=str(fields["scheduler_correlation_token"]),
        job_id="12345",
        action_evidence_path=str(fixture.evidence_path),
        evidence_document_sha256="4" * 64,
        evidence_mapping_digest=canonical_mapping_digest(evidence.to_mapping()),
        terminal_event_sequence=4,
        terminal_event_digest="5" * 64,
        terminal_state="FAILED",
        terminal_outcome="failed",
        terminal_observed_at=terminal_at,
        evidence=evidence,
    )
    occurred = (finished + timedelta(seconds=2)).astimezone(UTC).isoformat().replace("+00:00", "Z")
    return PhaseActionEvidenceAttestedEvent(
        sequence=5,
        phase_run_id=evidence.phase_run_id,
        attempt_id=evidence.attempt_id,
        occurred_at=occurred,
        payload=payload,
    )


def test_evidence_attestation_strict_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    event = _event(tmp_path, monkeypatch)
    assert phase_action_evidence_attested_event_from_mapping(event.to_mapping()) == event


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: data.update({"unknown": True}),
        lambda data: data["payload"].update({"evidence": None}),
        lambda data: data["payload"].update({"job_id": "job-123"}),
        lambda data: data["payload"].update({"terminal_event_digest": "0" * 64}),
    ],
)
def test_evidence_attestation_rejects_unknown_null_and_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: object,
) -> None:
    mapping = deepcopy(_event(tmp_path, monkeypatch).to_mapping())
    assert callable(mutation)
    mutation(mapping)
    with pytest.raises(ValueError):
        phase_action_evidence_attested_event_from_mapping(mapping)
