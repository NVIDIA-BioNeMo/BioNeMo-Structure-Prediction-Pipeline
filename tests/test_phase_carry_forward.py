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

"""Control derivation, typed replay, and crash-safe carry projection tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from bspp.orchestration.contract.database_direct_result import database_direct_result_digest
from bspp.orchestration.contract.database_placement import DatabaseSetSelection
from bspp.orchestration.contract.phase import (
    PhasePlan,
    PreprocessingPhasePlanPayload,
    canonical_mapping_digest,
    phase_runspec_from_mapping,
)
from bspp.orchestration.contract.phase_action_evidence_attestation import (
    PhaseActionEvidenceAttestedEvent,
)
from bspp.orchestration.contract.phase_carry_forward import (
    AttemptCarryForwardAdoptedContent,
    AttemptCarryForwardAdoptionEvidence,
    AttemptCarryForwardRequest,
    AttemptCarryForwardSelection,
    attempt_carry_forward_id,
    attempt_carry_forward_record_from_mapping,
)
from bspp.orchestration.contract.phase_retry import (
    PhaseAttemptRetriedEvent,
    PhaseAttemptRetriedPayload,
    phase_input_set_identity_digest,
    phase_retry_id,
    phase_scientific_identity_digest,
)
from bspp.orchestration.contract.phase_state import (
    PhaseAttempt,
    PhaseMaterializedEvent,
    PhaseMaterializedPayload,
    PhaseRun,
)
from bspp.orchestration.contract.preprocessing_action import (
    PreprocessingChunkActionEvidence,
    PreprocessingDatabasePlacementEvidence,
)
from bspp.orchestration.contract.preprocessing_execution import (
    preprocessing_chunk_execution_intent_from_plan,
)
from bspp.orchestration.control.phase_action_evidence_attestation import (
    ensure_phase_action_evidence_attestation,
)
from bspp.orchestration.control.phase_authority import PhaseAuthorityStore
from bspp.orchestration.control.phase_carry_forward import derive_attempt_carry_forward
from bspp.orchestration.control.phase_finalization import _carry_forward_closure
from bspp.orchestration.control.phase_rendering import render_phase_submission_intent
from bspp.orchestration.control.phase_retry import retry_phase
from bspp.orchestration.control.phase_status import status_phase
from bspp.orchestration.control.transport import RemoteSlurmTransport
from tests.support.preprocessing_execution import (
    LocalExecutionFixture,
    configure_preprocessing_fakes,
    invoke_preprocessing_execution,
    preprocessing_execution_fixture,
    skip_preprocessing_server_warmup,
)
from tests.test_phase_resume import _record_submission_state, _record_terminal


def _failed_attested_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[PhaseAuthorityStore, str]:
    fixture = preprocessing_execution_fixture(tmp_path / "work")
    site = fixture.action.payload.site.model_copy(
        update={"container_mounts": (f"{tmp_path / 'work'}:{tmp_path / 'work'}",)},
    )
    action = replace(fixture.action, payload=replace(fixture.action.payload, site=site))
    plan = PhasePlan(
        target_cluster=fixture.runspec.cluster.profile_name,
        input_location=fixture.runspec.input_location,
        payload=PreprocessingPhasePlanPayload(
            work_plan=fixture.runspec.payload.work_plan,
            chunk_execution_intent=preprocessing_chunk_execution_intent_from_plan(action.payload),
            database=DatabaseSetSelection(
                database_set=fixture.runspec.payload.database.database_set,
                requested_policy=fixture.runspec.payload.database.requested_policy,
            ),
        ),
    )
    runspec = replace(
        fixture.runspec,
        phase_plan_digest=plan.digest,
        payload=replace(fixture.runspec.payload, actions=(action,)),
    )
    attempt = PhaseAttempt(
        attempt_id=runspec.attempt_id,
        ordinal=1,
        phase_runspec_location="attempts/attempt-0001/phase-runspec.json",
        phase_runspec_digest=runspec.digest,
        created_at=runspec.materialized_at,
    )
    phase_run = PhaseRun(
        phase_run_id=runspec.phase_run_id,
        phase_plan_location="phase-plan.json",
        phase_plan_digest=plan.digest,
        created_at=runspec.materialized_at,
        current_attempt_id=attempt.attempt_id,
        attempts=(attempt,),
    )
    materialized = PhaseMaterializedEvent(
        phase_run_id=runspec.phase_run_id,
        attempt_id=runspec.attempt_id,
        occurred_at=runspec.materialized_at,
        payload=PhaseMaterializedPayload(phase_run=phase_run, phase_runspec=runspec),
    )
    authority_root = tmp_path / "authority"
    store = PhaseAuthorityStore(authority_root)
    authority = store.publish(
        phase_plan=plan,
        phase_run=phase_run,
        phase_runspec=runspec,
        materialized_event=materialized,
    )
    action_evidence_path = (
        Path(runspec.cluster.output_root)
        / "bspp-phase-runs"
        / runspec.phase_run_id
        / runspec.attempt_id
        / action.action_id
        / "action-evidence.json"
    )
    runtime_fixture = LocalExecutionFixture(
        runspec=runspec,
        runspec_path=authority.authority_path / attempt.phase_runspec_location,
        evidence_path=action_evidence_path,
        invocation_path=fixture.invocation_path,
        database_placement_result_path=fixture.database_placement_result_path,
        database_placement_failure_path=fixture.database_placement_failure_path,
        database_source_root=fixture.database_source_root,
        database_placement_paths=fixture.database_placement_paths,
    )
    configure_preprocessing_fakes(runtime_fixture, monkeypatch)
    skip_preprocessing_server_warmup(monkeypatch)
    executed = invoke_preprocessing_execution(runtime_fixture)
    assert executed.exit_code == 0, executed.output
    _record_submission_state(authority_root, runspec.phase_run_id, "submitted")
    _record_terminal(
        authority_root,
        runspec.phase_run_id,
        state="FAILED",
        exit_code="1:0",
        outcome="failed",
    )
    authority = store.validate(runspec.phase_run_id)
    attested = ensure_phase_action_evidence_attestation(
        authority,
        store=store,
        transport=RemoteSlurmTransport(kind="local-slurm", ssh_target=None),
        clock=lambda: datetime(2026, 8, 20, 13, 0, tzinfo=UTC),
    )
    assert attested.current_action_evidence_attestation is not None
    return store, runspec.phase_run_id


def _carried_retry_payload(
    store: PhaseAuthorityStore,
    phase_run_id: str,
    *,
    selected_names: tuple[str, ...] = ("AFDB_zeta.a3m",),
) -> PhaseAttemptRetriedPayload:
    authority = store.validate(phase_run_id)
    successor_ordinal = authority.current_attempt.ordinal + 1
    successor_attempt_id = f"attempt-{successor_ordinal:04d}"
    materialized_at = (
        "2026-08-20T13:01:00.000000Z"
        if successor_ordinal == 2
        else f"2026-08-{18 + successor_ordinal:02d}T13:01:00.000000Z"
    )
    successor_runspec = replace(
        authority.phase_runspec,
        attempt_id=successor_attempt_id,
        materialized_at=materialized_at,
        carry_forward=None,
        payload=replace(
            authority.phase_runspec.payload,
            database=replace(
                authority.phase_runspec.payload.database,
                source_manifest_projection=f"attempts/{successor_attempt_id}/database-source-manifest.json",
            ),
        ),
    )
    evidence = authority.current_action_evidence_attestation
    assert evidence is not None
    by_member = {item.member_name: item for item in evidence.payload.evidence.output_hashes}
    request = AttemptCarryForwardRequest(
        source_attempt_id=authority.current_attempt.attempt_id,
        content=tuple(
            AttemptCarryForwardSelection(
                member_name=member_name,
                size_bytes=selected.size_bytes,
                sha256=selected.sha256,
            )
            for member_name in selected_names
            for selected in (by_member[member_name],)
        ),
    )
    record, successor_runspec = derive_attempt_carry_forward(
        authority=authority,
        successor_runspec=successor_runspec,
        target_attempt_ordinal=successor_ordinal,
        request=request,
        declared_at=successor_runspec.materialized_at,
    )
    successor = PhaseAttempt(
        attempt_id=successor_attempt_id,
        ordinal=successor_ordinal,
        phase_runspec_location=f"attempts/{successor_attempt_id}/phase-runspec.json",
        phase_runspec_digest=successor_runspec.digest,
        created_at=successor_runspec.materialized_at,
    )
    input_digest = phase_input_set_identity_digest(authority.phase_plan)
    scientific_digest = phase_scientific_identity_digest(authority.phase_plan)
    return PhaseAttemptRetriedPayload(
        retry_id=phase_retry_id(
            phase_run_id=phase_run_id,
            predecessor_attempt_id=authority.current_attempt.attempt_id,
            successor_attempt_id=successor_attempt_id,
            predecessor_outcome="failed",
            predecessor_phase_runspec_digest=authority.phase_runspec.digest,
            phase_plan_digest=authority.phase_plan.digest,
            input_set_identity_digest=input_digest,
            scientific_identity_digest=scientific_digest,
            selected_cluster_profile=successor_runspec.cluster.profile_name,
            successor_phase_runspec_digest=successor_runspec.digest,
        ),
        predecessor_attempt_id=authority.current_attempt.attempt_id,
        predecessor_phase_runspec_digest=authority.phase_runspec.digest,
        predecessor_outcome="failed",
        phase_plan_digest=authority.phase_plan.digest,
        input_set_identity_digest=input_digest,
        scientific_identity_digest=scientific_digest,
        selected_cluster_profile=successor_runspec.cluster.profile_name,
        successor_attempt=successor,
        successor_phase_runspec=successor_runspec,
        carry_forward_record=record,
    )


def _append_carried_retry(
    store: PhaseAuthorityStore,
    phase_run_id: str,
    payload: PhaseAttemptRetriedPayload,
) -> None:
    store.append_attempt_retried_event(
        phase_run_id,
        lambda sequence: PhaseAttemptRetriedEvent(
            sequence=sequence,
            phase_run_id=phase_run_id,
            attempt_id=payload.successor_attempt.attempt_id,
            occurred_at=payload.successor_phase_runspec.materialized_at,
            payload=payload,
        ),
    )


def _rewrite_carried_retry_with_self_consistent_tamper(
    store: PhaseAuthorityStore,
    phase_run_id: str,
    tamper: str,
) -> None:
    authority = store.validate(phase_run_id)
    event_path = next((authority.authority_path / "events").glob("*-phase-attempt-retried.json"))
    mapping: dict[str, Any] = json.loads(event_path.read_text())
    payload = mapping["payload"]
    record = payload["carry_forward_record"]
    if tamper == "attempt-ordinals":
        record["source_attempt_ordinal"] = 8
        record["target_attempt_ordinal"] = 9
    elif tamper == "remaining-search-sha":
        record["remaining_search_input_sha256"] = "f" * 64
    elif tamper == "workspace-root":
        record["workspace"]["roots"][2]["physical_root"] += "-tampered"
    elif tamper == "private-source":
        record["content"][0]["source_private_mount_path"] = (
            "/run/bspp-carry/tampered/" + record["content"][0]["member_name"]
        )
    elif tamper == "target-action-path":
        record["content"][0]["target_declared_path"] = "/tampered/" + record["content"][0]["member_name"]
    elif tamper == "declared-at":
        record["declared_at"] = "2026-08-20T13:00:59.000000Z"
    else:
        raise AssertionError(f"unsupported tamper: {tamper}")
    record["content_digest"] = canonical_mapping_digest({"schema_version": 1, "content": record["content"]})
    identity = {key: value for key, value in record.items() if key != "attempt_carry_forward_id"}
    record["attempt_carry_forward_id"] = attempt_carry_forward_id(identity)
    typed_record = attempt_carry_forward_record_from_mapping(record)
    successor_mapping = payload["successor_phase_runspec"]
    successor_mapping["carry_forward"]["attempt_carry_forward_id"] = typed_record.attempt_carry_forward_id
    successor_mapping["carry_forward"]["digest"] = typed_record.digest
    successor = phase_runspec_from_mapping(successor_mapping)
    payload["successor_attempt"]["phase_runspec_digest"] = successor.digest
    payload["retry_id"] = phase_retry_id(
        phase_run_id=mapping["phase_run_id"],
        predecessor_attempt_id=payload["predecessor_attempt_id"],
        successor_attempt_id=payload["successor_attempt"]["attempt_id"],
        predecessor_outcome=payload["predecessor_outcome"],
        predecessor_phase_runspec_digest=payload["predecessor_phase_runspec_digest"],
        phase_plan_digest=payload["phase_plan_digest"],
        input_set_identity_digest=payload["input_set_identity_digest"],
        scientific_identity_digest=payload["scientific_identity_digest"],
        selected_cluster_profile=payload["selected_cluster_profile"],
        successor_phase_runspec_digest=successor.digest,
    )
    event_path.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n")


@pytest.mark.parametrize(
    "tamper",
    [
        "attempt-ordinals",
        "remaining-search-sha",
        "workspace-root",
        "private-source",
        "target-action-path",
        "declared-at",
    ],
)
def test_replay_rejects_self_consistently_rehashed_carry_authority_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    store, phase_run_id = _failed_attested_authority(tmp_path, monkeypatch)
    _append_carried_retry(store, phase_run_id, _carried_retry_payload(store, phase_run_id))
    _rewrite_carried_retry_with_self_consistent_tamper(store, phase_run_id, tamper)

    with pytest.raises(ValueError, match="carry-forward"):
        store.validate(phase_run_id)


def _adoption_for_record(
    store: PhaseAuthorityStore,
    phase_run_id: str,
    *,
    submission_id: str,
    content_sha256: str | None = None,
) -> AttemptCarryForwardAdoptionEvidence:
    authority = store.validate(phase_run_id)
    record = authority.current_carry_forward
    assert record is not None
    return AttemptCarryForwardAdoptionEvidence(
        phase_run_id=phase_run_id,
        attempt_id=authority.current_attempt.attempt_id,
        phase_runspec_digest=authority.phase_runspec.digest,
        phase_submission_id=submission_id,
        action_id=authority.phase_runspec.payload.actions[0].action_id,
        attempt_carry_forward_id=record.attempt_carry_forward_id,
        attempt_carry_forward_digest=record.digest,
        remaining_search_input_sha256=record.remaining_search_input_sha256,
        content=tuple(
            AttemptCarryForwardAdoptedContent(
                member_name=item.member_name,
                source_path=item.source_private_mount_path,
                target_path=item.target_declared_path,
                size_bytes=item.size_bytes,
                sha256=content_sha256 or item.sha256,
            )
            for item in record.content
        ),
        adopted_at="2026-08-21T12:02:00.000000Z",
    )


def _rebind_action_evidence_identity(
    evidence: PreprocessingChunkActionEvidence,
    *,
    attempt_id: str,
    phase_runspec_digest: str,
    carry_forward_adoption: AttemptCarryForwardAdoptionEvidence | None,
) -> PreprocessingChunkActionEvidence:
    placement = evidence.database_placement
    assert isinstance(placement, PreprocessingDatabasePlacementEvidence)
    if placement.result is not None:
        result = replace(
            placement.result,
            phase_run_id=evidence.phase_run_id,
            attempt_id=attempt_id,
            phase_runspec_digest=phase_runspec_digest,
            action_id=evidence.action_id,
        )
        placement = replace(
            placement,
            result=result,
            result_digest=database_direct_result_digest(result),
        )
    else:
        assert placement.failure is not None
        placement = replace(
            placement,
            failure=replace(
                placement.failure,
                phase_run_id=evidence.phase_run_id,
                attempt_id=attempt_id,
                phase_runspec_digest=phase_runspec_digest,
                action_id=evidence.action_id,
            ),
        )
    return replace(
        evidence,
        attempt_id=attempt_id,
        phase_runspec_digest=phase_runspec_digest,
        database_placement=placement,
        carry_forward_adoption=carry_forward_adoption,
    )


def _attest_carried_attempt_failure(
    store: PhaseAuthorityStore,
    phase_run_id: str,
    *,
    adoption_mode: str = "valid",
) -> None:
    _record_submission_state(
        store.authority_root,
        phase_run_id,
        "submitted",
        job_id="223456",
        intended_at="2026-08-21T12:00:00.000000Z",
    )
    authority = store.validate(phase_run_id)
    assert authority.submission is not None
    source_attestation = next(
        event for event in authority.events if isinstance(event, PhaseActionEvidenceAttestedEvent)
    )
    submission_id = authority.submission.submission_id
    drift_sha256 = hashlib.sha256(b">AFDB_zeta\nBBBB\n").hexdigest()
    adoption = None
    if adoption_mode != "missing":
        adoption = _adoption_for_record(
            store,
            phase_run_id,
            submission_id=("phase-submission-" + "0" * 64 if adoption_mode == "wrong-submission" else submission_id),
            content_sha256=drift_sha256 if adoption_mode == "hash-drift" else None,
        )
        if adoption_mode == "wrong-record":
            adoption = replace(adoption, attempt_carry_forward_digest="f" * 64)
    output_hashes = source_attestation.payload.evidence.output_hashes
    if adoption_mode == "hash-drift":
        output_hashes = tuple(
            replace(item, sha256=drift_sha256) if item.member_name == "AFDB_zeta.a3m" else item
            for item in output_hashes
        )
    evidence = replace(
        _rebind_action_evidence_identity(
            source_attestation.payload.evidence,
            attempt_id=authority.current_attempt.attempt_id,
            phase_runspec_digest=authority.phase_runspec.digest,
            carry_forward_adoption=adoption,
        ),
        output_hashes=output_hashes,
    )
    evidence_path = Path(authority.submission.actions[0].plan.action_evidence_path)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(evidence.to_mapping(), indent=2, sort_keys=True) + "\n")
    _record_terminal(
        store.authority_root,
        phase_run_id,
        state="FAILED",
        exit_code="1:0",
        outcome="failed",
        observed_at="2026-08-21T12:03:00.000000Z",
    )
    attested = ensure_phase_action_evidence_attestation(
        store.validate(phase_run_id),
        store=store,
        transport=RemoteSlurmTransport(kind="local-slurm", ssh_target=None),
        clock=lambda: datetime(2026, 8, 21, 13, 0, tzinfo=UTC),
    )
    assert attested.current_action_evidence_attestation is not None


def test_attestation_is_reused_only_after_exact_remote_reread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, phase_run_id = _failed_attested_authority(tmp_path, monkeypatch)
    authority = store.validate(phase_run_id)
    event_count = len(authority.events)
    reused = ensure_phase_action_evidence_attestation(
        authority,
        store=store,
        transport=RemoteSlurmTransport(kind="local-slurm", ssh_target=None),
        clock=lambda: datetime(2026, 8, 20, 13, 1, tzinfo=UTC),
    )
    assert len(reused.events) == event_count
    assert reused.current_action_evidence_attestation == authority.current_action_evidence_attestation

    attestation = authority.current_action_evidence_attestation
    assert attestation is not None
    evidence_path = Path(attestation.payload.action_evidence_path)
    evidence_path.chmod(0o644)
    evidence_path.write_bytes(evidence_path.read_bytes() + b" ")
    evidence_path.chmod(0o444)
    with pytest.raises(ValueError, match=r"canonical JSON|changed"):
        ensure_phase_action_evidence_attestation(
            authority,
            store=store,
            transport=RemoteSlurmTransport(kind="local-slurm", ssh_target=None),
            clock=lambda: datetime(2026, 8, 20, 13, 2, tzinfo=UTC),
        )


def test_carried_retry_projects_record_before_runspec_and_recovers_from_record_only_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, phase_run_id = _failed_attested_authority(tmp_path, monkeypatch)
    payload = _carried_retry_payload(store, phase_run_id)

    def stop_after_record(_path: Path) -> None:
        raise OSError("stop after carry record")

    interrupted_store = PhaseAuthorityStore(
        store.authority_root,
        after_retry_carry_forward_link=stop_after_record,
    )
    with pytest.raises(OSError, match="stop after carry record"):
        interrupted_store.append_attempt_retried_event(
            phase_run_id,
            lambda sequence: PhaseAttemptRetriedEvent(
                sequence=sequence,
                phase_run_id=phase_run_id,
                attempt_id="attempt-0002",
                occurred_at="2026-08-20T13:01:00.000000Z",
                payload=payload,
            ),
        )
    incomplete = store.validate(phase_run_id)
    assert not incomplete.current_runspec_projection_complete
    carry_path = incomplete.authority_path / "attempts/attempt-0002/attempt-carry-forward.json"
    runspec_path = incomplete.authority_path / "attempts/attempt-0002/phase-runspec.json"
    assert (
        carry_path.read_bytes()
        == (json.dumps(payload.carry_forward_record.to_mapping(), indent=2, sort_keys=True) + "\n").encode()
    )
    assert not runspec_path.exists()

    recovered = store.publish_current_attempt_runspec_projection(phase_run_id)
    assert recovered.current_runspec_projection_complete
    assert recovered.current_carry_forward == payload.carry_forward_record
    assert recovered.phase_runspec.carry_forward is not None
    assert hashlib.sha256(carry_path.read_bytes()).hexdigest()
    carry_sha256 = hashlib.sha256(carry_path.read_bytes()).hexdigest()
    runspec_sha256 = hashlib.sha256(runspec_path.read_bytes()).hexdigest()
    intent = render_phase_submission_intent(
        recovered.phase_runspec,
        phase_runspec_location=recovered.current_attempt.phase_runspec_location,
        phase_runspec_document_sha256=runspec_sha256,
        carry_forward_record=recovered.current_carry_forward,
        carry_forward_document_sha256=carry_sha256,
    )
    action_plan = intent.actions[0]
    assert action_plan.carry_forward_submission_id == intent.submission_id
    assert action_plan.carry_forward_record_sha256 == carry_sha256
    assert "--carry-forward-record" in action_plan.script_body
    assert "--phase-submission-id" in action_plan.script_body
    assert any(item.origin == "carry-source" and item.read_only for item in action_plan.carry_forward_mounts)
    status = status_phase(phase_run_id, authority_root=store.authority_root)
    assert status.carry_forward is not None
    assert status.carry_forward.attempt_carry_forward_id == payload.carry_forward_record.attempt_carry_forward_id
    assert status.to_mapping()["carry_forward"]["record_projection_complete"] is True

    attestation = next(
        event for event in recovered.events if getattr(event, "event_type", None) == "phase-action-evidence-attested"
    )
    record = payload.carry_forward_record
    assert record is not None
    adoption = AttemptCarryForwardAdoptionEvidence(
        phase_run_id=phase_run_id,
        attempt_id="attempt-0002",
        phase_runspec_digest=recovered.phase_runspec.digest,
        phase_submission_id=intent.submission_id,
        action_id=recovered.phase_runspec.payload.actions[0].action_id,
        attempt_carry_forward_id=record.attempt_carry_forward_id,
        attempt_carry_forward_digest=record.digest,
        remaining_search_input_sha256=record.remaining_search_input_sha256,
        content=tuple(
            AttemptCarryForwardAdoptedContent(
                member_name=item.member_name,
                source_path=item.source_private_mount_path,
                target_path=item.target_declared_path,
                size_bytes=item.size_bytes,
                sha256=item.sha256,
            )
            for item in record.content
        ),
        adopted_at="2026-08-20T13:02:00.000000Z",
    )
    current_evidence = _rebind_action_evidence_identity(
        attestation.payload.evidence,
        attempt_id="attempt-0002",
        phase_runspec_digest=recovered.phase_runspec.digest,
        carry_forward_adoption=adoption,
    )
    _record_submission_state(store.authority_root, phase_run_id, "submitted")
    recovered = store.validate(phase_run_id)
    closure = _carry_forward_closure(recovered, current_evidence)
    assert tuple(item.attempt_carry_forward_id for item in closure) == (record.attempt_carry_forward_id,)
    assert closure[0].target_adoption_evidence_digest == adoption.digest


def test_post_retry_projection_recovery_performs_no_evidence_or_configuration_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, phase_run_id = _failed_attested_authority(tmp_path, monkeypatch)
    payload = _carried_retry_payload(store, phase_run_id)

    def stop_after_event(_path: Path) -> None:
        raise OSError("stop after carried Retry event")

    interrupted = PhaseAuthorityStore(store.authority_root, after_retry_event_publish=stop_after_event)
    with pytest.raises(OSError, match="stop after carried Retry event"):
        _append_carried_retry(interrupted, phase_run_id, payload)

    def forbidden_read(*_args: object, **_kwargs: object) -> bytes:
        raise AssertionError("post-Retry recovery must not reread evidence")

    monkeypatch.setattr(RemoteSlurmTransport, "read_immutable_text_artifact_no_follow", forbidden_read)
    result = retry_phase(
        phase_run_id,
        authority_root=store.authority_root,
        config_path=tmp_path / "must-not-be-read.yaml",
        carry_forward_path=tmp_path / "must-not-be-read.json",
        clock=lambda: (_ for _ in ()).throw(AssertionError("clock must not be read")),
    )
    assert result.successor_attempt_id == "attempt-0002"
    assert result.attempt_carry_forward_id == payload.carry_forward_record.attempt_carry_forward_id
    assert store.validate(phase_run_id).current_runspec_projection_complete


def test_three_attempt_lineage_propagates_inbound_member_and_adds_verified_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, phase_run_id = _failed_attested_authority(tmp_path, monkeypatch)
    first_retry = _carried_retry_payload(store, phase_run_id)
    _append_carried_retry(store, phase_run_id, first_retry)
    _attest_carried_attempt_failure(store, phase_run_id)

    second_retry = _carried_retry_payload(
        store,
        phase_run_id,
        selected_names=("AFDB_zeta.a3m", "AFDB_alpha.a3m"),
    )
    _append_carried_retry(store, phase_run_id, second_retry)
    authority = store.validate(phase_run_id)
    assert authority.current_attempt.attempt_id == "attempt-0003"
    assert tuple(item.member_name for item in authority.current_carry_forward.content) == (
        "AFDB_zeta.a3m",
        "AFDB_alpha.a3m",
    )
    assert tuple(item.carry_forward.attempt_carry_forward_id for item in authority.prior_attempts[1:]) == (
        first_retry.carry_forward_record.attempt_carry_forward_id,
    )

    _record_submission_state(
        store.authority_root,
        phase_run_id,
        "submitted",
        job_id="323456",
        intended_at="2026-08-22T12:00:00.000000Z",
    )
    authority = store.validate(phase_run_id)
    assert authority.submission is not None
    current_adoption = _adoption_for_record(
        store,
        phase_run_id,
        submission_id=authority.submission.submission_id,
    )
    predecessor_attestation = next(
        event
        for event in authority.events
        if event.sequence == second_retry.carry_forward_record.verification.attestation_event_sequence
    )
    assert isinstance(predecessor_attestation, PhaseActionEvidenceAttestedEvent)
    current_evidence = _rebind_action_evidence_identity(
        predecessor_attestation.payload.evidence,
        attempt_id="attempt-0003",
        phase_runspec_digest=authority.phase_runspec.digest,
        carry_forward_adoption=current_adoption,
    )
    closure = _carry_forward_closure(authority, current_evidence)
    assert tuple(item.attempt_carry_forward_id for item in closure) == (
        first_retry.carry_forward_record.attempt_carry_forward_id,
        second_retry.carry_forward_record.attempt_carry_forward_id,
    )


@pytest.mark.parametrize(
    ("adoption_mode", "selected_names", "error"),
    [
        ("missing", ("AFDB_zeta.a3m", "AFDB_alpha.a3m"), "lacks passing inbound"),
        ("wrong-submission", ("AFDB_zeta.a3m", "AFDB_alpha.a3m"), "lacks passing inbound"),
        ("wrong-record", ("AFDB_zeta.a3m", "AFDB_alpha.a3m"), "lacks passing inbound"),
        ("valid", ("AFDB_alpha.a3m",), "propagate every inbound member"),
        ("valid", ("AFDB_alpha.a3m", "AFDB_zeta.a3m"), "declaration order"),
        ("hash-drift", ("AFDB_zeta.a3m", "AFDB_alpha.a3m"), "byte-identically"),
    ],
)
def test_three_attempt_lineage_rejects_invalid_inbound_propagation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    adoption_mode: str,
    selected_names: tuple[str, ...],
    error: str,
) -> None:
    store, phase_run_id = _failed_attested_authority(tmp_path, monkeypatch)
    _append_carried_retry(store, phase_run_id, _carried_retry_payload(store, phase_run_id))
    _attest_carried_attempt_failure(store, phase_run_id, adoption_mode=adoption_mode)

    with pytest.raises(ValueError, match=error):
        payload = _carried_retry_payload(store, phase_run_id, selected_names=selected_names)
        _append_carried_retry(store, phase_run_id, payload)
