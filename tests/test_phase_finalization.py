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

"""Control receipt, event replay, and sealing tests for preprocessing."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import pytest
from click.testing import CliRunner

from bspp.orchestration.contract.database_capacity_fallback_result import (
    DatabaseCapacityFallbackResult,
)
from bspp.orchestration.contract.database_direct_result import database_direct_result_digest
from bspp.orchestration.contract.database_placement import (
    DATABASE_SOURCE_ROOT,
    DatabaseAccessPolicy,
    DatabaseSetSelection,
)
from bspp.orchestration.contract.database_placement_result import (
    DatabasePostScienceObservation,
    DatabasePostScienceObservationFailure,
    database_placement_result_digest,
)
from bspp.orchestration.contract.phase import (
    PhasePlan,
    PreprocessingPhasePlanPayload,
    canonical_mapping_digest,
)
from bspp.orchestration.contract.phase_receipt import (
    PhaseFinalizedEvent,
    phase_finalized_event_from_mapping,
    phase_receipt_id,
)
from bspp.orchestration.contract.phase_reconciliation import (
    PhaseActionTerminalObservedEvent,
    PhaseActionTerminalObservedPayload,
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
from bspp.orchestration.contract.phase_submission import (
    PhaseActionDispatchIntendedEvent,
    PhaseActionDispatchIntendedPayload,
    PhaseActionDispatchRejectedEvent,
    PhaseActionDispatchRejectedPayload,
    PhaseActionSubmittedEvent,
    PhaseActionSubmittedPayload,
    PhaseSubmissionIntendedEvent,
)
from bspp.orchestration.contract.preprocessing_action import preprocessing_chunk_action_evidence_from_mapping
from bspp.orchestration.contract.preprocessing_execution import (
    preprocessing_chunk_execution_intent_from_plan,
)
from bspp.orchestration.contract.preprocessing_handoff import (
    preprocessing_content_validation_evidence_id,
)
from bspp.orchestration.control.cli import cli
from bspp.orchestration.control.phase_authority import (
    PhaseAuthorityEventHandler,
    PhaseAuthorityStore,
    PhaseAuthorityValidation,
)
from bspp.orchestration.control.phase_finalization import PhaseFinalizationResult, finalize_phase
from bspp.orchestration.control.phase_rendering import render_phase_submission_intent
from bspp.orchestration.control.phase_retry import retry_phase
from bspp.orchestration.control.phase_status import status_phase
from bspp.orchestration.control.phase_submission import submit_phase
from bspp.orchestration.control.transport import CommandResult, default_command_runner
from bspp.orchestration.runtime.preprocessing.finalization import finalize_preprocessing_chunk
from tests.support.database_cold_replica import _stage_required_fixture
from tests.support.database_cold_replica import cold_cache_root as cold_cache_root
from tests.support.preprocessing_execution import (
    LocalExecutionFixture,
    configure_preprocessing_fakes,
    invoke_preprocessing_execution,
    preprocessing_execution_fixture,
    skip_preprocessing_server_warmup,
)

FINALIZED_AT = datetime(2026, 8, 20, 14, 0, tzinfo=UTC)


def _record_submission(
    authority_root: Path,
    phase_run_id: str,
    *,
    job_id: str = "123456",
    outcome: Literal["dispatching", "submitted", "rejected"] = "submitted",
) -> None:
    store = PhaseAuthorityStore(authority_root)
    authority = store.validate(phase_run_id)
    attempt = authority.current_attempt
    document_sha256 = hashlib.sha256(
        (authority.authority_path / attempt.phase_runspec_location).read_bytes()
    ).hexdigest()
    intent = render_phase_submission_intent(
        authority.phase_runspec,
        phase_runspec_location=attempt.phase_runspec_location,
        phase_runspec_document_sha256=document_sha256,
    )
    authority = store.append_event(
        phase_run_id,
        lambda sequence: PhaseSubmissionIntendedEvent(
            sequence=sequence,
            phase_run_id=phase_run_id,
            attempt_id=authority.phase_runspec.attempt_id,
            occurred_at="2026-08-19T12:10:00.000000Z",
            payload=intent,
        ),
    )
    plan = intent.actions[0]
    dispatch = PhaseActionDispatchIntendedPayload(
        submission_id=intent.submission_id,
        action_id=plan.action_id,
        script_sha256=plan.script_sha256,
        scheduler_correlation_token=plan.scheduler_correlation_token,
        dependency_job_ids=(),
    )
    authority = store.append_event(
        phase_run_id,
        lambda sequence: PhaseActionDispatchIntendedEvent(
            sequence=sequence,
            phase_run_id=phase_run_id,
            attempt_id=authority.phase_runspec.attempt_id,
            occurred_at="2026-08-19T12:11:00.000000Z",
            payload=dispatch,
        ),
    )
    if outcome == "dispatching":
        return
    if outcome == "rejected":
        store.append_event(
            phase_run_id,
            lambda sequence: PhaseActionDispatchRejectedEvent(
                sequence=sequence,
                phase_run_id=phase_run_id,
                attempt_id=authority.phase_runspec.attempt_id,
                occurred_at="2026-08-19T12:12:00.000000Z",
                payload=PhaseActionDispatchRejectedPayload(
                    submission_id=intent.submission_id,
                    action_id=plan.action_id,
                    script_sha256=plan.script_sha256,
                    scheduler_correlation_token=plan.scheduler_correlation_token,
                    dependency_job_ids=(),
                    sbatch_argv=("sbatch", "--parsable", plan.cluster_script_path),
                    return_code=1,
                    stdout="",
                    stderr="partition unavailable\n",
                ),
            ),
        )
        return
    store.append_event(
        phase_run_id,
        lambda sequence: PhaseActionSubmittedEvent(
            sequence=sequence,
            phase_run_id=phase_run_id,
            attempt_id=authority.phase_runspec.attempt_id,
            occurred_at="2026-08-19T12:12:00.000000Z",
            payload=PhaseActionSubmittedPayload(
                submission_id=intent.submission_id,
                action_id=plan.action_id,
                script_sha256=plan.script_sha256,
                scheduler_correlation_token=plan.scheduler_correlation_token,
                dependency_job_ids=(),
                job_id=job_id,
                sbatch_argv=("sbatch", "--parsable", plan.cluster_script_path),
            ),
        ),
    )


@dataclass(frozen=True)
class RegisteredInterveningEvent:
    """Small test event proving that event replay is registry-driven."""

    sequence: int
    phase_run_id: str
    attempt_id: str
    event_type: str = "phase-submitted-test"

    def to_mapping(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "event_type": self.event_type,
            "phase_run_id": self.phase_run_id,
            "attempt_id": self.attempt_id,
        }


def _registered_intervening_event_from_mapping(
    mapping: Mapping[str, object],
) -> RegisteredInterveningEvent:
    if set(mapping) != {"sequence", "event_type", "phase_run_id", "attempt_id"}:
        raise ValueError("registered test event fields do not match its schema")
    sequence = mapping["sequence"]
    event_type = mapping["event_type"]
    phase_run_id = mapping["phase_run_id"]
    attempt_id = mapping["attempt_id"]
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 2:
        raise ValueError("registered test event sequence must follow materialization")
    if event_type != "phase-submitted-test":
        raise ValueError("registered test event has the wrong discriminator")
    if not isinstance(phase_run_id, str) or not isinstance(attempt_id, str):
        raise ValueError("registered test event requires string identities")
    return RegisteredInterveningEvent(
        sequence=sequence,
        phase_run_id=phase_run_id,
        attempt_id=attempt_id,
    )


def _replay_registered_intervening_event(
    event: object,
    authority: PhaseAuthorityValidation,
) -> PhaseAuthorityValidation:
    if not isinstance(event, RegisteredInterveningEvent):
        raise TypeError("registered test handler requires RegisteredInterveningEvent")
    if event.phase_run_id != authority.phase_run.phase_run_id or event.attempt_id != authority.phase_runspec.attempt_id:
        raise ValueError("registered test event does not bind the authoritative attempt")
    return replace(authority, events=(*authority.events, event))


def _accepted_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    submission_outcome: Literal["dispatching", "submitted", "rejected"] | None = "submitted",
    stage_required: bool = False,
    stage_preferred: bool = False,
    preferred_fallback: bool = True,
    cold_cache_root: Path | None = None,
    warm_reuse: bool = False,
) -> dict[str, object]:
    manifest_path: Path | None = None
    cache_root: Path | None = None
    if stage_required or stage_preferred:
        assert cold_cache_root is not None
        fixture, manifest_path, cache_root = _stage_required_fixture(
            tmp_path / "work",
            cold_cache_root,
            monkeypatch,
            requested_policy=(
                DatabaseAccessPolicy.STAGE_PREFERRED if stage_preferred else DatabaseAccessPolicy.STAGE_REQUIRED
            ),
        )
    else:
        fixture = preprocessing_execution_fixture(tmp_path / "work")
    plan = PhasePlan(
        target_cluster=fixture.runspec.cluster.profile_name,
        input_location=fixture.runspec.input_location,
        payload=PreprocessingPhasePlanPayload(
            work_plan=fixture.runspec.payload.work_plan,
            chunk_execution_intent=preprocessing_chunk_execution_intent_from_plan(fixture.action.payload),
            database=DatabaseSetSelection(
                database_set=fixture.runspec.payload.database.database_set,
                requested_policy=fixture.runspec.payload.database.requested_policy,
            ),
        ),
    )
    runspec = replace(fixture.runspec, phase_plan_digest=plan.digest)
    fixture = replace(fixture, runspec=runspec)
    fixture.runspec_path.write_text(json.dumps(runspec.to_mapping(), indent=2, sort_keys=True) + "\n")
    configure_preprocessing_fakes(fixture, monkeypatch)
    skip_preprocessing_server_warmup(monkeypatch)
    if stage_required or stage_preferred:
        assert manifest_path is not None and cache_root is not None
        fixture.database_placement_result_path.chmod(0o600)
        fixture.database_placement_result_path.unlink()
        monkeypatch.setattr(
            os,
            "fstatvfs",
            lambda _descriptor: SimpleNamespace(
                f_bavail=1 if stage_preferred and preferred_fallback else 2,
                f_frsize=1,
            ),
        )
        fixture.place_database(
            fixture.runspec,
            action_id=fixture.action.action_id,
            source_manifest_path=manifest_path,
            result_path=fixture.database_placement_result_path,
            failure_path=fixture.database_placement_failure_path,
        )
        if warm_reuse:
            assert stage_required or (stage_preferred and not preferred_fallback)
            fixture.database_placement_result_path.unlink()
            fixture.place_database(
                fixture.runspec,
                action_id=fixture.action.action_id,
                source_manifest_path=manifest_path,
                result_path=fixture.database_placement_result_path,
                failure_path=fixture.database_placement_failure_path,
            )
        if stage_required or (stage_preferred and not preferred_fallback):
            identity = fixture.runspec.payload.database.source_manifest_sha256
            fixture = replace(
                fixture,
                database_placement_paths=replace(
                    fixture.database_placement_paths,
                    selected_root=cache_root / "replicas" / identity,
                    lease=cache_root / ".locks" / f"{identity}.lock",
                ),
            )
    executed = invoke_preprocessing_execution(fixture)
    assert executed.exit_code == 0, executed.output
    action_evidence = preprocessing_chunk_action_evidence_from_mapping(json.loads(fixture.evidence_path.read_text()))
    handoff = tmp_path / "handoff"
    finalize_preprocessing_chunk(
        runspec,
        action_evidence,
        handoff_path=handoff,
        clock=lambda: FINALIZED_AT,
    )

    attempt = PhaseAttempt(
        attempt_id=runspec.attempt_id,
        ordinal=1,
        phase_runspec_location=f"attempts/{runspec.attempt_id}/phase-runspec.json",
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
    PhaseAuthorityStore(authority_root).publish(
        phase_plan=plan,
        phase_run=phase_run,
        phase_runspec=runspec,
        materialized_event=materialized,
    )
    if submission_outcome is not None:
        _record_submission(authority_root, runspec.phase_run_id, outcome=submission_outcome)
    scheduler_path = tmp_path / "scheduler.json"
    scheduler_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "evidence_kind": "provided-successful-terminal-observation",
                "phase_run_id": runspec.phase_run_id,
                "attempt_id": runspec.attempt_id,
                "phase_runspec_digest": runspec.digest,
                "action_id": fixture.action.action_id,
                "job_id": "123456",
                "source": "sacct",
                "state": "COMPLETED",
                "exit_code": "0:0",
                "observed_at": "2026-08-19T13:00:00.000000Z",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return {
        "fixture": fixture,
        "authority_root": authority_root,
        "handoff": handoff,
        "scheduler": scheduler_path,
    }


def _finalize(
    inputs: dict[str, object],
    *,
    authority_store: PhaseAuthorityStore | None = None,
) -> PhaseFinalizationResult:
    fixture = inputs["fixture"]
    authority_root = inputs["authority_root"]
    scheduler = inputs["scheduler"]
    handoff = inputs["handoff"]
    assert isinstance(fixture, LocalExecutionFixture)
    assert isinstance(authority_root, Path)
    assert isinstance(scheduler, Path)
    assert isinstance(handoff, Path)
    return finalize_phase(
        fixture.runspec.phase_run_id,
        authority_root=authority_root,
        scheduler_evidence_path=scheduler,
        action_evidence_path=fixture.evidence_path,
        handoff_path=handoff,
        clock=lambda: FINALIZED_AT,
        authority_store=authority_store,
    )


def _rewrite_fallback_authority(
    inputs: dict[str, object],
    transform: Callable[[DatabaseCapacityFallbackResult], DatabaseCapacityFallbackResult],
) -> None:
    fixture = inputs["fixture"]
    handoff = inputs["handoff"]
    assert isinstance(fixture, LocalExecutionFixture)
    assert isinstance(handoff, Path)
    evidence = preprocessing_chunk_action_evidence_from_mapping(json.loads(fixture.evidence_path.read_text()))
    placement = evidence.database_placement
    assert placement.result is not None
    assert isinstance(placement.result, DatabaseCapacityFallbackResult)
    forged_result = transform(placement.result)
    forged_placement = replace(
        placement,
        result=forged_result,
        result_digest=database_direct_result_digest(forged_result),
        post_science_observation=DatabasePostScienceObservation.from_pre_science(forged_result.pre_science_observation),
    )
    forged_evidence = replace(evidence, database_placement=forged_placement)
    fixture.evidence_path.chmod(0o644)
    fixture.evidence_path.write_text(json.dumps(forged_evidence.to_mapping(), indent=2, sort_keys=True) + "\n")
    fixture.evidence_path.chmod(0o444)

    validation_path = handoff / "content-validation.json"
    validation = json.loads(validation_path.read_text())
    validation["action_evidence_digest"] = canonical_mapping_digest(forged_evidence.to_mapping())
    validation["evidence_id"] = preprocessing_content_validation_evidence_id(
        {key: value for key, value in validation.items() if key != "evidence_id"}
    )
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")


def _rewrite_action_evidence_mapping(
    inputs: dict[str, object],
    transform: Callable[[dict[str, object]], None],
) -> None:
    """Forge serialized action evidence while retaining exact handoff linkage."""
    fixture = inputs["fixture"]
    handoff = inputs["handoff"]
    assert isinstance(fixture, LocalExecutionFixture)
    assert isinstance(handoff, Path)
    mapping = json.loads(fixture.evidence_path.read_text())
    assert isinstance(mapping, dict)
    transform(mapping)
    fixture.evidence_path.chmod(0o644)
    fixture.evidence_path.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n")
    fixture.evidence_path.chmod(0o444)

    validation_path = handoff / "content-validation.json"
    validation = json.loads(validation_path.read_text())
    validation["action_evidence_digest"] = canonical_mapping_digest(mapping)
    validation["evidence_id"] = preprocessing_content_validation_evidence_id(
        {key: value for key, value in validation.items() if key != "evidence_id"}
    )
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")


def _record_terminal_observation(
    authority_root: Path,
    phase_run_id: str,
    *,
    state: str,
    exit_code: str | None,
    outcome: Literal["succeeded", "failed"],
) -> None:
    store = PhaseAuthorityStore(authority_root)
    authority = store.validate(phase_run_id)
    assert authority.submission is not None
    action = authority.submission.actions[0]
    assert action.job_id is not None
    payload = PhaseActionTerminalObservedPayload(
        submission_id=authority.submission.submission_id,
        phase_runspec_digest=authority.phase_runspec.digest,
        action_id=action.action_id,
        runtime_action_digest=action.plan.runtime_action_digest,
        scheduler_correlation_token=action.scheduler_correlation_token,
        job_id=action.job_id,
        state=state,
        exit_code=exit_code,
        outcome=outcome,
    )
    store.append_event(
        phase_run_id,
        lambda sequence: PhaseActionTerminalObservedEvent(
            sequence=sequence,
            phase_run_id=phase_run_id,
            attempt_id=authority.phase_runspec.attempt_id,
            occurred_at="2026-08-19T13:00:00.000000Z",
            payload=payload,
        ),
    )


def test_control_finalization_issues_attempt_bound_receipt_and_seals_without_rewriting_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _accepted_inputs(tmp_path, monkeypatch)
    fixture = inputs["fixture"]
    assert isinstance(fixture, LocalExecutionFixture)
    authority_root = inputs["authority_root"]
    assert isinstance(authority_root, Path)
    run_root = authority_root / fixture.runspec.phase_run_id
    snapshot_before = (run_root / "phase-run.json").read_bytes()

    result = _finalize(inputs)

    assert result.status == "accepted"
    assert (run_root / "phase-run.json").read_bytes() == snapshot_before
    assert {path.name for path in (run_root / "events").iterdir()} == {
        "000001-phase-materialized.json",
        "000002-phase-submission-intended.json",
        "000003-phase-action-dispatch-intended.json",
        "000004-phase-action-submitted.json",
        "000005-phase-finalized.json",
    }
    replay = PhaseAuthorityStore(authority_root).validate(fixture.runspec.phase_run_id)
    assert replay.lifecycle.run_status == "accepted"
    assert replay.lifecycle.attempt_status == "succeeded"
    assert replay.lifecycle.sealed
    assert replay.receipt is not None
    assert replay.receipt.phase_receipt_id == result.phase_receipt_id
    assert replay.receipt.database_requested_policy.value == "direct"
    assert replay.receipt.database_source_manifest_sha256 == fixture.runspec.payload.database.source_manifest_sha256
    assert replay.receipt.database_placement_outcome.value == "direct-requested"
    assert replay.phase_run.status == "materialized"
    assert not replay.phase_run.sealed
    with pytest.raises(ValueError, match="accepted Phase Attempt boundary"):
        retry_phase(
            fixture.runspec.phase_run_id,
            authority_root=authority_root,
            config_path=tmp_path / "must-not-read.yaml",
            clock=lambda: (_ for _ in ()).throw(AssertionError("clock must not run")),
        )


def test_control_finalization_accepts_exact_cold_staged_lease_and_receipt_summary(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _accepted_inputs(
        tmp_path,
        monkeypatch,
        stage_required=True,
        cold_cache_root=cold_cache_root,
    )

    result = _finalize(inputs)

    assert result.status == "accepted"
    fixture = inputs["fixture"]
    authority_root = inputs["authority_root"]
    assert isinstance(fixture, LocalExecutionFixture)
    assert isinstance(authority_root, Path)
    replay = PhaseAuthorityStore(authority_root).validate(fixture.runspec.phase_run_id)
    assert replay.receipt is not None
    assert replay.receipt.database_requested_policy.value == "stage-required"
    assert replay.receipt.database_source_manifest_sha256 == fixture.runspec.payload.database.source_manifest_sha256
    assert replay.receipt.database_placement_outcome.value == "replica-cold"


def test_control_finalization_accepts_exact_warm_staged_lease_and_receipt_summary(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _accepted_inputs(
        tmp_path,
        monkeypatch,
        stage_required=True,
        cold_cache_root=cold_cache_root,
        warm_reuse=True,
    )

    result = _finalize(inputs)

    assert result.status == "accepted"
    fixture = inputs["fixture"]
    authority_root = inputs["authority_root"]
    assert isinstance(fixture, LocalExecutionFixture)
    assert isinstance(authority_root, Path)
    replay = PhaseAuthorityStore(authority_root).validate(fixture.runspec.phase_run_id)
    assert replay.receipt is not None
    assert replay.receipt.database_requested_policy.value == "stage-required"
    assert replay.receipt.database_source_manifest_sha256 == fixture.runspec.payload.database.source_manifest_sha256
    assert replay.receipt.database_placement_outcome.value == "replica-warm"


def test_control_finalization_accepts_stage_preferred_capacity_fallback_as_ordinary_receipt(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _accepted_inputs(
        tmp_path,
        monkeypatch,
        stage_preferred=True,
        cold_cache_root=cold_cache_root,
    )

    result = _finalize(inputs)

    assert result.status == "accepted"
    fixture = inputs["fixture"]
    authority_root = inputs["authority_root"]
    assert isinstance(fixture, LocalExecutionFixture)
    assert isinstance(authority_root, Path)
    replay = PhaseAuthorityStore(authority_root).validate(fixture.runspec.phase_run_id)
    assert replay.receipt is not None
    assert replay.receipt.database_requested_policy == DatabaseAccessPolicy.STAGE_PREFERRED
    assert replay.receipt.database_placement_outcome.value == "direct-capacity-fallback"


def test_control_finalization_rejects_fallback_observation_members_outside_runspec_authority(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _accepted_inputs(
        tmp_path,
        monkeypatch,
        stage_preferred=True,
        cold_cache_root=cold_cache_root,
    )

    def forge_observation(result: DatabaseCapacityFallbackResult) -> DatabaseCapacityFallbackResult:
        first, *remaining = result.pre_science_observation.members
        return replace(
            result,
            pre_science_observation=replace(
                result.pre_science_observation,
                members=(replace(first, mtime_ns=first.mtime_ns + 1), *remaining),
            ),
        )

    _rewrite_fallback_authority(inputs, forge_observation)

    with pytest.raises(ValueError, match="authoritative RunSpec"):
        _finalize(inputs)


def test_control_finalization_rejects_fallback_reserve_outside_runspec_authority(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _accepted_inputs(
        tmp_path,
        monkeypatch,
        stage_preferred=True,
        cold_cache_root=cold_cache_root,
    )

    def forge_reserve(result: DatabaseCapacityFallbackResult) -> DatabaseCapacityFallbackResult:
        gate = result.capacity_gate
        return replace(
            result,
            capacity_gate=replace(
                gate,
                reserved_bytes=gate.reserved_bytes + 1,
                required_bytes=gate.required_bytes + 1,
            ),
        )

    _rewrite_fallback_authority(inputs, forge_reserve)

    with pytest.raises(ValueError, match="authoritative RunSpec"):
        _finalize(inputs)


def test_control_finalization_rejects_fallback_cache_filesystem_outside_runspec_authority(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _accepted_inputs(
        tmp_path,
        monkeypatch,
        stage_preferred=True,
        cold_cache_root=cold_cache_root,
    )

    def forge_filesystem(result: DatabaseCapacityFallbackResult) -> DatabaseCapacityFallbackResult:
        return replace(
            result,
            cache_mount=replace(
                result.cache_mount,
                filesystem_type=f"{result.cache_mount.filesystem_type}-forged",
            ),
        )

    _rewrite_fallback_authority(inputs, forge_filesystem)

    with pytest.raises(ValueError, match="authoritative RunSpec"):
        _finalize(inputs)


@pytest.mark.parametrize(
    ("warm_reuse", "expected_outcome"),
    [(False, "replica-cold"), (True, "replica-warm")],
)
def test_control_finalization_accepts_stage_preferred_staged_families_with_lease(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    warm_reuse: bool,
    expected_outcome: str,
) -> None:
    inputs = _accepted_inputs(
        tmp_path,
        monkeypatch,
        stage_preferred=True,
        preferred_fallback=False,
        cold_cache_root=cold_cache_root,
        warm_reuse=warm_reuse,
    )

    result = _finalize(inputs)

    assert result.status == "accepted"
    fixture = inputs["fixture"]
    authority_root = inputs["authority_root"]
    assert isinstance(fixture, LocalExecutionFixture)
    assert isinstance(authority_root, Path)
    replay = PhaseAuthorityStore(authority_root).validate(fixture.runspec.phase_run_id)
    assert replay.receipt is not None
    assert replay.receipt.database_requested_policy == DatabaseAccessPolicy.STAGE_PREFERRED
    assert replay.receipt.database_placement_outcome.value == expected_outcome


def test_warm_finalized_payload_rejects_cold_lease_family_forgery(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _accepted_inputs(
        tmp_path,
        monkeypatch,
        stage_required=True,
        cold_cache_root=cold_cache_root,
        warm_reuse=True,
    )
    fixture = inputs["fixture"]
    authority_root = inputs["authority_root"]
    assert isinstance(fixture, LocalExecutionFixture)
    assert isinstance(authority_root, Path)
    _finalize(inputs)
    event_path = authority_root / fixture.runspec.phase_run_id / "events/000005-phase-finalized.json"
    mapping = json.loads(event_path.read_text())
    lease = mapping["payload"]["action_evidence"]["database_placement"]["lease_outcome"]
    lease["database_replica_cold_result_digest"] = lease.pop("database_replica_warm_result_digest")

    with pytest.raises(ValueError, match="warm Result requires the warm lease evidence family"):
        phase_finalized_event_from_mapping(mapping)


def test_cold_finalized_payload_rejects_warm_lease_family_forgery(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _accepted_inputs(
        tmp_path,
        monkeypatch,
        stage_required=True,
        cold_cache_root=cold_cache_root,
    )
    fixture = inputs["fixture"]
    authority_root = inputs["authority_root"]
    assert isinstance(fixture, LocalExecutionFixture)
    assert isinstance(authority_root, Path)
    _finalize(inputs)
    event_path = authority_root / fixture.runspec.phase_run_id / "events/000005-phase-finalized.json"
    mapping = json.loads(event_path.read_text())
    lease = mapping["payload"]["action_evidence"]["database_placement"]["lease_outcome"]
    lease["database_replica_warm_result_digest"] = lease.pop("database_replica_cold_result_digest")

    with pytest.raises(ValueError, match="cold Result requires the cold lease evidence family"):
        phase_finalized_event_from_mapping(mapping)


def test_warm_finalized_payload_rejects_cold_receipt_summary_forgery(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _accepted_inputs(
        tmp_path,
        monkeypatch,
        stage_required=True,
        cold_cache_root=cold_cache_root,
        warm_reuse=True,
    )
    fixture = inputs["fixture"]
    authority_root = inputs["authority_root"]
    assert isinstance(fixture, LocalExecutionFixture)
    assert isinstance(authority_root, Path)
    _finalize(inputs)
    event_path = authority_root / fixture.runspec.phase_run_id / "events/000005-phase-finalized.json"
    mapping = json.loads(event_path.read_text())
    receipt = mapping["payload"]["receipt"]
    receipt["database_placement_outcome"] = "replica-cold"
    identity = {key: item for key, item in receipt.items() if key != "phase_receipt_id"}
    receipt["phase_receipt_id"] = phase_receipt_id(identity)

    with pytest.raises(ValueError, match="exact successful staged database authority"):
        phase_finalized_event_from_mapping(mapping)


def test_control_finalization_rejects_no_kernel_staged_terminal_evidence(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _accepted_inputs(
        tmp_path,
        monkeypatch,
        stage_required=True,
        cold_cache_root=cold_cache_root,
    )
    fixture = inputs["fixture"]
    assert isinstance(fixture, LocalExecutionFixture)

    def forge_no_kernel(mapping: dict[str, object]) -> None:
        placement = mapping["database_placement"]
        assert isinstance(placement, dict)
        lease = placement["lease_outcome"]
        assert isinstance(lease, dict)
        lease["kernel_started"] = False
        lease["held_through_kernel_exit"] = False
        placement["science_started"] = False

    _rewrite_action_evidence_mapping(inputs, forge_no_kernel)

    with pytest.raises(ValueError, match="staged preprocessing"):
        _finalize(inputs)


def test_staged_finalized_payload_rejects_cross_branch_receipt_summary(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _accepted_inputs(
        tmp_path,
        monkeypatch,
        stage_required=True,
        cold_cache_root=cold_cache_root,
    )
    fixture = inputs["fixture"]
    authority_root = inputs["authority_root"]
    assert isinstance(fixture, LocalExecutionFixture)
    assert isinstance(authority_root, Path)
    _finalize(inputs)
    event_path = authority_root / fixture.runspec.phase_run_id / "events/000005-phase-finalized.json"
    mapping = json.loads(event_path.read_text())
    receipt = mapping["payload"]["receipt"]
    receipt["database_requested_policy"] = "direct"
    receipt["database_placement_outcome"] = "direct-requested"
    identity = {key: item for key, item in receipt.items() if key != "phase_receipt_id"}
    receipt["phase_receipt_id"] = phase_receipt_id(identity)

    with pytest.raises(ValueError, match="staged database authority"):
        phase_finalized_event_from_mapping(mapping)


def test_control_finalization_rejects_post_science_observation_failure_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _accepted_inputs(tmp_path, monkeypatch)
    fixture = inputs["fixture"]
    assert isinstance(fixture, LocalExecutionFixture)

    def forge_failed_post_science(mapping: dict[str, object]) -> None:
        placement = mapping["database_placement"]
        assert isinstance(placement, dict)
        result_envelope = placement["result"]
        assert isinstance(result_envelope, dict)
        result = result_envelope["database_placement_result"]
        assert isinstance(result, dict)
        placement["post_science_observation"] = DatabasePostScienceObservationFailure(
            source_container_root=DATABASE_SOURCE_ROOT,
            source_manifest_sha256=str(result["source_manifest_sha256"]),
            error="selected Database Source root became unavailable",
        ).to_mapping()

    _rewrite_action_evidence_mapping(inputs, forge_failed_post_science)

    with pytest.raises(ValueError, match="unchanged direct Database Source observation"):
        _finalize(inputs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("database_requested_policy", "stage-required"),
        ("database_source_manifest_sha256", "0" * 64),
        ("database_placement_outcome", "replica-cold"),
    ],
)
def test_finalized_payload_rejects_forged_database_receipt_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    inputs = _accepted_inputs(tmp_path, monkeypatch)
    fixture = inputs["fixture"]
    authority_root = inputs["authority_root"]
    assert isinstance(fixture, LocalExecutionFixture)
    assert isinstance(authority_root, Path)
    _finalize(inputs)
    event_path = authority_root / fixture.runspec.phase_run_id / "events" / "000005-phase-finalized.json"
    mapping = json.loads(event_path.read_text())
    receipt = mapping["payload"]["receipt"]
    receipt[field] = value
    identity = {key: item for key, item in receipt.items() if key != "phase_receipt_id"}
    receipt["phase_receipt_id"] = phase_receipt_id(identity)

    with pytest.raises(ValueError, match=r"Database|database"):
        phase_finalized_event_from_mapping(mapping)


def test_phase_receipt_loader_requires_exact_database_summary_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _accepted_inputs(tmp_path, monkeypatch)
    fixture = inputs["fixture"]
    authority_root = inputs["authority_root"]
    assert isinstance(fixture, LocalExecutionFixture)
    assert isinstance(authority_root, Path)
    _finalize(inputs)
    event_path = authority_root / fixture.runspec.phase_run_id / "events/000005-phase-finalized.json"
    mapping = json.loads(event_path.read_text())
    del mapping["payload"]["receipt"]["database_requested_policy"]

    with pytest.raises(ValueError, match="database_requested_policy"):
        phase_finalized_event_from_mapping(mapping)


def test_successor_attempt_finalization_and_status_bind_acceptance_to_attempt_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _accepted_inputs(tmp_path, monkeypatch, submission_outcome=None)
    fixture = inputs["fixture"]
    authority_root = inputs["authority_root"]
    assert isinstance(fixture, LocalExecutionFixture)
    assert isinstance(authority_root, Path)
    _record_submission(authority_root, fixture.runspec.phase_run_id)
    _record_terminal_observation(
        authority_root,
        fixture.runspec.phase_run_id,
        state="FAILED",
        exit_code="1:0",
        outcome="failed",
    )

    store = PhaseAuthorityStore(authority_root)
    predecessor = store.validate(fixture.runspec.phase_run_id)
    materialized_at = "2026-08-20T13:00:00.000000Z"
    successor_runspec = replace(
        predecessor.phase_runspec,
        attempt_id="attempt-0002",
        materialized_at=materialized_at,
        payload=replace(
            predecessor.phase_runspec.payload,
            database=replace(
                predecessor.phase_runspec.payload.database,
                source_manifest_projection="attempts/attempt-0002/database-source-manifest.json",
            ),
        ),
    )
    successor_attempt = PhaseAttempt(
        attempt_id="attempt-0002",
        ordinal=2,
        phase_runspec_location="attempts/attempt-0002/phase-runspec.json",
        phase_runspec_digest=successor_runspec.digest,
        created_at=materialized_at,
    )
    input_digest = phase_input_set_identity_digest(predecessor.phase_plan)
    scientific_digest = phase_scientific_identity_digest(predecessor.phase_plan)
    retry_identity = phase_retry_id(
        phase_run_id=fixture.runspec.phase_run_id,
        predecessor_attempt_id="attempt-0001",
        successor_attempt_id="attempt-0002",
        predecessor_outcome="failed",
        predecessor_phase_runspec_digest=predecessor.phase_runspec.digest,
        phase_plan_digest=predecessor.phase_plan.digest,
        input_set_identity_digest=input_digest,
        scientific_identity_digest=scientific_digest,
        selected_cluster_profile=successor_runspec.cluster.profile_name,
        successor_phase_runspec_digest=successor_runspec.digest,
    )
    payload = PhaseAttemptRetriedPayload(
        retry_id=retry_identity,
        predecessor_attempt_id="attempt-0001",
        predecessor_phase_runspec_digest=predecessor.phase_runspec.digest,
        predecessor_outcome="failed",
        phase_plan_digest=predecessor.phase_plan.digest,
        input_set_identity_digest=input_digest,
        scientific_identity_digest=scientific_digest,
        selected_cluster_profile=successor_runspec.cluster.profile_name,
        successor_attempt=successor_attempt,
        successor_phase_runspec=successor_runspec,
    )
    store.append_attempt_retried_event(
        fixture.runspec.phase_run_id,
        lambda sequence: PhaseAttemptRetriedEvent(
            sequence=sequence,
            phase_run_id=fixture.runspec.phase_run_id,
            attempt_id="attempt-0002",
            occurred_at=materialized_at,
            payload=payload,
        ),
    )

    original_evidence = preprocessing_chunk_action_evidence_from_mapping(json.loads(fixture.evidence_path.read_text()))
    original_placement_result = original_evidence.database_placement.result
    assert original_placement_result is not None
    successor_placement_result = replace(
        original_placement_result,
        attempt_id="attempt-0002",
        phase_runspec_digest=successor_runspec.digest,
    )
    successor_evidence = replace(
        original_evidence,
        attempt_id="attempt-0002",
        phase_runspec_digest=successor_runspec.digest,
        database_placement=replace(
            original_evidence.database_placement,
            result=successor_placement_result,
            result_digest=database_placement_result_digest(successor_placement_result),
        ),
    )
    successor_evidence_path = tmp_path / "successor-action-evidence.json"
    successor_evidence_path.write_text(json.dumps(successor_evidence.to_mapping(), indent=2, sort_keys=True) + "\n")
    successor_handoff = tmp_path / "successor-handoff"
    finalize_preprocessing_chunk(
        successor_runspec,
        successor_evidence,
        handoff_path=successor_handoff,
        clock=lambda: FINALIZED_AT,
    )
    _record_submission(authority_root, fixture.runspec.phase_run_id, job_id="654321")
    scheduler_path = tmp_path / "successor-scheduler.json"
    scheduler_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "evidence_kind": "provided-successful-terminal-observation",
                "phase_run_id": fixture.runspec.phase_run_id,
                "attempt_id": "attempt-0002",
                "phase_runspec_digest": successor_runspec.digest,
                "action_id": successor_runspec.payload.actions[0].action_id,
                "job_id": "654321",
                "source": "sacct",
                "state": "COMPLETED",
                "exit_code": "0:0",
                "observed_at": "2026-08-20T13:30:00.000000Z",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    result = finalize_phase(
        fixture.runspec.phase_run_id,
        authority_root=authority_root,
        scheduler_evidence_path=scheduler_path,
        action_evidence_path=successor_evidence_path,
        handoff_path=successor_handoff,
        clock=lambda: FINALIZED_AT,
    )

    def scheduler_runner(argv: tuple[str, ...]) -> CommandResult:
        return CommandResult(argv, 0, json.dumps({"jobs": []}), "")

    report = status_phase(
        fixture.runspec.phase_run_id,
        authority_root=authority_root,
        runner=scheduler_runner,
    )
    accepted = store.validate(fixture.runspec.phase_run_id)
    assert result.attempt_id == "attempt-0002"
    assert report.attempt_id == "attempt-0002"
    assert report.attempt_status == "succeeded"
    assert report.status == "accepted"
    assert report.sealed
    assert accepted.receipt is not None and accepted.receipt.attempt_id == "attempt-0002"


def test_successful_durable_terminal_observation_remains_compatible_with_finalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _accepted_inputs(tmp_path, monkeypatch)
    fixture = inputs["fixture"]
    authority_root = inputs["authority_root"]
    assert isinstance(fixture, LocalExecutionFixture)
    assert isinstance(authority_root, Path)
    _record_terminal_observation(
        authority_root,
        fixture.runspec.phase_run_id,
        state="COMPLETED",
        exit_code="0:0",
        outcome="succeeded",
    )

    result = _finalize(inputs)

    assert result.status == "accepted"
    replay = PhaseAuthorityStore(authority_root).validate(fixture.runspec.phase_run_id)
    assert replay.lifecycle.attempt_status == "succeeded"
    assert replay.terminal_observations[0].outcome == "succeeded"
    assert replay.finalized_event is not None and replay.finalized_event.sequence == 6


def test_failed_lifecycle_guard_precedes_all_finalization_evidence_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _accepted_inputs(tmp_path, monkeypatch)
    fixture = inputs["fixture"]
    authority_root = inputs["authority_root"]
    assert isinstance(fixture, LocalExecutionFixture)
    assert isinstance(authority_root, Path)
    _record_terminal_observation(
        authority_root,
        fixture.runspec.phase_run_id,
        state="FAILED",
        exit_code="1:0",
        outcome="failed",
    )
    inputs["fixture"] = replace(fixture, evidence_path=tmp_path / "missing-action-evidence.json")
    inputs["scheduler"] = tmp_path / "missing-scheduler-evidence.json"
    inputs["handoff"] = tmp_path / "missing-handoff"

    with pytest.raises(ValueError, match="failed Phase Attempt boundary"):
        _finalize(inputs)


def test_authority_replay_rejects_finalization_after_failed_terminal_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _accepted_inputs(tmp_path, monkeypatch)
    _finalize(inputs)
    fixture = inputs["fixture"]
    authority_root = inputs["authority_root"]
    assert isinstance(fixture, LocalExecutionFixture)
    assert isinstance(authority_root, Path)
    store = PhaseAuthorityStore(authority_root)
    accepted = store.validate(fixture.runspec.phase_run_id)
    assert accepted.submission is not None
    action = accepted.submission.actions[0]
    assert action.job_id is not None
    terminal = PhaseActionTerminalObservedEvent(
        sequence=5,
        phase_run_id=fixture.runspec.phase_run_id,
        attempt_id=fixture.runspec.attempt_id,
        occurred_at="2026-08-19T13:00:00.000000Z",
        payload=PhaseActionTerminalObservedPayload(
            submission_id=accepted.submission.submission_id,
            phase_runspec_digest=accepted.phase_runspec.digest,
            action_id=action.action_id,
            runtime_action_digest=action.plan.runtime_action_digest,
            scheduler_correlation_token=action.scheduler_correlation_token,
            job_id=action.job_id,
            state="FAILED",
            exit_code="1:0",
            outcome="failed",
        ),
    )
    events = accepted.authority_path / "events"
    finalized_path = events / "000005-phase-finalized.json"
    finalized_mapping = json.loads(finalized_path.read_text())
    finalized_mapping["sequence"] = 6
    finalized_path.rename(events / "000006-phase-finalized.json")
    (events / "000006-phase-finalized.json").write_text(json.dumps(finalized_mapping, indent=2, sort_keys=True) + "\n")
    (events / "000005-phase-action-terminal-observed.json").write_text(
        json.dumps(terminal.to_mapping(), indent=2, sort_keys=True) + "\n"
    )

    with pytest.raises(ValueError, match="current active Phase Attempt"):
        store.validate(fixture.runspec.phase_run_id)


@pytest.mark.parametrize("submission_outcome", [None, "dispatching", "rejected"])
def test_finalization_rejects_absent_incomplete_or_rejected_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    submission_outcome: Literal["dispatching", "rejected"] | None,
) -> None:
    inputs = _accepted_inputs(
        tmp_path,
        monkeypatch,
        submission_outcome=submission_outcome,
    )
    fixture = inputs["fixture"]
    authority_root = inputs["authority_root"]
    assert isinstance(fixture, LocalExecutionFixture)
    assert isinstance(authority_root, Path)
    events = authority_root / fixture.runspec.phase_run_id / "events"
    before = {path.name: path.read_bytes() for path in events.iterdir()}

    expected = (
        "failed Phase Attempt boundary" if submission_outcome == "rejected" else "complete durable Phase Submission"
    )
    with pytest.raises(ValueError, match=expected):
        _finalize(inputs)

    assert {path.name: path.read_bytes() for path in events.iterdir()} == before


def test_finalization_rejects_numeric_scheduler_job_other_than_durable_assignment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _accepted_inputs(tmp_path, monkeypatch)
    scheduler = inputs["scheduler"]
    fixture = inputs["fixture"]
    authority_root = inputs["authority_root"]
    assert isinstance(scheduler, Path)
    assert isinstance(fixture, LocalExecutionFixture)
    assert isinstance(authority_root, Path)
    mapping = json.loads(scheduler.read_text())
    mapping["job_id"] = "999999"
    scheduler.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="assigned Slurm job"):
        _finalize(inputs)

    assert not (authority_root / fixture.runspec.phase_run_id / "events/000005-phase-finalized.json").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("phase_runspec_digest", "0" * 64),
        ("action_id", "preprocessing-chunk-999999"),
    ],
)
def test_finalization_rejects_scheduler_evidence_for_wrong_runspec_or_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    inputs = _accepted_inputs(tmp_path, monkeypatch)
    scheduler = inputs["scheduler"]
    fixture = inputs["fixture"]
    authority_root = inputs["authority_root"]
    assert isinstance(scheduler, Path)
    assert isinstance(fixture, LocalExecutionFixture)
    assert isinstance(authority_root, Path)
    mapping = json.loads(scheduler.read_text())
    mapping[field] = value
    scheduler.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="current authoritative attempt"):
        _finalize(inputs)

    assert not (authority_root / fixture.runspec.phase_run_id / "events/000005-phase-finalized.json").exists()


def test_sealed_phase_submit_is_remote_call_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _accepted_inputs(tmp_path, monkeypatch)
    fixture = inputs["fixture"]
    authority_root = inputs["authority_root"]
    assert isinstance(fixture, LocalExecutionFixture)
    assert isinstance(authority_root, Path)
    _finalize(inputs)

    def reject_remote(_argv: tuple[str, ...]) -> CommandResult:
        raise AssertionError("sealed Phase Submission must not contact the cluster")

    with pytest.raises(ValueError, match="already sealed"):
        submit_phase(
            fixture.runspec.phase_run_id,
            authority_root=authority_root,
            clock=lambda: FINALIZED_AT,
            runner=reject_remote,
        )


def test_phase_operation_lock_retains_stable_zero_byte_sentinel(tmp_path: Path) -> None:
    authority_root = tmp_path / "authority"
    store = PhaseAuthorityStore(authority_root)
    phase_run_id = "phase-run-0123456789abcdef0123456789abcdef"
    lock_path = authority_root / f".phase-operation-{phase_run_id}.lock"

    with store.phase_operation_lock(phase_run_id):
        first_stat = lock_path.stat()

    assert lock_path.exists()
    assert lock_path.stat().st_size == 0

    with store.phase_operation_lock(phase_run_id):
        reacquired_stat = lock_path.stat()

    assert (reacquired_stat.st_dev, reacquired_stat.st_ino) == (first_stat.st_dev, first_stat.st_ino)
    assert lock_path.stat().st_size == 0


def test_submitter_and_finalizer_share_operation_lock_across_sbatch_assignment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _accepted_inputs(tmp_path, monkeypatch, submission_outcome=None)
    fixture = inputs["fixture"]
    authority_root = inputs["authority_root"]
    assert isinstance(fixture, LocalExecutionFixture)
    assert isinstance(authority_root, Path)
    sbatch_entered = threading.Event()
    release_sbatch = threading.Event()
    finalizer_started = threading.Event()
    finalizer_done = threading.Event()
    failures: list[BaseException] = []

    def blocked_runner(argv: tuple[str, ...]) -> CommandResult:
        if argv[0] != "sbatch":
            return default_command_runner(argv)
        sbatch_entered.set()
        if not release_sbatch.wait(timeout=5):
            raise TimeoutError("test did not release blocked sbatch")
        return CommandResult(argv=argv, returncode=0, stdout="123456\n", stderr="")

    def submission_worker() -> None:
        try:
            submit_phase(
                fixture.runspec.phase_run_id,
                authority_root=authority_root,
                clock=lambda: datetime(2026, 8, 20, 12, 0, tzinfo=UTC),
                runner=blocked_runner,
            )
        except BaseException as exc:  # pragma: no cover - reported in the parent thread
            failures.append(exc)

    def finalization_worker() -> None:
        finalizer_started.set()
        try:
            _finalize(inputs)
        except BaseException as exc:  # pragma: no cover - reported in the parent thread
            failures.append(exc)
        finally:
            finalizer_done.set()

    submitter = threading.Thread(target=submission_worker, name="phase-submitter")
    finalizer = threading.Thread(target=finalization_worker, name="phase-finalizer")
    submitter.start()
    assert sbatch_entered.wait(timeout=5)
    finalizer.start()
    assert finalizer_started.wait(timeout=1)
    assert not finalizer_done.wait(timeout=0.2)
    release_sbatch.set()
    submitter.join(timeout=5)
    finalizer.join(timeout=5)

    assert not submitter.is_alive()
    assert not finalizer.is_alive()
    assert failures == []
    replay = PhaseAuthorityStore(authority_root).validate(fixture.runspec.phase_run_id)
    assert [type(event) for event in replay.events[1:]] == [
        PhaseSubmissionIntendedEvent,
        PhaseActionDispatchIntendedEvent,
        PhaseActionSubmittedEvent,
        PhaseFinalizedEvent,
    ]
    assert replay.submission is not None
    assert replay.submission.actions[0].job_id == "123456"
    assert replay.lifecycle.sealed


def test_registered_intervening_event_replays_before_terminal_finalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _accepted_inputs(tmp_path, monkeypatch)
    fixture = inputs["fixture"]
    authority_root = inputs["authority_root"]
    assert isinstance(fixture, LocalExecutionFixture)
    assert isinstance(authority_root, Path)
    store = PhaseAuthorityStore(
        authority_root,
        event_handlers=(
            PhaseAuthorityEventHandler(
                event_type="phase-submitted-test",
                loader=_registered_intervening_event_from_mapping,
                replay=_replay_registered_intervening_event,
            ),
        ),
    )
    intervening = RegisteredInterveningEvent(
        sequence=5,
        phase_run_id=fixture.runspec.phase_run_id,
        attempt_id=fixture.runspec.attempt_id,
    )
    events_path = authority_root / fixture.runspec.phase_run_id / "events"
    (events_path / "000005-phase-submitted-test.json").write_text(
        json.dumps(intervening.to_mapping(), indent=2, sort_keys=True) + "\n"
    )

    replay = store.validate(fixture.runspec.phase_run_id)
    assert not replay.lifecycle.sealed
    assert len(replay.events) == 5
    assert replay.events[-1] == intervening

    result = _finalize(inputs, authority_store=store)

    assert result.status == "accepted"
    assert {path.name for path in events_path.iterdir()} == {
        "000001-phase-materialized.json",
        "000002-phase-submission-intended.json",
        "000003-phase-action-dispatch-intended.json",
        "000004-phase-action-submitted.json",
        "000005-phase-submitted-test.json",
        "000006-phase-finalized.json",
    }
    accepted = store.validate(fixture.runspec.phase_run_id)
    assert accepted.lifecycle.sealed
    assert accepted.finalized_event is not None
    assert accepted.finalized_event.sequence == 6
    assert accepted.events[-2:] == (intervening, accepted.finalized_event)


def test_identical_retry_returns_receipt_but_divergent_retry_cannot_mutate_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _accepted_inputs(tmp_path, monkeypatch)
    first = _finalize(inputs)
    second = _finalize(inputs)
    assert second == first
    authority_root = inputs["authority_root"]
    assert isinstance(authority_root, Path)
    before = {
        path.relative_to(authority_root): path.read_bytes() for path in authority_root.rglob("*") if path.is_file()
    }
    scheduler = inputs["scheduler"]
    assert isinstance(scheduler, Path)
    changed = json.loads(scheduler.read_text())
    changed["job_id"] = "999999"
    scheduler.write_text(json.dumps(changed))

    with pytest.raises(ValueError, match=r"assigned Slurm job|differ"):
        _finalize(inputs)

    after = {
        path.relative_to(authority_root): path.read_bytes() for path in authority_root.rglob("*") if path.is_file()
    }
    assert after == before


@pytest.mark.parametrize(
    ("field", "value"),
    [("state", "FAILED"), ("exit_code", "1:0"), ("job_id", "not-a-job")],
)
def test_invalid_scheduler_attestation_cannot_receive_a_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    inputs = _accepted_inputs(tmp_path, monkeypatch)
    scheduler = inputs["scheduler"]
    assert isinstance(scheduler, Path)
    mapping = json.loads(scheduler.read_text())
    mapping[field] = value
    scheduler.write_text(json.dumps(mapping))
    fixture = inputs["fixture"]
    authority_root = inputs["authority_root"]
    assert isinstance(fixture, LocalExecutionFixture)
    assert isinstance(authority_root, Path)

    with pytest.raises(ValueError):
        _finalize(inputs)

    assert {path.name for path in (authority_root / fixture.runspec.phase_run_id / "events").iterdir()} == {
        "000001-phase-materialized.json",
        "000002-phase-submission-intended.json",
        "000003-phase-action-dispatch-intended.json",
        "000004-phase-action-submitted.json",
    }


def test_prepublication_fault_leaves_authority_byte_identical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _accepted_inputs(tmp_path, monkeypatch)
    authority_root = inputs["authority_root"]
    assert isinstance(authority_root, Path)
    fixture = inputs["fixture"]
    assert isinstance(fixture, LocalExecutionFixture)
    run_root = authority_root / fixture.runspec.phase_run_id
    before = {path.relative_to(run_root): path.read_bytes() for path in run_root.rglob("*") if path.is_file()}

    def fail(_temporary: Path) -> None:
        raise RuntimeError("injected event publication fault")

    with pytest.raises(RuntimeError, match="injected event publication fault"):
        _finalize(inputs, authority_store=PhaseAuthorityStore(authority_root, before_event_publish=fail))

    after = {path.relative_to(run_root): path.read_bytes() for path in run_root.rglob("*") if path.is_file()}
    assert after == before


@pytest.mark.parametrize("damage", ["gap", "unknown", "after-seal"])
def test_authority_event_discovery_rejects_gaps_unknown_files_and_post_seal_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    inputs = _accepted_inputs(tmp_path, monkeypatch)
    _finalize(inputs)
    fixture = inputs["fixture"]
    authority_root = inputs["authority_root"]
    assert isinstance(fixture, LocalExecutionFixture)
    assert isinstance(authority_root, Path)
    events = authority_root / fixture.runspec.phase_run_id / "events"
    finalized = events / "000005-phase-finalized.json"
    if damage == "gap":
        finalized.rename(events / "000006-phase-finalized.json")
    elif damage == "unknown":
        (events / "000006-unknown.json").write_text("{}\n")
    else:
        mapping = json.loads(finalized.read_text())
        mapping["sequence"] = 6
        (events / "000006-phase-finalized.json").write_text(json.dumps(mapping))

    with pytest.raises(ValueError):
        PhaseAuthorityStore(authority_root).validate(fixture.runspec.phase_run_id)


def test_phase_finalize_cli_emits_receipt_json_without_changing_legacy_run_help(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _accepted_inputs(tmp_path, monkeypatch)
    fixture = inputs["fixture"]
    assert isinstance(fixture, LocalExecutionFixture)
    runner = CliRunner()
    legacy_help = runner.invoke(cli, ["run", "--help"]).output
    result = runner.invoke(
        cli,
        [
            "phase",
            "finalize",
            fixture.runspec.phase_run_id,
            "--authority-root",
            str(inputs["authority_root"]),
            "--scheduler-evidence",
            str(inputs["scheduler"]),
            "--action-evidence",
            str(fixture.evidence_path),
            "--handoff",
            str(inputs["handoff"]),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "accepted"
    assert runner.invoke(cli, ["run", "--help"]).output == legacy_help


def test_finalization_warning_emitted_once_in_sequential_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The advisory warning fires exactly once in sequential operation (criterion #22)."""
    import logging as _logging

    inputs = _accepted_inputs(tmp_path, monkeypatch)
    with caplog.at_level(_logging.WARNING, logger="bspp.orchestration.control.phase_finalization"):
        first = _finalize(inputs)
        second = _finalize(inputs)
    assert second == first
    warning_records = [r for r in caplog.records if r.levelno == _logging.WARNING and "first-time" in r.getMessage()]
    assert len(warning_records) == 1, (
        f"expected exactly one first-time finalization warning, got {len(warning_records)}"
    )
