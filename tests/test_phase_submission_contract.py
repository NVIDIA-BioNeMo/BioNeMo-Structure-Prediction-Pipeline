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

from __future__ import annotations

import hashlib
from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace

import pytest

from bspp.orchestration.contract.phase_submission import (
    PhaseActionDispatchIntendedEvent,
    PhaseActionDispatchIntendedPayload,
    PhaseActionDispatchRejectedEvent,
    PhaseActionDispatchRejectedPayload,
    PhaseActionSubmissionPlan,
    PhaseActionSubmissionView,
    PhaseActionSubmittedEvent,
    PhaseActionSubmittedPayload,
    PhaseSubmissionIntendedEvent,
    PhaseSubmissionIntendedPayload,
    PhaseSubmissionLifecycleView,
    phase_action_dispatch_intended_event_from_mapping,
    phase_action_dispatch_rejected_event_from_mapping,
    phase_action_scheduler_correlation_token,
    phase_action_submission_identity_mapping,
    phase_action_submission_plan_from_mapping,
    phase_action_submission_view_from_mapping,
    phase_action_submitted_event_from_mapping,
    phase_submission_id,
    phase_submission_intended_event_from_mapping,
    phase_submission_lifecycle_view_from_mapping,
)

RUN_ID = "phase-run-0123456789abcdef0123456789abcdef"
ATTEMPT_ID = "attempt-0001"
ACTION_ID = "preprocessing-chunk-000000"
RUNSPEC_DIGEST = "1" * 64
RUNSPEC_DOCUMENT_SHA256 = "2" * 64
QUALIFICATION_TUPLE_ID = "3" * 64
RUNTIME_ACTION_DIGEST = "4" * 64
OCCURRED_AT = "2026-08-20T12:34:56.123456Z"


def _identity(*, action_id: str = ACTION_ID, dependency_action_ids: tuple[str, ...] = ()) -> dict[str, object]:
    return phase_action_submission_identity_mapping(
        action_id=action_id,
        runtime_action_digest=RUNTIME_ACTION_DIGEST,
        dependency_action_ids=dependency_action_ids,
        cluster_script_path=f"/cluster/staging/actions/{action_id}.sbatch",
        action_evidence_path=f"/cluster/output/{action_id}/action-evidence.json",
        handoff_path=f"/cluster/output/{action_id}/handoff",
    )


def _submission_id(*, actions: tuple[dict[str, object], ...] | None = None) -> str:
    return phase_submission_id(
        phase_run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        phase_runspec_digest=RUNSPEC_DIGEST,
        phase_runspec_document_sha256=RUNSPEC_DOCUMENT_SHA256,
        qualification_tuple_id=QUALIFICATION_TUPLE_ID,
        actions=actions or (_identity(),),
    )


def _plan() -> PhaseActionSubmissionPlan:
    submission_id = _submission_id()
    token = phase_action_scheduler_correlation_token(submission_id, ACTION_ID)
    script = "\n".join(
        (
            "#!/usr/bin/env bash",
            f"#SBATCH --job-name={token}",
            f"#SBATCH --comment={token}",
            "set -euo pipefail",
            "",
        )
    )
    return PhaseActionSubmissionPlan(
        action_id=ACTION_ID,
        runtime_action_digest=RUNTIME_ACTION_DIGEST,
        dependency_action_ids=(),
        cluster_script_path=f"/cluster/staging/actions/{ACTION_ID}.sbatch",
        script_sha256=hashlib.sha256(script.encode()).hexdigest(),
        script_body=script,
        job_name=token,
        scheduler_correlation_token=token,
        action_evidence_path=f"/cluster/output/{ACTION_ID}/action-evidence.json",
        handoff_path=f"/cluster/output/{ACTION_ID}/handoff",
    )


def _intended_event() -> PhaseSubmissionIntendedEvent:
    return PhaseSubmissionIntendedEvent(
        sequence=2,
        phase_run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        occurred_at=OCCURRED_AT,
        payload=PhaseSubmissionIntendedPayload(
            submission_id=_submission_id(),
            phase_runspec_location=f"attempts/{ATTEMPT_ID}/phase-runspec.json",
            phase_runspec_digest=RUNSPEC_DIGEST,
            phase_runspec_document_sha256=RUNSPEC_DOCUMENT_SHA256,
            qualification_tuple_id=QUALIFICATION_TUPLE_ID,
            actions=(_plan(),),
        ),
    )


def _dispatch_payload() -> PhaseActionDispatchIntendedPayload:
    plan = _plan()
    return PhaseActionDispatchIntendedPayload(
        submission_id=_submission_id(),
        action_id=ACTION_ID,
        script_sha256=plan.script_sha256,
        scheduler_correlation_token=plan.scheduler_correlation_token,
        dependency_job_ids=(),
    )


def test_submission_id_preimage_is_explicit_non_cyclic_and_ordered() -> None:
    submission_id = _submission_id()
    assert submission_id.startswith("phase-submission-")
    assert len(submission_id) == len("phase-submission-") + 64

    token = phase_action_scheduler_correlation_token(submission_id, ACTION_ID)
    assert token.startswith("bspp-phase-")
    assert token == phase_action_scheduler_correlation_token(submission_id, ACTION_ID)
    assert len(token) <= 128

    changed_path = _identity()
    changed_path["handoff_path"] = "/cluster/output/different/handoff"
    assert _submission_id(actions=(changed_path,)) != submission_id

    second = _identity(action_id="preprocessing-chunk-000001")
    assert _submission_id(actions=(_identity(), second)) != _submission_id(actions=(second, _identity()))


def test_intended_event_and_plan_strict_round_trip() -> None:
    event = _intended_event()
    assert phase_submission_intended_event_from_mapping(event.to_mapping()) == event
    assert phase_action_submission_plan_from_mapping(_plan().to_mapping()) == _plan()
    assert event.payload.actions[0].job_name == event.payload.actions[0].scheduler_correlation_token


def test_dispatch_outcome_events_strict_round_trip() -> None:
    intended = PhaseActionDispatchIntendedEvent(
        sequence=3,
        phase_run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        occurred_at=OCCURRED_AT,
        payload=_dispatch_payload(),
    )
    submitted = PhaseActionSubmittedEvent(
        sequence=4,
        phase_run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        occurred_at=OCCURRED_AT,
        payload=PhaseActionSubmittedPayload(
            **_dispatch_payload().__dict__,
            job_id="12345",
            sbatch_argv=("sbatch", "--parsable", _plan().cluster_script_path),
        ),
    )
    rejected = PhaseActionDispatchRejectedEvent(
        sequence=4,
        phase_run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        occurred_at=OCCURRED_AT,
        payload=PhaseActionDispatchRejectedPayload(
            **_dispatch_payload().__dict__,
            sbatch_argv=("sbatch", "--parsable", _plan().cluster_script_path),
            return_code=1,
            stdout="",
            stderr="invalid account\n",
        ),
    )

    assert phase_action_dispatch_intended_event_from_mapping(intended.to_mapping()) == intended
    assert phase_action_submitted_event_from_mapping(submitted.to_mapping()) == submitted
    assert phase_action_dispatch_rejected_event_from_mapping(rejected.to_mapping()) == rejected


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda mapping: mapping.update({"future": True}), "Unknown PhaseSubmissionIntendedEvent"),
        (lambda mapping: mapping.pop("schema_version"), "missing explicit schema_version"),
        (lambda mapping: mapping.update({"schema_version": None}), "must be declared explicitly"),
        (lambda mapping: mapping.update({"schema_version": 999}), "Unsupported"),
        (lambda mapping: mapping.update({"sequence": True}), "sequence must be an integer"),
        (lambda mapping: mapping.update({"occurred_at": "2026-08-20"}), "explicit UTC timestamp"),
        (lambda mapping: mapping.update({"event_type": "phase-finalized"}), "unsupported"),
    ),
)
def test_intended_event_loader_rejects_envelope_schema_drift(
    mutation: Callable[[dict[str, object]], object],
    message: str,
) -> None:
    mapping = deepcopy(_intended_event().to_mapping())
    mutation(mapping)
    with pytest.raises(ValueError, match=message):
        phase_submission_intended_event_from_mapping(mapping)


def test_intent_rejects_canonical_id_token_and_runspec_envelope_drift() -> None:
    event = _intended_event()
    with pytest.raises(ValueError, match="canonical event-bound content"):
        replace(event, payload=replace(event.payload, phase_runspec_document_sha256="5" * 64))
    with pytest.raises(ValueError, match="scheduler identity"):
        replace(event.payload, submission_id=f"phase-submission-{'6' * 64}")
    with pytest.raises(ValueError, match="RunSpec location"):
        replace(event, attempt_id="attempt-0002")


def test_plan_rejects_script_hash_directive_and_scheduler_identity_drift() -> None:
    plan = _plan()
    with pytest.raises(ValueError, match="exact UTF-8 body"):
        replace(plan, script_body=f"{plan.script_body}# drift\n")
    with pytest.raises(ValueError, match="exactly one"):
        changed = plan.script_body.replace("#SBATCH --comment=", "# comment was ")
        replace(plan, script_body=changed, script_sha256=hashlib.sha256(changed.encode()).hexdigest())
    with pytest.raises(ValueError, match="must equal"):
        replace(plan, job_name="different-safe-name")
    with pytest.raises(ValueError, match="safe absolute POSIX"):
        replace(plan, cluster_script_path="../escape.sbatch")


def test_action_graph_identity_rejects_missing_dependencies_duplicates_and_cycles() -> None:
    parent = _identity()
    child = _identity(
        action_id="preprocessing-chunk-000001",
        dependency_action_ids=(ACTION_ID,),
    )
    assert _submission_id(actions=(child, parent)).startswith("phase-submission-")

    missing = _identity(
        action_id="preprocessing-chunk-000001",
        dependency_action_ids=("preprocessing-chunk-999999",),
    )
    with pytest.raises(ValueError, match="declared actions"):
        _submission_id(actions=(parent, missing))
    with pytest.raises(ValueError, match="unique"):
        _submission_id(actions=(parent, parent))

    cycle_a = _identity(dependency_action_ids=("preprocessing-chunk-000001",))
    cycle_b = _identity(
        action_id="preprocessing-chunk-000001",
        dependency_action_ids=(ACTION_ID,),
    )
    with pytest.raises(ValueError, match="acyclic"):
        _submission_id(actions=(cycle_a, cycle_b))


def test_dispatch_contracts_bind_exact_submission_action_and_command_result() -> None:
    dispatch = _dispatch_payload()
    with pytest.raises(ValueError, match="does not match"):
        replace(dispatch, action_id="preprocessing-chunk-000001")
    with pytest.raises(ValueError, match="numeric"):
        PhaseActionSubmittedPayload(**dispatch.__dict__, job_id="123.batch", sbatch_argv=("sbatch",))
    with pytest.raises(ValueError, match="nonzero"):
        PhaseActionDispatchRejectedPayload(
            **dispatch.__dict__,
            sbatch_argv=("sbatch",),
            return_code=0,
            stdout="",
            stderr="",
        )
    with pytest.raises(ValueError, match="unique"):
        replace(dispatch, dependency_job_ids=("123", "123"))


def test_replay_views_round_trip_and_enforce_derived_status() -> None:
    intended = _intended_event()
    planned_action = PhaseActionSubmissionView(plan=_plan(), status="planned")
    submitting = PhaseSubmissionLifecycleView(
        phase_run_id=RUN_ID,
        attempt_id=ATTEMPT_ID,
        submission_id=intended.payload.submission_id,
        phase_runspec_location=intended.payload.phase_runspec_location,
        phase_runspec_digest=intended.payload.phase_runspec_digest,
        phase_runspec_document_sha256=intended.payload.phase_runspec_document_sha256,
        qualification_tuple_id=intended.payload.qualification_tuple_id,
        actions=(planned_action,),
        status="submitting",
    )
    assert phase_action_submission_view_from_mapping(planned_action.to_mapping()) == planned_action
    assert phase_submission_lifecycle_view_from_mapping(submitting.to_mapping()) == submitting

    submitted_action = PhaseActionSubmissionView(
        plan=_plan(),
        status="submitted",
        job_id="12345",
        sbatch_argv=("sbatch", "--parsable", _plan().cluster_script_path),
    )
    submitted = replace(submitting, actions=(submitted_action,), status="submitted")
    assert phase_submission_lifecycle_view_from_mapping(submitted.to_mapping()) == submitted
    with pytest.raises(ValueError, match="does not match"):
        replace(submitting, status="submitted")


def test_strict_nested_loaders_reject_unknown_missing_and_wrong_collection_types() -> None:
    mapping = deepcopy(_intended_event().to_mapping())
    payload = mapping["payload"]
    assert isinstance(payload, dict)
    actions = payload["actions"]
    assert isinstance(actions, list)
    action = actions[0]
    assert isinstance(action, dict)
    action["unknown"] = "drift"
    with pytest.raises(ValueError, match="Unknown PhaseActionSubmissionPlan"):
        phase_submission_intended_event_from_mapping(mapping)

    mapping = deepcopy(_intended_event().to_mapping())
    payload = mapping["payload"]
    assert isinstance(payload, dict)
    payload["actions"] = "not-a-list"
    with pytest.raises(ValueError, match="actions must be a list"):
        phase_submission_intended_event_from_mapping(mapping)

    mapping = deepcopy(_intended_event().to_mapping())
    payload = mapping["payload"]
    assert isinstance(payload, dict)
    payload.pop("phase_runspec_digest")
    with pytest.raises(ValueError, match="phase_runspec_digest must be"):
        phase_submission_intended_event_from_mapping(mapping)
