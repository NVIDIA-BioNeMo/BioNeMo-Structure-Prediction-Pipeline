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

"""Read-only validation of historical postprocessing V1 authority."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from bspp.orchestration.contract.postprocessing_acceptance_policy import (
    postprocessing_acceptance_policy_from_mapping,
)
from bspp.orchestration.contract.postprocessing_action_contract import (
    postprocessing_action_semantics_digest,
)
from bspp.orchestration.contract.postprocessing_plan import postprocessing_phase_plan_from_mapping
from bspp.orchestration.contract.postprocessing_retry_events import (
    PostprocessingAttemptRetriedPayload,
)
from bspp.orchestration.contract.postprocessing_runspec import read_postprocessing_phase_runspec_from_mapping
from bspp.orchestration.contract.postprocessing_runspec_v1 import HistoricalPostprocessingPhaseRunSpecV1
from bspp.orchestration.contract.postprocessing_submission_events import (
    PostprocessingSubmissionIntendedPayload,
)
from bspp.orchestration.contract.runspec import RunSpec
from bspp.orchestration.control.postprocessing_authority_store import (
    canonical_json_bytes,
    read_canonical_json,
    required_string,
    verify_bytes,
)
from bspp.orchestration.control.postprocessing_authority_validation import (
    legacy_runspec_from_bytes,
    projection_bytes_with_initial_fallback,
    read_postprocessing_events,
    replay_qualified_postprocessing_runtime,
    validate_postprocessing_phase_run_mapping,
    verified_regular_bytes,
    verify_optional_projection,
)
from bspp.orchestration.control.postprocessing_event_replay import replay_postprocessing_events
from bspp.orchestration.control.postprocessing_phase_rendering_v1 import (
    PostprocessingRenderInput,
    historical_postprocessing_renderer,
)
from bspp.orchestration.control.postprocessing_phase_types import (
    HistoricalPostprocessingAuthorityV1,
    PostprocessingInitialAttemptSnapshot,
    PostprocessingPhaseRunSnapshot,
)


def validate_historical_postprocessing_authority_v1(
    authority_root: Path,
    phase_run_id: str,
) -> HistoricalPostprocessingAuthorityV1:
    """Authenticate a complete V1 authority without making it executable."""
    authority_path = authority_root / phase_run_id
    plan = postprocessing_phase_plan_from_mapping(read_canonical_json(authority_path / "phase-plan.json"))
    run_mapping = read_canonical_json(authority_path / "phase-run.json")
    validate_postprocessing_phase_run_mapping(run_mapping, plan=plan, expected_phase_run_id=phase_run_id)
    initial_attempt_id = required_string(run_mapping, "current_attempt_id")
    initial_mapping = read_canonical_json(authority_path / "attempts" / initial_attempt_id / "phase-runspec.json")
    initial_runspec = read_postprocessing_phase_runspec_from_mapping(initial_mapping)
    if not isinstance(initial_runspec, HistoricalPostprocessingPhaseRunSpecV1):
        raise ValueError("stored postprocessing authority is not historical V1 authority")
    if (
        initial_runspec.phase_run_id != phase_run_id
        or initial_runspec.attempt_id != initial_attempt_id
        or initial_runspec.phase_plan_digest != plan.digest
    ):
        raise ValueError("stored historical postprocessing V1 RunSpec identity differs from authority")
    declared_attempt = cast("Mapping[str, object]", cast("list[object]", run_mapping["attempts"])[0])
    if (
        declared_attempt["phase_runspec_digest"] != initial_runspec.digest
        or declared_attempt["created_at"] != initial_runspec.materialized_at
        or run_mapping["created_at"] != initial_runspec.materialized_at
    ):
        raise ValueError("stored historical postprocessing V1 snapshot differs from its initial RunSpec")

    initial_projection = initial_runspec.payload.execution_projection
    initial_projection_path = authority_path / initial_projection.document_location
    initial_legacy_bytes = verified_regular_bytes(
        initial_projection_path,
        expected_sha256=initial_projection.document_sha256,
        expected_size=initial_projection.document_size_bytes,
        label="initial historical postprocessing V1 legacy RunSpec projection",
    )
    initial_legacy_runspec = legacy_runspec_from_bytes(
        initial_legacy_bytes,
        source_path=initial_projection_path,
        expected_sha256=initial_projection.document_sha256,
    )
    _validate_v1_action_semantics(initial_runspec, initial_legacy_runspec)
    events = read_postprocessing_events(authority_path)
    replay = replay_postprocessing_events(events, initial_runspec=initial_runspec)
    runspec = replay.runspec
    if not isinstance(runspec, HistoricalPostprocessingPhaseRunSpecV1):
        raise ValueError("historical postprocessing authority cannot change contract families during Retry")
    if runspec.phase_plan_digest != plan.digest:
        raise ValueError("current historical postprocessing V1 RunSpec does not bind the Phase Plan")

    projection = runspec.payload.execution_projection
    projection_path = authority_path / projection.document_location
    projection_complete = True
    if replay.embedded_legacy_runspec_bytes is None:
        projection_bytes = verified_regular_bytes(
            projection_path,
            expected_sha256=projection.document_sha256,
            expected_size=projection.document_size_bytes,
            label="historical postprocessing V1 legacy RunSpec projection",
        )
    else:
        projection_bytes = replay.embedded_legacy_runspec_bytes
        verify_bytes(
            projection_bytes,
            expected_sha256=projection.document_sha256,
            expected_size=projection.document_size_bytes,
            label="embedded historical postprocessing V1 Retry legacy RunSpec projection",
        )
        projection_complete = verify_optional_projection(projection_path, projection_bytes)
        runspec_projection = authority_path / "attempts" / runspec.attempt_id / "phase-runspec.json"
        projection_complete = (
            verify_optional_projection(runspec_projection, canonical_json_bytes(runspec.to_mapping()))
            and projection_complete
        )
    legacy_runspec = legacy_runspec_from_bytes(
        projection_bytes,
        source_path=projection_path,
        expected_sha256=projection.document_sha256,
    )
    _validate_v1_action_semantics(runspec, legacy_runspec)

    initial_policy_path = authority_path / initial_runspec.payload.acceptance_policy.location
    policy_bytes, policy_complete = projection_bytes_with_initial_fallback(
        authority_path / runspec.payload.acceptance_policy.location,
        initial_path=initial_policy_path,
        expected_sha256=runspec.payload.acceptance_policy.sha256,
        expected_size=runspec.payload.acceptance_policy.size_bytes,
        label="historical postprocessing V1 acceptance policy snapshot",
    )
    projection_complete = projection_complete and policy_complete
    policy_mapping = json.loads(policy_bytes)
    if not isinstance(policy_mapping, Mapping):
        raise TypeError("historical postprocessing V1 acceptance policy must be a JSON mapping")
    policy = postprocessing_acceptance_policy_from_mapping(policy_mapping)
    if policy.semantic_digest != runspec.payload.acceptance_policy.semantic_digest:
        raise ValueError("historical postprocessing V1 acceptance policy semantic digest differs")

    qualification_path = authority_path / runspec.payload.qualified_runtime.qualification_location
    if replay.embedded_runtime_qualification_bytes is None:
        qualification_bytes = verified_regular_bytes(
            qualification_path,
            expected_sha256=runspec.payload.qualified_runtime.qualification_sha256,
            expected_size=runspec.payload.qualified_runtime.qualification_size_bytes,
            label="historical postprocessing V1 Runtime Qualification",
        )
        qualification_complete = True
    else:
        qualification_bytes = replay.embedded_runtime_qualification_bytes
        verify_bytes(
            qualification_bytes,
            expected_sha256=runspec.payload.qualified_runtime.qualification_sha256,
            expected_size=runspec.payload.qualified_runtime.qualification_size_bytes,
            label="embedded historical postprocessing V1 Runtime Qualification",
        )
        qualification_complete = verify_optional_projection(qualification_path, qualification_bytes)
    projection_complete = projection_complete and qualification_complete
    replayed_runtime = replay_qualified_postprocessing_runtime(
        qualification_bytes,
        attempt_id=runspec.attempt_id,
        profile_name=runspec.cluster.profile_name,
    )
    if replayed_runtime != runspec.payload.qualified_runtime:
        raise ValueError("historical postprocessing V1 Runtime Qualification selection differs")

    phase_run = PostprocessingPhaseRunSnapshot(
        phase_run_id=phase_run_id,
        phase_plan_location=required_string(run_mapping, "phase_plan_location"),
        phase_plan_digest=required_string(run_mapping, "phase_plan_digest"),
        created_at=required_string(run_mapping, "created_at"),
        current_attempt_id=required_string(run_mapping, "current_attempt_id"),
        initial_attempt=PostprocessingInitialAttemptSnapshot(
            attempt_id=required_string(declared_attempt, "attempt_id"),
            ordinal=cast("int", declared_attempt["ordinal"]),
            phase_runspec_location=required_string(declared_attempt, "phase_runspec_location"),
            phase_runspec_digest=required_string(declared_attempt, "phase_runspec_digest"),
            created_at=required_string(declared_attempt, "created_at"),
        ),
    )
    authority = HistoricalPostprocessingAuthorityV1(
        authority_path=authority_path,
        phase_plan=plan,
        phase_run=phase_run,
        runspec=runspec,
        legacy_runspec=legacy_runspec,
        acceptance_policy=policy,
        legacy_runspec_bytes=projection_bytes,
        acceptance_policy_bytes=policy_bytes,
        runtime_qualification_bytes=qualification_bytes,
        events=events,
        status=replay.status,
        sealed=replay.sealed,
        current_attempt_projection_complete=projection_complete,
        submission_state=replay.submission_state,
        terminal_payloads=replay.terminal_payloads,
    )
    _validate_historical_submission_scripts(
        authority,
        initial_runspec=initial_runspec,
        initial_legacy_runspec=initial_legacy_runspec,
        initial_legacy_bytes=initial_legacy_bytes,
    )
    return authority


def _validate_v1_action_semantics(
    runspec: HistoricalPostprocessingPhaseRunSpecV1,
    legacy_runspec: RunSpec,
) -> None:
    """Recompute the frozen V1 preimage from exact persisted workflow modes."""
    if legacy_runspec.workflow is None:
        raise ValueError("historical postprocessing V1 legacy RunSpec must contain a workflow")
    workflow_steps = {step.name: step for step in legacy_runspec.workflow.steps}
    step_modes: dict[str, str | None] = {}
    for action in runspec.payload.actions:
        if action.step_name == "acceptance-adjudication":
            step_modes[action.step_name] = None
            continue
        step = workflow_steps.get(action.step_name)
        if step is None or not step.run:
            raise ValueError("historical postprocessing V1 actions differ from stored active workflow steps")
        step_modes[action.step_name] = step.mode
    expected = postprocessing_action_semantics_digest(
        runspec.payload.actions,
        step_modes=step_modes,
        scientific_identity=runspec.payload.scientific_identity,
    )
    if runspec.payload.action_semantics_digest != expected:
        raise ValueError("historical postprocessing V1 action semantics digest differs from its stored preimage")


def _validate_historical_submission_scripts(
    authority: HistoricalPostprocessingAuthorityV1,
    *,
    initial_runspec: HistoricalPostprocessingPhaseRunSpecV1,
    initial_legacy_runspec: RunSpec,
    initial_legacy_bytes: bytes,
) -> None:
    runspec = initial_runspec
    legacy_runspec = initial_legacy_runspec
    legacy_bytes = initial_legacy_bytes
    for event in authority.events:
        payload = event.payload
        if isinstance(payload, PostprocessingAttemptRetriedPayload):
            successor = payload.successor_phase_runspec
            if not isinstance(successor, HistoricalPostprocessingPhaseRunSpecV1):
                raise ValueError("historical postprocessing authority cannot change contract families during Retry")
            runspec = successor
            legacy_bytes = payload.successor_legacy_runspec_yaml.encode()
            legacy_runspec = legacy_runspec_from_bytes(
                legacy_bytes,
                source_path=authority.authority_path / runspec.payload.execution_projection.document_location,
                expected_sha256=runspec.payload.execution_projection.document_sha256,
            )
            continue
        if not isinstance(payload, PostprocessingSubmissionIntendedPayload):
            continue
        render_input = PostprocessingRenderInput(runspec=runspec, legacy_runspec=legacy_runspec)
        versions = {plan.renderer_contract_version for plan in payload.actions}
        if len(versions) != 1:
            raise ValueError("historical postprocessing V1 submission mixes renderer contract versions")
        renderer = historical_postprocessing_renderer(next(iter(versions)))
        scripts = {action.action_id: renderer(render_input, action) for action in runspec.payload.actions}
        expected_submission_id = _submission_id(authority, runspec, scripts)
        if payload.submission_id != expected_submission_id or len(payload.actions) != len(runspec.payload.actions):
            raise ValueError("historical postprocessing V1 submission differs from frozen renderer authority")
        for action, plan in zip(runspec.payload.actions, payload.actions, strict=True):
            expected_sha256 = hashlib.sha256(scripts[action.action_id].encode()).hexdigest()
            if plan.script_sha256 != expected_sha256:
                raise ValueError("historical postprocessing V1 script hash differs from frozen renderer output")


def _submission_id(
    authority: HistoricalPostprocessingAuthorityV1,
    runspec: HistoricalPostprocessingPhaseRunSpecV1,
    scripts: Mapping[str, str],
) -> str:
    from bspp.orchestration.control.postprocessing_authority_store import mapping_digest

    return "postprocessing-submission-" + mapping_digest(
        {
            "schema_version": 1,
            "phase_run_id": authority.phase_run_id,
            "attempt_id": runspec.attempt_id,
            "phase_runspec_digest": runspec.digest,
            "actions": [
                {
                    "action_id": action.action_id,
                    "script_sha256": hashlib.sha256(scripts[action.action_id].encode()).hexdigest(),
                    "renderer_contract_version": 1,
                }
                for action in runspec.payload.actions
            ],
        }
    )


__all__ = ["validate_historical_postprocessing_authority_v1"]
