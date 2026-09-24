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

"""Strict validation and event replay for postprocessing V2 Phase authority."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import cast

import yaml

from bspp.orchestration.contract.postprocessing_acceptance_policy import (
    postprocessing_acceptance_policy_from_mapping,
)
from bspp.orchestration.contract.postprocessing_plan import (
    postprocessing_phase_plan_from_mapping,
)
from bspp.orchestration.contract.postprocessing_retry_events import (
    PostprocessingAttemptRetriedPayload,
)
from bspp.orchestration.contract.postprocessing_runspec import (
    ExecutablePostprocessingPhaseRunSpec,
    postprocessing_phase_runspec_from_mapping,
)
from bspp.orchestration.contract.postprocessing_runspec_v2 import (
    PostprocessingPhaseRunSpec,
)
from bspp.orchestration.contract.postprocessing_runspec_v3 import PostprocessingPhaseRunSpecV3
from bspp.orchestration.contract.postprocessing_submission_events import (
    PostprocessingSubmissionIntendedPayload,
)
from bspp.orchestration.contract.runspec import RunSpec, runspec_from_mapping
from bspp.orchestration.contract.runspec_validation import validate_active_workflow_static
from bspp.orchestration.control.plan import render_runspec_yaml
from bspp.orchestration.control.postprocessing_authority_store import (
    canonical_json_bytes as _canonical_json_bytes,
)
from bspp.orchestration.control.postprocessing_authority_store import (
    read_canonical_json as _read_canonical_json,
)
from bspp.orchestration.control.postprocessing_authority_store import (
    required_string as _required_string,
)
from bspp.orchestration.control.postprocessing_authority_store import (
    verify_bytes as _verify_bytes,
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
from bspp.orchestration.control.postprocessing_phase_lifecycle import (
    _postprocessing_submission_id,
    _rendered_action_scripts,
)
from bspp.orchestration.control.postprocessing_phase_types import (
    PostprocessingAuthority,
    PostprocessingInitialAttemptSnapshot,
    PostprocessingPhaseRunSnapshot,
)
from bspp.orchestration.control.postprocessing_renderer3_compatibility import (
    render_authenticated_renderer3_scripts,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_JOB_ID = re.compile(r"[0-9]+")


def validate_postprocessing_authority(authority_root: Path, phase_run_id: str) -> PostprocessingAuthority:
    """Strictly validate immutable projections and replay postprocessing events."""
    authority_path = authority_root / phase_run_id
    plan_mapping = _read_canonical_json(authority_path / "phase-plan.json")
    plan = postprocessing_phase_plan_from_mapping(plan_mapping)
    run_mapping = _read_canonical_json(authority_path / "phase-run.json")
    validate_postprocessing_phase_run_mapping(run_mapping, plan=plan, expected_phase_run_id=phase_run_id)
    initial_attempt_id = _required_string(run_mapping, "current_attempt_id")
    initial_runspec_mapping = _read_canonical_json(
        authority_path / "attempts" / initial_attempt_id / "phase-runspec.json"
    )
    initial_runspec = postprocessing_phase_runspec_from_mapping(initial_runspec_mapping)
    if initial_runspec.phase_run_id != phase_run_id or initial_runspec.attempt_id != initial_attempt_id:
        raise ValueError("stored postprocessing Phase RunSpec identity does not match authority")
    if initial_runspec.phase_plan_digest != plan.digest:
        raise ValueError("stored postprocessing Phase RunSpec does not bind the Phase Plan")
    declared_attempt = cast("Mapping[str, object]", cast("list[object]", run_mapping["attempts"])[0])
    if (
        declared_attempt["phase_runspec_digest"] != initial_runspec.digest
        or declared_attempt["created_at"] != initial_runspec.materialized_at
        or run_mapping["created_at"] != initial_runspec.materialized_at
    ):
        raise ValueError("stored postprocessing Phase Run snapshot differs from its initial RunSpec")
    initial_projection_path = authority_path / initial_runspec.payload.execution_projection.document_location
    initial_legacy_bytes = verified_regular_bytes(
        initial_projection_path,
        expected_sha256=initial_runspec.payload.execution_projection.document_sha256,
        expected_size=initial_runspec.payload.execution_projection.document_size_bytes,
        label="initial stored legacy RunSpec projection",
    )
    initial_legacy_runspec = legacy_runspec_from_bytes(
        initial_legacy_bytes,
        source_path=initial_projection_path,
        expected_sha256=initial_runspec.payload.execution_projection.document_sha256,
    )
    events = read_postprocessing_events(authority_path)
    replay = replay_postprocessing_events(
        events,
        initial_runspec=initial_runspec,
    )
    status = replay.status
    sealed = replay.sealed
    runspec = replay.runspec
    if not isinstance(runspec, (PostprocessingPhaseRunSpec, PostprocessingPhaseRunSpecV3)):
        raise ValueError("postprocessing authority cannot change RunSpec contract families during Retry")
    embedded_legacy_bytes = replay.embedded_legacy_runspec_bytes
    embedded_qualification_bytes = replay.embedded_runtime_qualification_bytes
    if runspec.phase_plan_digest != plan.digest:
        raise ValueError("current postprocessing Phase RunSpec does not bind the Phase Plan")

    projection_path = authority_path / runspec.payload.execution_projection.document_location
    projection_complete = True
    if embedded_legacy_bytes is None:
        projection_bytes = verified_regular_bytes(
            projection_path,
            expected_sha256=runspec.payload.execution_projection.document_sha256,
            expected_size=runspec.payload.execution_projection.document_size_bytes,
            label="stored legacy RunSpec projection",
        )
    else:
        projection_bytes = embedded_legacy_bytes
        _verify_bytes(
            projection_bytes,
            expected_sha256=runspec.payload.execution_projection.document_sha256,
            expected_size=runspec.payload.execution_projection.document_size_bytes,
            label="embedded Retry legacy RunSpec projection",
        )
        projection_complete = verify_optional_projection(projection_path, projection_bytes)
        runspec_projection = authority_path / "attempts" / runspec.attempt_id / "phase-runspec.json"
        projection_complete = (
            verify_optional_projection(runspec_projection, _canonical_json_bytes(runspec.to_mapping()))
            and projection_complete
        )
    legacy_mapping = yaml.safe_load(projection_bytes)
    if not isinstance(legacy_mapping, Mapping):
        raise TypeError("stored legacy RunSpec projection must be a YAML mapping")
    legacy_runspec = runspec_from_mapping(
        legacy_mapping,
        source_path=projection_path,
        source_hash=runspec.payload.execution_projection.document_sha256,
    )
    validate_active_workflow_static(legacy_runspec).raise_if_invalid()
    if render_runspec_yaml(legacy_mapping).encode() != projection_bytes:
        raise ValueError("stored legacy RunSpec projection bytes are not the once-rendered canonical YAML")

    initial_policy_path = authority_path / initial_runspec.payload.acceptance_policy.location
    policy_bytes, policy_complete = projection_bytes_with_initial_fallback(
        authority_path / runspec.payload.acceptance_policy.location,
        initial_path=initial_policy_path,
        expected_sha256=runspec.payload.acceptance_policy.sha256,
        expected_size=runspec.payload.acceptance_policy.size_bytes,
        label="stored acceptance policy snapshot",
    )
    projection_complete = projection_complete and policy_complete
    policy_mapping = json.loads(policy_bytes)
    if not isinstance(policy_mapping, Mapping):
        raise TypeError("stored acceptance policy snapshot must be a JSON mapping")
    policy = postprocessing_acceptance_policy_from_mapping(policy_mapping)
    if policy.semantic_digest != runspec.payload.acceptance_policy.semantic_digest:
        raise ValueError("stored acceptance policy semantic digest differs from the RunSpec")

    qualification_path = authority_path / runspec.payload.qualified_runtime.qualification_location
    if embedded_qualification_bytes is None:
        qualification_bytes = verified_regular_bytes(
            qualification_path,
            expected_sha256=runspec.payload.qualified_runtime.qualification_sha256,
            expected_size=runspec.payload.qualified_runtime.qualification_size_bytes,
            label="stored postprocessing Runtime Qualification",
        )
        qualification_complete = True
    else:
        qualification_bytes = embedded_qualification_bytes
        _verify_bytes(
            qualification_bytes,
            expected_sha256=runspec.payload.qualified_runtime.qualification_sha256,
            expected_size=runspec.payload.qualified_runtime.qualification_size_bytes,
            label="embedded Retry Runtime Qualification",
        )
        qualification_complete = verify_optional_projection(qualification_path, qualification_bytes)
    projection_complete = projection_complete and qualification_complete
    replayed_runtime = replay_qualified_postprocessing_runtime(
        qualification_bytes,
        attempt_id=runspec.attempt_id,
        profile_name=runspec.cluster.profile_name,
    )
    if replayed_runtime != runspec.payload.qualified_runtime:
        raise ValueError("stored postprocessing Runtime Qualification selection differs from the RunSpec")

    phase_run = PostprocessingPhaseRunSnapshot(
        phase_run_id=phase_run_id,
        phase_plan_location=_required_string(run_mapping, "phase_plan_location"),
        phase_plan_digest=_required_string(run_mapping, "phase_plan_digest"),
        created_at=_required_string(run_mapping, "created_at"),
        current_attempt_id=_required_string(run_mapping, "current_attempt_id"),
        initial_attempt=PostprocessingInitialAttemptSnapshot(
            attempt_id=_required_string(declared_attempt, "attempt_id"),
            ordinal=cast("int", declared_attempt["ordinal"]),
            phase_runspec_location=_required_string(declared_attempt, "phase_runspec_location"),
            phase_runspec_digest=_required_string(declared_attempt, "phase_runspec_digest"),
            created_at=_required_string(declared_attempt, "created_at"),
        ),
    )
    authority = PostprocessingAuthority(
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
        status=status,
        sealed=sealed,
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


def _validate_historical_submission_scripts(
    authority: PostprocessingAuthority,
    *,
    initial_runspec: ExecutablePostprocessingPhaseRunSpec,
    initial_legacy_runspec: RunSpec,
    initial_legacy_bytes: bytes,
) -> None:
    """Authenticate every historical submission against the frozen v1 renderer."""
    runspec = initial_runspec
    legacy_runspec = initial_legacy_runspec
    legacy_bytes = initial_legacy_bytes
    for event in authority.events:
        payload = event.payload
        if isinstance(payload, PostprocessingAttemptRetriedPayload):
            successor = payload.successor_phase_runspec
            if type(successor) is not type(runspec):
                raise ValueError("postprocessing authority cannot change RunSpec contract families during Retry")
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
        historical = replace(
            authority,
            runspec=runspec,
            legacy_runspec=legacy_runspec,
            legacy_runspec_bytes=legacy_bytes,
        )
        versions = {plan.renderer_contract_version for plan in payload.actions}
        if len(versions) != 1:
            raise ValueError("postprocessing V2 submission mixes renderer contract versions")
        renderer_contract_version = next(iter(versions))
        scripts = (
            render_authenticated_renderer3_scripts(historical, payload)
            if isinstance(runspec, PostprocessingPhaseRunSpecV3) and renderer_contract_version == 3
            else _rendered_action_scripts(historical, renderer_contract_version=renderer_contract_version)
        )
        if payload.submission_id != _postprocessing_submission_id(
            historical, scripts, renderer_contract_version=renderer_contract_version
        ):
            raise ValueError("postprocessing submission id differs from the frozen rendered scripts")
        if len(payload.actions) != len(runspec.payload.actions):
            raise ValueError("postprocessing submission script plans differ from frozen actions")
        for action, plan in zip(runspec.payload.actions, payload.actions, strict=True):
            expected_sha256 = hashlib.sha256(scripts[action.action_id].encode()).hexdigest()
            if plan.renderer_contract_version != renderer_contract_version or plan.script_sha256 != expected_sha256:
                raise ValueError("postprocessing submission script hash differs from frozen renderer output")


__all__ = [
    "PostprocessingAuthority",
    "validate_postprocessing_authority",
]
