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

"""Strict immutable Phase Retry contract tests."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from bspp.orchestration.contract.phase import PhasePlan, PhaseRunSpec
from bspp.orchestration.contract.phase_retry import (
    PhaseAttemptRetriedEvent,
    PhaseAttemptRetriedPayload,
    compare_retry_invariants,
    phase_attempt_retried_event_from_mapping,
    phase_input_set_identity_digest,
    phase_retry_id,
    phase_scientific_identity_digest,
)
from bspp.orchestration.contract.phase_state import PhaseAttempt, phase_run_from_mapping
from bspp.orchestration.control.phase_authority import PhaseAuthorityStore
from bspp.orchestration.runtime.preprocessing.commands import plan_preprocessing_chunk_execution
from tests.test_phase_resume import _materialized_authority


def _contract_fixture(tmp_path: Path) -> tuple[PhaseAttemptRetriedEvent, PhasePlan, PhaseRunSpec]:
    authority_root, phase_run_id = _materialized_authority(tmp_path / "authority")
    authority = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    predecessor = authority.phase_runspec
    plan = authority.phase_plan
    successor = replace(
        predecessor,
        attempt_id="attempt-0002",
        materialized_at="2026-08-20T13:00:00.000000Z",
        payload=replace(
            predecessor.payload,
            database=replace(
                predecessor.payload.database,
                source_manifest_projection="attempts/attempt-0002/database-source-manifest.json",
            ),
        ),
    )
    input_digest = phase_input_set_identity_digest(plan)
    scientific_digest = phase_scientific_identity_digest(plan)
    attempt = PhaseAttempt(
        attempt_id="attempt-0002",
        ordinal=2,
        phase_runspec_location="attempts/attempt-0002/phase-runspec.json",
        phase_runspec_digest=successor.digest,
        created_at=successor.materialized_at,
    )
    retry_id = phase_retry_id(
        phase_run_id=successor.phase_run_id,
        predecessor_attempt_id="attempt-0001",
        successor_attempt_id="attempt-0002",
        predecessor_outcome="failed",
        predecessor_phase_runspec_digest=predecessor.digest,
        phase_plan_digest=plan.digest,
        input_set_identity_digest=input_digest,
        scientific_identity_digest=scientific_digest,
        selected_cluster_profile=successor.cluster.profile_name,
        successor_phase_runspec_digest=successor.digest,
    )
    event = PhaseAttemptRetriedEvent(
        sequence=5,
        phase_run_id=successor.phase_run_id,
        attempt_id="attempt-0002",
        occurred_at=successor.materialized_at,
        payload=PhaseAttemptRetriedPayload(
            retry_id=retry_id,
            predecessor_attempt_id="attempt-0001",
            predecessor_phase_runspec_digest=predecessor.digest,
            predecessor_outcome="failed",
            phase_plan_digest=plan.digest,
            input_set_identity_digest=input_digest,
            scientific_identity_digest=scientific_digest,
            selected_cluster_profile=successor.cluster.profile_name,
            successor_attempt=attempt,
            successor_phase_runspec=successor,
        ),
    )
    return event, plan, predecessor


def test_retry_event_round_trips_with_deterministic_identity(tmp_path: Path) -> None:
    event, _plan, _predecessor = _contract_fixture(tmp_path)
    assert phase_attempt_retried_event_from_mapping(event.to_mapping()) == event
    assert event.payload.retry_id == phase_attempt_retried_event_from_mapping(event.to_mapping()).payload.retry_id


def test_initial_phase_run_rejects_successor_attempt_while_retry_attempt_remains_valid(tmp_path: Path) -> None:
    event, _plan, _predecessor = _contract_fixture(tmp_path)
    authority_root, phase_run_id = _materialized_authority(tmp_path / "initial")
    initial = PhaseAuthorityStore(authority_root).validate(phase_run_id).phase_run
    successor = event.payload.successor_attempt
    assert successor.attempt_id == "attempt-0002"

    with pytest.raises(ValueError, match="initial Phase Run"):
        replace(initial, current_attempt_id=successor.attempt_id, attempts=(successor,))

    mapping = initial.to_mapping()
    mapping["current_attempt_id"] = successor.attempt_id
    mapping["attempts"] = [successor.to_mapping()]
    with pytest.raises(ValueError, match="initial Phase Run"):
        phase_run_from_mapping(mapping)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: data.update({"unknown": True}),
        lambda data: data["payload"].pop("schema_version"),
        lambda data: data["payload"].update({"predecessor_outcome": "running"}),
        lambda data: data["payload"]["successor_attempt"].update({"ordinal": 3}),
        lambda data: data["payload"]["successor_phase_runspec"].update({"attempt_id": "attempt-0003"}),
    ],
)
def test_retry_event_loader_rejects_unknown_missing_and_drifted_bindings(tmp_path: Path, mutation: object) -> None:
    event, _plan, _predecessor = _contract_fixture(tmp_path)
    mapping = deepcopy(event.to_mapping())
    assert callable(mutation)
    mutation(mapping)
    with pytest.raises(ValueError):
        phase_attempt_retried_event_from_mapping(mapping)


@pytest.mark.parametrize("predecessor_attempt_id", ["attempt-one", "attempt-0000", "attempt-0002"])
def test_retry_payload_rejects_malformed_boundary_and_noncontiguous_predecessor_ids(
    tmp_path: Path,
    predecessor_attempt_id: str,
) -> None:
    event, _plan, _predecessor = _contract_fixture(tmp_path)
    payload = event.payload
    retry_id = phase_retry_id(
        phase_run_id=event.phase_run_id,
        predecessor_attempt_id=predecessor_attempt_id,
        successor_attempt_id=payload.successor_attempt.attempt_id,
        predecessor_outcome=payload.predecessor_outcome,
        predecessor_phase_runspec_digest=payload.predecessor_phase_runspec_digest,
        phase_plan_digest=payload.phase_plan_digest,
        input_set_identity_digest=payload.input_set_identity_digest,
        scientific_identity_digest=payload.scientific_identity_digest,
        selected_cluster_profile=payload.selected_cluster_profile,
        successor_phase_runspec_digest=payload.successor_phase_runspec.digest,
    )
    with pytest.raises(ValueError, match="predecessor Attempt id"):
        replace(
            payload,
            predecessor_attempt_id=predecessor_attempt_id,
            retry_id=retry_id,
        )

    mapping = event.to_mapping()
    mapping["payload"]["predecessor_attempt_id"] = predecessor_attempt_id
    mapping["payload"]["retry_id"] = retry_id
    with pytest.raises(ValueError, match="predecessor Attempt id"):
        phase_attempt_retried_event_from_mapping(mapping)


def test_retry_invariant_comparison_rejects_non_allowlisted_scientific_drift(tmp_path: Path) -> None:
    event, plan, predecessor = _contract_fixture(tmp_path)
    successor = event.payload.successor_phase_runspec
    compare_retry_invariants(plan, predecessor, successor)
    action = successor.payload.actions[0]
    changed_site = action.payload.site.model_copy(update={"container_mounts": ("/different:/different",)})
    changed_execution = replace(action.payload, site=changed_site)
    changed_action = replace(action, payload=changed_execution)
    changed = replace(successor, payload=replace(successor.payload, actions=(changed_action,)))
    with pytest.raises(ValueError, match="non-allowlisted"):
        compare_retry_invariants(plan, predecessor, changed)


@pytest.mark.parametrize(
    "field",
    ["input-content", "scientific", "expected-a3ms", "runtime", "site-path"],
)
def test_retry_invariant_allowlist_rejects_each_frozen_identity_class(
    tmp_path: Path,
    field: str,
) -> None:
    event, plan, predecessor = _contract_fixture(tmp_path)
    successor = event.payload.successor_phase_runspec
    if field == "input-content":
        changed = replace(
            successor,
            input_location=replace(successor.input_location, sha256="0" * 64),
        )
    else:
        execution = successor.payload.actions[0].payload
        scientific = execution.scientific
        site = execution.site
        runtime = execution.runtime
        expected_names = tuple(item.member_name for item in execution.expected_a3ms)
        if field == "scientific":
            scientific = scientific.model_copy(update={"max_sequences": 4321})
        elif field == "expected-a3ms":
            expected_names = ("changed.a3m", *expected_names[1:])
        elif field == "runtime":
            runtime = replace(runtime, submission_counter=runtime.submission_counter + 1)
        else:
            site = site.model_copy(update={"input_root": "/different-input"})
        regenerated = plan_preprocessing_chunk_execution(
            chunk=successor.payload.work_plan.chunks[0],
            records=successor.payload.work_plan.input.records,
            expected_a3m_members=expected_names,
            scientific=scientific,
            site=site,
            runtime=runtime,
        )
        action = replace(successor.payload.actions[0], payload=regenerated)
        database = replace(
            successor.payload.database,
            branches=tuple(
                replace(
                    branch,
                    gpuserver_argv=regenerated.gpuserver_argv,
                    search_argv=regenerated.search_argv,
                )
                for branch in successor.payload.database.branches
            ),
        )
        changed = replace(
            successor,
            payload=replace(successor.payload, actions=(action,), database=database),
        )

    with pytest.raises(ValueError, match=r"complete input location|non-allowlisted"):
        compare_retry_invariants(plan, predecessor, changed)


def test_retry_invariant_allowlist_accepts_cluster_and_resource_changes(tmp_path: Path) -> None:
    event, plan, predecessor = _contract_fixture(tmp_path)
    successor = event.payload.successor_phase_runspec
    action = successor.payload.actions[0]
    changed_action = replace(
        action,
        resources=replace(
            action.resources,
            partition="rotated",
            cpus_per_task=action.resources.cpus_per_task + 1,
            memory="48G",
            time="02:00:00",
        ),
    )
    changed = replace(
        successor,
        cluster=replace(
            successor.cluster,
            project_root="/rotated/project",
            output_root="/rotated/output",
            staging_root="/rotated/staging",
        ),
        payload=replace(successor.payload, actions=(changed_action,)),
    )

    compare_retry_invariants(plan, predecessor, changed)
