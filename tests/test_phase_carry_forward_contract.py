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

"""Strict acyclic Attempt carry-forward contract tests."""

from __future__ import annotations

from copy import deepcopy

import pytest

from bspp.orchestration.contract.phase import canonical_mapping_digest
from bspp.orchestration.contract.phase_carry_forward import (
    AttemptCarryForwardContent,
    AttemptCarryForwardRecord,
    AttemptCarryForwardVerificationReference,
    AttemptWorkspaceBinding,
    AttemptWorkspaceRootBinding,
    attempt_carry_forward_id,
    attempt_carry_forward_record_from_mapping,
)


def _record() -> AttemptCarryForwardRecord:
    content = (
        AttemptCarryForwardContent(
            source_action_id="preprocessing-chunk-000000",
            target_action_id="preprocessing-chunk-000000",
            member_name="AFDB_zeta.a3m",
            source_ordinal=0,
            record_identity="protein-zeta",
            source_header=">protein-zeta",
            source_declared_path="/logical/scratch/n0g0_0/AFDB_zeta.a3m",
            target_declared_path="/logical/scratch/n0g0_0/AFDB_zeta.a3m",
            source_physical_path="/host/source/AFDB_zeta.a3m",
            target_physical_path="/host/target/scratch/n0g0_0/AFDB_zeta.a3m",
            source_private_mount_path=(
                "/run/bspp-carry/sources/attempt-0001/attempt-0002/preprocessing-chunk-000000/000000/AFDB_zeta.a3m"
            ),
            size_bytes=23,
            sha256="1" * 64,
        ),
    )
    workspace = AttemptWorkspaceBinding(
        phase_run_id="phase-run-0123456789abcdef0123456789abcdef",
        target_attempt_id="attempt-0002",
        target_action_id="preprocessing-chunk-000000",
        workspace_root="/host/target",
        roots=(
            AttemptWorkspaceRootBinding(
                role="scratch-output",
                logical_root="/logical/scratch",
                physical_root="/host/target/scratch",
            ),
            AttemptWorkspaceRootBinding(
                role="input",
                logical_root="/logical/input",
                physical_root="/host/target/input",
            ),
        ),
        search_input_physical_path="/host/target/input/n0g0/chunk.fa",
        split_input_physical_path="/host/target/input/split/chunk.fa",
        identity_sentinel_path="/host/target/carry-forward-identity.json",
        private_workspace_mount_path="/run/bspp-carry/workspace/run/attempt/action",
        source_path_mode="frozen-mount-inverse",
    )
    verification = AttemptCarryForwardVerificationReference(
        attestation_id="phase-action-evidence-attestation-" + "2" * 64,
        attestation_digest="3" * 64,
        attestation_event_sequence=5,
        evidence_path="/host/source/action-evidence.json",
        evidence_document_sha256="4" * 64,
        evidence_mapping_digest="5" * 64,
        terminal_event_digest="6" * 64,
        evidence_finished_at="2026-08-20T12:00:00.000000Z",
    )
    content_digest = canonical_mapping_digest({"schema_version": 1, "content": [item.to_mapping() for item in content]})
    identity = {
        "schema_version": 1,
        "phase_run_id": "phase-run-0123456789abcdef0123456789abcdef",
        "phase_kind": "preprocessing",
        "phase_plan_digest": "7" * 64,
        "input_set_identity_digest": "8" * 64,
        "scientific_identity_digest": "9" * 64,
        "source_attempt_id": "attempt-0001",
        "source_attempt_ordinal": 1,
        "source_runspec_digest": "a" * 64,
        "target_attempt_id": "attempt-0002",
        "target_attempt_ordinal": 2,
        "verification": verification.to_mapping(),
        "workspace": workspace.to_mapping(),
        "content": [item.to_mapping() for item in content],
        "content_digest": content_digest,
        "remaining_record_ordinals": [1, 2],
        "remaining_search_input_sha256": "b" * 64,
        "declared_at": "2026-08-20T12:01:00.000000Z",
    }
    return AttemptCarryForwardRecord(
        attempt_carry_forward_id=attempt_carry_forward_id(identity),
        phase_run_id=str(identity["phase_run_id"]),
        phase_kind="preprocessing",
        phase_plan_digest="7" * 64,
        input_set_identity_digest="8" * 64,
        scientific_identity_digest="9" * 64,
        source_attempt_id="attempt-0001",
        source_attempt_ordinal=1,
        source_runspec_digest="a" * 64,
        target_attempt_id="attempt-0002",
        target_attempt_ordinal=2,
        verification=verification,
        workspace=workspace,
        content=content,
        content_digest=content_digest,
        remaining_record_ordinals=(1, 2),
        remaining_search_input_sha256="b" * 64,
        declared_at="2026-08-20T12:01:00.000000Z",
    )


def test_carry_record_round_trips_and_has_no_target_runspec_digest_cycle() -> None:
    record = _record()
    assert attempt_carry_forward_record_from_mapping(record.to_mapping()) == record
    assert "target_runspec_digest" not in record.to_mapping()
    assert (
        record.attempt_carry_forward_id
        == attempt_carry_forward_record_from_mapping(record.to_mapping()).attempt_carry_forward_id
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: data.update({"unknown": True}),
        lambda data: data.update({"content": None}),
        lambda data: data["workspace"].update({"roots": []}),
        lambda data: data["content"][0].update({"member_name": "undeclared.log"}),
        lambda data: data.update({"remaining_record_ordinals": [0, 1]}),
    ],
)
def test_carry_record_rejects_unknown_null_empty_and_overlapping_content(mutation: object) -> None:
    mapping = deepcopy(_record().to_mapping())
    assert callable(mutation)
    mutation(mapping)
    with pytest.raises(ValueError):
        attempt_carry_forward_record_from_mapping(mapping)
