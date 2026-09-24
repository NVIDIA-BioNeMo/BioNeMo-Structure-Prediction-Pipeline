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

"""Family-dispatched attempt-bound Phase Receipt and finalized-payload contracts.

These tests cover the additive folding receipt generalization in
``contract/phase_receipt.py`` while proving preprocessing acceptance, rejection,
and identity bytes remain byte-identical. They are self-contained and use only
string-literal shapes (no fixtures).
"""

from __future__ import annotations

import pytest

from bspp.orchestration.contract.phase_receipt import (
    _ACTION_ID,
    PhaseFinalizedPayload,
    PhaseReceipt,
    _receipt_family,
    phase_finalized_payload_from_mapping,
    phase_receipt_from_mapping,
    phase_receipt_id,
)

SHA = "a" * 64
RUN_ID = f"phase-run-{SHA[:32]}"
ATTEMPT_ID = "attempt-0001"
RUNSPEC_LOCATION = "attempts/attempt-0001/phase-runspec.json"
FINALIZED_AT = "2026-09-11T00:00:00Z"

FOLDING_ACTION_KINDS = ("msa-flatten", "split", "preprocess", "fold", "canonical-pair")


def _receipt_base() -> dict[str, object]:
    return {
        "schema_version": 1,
        "phase_run_id": RUN_ID,
        "attempt_id": ATTEMPT_ID,
        "phase_plan_digest": SHA,
        "phase_runspec_location": RUNSPEC_LOCATION,
        "phase_runspec_digest": SHA,
        "input_sha256": SHA,
        "input_size_bytes": 1,
        "input_location_digest": SHA,
        "cluster_snapshot_digest": SHA,
        "runtime_action_digest": SHA,
        "adapter_version": "1.0.0",
        "slurm_job_id": "123456",
        "finalized_at": FINALIZED_AT,
        "result": "passed",
    }


def _folding_receipt_body() -> dict[str, object]:
    return {
        **_receipt_base(),
        "action_id": "canonical-pair-000001",
        "canonical_pair_index_digest": SHA,
        "folding_action_evidence_digests": [SHA] * 5,
    }


def _folding_receipt_mapping() -> dict[str, object]:
    body = _folding_receipt_body()
    return {**body, "phase_receipt_id": phase_receipt_id(body)}


def _preprocessing_receipt_body() -> dict[str, object]:
    return {
        **_receipt_base(),
        "action_id": "preprocessing-chunk-000001",
        "output_artifact_set_id": f"sha256:{SHA}",
        "artifact_location_id": f"artifact-location-{SHA}",
        "scheduler_evidence_digest": SHA,
        "action_evidence_digest": SHA,
        "content_validation_evidence_digest": SHA,
        "database_requested_policy": "direct",
        "database_source_manifest_sha256": SHA,
        "database_placement_outcome": "direct-requested",
        "member_count": 1,
        "logical_bytes": 1,
    }


def _preprocessing_receipt_mapping() -> dict[str, object]:
    body = _preprocessing_receipt_body()
    return {**body, "phase_receipt_id": phase_receipt_id(body)}


def _scheduler_evidence_mapping(action_id: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "evidence_kind": "provided-successful-terminal-observation",
        "phase_run_id": RUN_ID,
        "attempt_id": ATTEMPT_ID,
        "phase_runspec_digest": SHA,
        "action_id": action_id,
        "job_id": "123456",
        "source": "sacct",
        "state": "COMPLETED",
        "exit_code": "0:0",
        "observed_at": FINALIZED_AT,
    }


def _folding_payload_mapping() -> dict[str, object]:
    return {
        "schema_version": 1,
        "scheduler_evidence": _scheduler_evidence_mapping("canonical-pair-000001"),
        "folding_action_evidence_digests": [SHA] * 5,
        "canonical_pair_index_digest": SHA,
        "receipt": _folding_receipt_mapping(),
    }


def test_action_id_grammar_accepts_all_six_kinds_and_rejects_unknown() -> None:
    accepted = ("preprocessing-chunk-000001", *(f"{kind}-000001" for kind in FOLDING_ACTION_KINDS))
    for action_id in accepted:
        assert _ACTION_ID.fullmatch(action_id) is not None, action_id
    for action_id in ("bogus-000001", "fold-00001", "fold-0000001"):
        assert _ACTION_ID.fullmatch(action_id) is None, action_id


def test_receipt_family_dispatches_preprocessing_and_folding() -> None:
    assert _receipt_family("preprocessing-chunk-000001") == "preprocessing"
    for kind in FOLDING_ACTION_KINDS:
        assert _receipt_family(f"{kind}-000001") == "folding", kind
    with pytest.raises(ValueError, match="record has invalid Runtime Action identity"):
        _receipt_family("bogus-000001")


def test_folding_receipt_round_trips_through_mapping() -> None:
    mapping = _folding_receipt_mapping()
    receipt = phase_receipt_from_mapping(mapping)
    assert isinstance(receipt, PhaseReceipt)
    assert receipt.action_id == "canonical-pair-000001"
    assert receipt.canonical_pair_index_digest == SHA
    assert receipt.folding_action_evidence_digests == (SHA,) * 5
    assert receipt.output_artifact_set_id is None
    assert receipt.database_requested_policy is None
    assert receipt.phase_receipt_id == phase_receipt_id(receipt.identity_mapping())
    assert phase_receipt_from_mapping(receipt.to_mapping()) == receipt


def test_preprocessing_receipt_round_trip_is_byte_identical() -> None:
    mapping = _preprocessing_receipt_mapping()
    receipt = phase_receipt_from_mapping(mapping)
    assert receipt.database_requested_policy is not None
    assert receipt.database_requested_policy.value == "direct"
    assert receipt.database_placement_outcome is not None
    assert receipt.database_placement_outcome.value == "direct-requested"
    assert receipt.member_count == 1
    assert receipt.canonical_pair_index_digest is None
    assert receipt.folding_action_evidence_digests == ()
    assert receipt.phase_receipt_id == phase_receipt_id(receipt.identity_mapping())
    assert receipt.to_mapping() == mapping
    assert phase_receipt_from_mapping(receipt.to_mapping()) == receipt


def test_folding_receipt_rejects_preprocessing_chunk_and_database_fields() -> None:
    mapping = _folding_receipt_mapping()
    mapping.update(
        {
            "database_requested_policy": "direct",
            "output_artifact_set_id": f"sha256:{SHA}",
            "member_count": 1,
        }
    )
    with pytest.raises(ValueError, match="Unknown PhaseReceipt field"):
        phase_receipt_from_mapping(mapping)


def test_preprocessing_receipt_requires_database_fields() -> None:
    mapping = _preprocessing_receipt_mapping()
    del mapping["database_requested_policy"]
    with pytest.raises(ValueError, match="database_requested_policy must be a non-empty string"):
        phase_receipt_from_mapping(mapping)


def test_preprocessing_receipt_rejects_folding_fields() -> None:
    mapping = _preprocessing_receipt_mapping()
    mapping["canonical_pair_index_digest"] = SHA
    with pytest.raises(ValueError, match="Unknown PhaseReceipt field"):
        phase_receipt_from_mapping(mapping)


def test_folding_receipt_requires_folding_digests() -> None:
    mapping = _folding_receipt_mapping()
    del mapping["canonical_pair_index_digest"]
    with pytest.raises(ValueError, match="canonical_pair_index_digest must be a non-empty string"):
        phase_receipt_from_mapping(mapping)


def test_folding_receipt_rejects_malformed_action_digest_collection() -> None:
    mapping = _folding_receipt_mapping()
    mapping["folding_action_evidence_digests"] = "not-a-list"
    with pytest.raises(ValueError, match="folding_action_evidence_digests must be a list of non-empty strings"):
        phase_receipt_from_mapping(mapping)


def test_folding_finalized_payload_loads_and_omits_chunk_handoff_keys() -> None:
    payload = phase_finalized_payload_from_mapping(_folding_payload_mapping())
    assert isinstance(payload, PhaseFinalizedPayload)
    assert payload.receipt.action_id == "canonical-pair-000001"
    assert payload.canonical_pair_index_digest == SHA
    assert payload.folding_action_evidence_digests == (SHA,) * 5
    mapping = payload.to_mapping()
    assert set(mapping) == {
        "schema_version",
        "scheduler_evidence",
        "folding_action_evidence_digests",
        "canonical_pair_index_digest",
        "receipt",
    }
    for key in ("action_evidence", "chunk_manifest", "artifact_set", "artifact_location", "content_validation"):
        assert key not in mapping


def test_folding_finalized_payload_rejects_chunk_manifest() -> None:
    payload = _folding_payload_mapping()
    payload["chunk_manifest"] = {}
    with pytest.raises(ValueError, match="Unknown PhaseFinalizedPayload field"):
        phase_finalized_payload_from_mapping(payload)


def test_preprocessing_finalized_payload_requires_chunk_handoff_fields() -> None:
    payload: dict[str, object] = {
        "schema_version": 1,
        "scheduler_evidence": _scheduler_evidence_mapping("preprocessing-chunk-000001"),
        "receipt": _preprocessing_receipt_mapping(),
    }
    # action_evidence is the first required preprocessing chunk/handoff field in
    # loader evaluation order; its absence proves those fields are not optional.
    with pytest.raises(ValueError, match="action_evidence must be a mapping"):
        phase_finalized_payload_from_mapping(payload)
