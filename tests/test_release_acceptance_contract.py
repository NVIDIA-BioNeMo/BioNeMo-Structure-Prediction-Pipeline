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
import json
import os
from pathlib import Path

import pytest

from bspp.orchestration.contract.release_acceptance import (
    AcceptanceEvidence,
    CandidateInventoryReport,
    LocalIntegrityReport,
    NoUploadReport,
    ProvenanceIdentityReport,
    PublicationApproval,
    PublicationApprovalStore,
    SemanticValidationReport,
    TerminalFailureReport,
    acceptance_evidence_from_mapping,
    validate_acceptance,
    validate_processing_upload_policy,
)
from bspp.orchestration.contract.runspec import runspec_from_mapping
from bspp.orchestration.contract.submission_evidence import EvidenceIndex, EvidenceIndexEntry, build_evidence_index
from tests.runspec_workflow_helpers import workflow_runspec_data

SHA_A = "a" * 64
SHA_B = "b" * 64


def _evidence(root: Path) -> AcceptanceEvidence:
    execution = root / "submissions" / "3-slurm" / "attempt-1" / "execution.json"
    execution.parent.mkdir(parents=True)
    execution.write_text("{}")
    index = build_evidence_index(root)
    index_path = root / "processing-provenance-index.json"
    index_bytes = (json.dumps(index.to_mapping(), sort_keys=True, separators=(",", ":")) + "\n").encode()
    index_path.write_bytes(index_bytes)
    return AcceptanceEvidence(
        candidate_inventory=CandidateInventoryReport(1, ("item-1",), ("item-1",)),
        local_integrity=LocalIntegrityReport(1, ("result.json",), ("result.json",), ()),
        semantic_validation=SemanticValidationReport(1, 1, 1, 1, 0),
        terminal_failures=TerminalFailureReport(1, 1, ("item-1",), (), ()),
        no_upload=NoUploadReport(1, 0),
        provenance_identity=ProvenanceIdentityReport(
            1,
            index_path.name,
            hashlib.sha256(index_bytes).hexdigest(),
            (3,),
            SHA_A,
        ),
    )


def test_reports_are_strict_independently_versioned_and_round_trip(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    assert acceptance_evidence_from_mapping(evidence.to_mapping()) == evidence
    payload = evidence.to_mapping()
    inventory = dict(payload["candidate_inventory"], extra=True)  # type: ignore[arg-type]
    payload["candidate_inventory"] = inventory
    with pytest.raises(ValueError, match="CandidateInventoryReport fields"):
        acceptance_evidence_from_mapping(payload)
    payload = evidence.to_mapping()
    semantic = dict(payload["semantic_validation"], format_version=2)  # type: ignore[arg-type]
    payload["semantic_validation"] = semantic
    with pytest.raises(ValueError, match="SemanticValidationReport format_version"):
        acceptance_evidence_from_mapping(payload)


@pytest.mark.parametrize(
    ("fixture", "issue"),
    [
        ("inventory-mismatch.json", "candidate inventory mismatch"),
        ("extra-archive-member.json", "unexpected archive member"),
        ("missing-archive-member.json", "missing archive member"),
        ("archive-digest-mismatch.json", "local package digest mismatches"),
        ("semantic-mismatch.json", "semantic mismatches"),
        ("terminal-failure.json", "terminal failures"),
        ("missing-execution-result.json", "missing execution results"),
        ("upload-before-approval.json", "external uploads must be zero"),
    ],
)
def test_synthetic_failures_are_recomputed_from_observations(tmp_path: Path, fixture: str, issue: str) -> None:
    evidence = _evidence(tmp_path)
    mutation = json.loads((Path(__file__).parent / "fixtures" / "release_acceptance" / fixture).read_text())
    payload = evidence.to_mapping()
    payload[mutation["report"]].update(mutation["changes"])  # type: ignore[union-attr]
    result = validate_acceptance(
        acceptance_evidence_from_mapping(payload), evidence_root=tmp_path, expected_runspec_sha256=SHA_A
    )
    assert any(issue in value for value in result.issues)


def test_altered_evidence_or_index_blocks_acceptance(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    (tmp_path / "submissions" / "3-slurm" / "attempt-1" / "execution.json").write_text("altered")
    assert (
        "provenance tree verification failed"
        in validate_acceptance(evidence, evidence_root=tmp_path, expected_runspec_sha256=SHA_A).issues[0]
    )

    other = tmp_path / "other"
    other.mkdir()
    _evidence(other)
    payload = evidence.to_mapping()
    payload["provenance_identity"]["index_sha256"] = SHA_B  # type: ignore[index]
    assert not validate_acceptance(
        acceptance_evidence_from_mapping(payload), evidence_root=other, expected_runspec_sha256=SHA_A
    ).ok


def test_thresholds_and_expected_counts_are_parameters_not_campaign_constants(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    payload = evidence.to_mapping()
    payload["semantic_validation"].update({"minimum_pass_count": 2})  # type: ignore[union-attr]
    assert (
        "semantic pass count below minimum"
        in validate_acceptance(
            acceptance_evidence_from_mapping(payload), evidence_root=tmp_path, expected_runspec_sha256=SHA_A
        ).issues
    )


def test_publication_approval_store_rejects_absent_stale_altered_and_consumed(tmp_path: Path) -> None:
    store = PublicationApprovalStore(tmp_path / "approval.json")
    with pytest.raises(ValueError, match="absent"):
        store.require(acceptance_sha256=SHA_A, destination="s3://destination/prefix")
    approval = PublicationApproval.create(acceptance_sha256=SHA_A, destination="s3://destination/prefix")
    store.create(approval)
    with pytest.raises(FileExistsError):
        store.create(approval)
    with pytest.raises(ValueError, match="stale"):
        store.require(acceptance_sha256=SHA_B, destination=approval.destination)
    store.path.write_text(store.path.read_text().replace(approval.destination, "s3://altered/prefix"))
    with pytest.raises(ValueError, match="altered"):
        store.require(acceptance_sha256=SHA_A, destination=approval.destination)
    store.path.write_text(json.dumps(approval.to_mapping(), sort_keys=True) + "\n")
    assert store.consume(acceptance_sha256=SHA_A, destination=approval.destination) == approval
    with pytest.raises(ValueError, match="consumed"):
        store.require(acceptance_sha256=SHA_A, destination=approval.destination)


def test_processing_runspec_cannot_upload() -> None:
    validate_processing_upload_policy(self_upload=False, workflow_steps=("process", "accept"))
    with pytest.raises(ValueError, match="cannot upload"):
        validate_processing_upload_policy(self_upload=True, workflow_steps=("process",))
    with pytest.raises(ValueError, match="cannot upload"):
        validate_processing_upload_policy(self_upload=False, workflow_steps=("process", "upload-s3"))


def test_acceptance_rejects_cross_report_candidate_contradictions(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    payload = evidence.to_mapping()
    payload["terminal_failures"].update(  # type: ignore[union-attr]
        {
            "completed_candidates": ["item-2"],
            "failed_candidates": ["item-2"],
            "missing_execution_results": ["item-2"],
        }
    )

    result = validate_acceptance(
        acceptance_evidence_from_mapping(payload), evidence_root=tmp_path, expected_runspec_sha256=SHA_A
    )

    assert "terminal candidate sets overlap" in result.issues
    assert "completed candidates differ from observed inventory" in result.issues


def test_acceptance_rejects_impossible_parameterized_semantic_thresholds(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    payload = evidence.to_mapping()
    payload["semantic_validation"].update(  # type: ignore[union-attr]
        {"expected_count": 1, "minimum_pass_count": 2, "passed_count": 1, "mismatch_count": 0}
    )

    result = validate_acceptance(
        acceptance_evidence_from_mapping(payload), evidence_root=tmp_path, expected_runspec_sha256=SHA_A
    )

    assert "semantic minimum exceeds expected count" in result.issues


def test_acceptance_rejects_semantic_count_not_bound_to_inventory(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    payload = evidence.to_mapping()
    payload["semantic_validation"].update(  # type: ignore[union-attr]
        {"expected_count": 2, "minimum_pass_count": 1, "passed_count": 2, "mismatch_count": 0}
    )

    result = validate_acceptance(
        acceptance_evidence_from_mapping(payload), evidence_root=tmp_path, expected_runspec_sha256=SHA_A
    )

    assert "semantic expected count differs from candidate inventory" in result.issues


def test_acceptance_rejects_wrong_runspec_identity_even_when_reports_and_tree_pass(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)

    result = validate_acceptance(evidence, evidence_root=tmp_path, expected_runspec_sha256=SHA_B)

    assert not result.ok
    assert "provenance RunSpec identity mismatch" in result.issues


def test_publication_approval_can_only_be_created_from_successful_acceptance(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    accepted = validate_acceptance(evidence, evidence_root=tmp_path, expected_runspec_sha256=SHA_A)
    approval = PublicationApproval.create_from_acceptance(accepted, destination="s3://destination/prefix")
    assert approval.acceptance_sha256 == accepted.acceptance_sha256

    rejected = validate_acceptance(evidence, evidence_root=tmp_path, expected_runspec_sha256=SHA_B)
    with pytest.raises(ValueError, match="successful acceptance"):
        PublicationApproval.create_from_acceptance(rejected, destination="s3://destination/prefix")


def test_publication_approval_read_rejects_symlink_and_oversized_record(tmp_path: Path) -> None:
    acceptance_sha256 = SHA_A
    destination = "s3://destination/prefix"
    approval = PublicationApproval.create(acceptance_sha256=acceptance_sha256, destination=destination)
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps(approval.to_mapping()))
    store = PublicationApprovalStore(tmp_path / "approval.json")
    os.symlink(outside, store.path)
    with pytest.raises(ValueError, match="altered"):
        store.require(acceptance_sha256=acceptance_sha256, destination=destination)

    store.path.unlink()
    store.path.write_bytes(b" " * (1024 * 1024 + 1))
    with pytest.raises(ValueError, match="altered"):
        store.require(acceptance_sha256=acceptance_sha256, destination=destination)


def test_publication_approval_consumption_has_exactly_one_winner(tmp_path: Path) -> None:
    store = PublicationApprovalStore(tmp_path / "approval.json")
    approval = PublicationApproval.create(acceptance_sha256=SHA_A, destination="s3://destination/prefix")
    store.create(approval)

    assert store.consume(acceptance_sha256=SHA_A, destination=approval.destination) == approval
    with pytest.raises(ValueError, match="consumed"):
        store.consume(acceptance_sha256=SHA_A, destination=approval.destination)


def test_publication_approval_rejects_symlinked_parent_without_outside_write(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(outside, target_is_directory=True)
    store = PublicationApprovalStore(linked_parent / "approval.json")
    approval = PublicationApproval.create(acceptance_sha256=SHA_A, destination="s3://destination/prefix")

    with pytest.raises(ValueError, match=r"parent.*real directory"):
        store.create(approval)

    assert not (outside / "approval.json").exists()


def test_publication_approval_rejects_hardlinked_record(tmp_path: Path) -> None:
    approval = PublicationApproval.create(acceptance_sha256=SHA_A, destination="s3://destination/prefix")
    original = tmp_path / "original.json"
    original.write_text(json.dumps(approval.to_mapping(), sort_keys=True) + "\n")
    store = PublicationApprovalStore(tmp_path / "approval.json")
    os.link(original, store.path)

    with pytest.raises(ValueError, match="altered"):
        store.require(acceptance_sha256=SHA_A, destination=approval.destination)


def test_acceptance_consumes_mr3_processing_index_through_public_verifier(tmp_path: Path) -> None:
    result_path = tmp_path / "submissions" / "3-slurm" / "attempt-1" / "result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text("{}")
    relative_result = result_path.relative_to(tmp_path).as_posix()
    index = EvidenceIndex(
        1,
        (EvidenceIndexEntry(relative_result, hashlib.sha256(result_path.read_bytes()).hexdigest()),),
    )
    index_path = tmp_path / "processing-provenance-index.json"
    index_bytes = (json.dumps(index.to_mapping(), sort_keys=True, separators=(",", ":")) + "\n").encode()
    index_path.write_bytes(index_bytes)
    evidence = AcceptanceEvidence(
        candidate_inventory=CandidateInventoryReport(1, ("item-1",), ("item-1",)),
        local_integrity=LocalIntegrityReport(1, ("result.json",), ("result.json",), ()),
        semantic_validation=SemanticValidationReport(1, 1, 1, 1, 0),
        terminal_failures=TerminalFailureReport(1, 1, ("item-1",), (), ()),
        no_upload=NoUploadReport(1, 0),
        provenance_identity=ProvenanceIdentityReport(
            1,
            "processing-provenance-index.json",
            hashlib.sha256(index_bytes).hexdigest(),
            (3,),
            SHA_A,
        ),
    )

    assert validate_acceptance(evidence, evidence_root=tmp_path, expected_runspec_sha256=SHA_A).ok

    result_path.write_text('{"forged":true}')
    validation = validate_acceptance(evidence, evidence_root=tmp_path, expected_runspec_sha256=SHA_A)
    assert not validation.ok
    assert any("processing evidence digest mismatch" in issue for issue in validation.issues)


@pytest.mark.parametrize("run_kind", ["canary", "production"])
def test_governed_processing_runspec_rejects_upload_steps(tmp_path: Path, run_kind: str) -> None:
    data = workflow_runspec_data(tmp_path)
    data["run_kind"] = run_kind
    data["data_placement"] = {"s3": {"tool": "dm"}}
    data["workflow"]["steps"].append({"name": "upload-s3", "run": True})  # type: ignore[index]

    with pytest.raises(ValueError, match="processing RunSpec cannot upload"):
        runspec_from_mapping(data)


@pytest.mark.parametrize("run_kind", ["canary", "production"])
def test_governed_processing_runspec_rejects_self_upload_without_workflow(tmp_path: Path, run_kind: str) -> None:
    data = workflow_runspec_data(tmp_path)
    spec = runspec_from_mapping(data)
    spec = spec.model_copy(
        update={
            "run_kind": run_kind,
            "workflow": None,
            "worker": spec.worker.model_copy(update={"self_upload": True}),
        }
    )

    with pytest.raises(ValueError, match="processing RunSpec cannot upload"):
        spec.validate()


def test_development_runspec_keeps_existing_upload_behavior(tmp_path: Path) -> None:
    data = workflow_runspec_data(tmp_path)
    data["run_kind"] = "dev"
    data["data_placement"] = {"s3": {"tool": "dm"}}
    data["workflow"]["steps"].append({"name": "upload-s3", "run": True})  # type: ignore[index]

    assert runspec_from_mapping(data).run_kind == "dev"
