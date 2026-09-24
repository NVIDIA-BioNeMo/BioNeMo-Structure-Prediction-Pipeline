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

"""Tiny test-only harness for the generic acceptance-to-publication chain."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

from bspp.orchestration.contract.release_acceptance import (
    AcceptanceEvidence,
    AcceptanceValidation,
    CandidateInventoryReport,
    LocalIntegrityReport,
    NoUploadReport,
    ProvenanceIdentityReport,
    PublicationApproval,
    PublicationApprovalStore,
    SemanticValidationReport,
    TerminalFailureReport,
    validate_acceptance,
)
from bspp.orchestration.contract.submission_evidence import (
    SubmissionExpectation,
    SubmissionResult,
    SubmissionToken,
    build_evidence_index,
)

_RUNSPEC_SHA256 = "a" * 64
_DEFAULT_DESTINATION = "s3://release/location"


@dataclass
class SyntheticReleaseHarness:
    root: Path
    evidence: AcceptanceEvidence
    approvals: PublicationApprovalStore
    destination: str = _DEFAULT_DESTINATION

    @classmethod
    def create(cls, root: Path) -> SyntheticReleaseHarness:
        evidence_root = root / "evidence"
        evidence_root.mkdir()
        attempt_root = evidence_root / "submissions" / "3-process" / "attempt-1"
        attempt_root.mkdir(parents=True)
        token = SubmissionToken.create(
            run_id="synthetic-release",
            runspec_sha256=_RUNSPEC_SHA256,
            step_index=3,
            step_name="process",
            slice_id="0",
            attempt=1,
            script_sha256="b" * 64,
            bootstrap_sha256="c" * 64,
            control_state_sha256="d" * 64,
            runtime_qualification_sha256="e" * 64,
        )
        expectation = SubmissionExpectation(
            token,
            token.script_sha256,
            token.control_state_sha256,
            token.bootstrap_sha256,
            token.runtime_qualification_sha256,
            "f" * 40,
            "f" * 64,
        )
        result = SubmissionResult(
            token,
            token.script_sha256,
            token.control_state_sha256,
            token.bootstrap_sha256,
            token.runtime_qualification_sha256,
            expectation.runtime_ipsae_source_revision,
            expectation.runtime_ipsae_binary_sha256,
            1,
            "4242",
            "COMPLETED",
            (
                ("runtime_ipsae_binary_sha256", expectation.runtime_ipsae_binary_sha256),
                ("runtime_ipsae_source_revision", expectation.runtime_ipsae_source_revision),
            ),
        )
        (attempt_root / "expectation.json").write_bytes(expectation.canonical_bytes())
        (attempt_root / "result.json").write_bytes(result.canonical_bytes())
        index = build_evidence_index(evidence_root)
        index_path = evidence_root / "processing-provenance-index.json"
        index_bytes = (json.dumps(index.to_mapping(), sort_keys=True, separators=(",", ":")) + "\n").encode()
        index_path.write_bytes(index_bytes)
        evidence = AcceptanceEvidence(
            candidate_inventory=CandidateInventoryReport(1, ("candidate-1",), ("candidate-1",)),
            local_integrity=LocalIntegrityReport(1, ("artifact.json",), ("artifact.json",), ()),
            semantic_validation=SemanticValidationReport(1, 1, 1, 1, 0),
            terminal_failures=TerminalFailureReport(1, 1, ("candidate-1",), (), ()),
            no_upload=NoUploadReport(1, 0),
            provenance_identity=ProvenanceIdentityReport(
                1,
                index_path.name,
                hashlib.sha256(index_bytes).hexdigest(),
                (3,),
                _RUNSPEC_SHA256,
            ),
        )
        return cls(evidence_root, evidence, PublicationApprovalStore(root / "publication-approval.json"))

    def validate(self) -> AcceptanceValidation:
        return validate_acceptance(
            self.evidence,
            evidence_root=self.root,
            expected_runspec_sha256=_RUNSPEC_SHA256,
        )

    def approve(self, acceptance_sha256: str) -> None:
        acceptance = self.validate()
        if acceptance.acceptance_sha256 != acceptance_sha256:
            raise ValueError("acceptance digest does not match current validation")
        self.approvals.create(PublicationApproval.create_from_acceptance(acceptance, destination=self.destination))

    def publish(self, acceptance_sha256: str, *, destination: str | None = None) -> None:
        self.approvals.consume(
            acceptance_sha256=acceptance_sha256,
            destination=self.destination if destination is None else destination,
        )

    def mutate(self, name: str) -> None:
        if name == "inventory":
            report = replace(self.evidence.candidate_inventory, observed_candidates=("candidate-2",))
            self.evidence = replace(self.evidence, candidate_inventory=report)
        elif name == "extra_member":
            report = replace(self.evidence.local_integrity, observed_members=("artifact.json", "extra.json"))
            self.evidence = replace(self.evidence, local_integrity=report)
        elif name == "missing_member":
            report = replace(self.evidence.local_integrity, observed_members=())
            self.evidence = replace(self.evidence, local_integrity=report)
        elif name == "digest_mismatch":
            report = replace(self.evidence.local_integrity, digest_mismatches=("artifact.json",))
            self.evidence = replace(self.evidence, local_integrity=report)
        elif name == "semantic":
            report = replace(self.evidence.semantic_validation, passed_count=0, mismatch_count=1)
            self.evidence = replace(self.evidence, semantic_validation=report)
        elif name == "terminal_failure":
            report = replace(self.evidence.terminal_failures, failed_candidates=("candidate-1",))
            self.evidence = replace(self.evidence, terminal_failures=report)
        elif name == "missing_execution":
            report = replace(
                self.evidence.terminal_failures,
                completed_candidates=(),
                missing_execution_results=("candidate-1",),
            )
            self.evidence = replace(self.evidence, terminal_failures=report)
        elif name == "index":
            (self.root / "submissions" / "3-process" / "attempt-1" / "result.json").write_text("altered")
        elif name == "extra_evidence":
            (self.root / "submissions" / "3-process" / "attempt-1" / "unindexed.json").write_text("{}")
        elif name == "symlink_evidence":
            (self.root / "outside.json").write_text("{}")
            os.symlink(self.root / "outside.json", self.root / "linked.json")
        elif name == "upload":
            self.evidence = replace(self.evidence, no_upload=NoUploadReport(1, 1))
        else:
            raise ValueError(f"unknown synthetic mutation: {name}")
