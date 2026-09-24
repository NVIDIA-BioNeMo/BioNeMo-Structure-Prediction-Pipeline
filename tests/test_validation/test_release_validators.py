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
import io
import os
import tarfile
from pathlib import Path

import pytest

from bspp.orchestration.contract.release_acceptance import (
    LocalIntegrityReport,
    NoUploadReport,
    ProvenanceIdentityReport,
)
from bspp.orchestration.runtime.validation.candidate_semantic import validate_candidate_semantics
from bspp.orchestration.runtime.validation.local_package_integrity import validate_local_tar_integrity
from bspp.orchestration.runtime.validation.release_evidence import (
    build_release_acceptance_evidence,
    write_release_acceptance_evidence,
)
from bspp.orchestration.runtime.validation.terminal_failures import validate_terminal_results


def _tar(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def test_local_tar_integrity_is_computed_from_member_bytes(tmp_path: Path) -> None:
    archive = tmp_path / "candidate.tar"
    _tar(archive, {"a.json": b"a", "extra.json": b"extra"})

    report = validate_local_tar_integrity(
        archive,
        {
            "a.json": hashlib.sha256(b"different").hexdigest(),
            "missing.json": hashlib.sha256(b"missing").hexdigest(),
        },
    )

    assert report.expected_members == ("a.json", "missing.json")
    assert report.observed_members == ("a.json", "extra.json")
    assert report.digest_mismatches == ("a.json",)


def test_local_tar_integrity_rejects_final_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.tar"
    _tar(target, {"a.json": b"a"})
    linked = tmp_path / "linked.tar"
    os.symlink(target, linked)

    try:
        validate_local_tar_integrity(linked, {"a.json": hashlib.sha256(b"a").hexdigest()})
    except ValueError as error:
        assert "exclusive regular file" in str(error)
    else:
        raise AssertionError("symlinked archive unexpectedly passed integrity validation")


def test_candidate_semantics_are_recomputed_without_worker_ok_flag() -> None:
    report = validate_candidate_semantics(
        expected={"candidate-1": "good", "candidate-2": "good"},
        observed={"candidate-1": "good", "candidate-2": "wrong"},
        minimum_pass_count=2,
    )

    assert report.expected_count == 2
    assert report.passed_count == 1
    assert report.mismatch_count == 1


def test_candidate_semantics_rejects_unexpected_observations() -> None:
    with pytest.raises(ValueError, match="unexpected semantic candidates"):
        validate_candidate_semantics(
            expected={"candidate-1": "good"},
            observed={"candidate-1": "good", "extra": "good"},
            minimum_pass_count=1,
        )


def test_terminal_results_require_scheduler_completion_and_immutable_result() -> None:
    report = validate_terminal_results(
        expected_candidates=("candidate-1", "candidate-2", "candidate-3"),
        scheduler_states={"candidate-1": "COMPLETED", "candidate-2": "FAILED", "candidate-3": "COMPLETED"},
        result_candidates=("candidate-1",),
    )

    assert report.completed_candidates == ("candidate-1",)
    assert report.failed_candidates == ("candidate-2",)
    assert report.missing_execution_results == ("candidate-3",)


def test_terminal_results_reject_unexpected_scheduler_or_result_candidates() -> None:
    for scheduler_states, result_candidates in (
        ({"candidate-1": "COMPLETED", "extra": "COMPLETED"}, ("candidate-1",)),
        ({"candidate-1": "COMPLETED"}, ("candidate-1", "extra")),
    ):
        with pytest.raises(ValueError, match="unexpected"):
            validate_terminal_results(
                expected_candidates=("candidate-1",),
                scheduler_states=scheduler_states,
                result_candidates=result_candidates,
            )


def test_release_evidence_composes_independent_reports() -> None:
    inventory_expected = ("candidate-1",)
    inventory_observed = ("candidate-1",)
    semantic = validate_candidate_semantics(
        expected={"candidate-1": "good"},
        observed={"candidate-1": "good"},
        minimum_pass_count=1,
    )
    terminal = validate_terminal_results(
        expected_candidates=inventory_expected,
        scheduler_states={"candidate-1": "COMPLETED"},
        result_candidates=inventory_observed,
    )
    integrity = LocalIntegrityReport(1, ("artifact.json",), ("artifact.json",), ())
    provenance = ProvenanceIdentityReport(
        1,
        "processing-provenance-index.json",
        "a" * 64,
        (3,),
        "b" * 64,
    )

    evidence = build_release_acceptance_evidence(
        expected_candidates=inventory_expected,
        observed_candidates=inventory_observed,
        local_integrity=integrity,
        semantic_validation=semantic,
        terminal_failures=terminal,
        no_upload=NoUploadReport(1, 0),
        provenance_identity=provenance,
    )

    assert evidence.candidate_inventory.expected_candidates == inventory_expected


def test_release_acceptance_record_is_create_once_and_canonical(tmp_path: Path) -> None:
    semantic = validate_candidate_semantics(
        expected={"candidate-1": "good"},
        observed={"candidate-1": "good"},
        minimum_pass_count=1,
    )
    terminal = validate_terminal_results(
        expected_candidates=("candidate-1",),
        scheduler_states={"candidate-1": "COMPLETED"},
        result_candidates=("candidate-1",),
    )
    evidence = build_release_acceptance_evidence(
        expected_candidates=("candidate-1",),
        observed_candidates=("candidate-1",),
        local_integrity=LocalIntegrityReport(1, ("artifact.json",), ("artifact.json",), ()),
        semantic_validation=semantic,
        terminal_failures=terminal,
        no_upload=NoUploadReport(1, 0),
        provenance_identity=ProvenanceIdentityReport(
            1,
            "processing-provenance-index.json",
            "a" * 64,
            (3,),
            "b" * 64,
        ),
    )
    path = tmp_path / "acceptance.json"

    assert write_release_acceptance_evidence(evidence, path) == path
    assert path.read_bytes() == evidence.canonical_bytes()
    try:
        write_release_acceptance_evidence(evidence, path)
    except FileExistsError:
        pass
    else:
        raise AssertionError("acceptance record was overwritten")
