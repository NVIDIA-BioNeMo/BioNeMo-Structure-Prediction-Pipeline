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
import os
import subprocess
from pathlib import Path

import pytest

from bspp.orchestration.contract.execution_provenance_verify import (
    verify_execution_provenance_tree,
    verify_processing_provenance_boundary,
)
from bspp.orchestration.contract.submission_evidence import (
    SubmissionExpectation,
    SubmissionResult,
    SubmissionToken,
    submission_result_from_bytes,
)
from bspp.orchestration.control.execution_provenance_events import (
    ImmutableExecutionProvenanceStore,
    render_runtime_result_epilogue,
)
from bspp.orchestration.control.governed_submission import finalize_selected_evidence_index
from bspp.orchestration.control.submission_coordinator import SubmissionCoordinator


def _expectation() -> SubmissionExpectation:
    token = SubmissionToken.create(
        run_id="run-1",
        runspec_sha256="a" * 64,
        step_index=0,
        step_name="preflight",
        slice_id="0-9",
        attempt=1,
        script_sha256="b" * 64,
        bootstrap_sha256="c" * 64,
        control_state_sha256="d" * 64,
        runtime_qualification_sha256="e" * 64,
    )
    return SubmissionExpectation(
        token, "b" * 64, "d" * 64, "c" * 64, "e" * 64, "f" * 40, hashlib.sha256(b"binary").hexdigest()
    )


def _runtime_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    qualification = tmp_path / "qualification.json"
    binary = tmp_path / "ipsae_cpp"
    qualification.write_text('{"smoke_evidence":{"runtime_ipsae":{"source_revision":"' + "f" * 40 + '"}}}')
    binary.write_bytes(b"binary")
    return qualification, binary


def test_store_publishes_create_once_expectation_result_and_exact_index(tmp_path: Path) -> None:
    expectation = _expectation()
    result = SubmissionResult(
        expectation.token,
        expectation.rendered_script_sha256,
        expectation.control_state_sha256,
        expectation.bootstrap_sha256,
        expectation.runtime_qualification_sha256,
        expectation.runtime_ipsae_source_revision,
        expectation.runtime_ipsae_binary_sha256,
        job_id="123",
        scheduler_status="COMPLETED",
    )
    store = ImmutableExecutionProvenanceStore(tmp_path)

    assert store.write_expectation(expectation).exists()
    assert store.write_result(result).exists()
    finalized = store.finalize()
    assert verify_execution_provenance_tree(tmp_path, finalized.index).ok

    (tmp_path / "unexpected").write_text("x")
    assert not verify_execution_provenance_tree(tmp_path, finalized.index).ok


def test_processing_boundary_verifier_allows_acceptance_but_rejects_late_processing_attempt(tmp_path: Path) -> None:
    expectation = _expectation()
    result = SubmissionResult(
        expectation.token,
        expectation.rendered_script_sha256,
        expectation.control_state_sha256,
        expectation.bootstrap_sha256,
        expectation.runtime_qualification_sha256,
        expectation.runtime_ipsae_source_revision,
        expectation.runtime_ipsae_binary_sha256,
        job_id="123",
        scheduler_status="COMPLETED",
    )
    store = ImmutableExecutionProvenanceStore(tmp_path)
    expectation_path = store.write_expectation(expectation)
    result_path = store.write_result(result)
    coordinator = SubmissionCoordinator(tmp_path)
    coordinator.claim(expectation.token)
    coordinator.bind_job(expectation.token, job_id="123", scheduler_status="SUBMITTED")
    coordinator_path = coordinator.record_path(expectation.token)
    finalized = finalize_selected_evidence_index(
        tmp_path,
        tuple(path.relative_to(tmp_path) for path in (expectation_path, result_path, coordinator_path)),
    )
    acceptance = tmp_path / "submissions/0001-acceptance-semantic-attempt-0001/expectation.json"
    acceptance.parent.mkdir()
    acceptance.write_text("later acceptance\n")

    assert verify_processing_provenance_boundary(tmp_path, finalized.index, processing_step_indices=(0,)).ok

    forged = tmp_path / "submissions/0000-forged-attempt-0002/result.json"
    forged.parent.mkdir()
    forged.write_text("forged\n")
    assert not verify_processing_provenance_boundary(tmp_path, finalized.index, processing_step_indices=(0,)).ok


def test_store_rejects_replacement_of_immutable_record(tmp_path: Path) -> None:
    store = ImmutableExecutionProvenanceStore(tmp_path)
    expectation = _expectation()
    store.write_expectation(expectation)
    replacement = SubmissionExpectation(
        expectation.token,
        expectation.rendered_script_sha256,
        expectation.control_state_sha256,
        expectation.bootstrap_sha256,
        expectation.runtime_qualification_sha256,
        "0" * 40,
        expectation.runtime_ipsae_binary_sha256,
    )
    with pytest.raises(ValueError, match="immutable"):
        store.write_expectation(replacement)


def test_store_rejects_intermediate_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "submissions").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match=r"directory|symlink"):
        ImmutableExecutionProvenanceStore(evidence).write_expectation(_expectation())

    assert list(outside.iterdir()) == []


def test_runtime_result_epilogue_is_token_and_expectation_hash_authenticated(tmp_path: Path) -> None:
    expectation_path = tmp_path / "expectation.json"
    result_path = tmp_path / "result.json"
    qualification, binary = _runtime_artifacts(tmp_path)
    rendered = render_runtime_result_epilogue(
        expectation_path, result_path, runtime_qualification_path=qualification, runtime_ipsae_binary_path=binary
    )

    assert "BSPP_SUBMISSION_TOKEN" in rendered
    assert "BSPP_EXPECTATION_SHA256" in rendered
    assert str(expectation_path) in rendered
    assert str(result_path) in rendered
    assert "O_EXCL" in rendered


def test_runtime_result_epilogue_writes_canonical_bound_result(tmp_path: Path) -> None:
    expectation = _expectation()
    expectation_path = tmp_path / "expectation.json"
    result_path = tmp_path / "result.json"
    expectation_path.write_bytes(expectation.canonical_bytes())
    qualification, binary = _runtime_artifacts(tmp_path)
    environment = dict(os.environ)
    environment.update(
        BSPP_SUBMISSION_TOKEN=expectation.token.token,
        BSPP_EXPECTATION_SHA256=hashlib.sha256(expectation.canonical_bytes()).hexdigest(),
        BSPP_PAYLOAD_STATUS="0",
        SLURM_JOB_ID="4242",
        SLURM_ARRAY_JOB_ID="4000",
    )

    completed = subprocess.run(
        (
            "bash",
            "-c",
            render_runtime_result_epilogue(
                expectation_path,
                result_path,
                runtime_qualification_path=qualification,
                runtime_ipsae_binary_path=binary,
            ),
        ),
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    result = submission_result_from_bytes(result_path.read_bytes())
    assert result.job_id == "4000"
    assert result.token == expectation.token
    assert dict(result.runtime_observations)["runtime_ipsae_binary_sha256"] == expectation.runtime_ipsae_binary_sha256

    repeated = subprocess.run(
        (
            "bash",
            "-c",
            render_runtime_result_epilogue(
                expectation_path,
                result_path,
                runtime_qualification_path=qualification,
                runtime_ipsae_binary_path=binary,
            ),
        ),
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert repeated.returncode == 0, repeated.stderr


def test_runtime_result_epilogue_rejects_intermediate_symlink_escape(tmp_path: Path) -> None:
    expectation = _expectation()
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    outside = tmp_path / "outside"
    attempt = outside / "attempt"
    attempt.mkdir(parents=True)
    (evidence / "submissions").symlink_to(outside, target_is_directory=True)
    expectation_path = evidence / "submissions" / "attempt" / "expectation.json"
    result_path = evidence / "submissions" / "attempt" / "result.json"
    expectation_path.write_bytes(expectation.canonical_bytes())
    qualification, binary = _runtime_artifacts(tmp_path)
    environment = dict(os.environ)
    environment.update(
        BSPP_SUBMISSION_TOKEN=expectation.token.token,
        BSPP_EXPECTATION_SHA256=hashlib.sha256(expectation.canonical_bytes()).hexdigest(),
        BSPP_PAYLOAD_STATUS="0",
        SLURM_JOB_ID="4242",
    )

    completed = subprocess.run(
        (
            "bash",
            "-c",
            render_runtime_result_epilogue(
                expectation_path,
                result_path,
                runtime_qualification_path=qualification,
                runtime_ipsae_binary_path=binary,
                evidence_root=evidence,
            ),
        ),
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )

    assert completed.returncode != 0
    assert not result_path.exists()
