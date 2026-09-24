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

import importlib
import os
from pathlib import Path

import pytest

from bspp.orchestration.contract.submission_evidence import SubmissionExpectation, SubmissionResult, SubmissionToken


def _expectation(
    *, runtime_ipsae_source_revision: str = "a" * 40, runtime_ipsae_binary_sha256: str = "b" * 64
) -> SubmissionExpectation:
    token = SubmissionToken.create(
        run_id="run-1",
        runspec_sha256="a" * 64,
        step_index=3,
        step_name="slurm",
        slice_id="0-9",
        attempt=1,
        script_sha256="b" * 64,
        bootstrap_sha256="c" * 64,
        control_state_sha256="d" * 64,
        runtime_qualification_sha256="e" * 64,
    )
    return SubmissionExpectation(
        token,
        "b" * 64,
        "d" * 64,
        "c" * 64,
        "e" * 64,
        runtime_ipsae_source_revision,
        runtime_ipsae_binary_sha256,
    )


def _result(
    expectation: SubmissionExpectation,
    *,
    job_id: str,
    runtime_ipsae_source_revision: str,
    runtime_ipsae_binary_sha256: str,
) -> SubmissionResult:
    return SubmissionResult(
        expectation.token,
        expectation.rendered_script_sha256,
        expectation.control_state_sha256,
        expectation.bootstrap_sha256,
        expectation.runtime_qualification_sha256,
        expectation.runtime_ipsae_source_revision,
        expectation.runtime_ipsae_binary_sha256,
        job_id=job_id,
        scheduler_status="COMPLETED",
        runtime_observations=(
            ("runtime_ipsae_binary_sha256", runtime_ipsae_binary_sha256),
            ("runtime_ipsae_source_revision", runtime_ipsae_source_revision),
        ),
    )


def _integration() -> object:
    return importlib.import_module("bspp.orchestration.control.governed_submission")


def test_expectation_binds_exact_governed_artifact_bytes(tmp_path: Path) -> None:
    module = _integration()
    paths = {}
    for name, contents in {
        "runspec": b"canonical runspec\n",
        "script": b"#!/bin/sh\n",
        "bootstrap": b"bootstrap bytes\n",
        "control_state": b'{"state":"materialized"}\n',
        "runtime_qualification": (
            b'{"smoke_evidence":{"runtime_ipsae":{"binary":{"sha256":"'
            + b"b" * 64
            + b'"},"source_revision":"'
            + b"a" * 40
            + b'"}}}\n'
        ),
    }.items():
        path = tmp_path / name
        path.write_bytes(contents)
        paths[name] = path

    expectation = module.build_governed_submission_expectation(
        run_id="run-1",
        step_index=3,
        step_name="slurm",
        slice_id="0-9",
        attempt=1,
        **{f"{name}_path": path for name, path in paths.items()},
    )

    assert expectation.token.runspec_sha256 == module.stable_sha256(paths["runspec"])
    assert expectation.rendered_script_sha256 == module.stable_sha256(paths["script"])
    assert expectation.bootstrap_sha256 == module.stable_sha256(paths["bootstrap"])
    assert expectation.control_state_sha256 == module.stable_sha256(paths["control_state"])
    assert expectation.runtime_qualification_sha256 == module.stable_sha256(paths["runtime_qualification"])


def test_coordinator_claim_precedes_transport_submit(tmp_path: Path) -> None:
    module = _integration()
    events: list[str] = []

    class Coordinator:
        def claim(self, _token: object) -> object:
            events.append("claim")
            return module.SubmissionClaim(created=True, record=None)

        def bind_job(self, _token: object, *, job_id: str, scheduler_status: str) -> object:
            events.append(f"bind:{job_id}:{scheduler_status}")
            return object()

    def submit() -> object:
        events.append("submit")
        return module.SchedulerSubmission(job_id="123", status="PENDING")

    outcome = module.submit_claimed_once(expectation=_expectation(), coordinator=Coordinator(), submit=submit)

    assert outcome.job_id == "123"
    assert events == ["claim", "submit", "bind:123:PENDING"]


def test_bind_failure_after_scheduler_acceptance_is_ambiguous() -> None:
    module = _integration()
    events: list[str] = []

    class Coordinator:
        def claim(self, _token: object) -> object:
            events.append("claim")
            return module.SubmissionClaim(created=True, record=None)

        def bind_job(self, _token: object, *, job_id: str, scheduler_status: str) -> object:
            events.append(f"bind:{job_id}:{scheduler_status}")
            raise OSError("durable coordinator write failed")

    def submit() -> object:
        events.append("submit")
        return module.SchedulerSubmission(job_id="123", status="PENDING")

    with pytest.raises(module.SubmissionAmbiguousError, match=r"accepted.*binding"):
        module.submit_claimed_once(expectation=_expectation(), coordinator=Coordinator(), submit=submit)

    assert events == ["claim", "submit", "bind:123:PENDING"]


def test_directory_fsync_uses_nofollow_directory_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _integration()
    original_open = os.open
    observed_flags: list[int] = []

    def recording_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        observed_flags.append(flags)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(module.os, "open", recording_open)
    module._fsync_directory(tmp_path)

    assert observed_flags == [os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)]


def test_uncertain_restart_never_calls_submit_twice(tmp_path: Path) -> None:
    module = _integration()
    calls = 0
    coordinator = module.SubmissionCoordinator(tmp_path)
    expectation = _expectation()

    def uncertain_submit() -> object:
        nonlocal calls
        calls += 1
        raise TimeoutError("scheduler response lost")

    with pytest.raises(module.SubmissionAmbiguousError):
        module.submit_claimed_once(expectation=expectation, coordinator=coordinator, submit=uncertain_submit)
    with pytest.raises(module.SubmissionAmbiguousError):
        module.submit_claimed_once(expectation=expectation, coordinator=coordinator, submit=uncertain_submit)

    assert calls == 1


def test_uncertain_restart_reconciles_original_job_without_resubmit(tmp_path: Path) -> None:
    module = _integration()
    calls = 0
    coordinator = module.SubmissionCoordinator(tmp_path)
    expectation = _expectation()

    def submit() -> object:
        nonlocal calls
        calls += 1
        raise TimeoutError("response lost after acceptance")

    with pytest.raises(module.SubmissionAmbiguousError):
        module.submit_claimed_once(expectation=expectation, coordinator=coordinator, submit=submit)
    outcome = module.submit_claimed_once(
        expectation=expectation,
        coordinator=coordinator,
        submit=submit,
        reconcile=lambda token: "4242" if token == expectation.token.token else None,
    )

    assert outcome.job_id == "4242"
    assert outcome.submitted_now is False
    assert calls == 1


def test_reconciled_job_binding_failure_remains_ambiguous(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _integration()
    coordinator = module.SubmissionCoordinator(tmp_path)
    expectation = _expectation()
    coordinator.claim(expectation.token)

    def failed_bind(_token: object, *, job_id: str, scheduler_status: str) -> object:
        raise OSError(f"cannot bind {job_id}:{scheduler_status}")

    monkeypatch.setattr(coordinator, "bind_job", failed_bind)
    with pytest.raises(module.SubmissionAmbiguousError, match=r"reconciled.*binding"):
        module.submit_claimed_once(
            expectation=expectation,
            coordinator=coordinator,
            submit=lambda: pytest.fail("must not resubmit"),
            reconcile=lambda _token: "4242",
        )


def test_result_requires_exact_token_job_and_runtime_ipsae_binding() -> None:
    module = _integration()
    expectation = _expectation(
        runtime_ipsae_source_revision="a" * 40,
        runtime_ipsae_binary_sha256="b" * 64,
    )
    result = _result(
        expectation,
        job_id="123",
        runtime_ipsae_source_revision="a" * 40,
        runtime_ipsae_binary_sha256="b" * 64,
    )

    module.validate_bound_submission_result(expectation, result, coordinator_job_id="123")
    with pytest.raises(ValueError, match="job"):
        module.validate_bound_submission_result(expectation, result, coordinator_job_id="999")
    with pytest.raises(ValueError, match=r"runtime.*iPSAE"):
        module.validate_bound_submission_result(
            expectation,
            _result(
                expectation,
                job_id="123",
                runtime_ipsae_source_revision="a" * 40,
                runtime_ipsae_binary_sha256="c" * 64,
            ),
            coordinator_job_id="123",
        )


def test_final_index_is_deterministic_and_verifies_exact_tree(tmp_path: Path) -> None:
    module = _integration()
    (tmp_path / "attempt").mkdir()
    (tmp_path / "attempt" / "result.json").write_text("result\n")
    (tmp_path / "expectation.json").write_text("expectation\n")

    first = module.finalize_evidence_index(tmp_path)
    second = module.read_evidence_index(first.index_path)
    assert first.index == second
    assert [entry.path for entry in first.index.entries] == ["attempt/result.json", "expectation.json"]
    assert module.verify_final_evidence_index(tmp_path, first.index).ok

    (tmp_path / "unexpected").write_text("extra\n")
    assert not module.verify_final_evidence_index(tmp_path, first.index).ok


def test_final_index_rejects_intermediate_symlink_escape(tmp_path: Path) -> None:
    module = _integration()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "forged.json").write_text("forged\n")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "submissions").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match=r"regular files|directory|symlink"):
        module.finalize_evidence_index(evidence)

    assert not (evidence / module.FINAL_INDEX_NAME).exists()


def test_final_index_streams_large_retained_artifacts_without_generic_size_policy(tmp_path: Path) -> None:
    module = _integration()
    retained = tmp_path / "slurm.log"
    with retained.open("wb") as handle:
        handle.truncate(module.MAX_BOUND_ARTIFACT_BYTES + 1)

    finalized = module.finalize_evidence_index(tmp_path)

    assert finalized.index.entries[0].path == "slurm.log"


def test_selected_processing_index_rejects_missing_and_extra_attempt_records(tmp_path: Path) -> None:
    module = _integration()
    expected = Path("submissions/0000-step-attempt-0001/expectation.json")
    (tmp_path / expected).parent.mkdir(parents=True)
    (tmp_path / expected).write_text("expectation\n")
    extra = tmp_path / "submission-coordinator/unexpected.json"
    extra.parent.mkdir()
    extra.write_text("extra\n")

    with pytest.raises(ValueError, match="inventory mismatch"):
        module.finalize_selected_evidence_index(tmp_path, (expected,))

    extra.unlink()
    result = Path("submissions/0000-step-attempt-0001/result.json")
    with pytest.raises(ValueError, match="inventory mismatch"):
        module.finalize_selected_evidence_index(tmp_path, (expected, result))


def test_selected_processing_index_rejects_preexisting_symlink(tmp_path: Path) -> None:
    module = _integration()
    expected = Path("submissions/0000-step-attempt-0001/expectation.json")
    (tmp_path / expected).parent.mkdir(parents=True)
    (tmp_path / expected).write_text("expectation\n")
    outside = tmp_path / "outside-index.json"
    outside.write_text('{"entries":[],"format_version":1}\n')
    (tmp_path / module.PROCESSING_INDEX_NAME).symlink_to(outside)

    with pytest.raises(ValueError, match=r"regular file|symlink|bound artifact"):
        module.finalize_selected_evidence_index(tmp_path, (expected,))


def test_immutable_write_detects_parent_replacement_race(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _integration()
    evidence = tmp_path / "evidence"
    (evidence / "submissions").mkdir(parents=True)
    moved = tmp_path / "moved"
    original_fsync = module.os.fsync
    replaced = False

    def replacing_fsync(descriptor: int) -> None:
        nonlocal replaced
        original_fsync(descriptor)
        if not replaced and module.stat.S_ISDIR(module.os.fstat(descriptor).st_mode):
            replaced = True
            (evidence / "submissions").rename(moved)
            (evidence / "submissions").mkdir()

    monkeypatch.setattr(module.os, "fsync", replacing_fsync)
    with pytest.raises(ValueError, match=r"replaced|unsafe"):
        module.write_immutable_descendant(evidence, Path("submissions/attempt/result.json"), b"{}\n")
