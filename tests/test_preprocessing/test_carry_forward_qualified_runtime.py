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

"""Carried Runtime complement-search and verified-adoption tests."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from tests.support.preprocessing_execution import (
    LocalExecutionFixture,
    configure_preprocessing_fakes,
    invoke_preprocessing_execution,
    load_failed_preprocessing_evidence,
    preprocessing_execution_fixture,
    preprocessing_invocations,
    skip_preprocessing_server_warmup,
)

from bspp.orchestration.contract.phase import canonical_mapping_digest
from bspp.orchestration.contract.phase_carry_forward import (
    AttemptCarryForwardContent,
    AttemptCarryForwardRecord,
    AttemptCarryForwardReference,
    AttemptCarryForwardVerificationReference,
    AttemptWorkspaceBinding,
    AttemptWorkspaceRootBinding,
    attempt_carry_forward_id,
)
from bspp.orchestration.contract.preprocessing_action import (
    PreprocessingDatabasePlacementCommandFailureEvidence,
    preprocessing_chunk_action_evidence_from_mapping,
)
from bspp.orchestration.runtime.preprocessing.carry_forward import (
    adopt_carried_a3ms,
    load_attempt_carry_forward_record,
)
from bspp.orchestration.runtime.preprocessing.execution import (
    reconcile_preprocessing_chunk_action_evidence,
    reconcile_preprocessing_chunk_action_evidence_for_finalization,
)

SUBMISSION_ID = "phase-submission-" + "8" * 64


def _carried_fixture(
    tmp_path: Path,
    *,
    legacy_paired: bool = False,
) -> tuple[LocalExecutionFixture, AttemptCarryForwardRecord, Path]:
    fixture = preprocessing_execution_fixture(tmp_path / "work", legacy_paired=legacy_paired)
    runspec = replace(
        fixture.runspec,
        attempt_id="attempt-0002",
        materialized_at="2026-08-20T13:00:00.000000Z",
        payload=replace(
            fixture.runspec.payload,
            database=replace(
                fixture.runspec.payload.database,
                source_manifest_projection="attempts/attempt-0002/database-source-manifest.json",
            ),
        ),
    )
    action = runspec.payload.actions[0]
    site = action.payload.site
    workspace_root = tmp_path / "work"
    roots = tuple(
        AttemptWorkspaceRootBinding(
            role=role,
            logical_root=logical,
            physical_root=logical,
        )
        for role, logical in (
            ("input", site.input_root),
            ("scratch-output", site.scratch_output_root),
            ("project-logs", site.project_logs_root),
            ("finished-msa", site.finished_msa_root),
            ("split-input", site.split_input_root),
            ("finished-input", site.finished_input_root),
        )
    )
    for root in roots:
        directory = Path(root.logical_root)
        directory.mkdir(parents=True, exist_ok=True)
        for candidate in tuple(directory.iterdir()):
            if candidate.is_file():
                candidate.unlink()
            elif candidate.is_dir():
                for child in tuple(candidate.iterdir()):
                    child.unlink()
                candidate.rmdir()
    workspace = AttemptWorkspaceBinding(
        phase_run_id=runspec.phase_run_id,
        target_attempt_id=runspec.attempt_id,
        target_action_id=action.action_id,
        workspace_root=str(workspace_root),
        roots=roots,
        search_input_physical_path=action.payload.search_argv[3],
        split_input_physical_path=action.payload.package.completed_input_source_path,
        identity_sentinel_path=str(workspace_root / "carry-forward-identity.json"),
        private_workspace_mount_path=str(workspace_root),
        source_path_mode="frozen-mount-inverse",
    )
    member = action.payload.expected_a3ms[0]
    source_bytes = b"#4,2\t1,1\n>AFDB_zeta\nAAAATT\n"
    private_source = workspace_root / "private-source" / "000000" / member.member_name
    private_source.parent.mkdir(parents=True)
    private_source.write_bytes(source_bytes)
    target_path = Path(action.payload.evidence.scratch_output_directory) / member.member_name
    content = (
        AttemptCarryForwardContent(
            source_action_id=action.action_id,
            target_action_id=action.action_id,
            member_name=member.member_name,
            source_ordinal=member.source_ordinal,
            record_identity=member.record_identity,
            source_header=member.source_header,
            source_declared_path=f"/predecessor/scratch/{member.member_name}",
            target_declared_path=str(target_path),
            source_physical_path=str(private_source),
            target_physical_path=str(target_path),
            source_private_mount_path=str(private_source),
            size_bytes=len(source_bytes),
            sha256=hashlib.sha256(source_bytes).hexdigest(),
        ),
    )
    verification = AttemptCarryForwardVerificationReference(
        attestation_id="phase-action-evidence-attestation-" + "1" * 64,
        attestation_digest="2" * 64,
        attestation_event_sequence=5,
        evidence_path="/predecessor/action-evidence.json",
        evidence_document_sha256="3" * 64,
        evidence_mapping_digest="4" * 64,
        terminal_event_digest="5" * 64,
        evidence_finished_at="2026-08-20T12:00:00.000000Z",
    )
    content_digest = canonical_mapping_digest({"schema_version": 1, "content": [item.to_mapping() for item in content]})
    remaining_bytes = b">protein-alpha\nCCCC:AAA\n>protein-mu\nGGGG:CC\n"
    identity = {
        "schema_version": 1,
        "phase_run_id": runspec.phase_run_id,
        "phase_kind": "preprocessing",
        "phase_plan_digest": runspec.phase_plan_digest,
        "input_set_identity_digest": "6" * 64,
        "scientific_identity_digest": "7" * 64,
        "source_attempt_id": "attempt-0001",
        "source_attempt_ordinal": 1,
        "source_runspec_digest": "9" * 64,
        "target_attempt_id": runspec.attempt_id,
        "target_attempt_ordinal": 2,
        "verification": verification.to_mapping(),
        "workspace": workspace.to_mapping(),
        "content": [item.to_mapping() for item in content],
        "content_digest": content_digest,
        "remaining_record_ordinals": [1, 2],
        "remaining_search_input_sha256": hashlib.sha256(remaining_bytes).hexdigest(),
        "declared_at": runspec.materialized_at,
    }
    record = AttemptCarryForwardRecord(
        attempt_carry_forward_id=attempt_carry_forward_id(identity),
        phase_run_id=runspec.phase_run_id,
        phase_kind="preprocessing",
        phase_plan_digest=runspec.phase_plan_digest,
        input_set_identity_digest="6" * 64,
        scientific_identity_digest="7" * 64,
        source_attempt_id="attempt-0001",
        source_attempt_ordinal=1,
        source_runspec_digest="9" * 64,
        target_attempt_id=runspec.attempt_id,
        target_attempt_ordinal=2,
        verification=verification,
        workspace=workspace,
        content=content,
        content_digest=content_digest,
        remaining_record_ordinals=(1, 2),
        remaining_search_input_sha256=hashlib.sha256(remaining_bytes).hexdigest(),
        declared_at=runspec.materialized_at,
    )
    reference = AttemptCarryForwardReference(
        attempt_carry_forward_id=record.attempt_carry_forward_id,
        digest=record.digest,
        location="attempts/attempt-0002/attempt-carry-forward.json",
    )
    runspec = replace(runspec, carry_forward=reference)
    runspec_path = tmp_path / "phase-runspec.json"
    runspec_path.write_text(json.dumps(runspec.to_mapping(), indent=2, sort_keys=True) + "\n")
    record_path = tmp_path / "attempt-carry-forward.json"
    record_path.write_text(json.dumps(record.to_mapping(), indent=2, sort_keys=True) + "\n")
    sentinel = {
        "schema_version": 1,
        "phase_run_id": record.phase_run_id,
        "attempt_id": record.target_attempt_id,
        "action_id": action.action_id,
        "attempt_carry_forward_id": record.attempt_carry_forward_id,
        "attempt_carry_forward_digest": record.digest,
        "phase_submission_id": SUBMISSION_ID,
        "workspace_mapping_digest": workspace.digest,
    }
    Path(workspace.identity_sentinel_path).write_text(json.dumps(sentinel, indent=2, sort_keys=True) + "\n")
    return (
        LocalExecutionFixture(
            runspec=runspec,
            runspec_path=runspec_path,
            evidence_path=fixture.evidence_path,
            invocation_path=fixture.invocation_path,
            database_placement_result_path=fixture.database_placement_result_path,
            database_placement_failure_path=fixture.database_placement_failure_path,
            database_source_root=fixture.database_source_root,
            database_placement_paths=fixture.database_placement_paths,
        ),
        record,
        record_path,
    )


@pytest.mark.parametrize("legacy_paired", [False, True], ids=["unpaired_paired", "legacy-paired"])
def test_carried_runtime_searches_only_complement_then_adopts_exact_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    legacy_paired: bool,
) -> None:
    fixture, record, record_path = _carried_fixture(tmp_path, legacy_paired=legacy_paired)
    configure_preprocessing_fakes(fixture, monkeypatch)
    monkeypatch.setenv("BSPP_FAKE_A3MS", "AFDB_alpha.a3m|AFDB_mu.a3m")
    skip_preprocessing_server_warmup(monkeypatch)

    result = invoke_preprocessing_execution(
        fixture,
        extra_args=(
            "--carry-forward-record",
            str(record_path),
            "--phase-submission-id",
            SUBMISSION_ID,
        ),
    )

    assert result.exit_code == 0, result.output
    evidence = preprocessing_chunk_action_evidence_from_mapping(json.loads(fixture.evidence_path.read_text()))
    assert evidence.carry_forward_adoption is not None
    assert evidence.carry_forward_adoption.attempt_carry_forward_id == record.attempt_carry_forward_id
    assert evidence.carry_forward_adoption.phase_submission_id == SUBMISSION_ID
    assert evidence.raw_search_evidence is not None
    assert evidence.raw_search_evidence.searched_source_ordinals == (1, 2)
    expected_members = (
        "AFDB_alpha.a3m",
        "AFDB_mu.a3m",
    )
    if legacy_paired:
        expected_members += ("2.a3m", "3.a3m")
    assert tuple(item.member_name for item in evidence.raw_search_evidence.artifacts) == expected_members
    assert tuple(item.modeled_chain_length for item in evidence.raw_search_evidence.artifacts[2:]) == (
        (4, 2) if legacy_paired else ()
    )
    action = fixture.action
    search_input = Path(action.payload.search_argv[3])
    assert search_input.read_bytes() == b">protein-alpha\nCCCC:AAA\n>protein-mu\nGGGG:CC\n"
    assert tuple(item.member_name for item in evidence.output_hashes if item.role == "a3m") == (
        "AFDB_zeta.a3m",
        "AFDB_alpha.a3m",
        "AFDB_mu.a3m",
    )
    search = next(item for item in preprocessing_invocations(fixture.invocation_path) if item["kind"] == "search")
    assert search["argv"][2] == action.payload.search_argv[3]
    carried = record.content[0]
    assert Path(carried.source_private_mount_path).stat().st_ino != Path(carried.target_declared_path).stat().st_ino
    reconcile_preprocessing_chunk_action_evidence(fixture.runspec, evidence, carry_forward_record=record)
    shutil.rmtree(evidence.raw_search_evidence.raw_search_output_directory)
    reconcile_preprocessing_chunk_action_evidence_for_finalization(fixture.runspec, evidence)


def test_carried_runspec_placement_failure_precedes_and_excludes_adoption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, record, _record_path = _carried_fixture(tmp_path)
    configure_preprocessing_fakes(fixture, monkeypatch)

    result = invoke_preprocessing_execution(fixture, placement_process_status=19)

    assert result.exit_code != 0
    evidence = load_failed_preprocessing_evidence(fixture.evidence_path)
    assert isinstance(
        evidence.database_placement,
        PreprocessingDatabasePlacementCommandFailureEvidence,
    )
    assert evidence.carry_forward_adoption is None
    assert evidence.command_outcomes == ()
    assert evidence.output_hashes == ()
    assert not fixture.invocation_path.exists()
    Path(record.content[0].target_declared_path).parent.mkdir(parents=True, exist_ok=True)
    adoption = adopt_carried_a3ms(
        fixture.runspec,
        record,
        phase_submission_id=SUBMISSION_ID,
        adopted_at="2026-08-20T13:01:00.000000Z",
    )
    with pytest.raises(ValueError, match="payload-free"):
        replace(evidence, carry_forward_adoption=adoption)


@pytest.mark.parametrize("sentinel_fault", ["drift", "missing", "symlink"])
def test_carried_runtime_rejects_sentinel_fault_before_search(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sentinel_fault: str,
) -> None:
    fixture, record, record_path = _carried_fixture(tmp_path)
    configure_preprocessing_fakes(fixture, monkeypatch)
    skip_preprocessing_server_warmup(monkeypatch)
    sentinel = Path(record.workspace.identity_sentinel_path)
    if sentinel_fault == "drift":
        mapping = json.loads(sentinel.read_text())
        mapping["phase_submission_id"] = "phase-submission-" + "0" * 64
        sentinel.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n")
    elif sentinel_fault == "missing":
        sentinel.unlink()
    else:
        document = sentinel.read_bytes()
        target = sentinel.with_name("sentinel-target.json")
        sentinel.unlink()
        target.write_bytes(document)
        sentinel.symlink_to(target)

    result = invoke_preprocessing_execution(
        fixture,
        extra_args=(
            "--carry-forward-record",
            str(record_path),
            "--phase-submission-id",
            SUBMISSION_ID,
        ),
    )

    assert result.exit_code != 0
    assert "identity sentinel" in result.output
    assert not fixture.invocation_path.exists()


def test_staged_carry_record_rejects_noncanonical_bytes(tmp_path: Path) -> None:
    _fixture, _record, record_path = _carried_fixture(tmp_path)
    mapping = json.loads(record_path.read_text())
    record_path.write_text(json.dumps(mapping, separators=(",", ":")) + "\n")
    with pytest.raises(ValueError, match="canonical JSON"):
        load_attempt_carry_forward_record(record_path)


@pytest.mark.parametrize(
    ("fault", "target_published"),
    [
        ("source-hash", False),
        ("temporary-write", False),
        ("temporary-fsync", False),
        ("no-replace-link", False),
        ("directory-fsync", True),
        ("published-mismatch", True),
        ("existing-target", True),
    ],
)
def test_verified_copy_faults_never_replace_source_or_leave_temporary_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
    target_published: bool,
) -> None:
    fixture, record, _record_path = _carried_fixture(tmp_path)
    import bspp.orchestration.runtime.preprocessing.carry_forward as carry_runtime

    content = record.content[0]
    source = Path(content.source_private_mount_path)
    target = Path(content.target_declared_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    source_bytes = source.read_bytes()
    existing_bytes = b">existing\nCCCC\n"

    if fault == "source-hash":
        source.write_bytes(b">AFDB_zeta\nBBBB\n")
    elif fault == "temporary-write":

        def fail_partial_write(handle: object, data: bytes) -> None:
            handle.write(data[:4])  # type: ignore[attr-defined]
            raise OSError("injected temporary write failure")

        monkeypatch.setattr(carry_runtime, "_write_all", fail_partial_write)
    elif fault in {"temporary-fsync", "directory-fsync"}:
        real_fsync = carry_runtime.os.fsync
        calls = 0

        def fail_selected_fsync(descriptor: int) -> None:
            nonlocal calls
            calls += 1
            if calls == (1 if fault == "temporary-fsync" else 2):
                raise OSError(f"injected {fault}")
            real_fsync(descriptor)

        monkeypatch.setattr(carry_runtime.os, "fsync", fail_selected_fsync)
    elif fault == "no-replace-link":

        def fail_link(_source: Path, _target: Path) -> None:
            raise OSError("injected no-replace link failure")

        monkeypatch.setattr(carry_runtime.os, "link", fail_link)
    elif fault == "published-mismatch":
        real_read = carry_runtime._read_exact_regular_no_follow

        def changed_target_read(path: Path, *, expected_size: int) -> bytes:
            data = real_read(path, expected_size=expected_size)
            if path == target:
                return b"X" + data[1:]
            return data

        monkeypatch.setattr(carry_runtime, "_read_exact_regular_no_follow", changed_target_read)
    elif fault == "existing-target":
        target.write_bytes(existing_bytes)

    with pytest.raises((OSError, ValueError)):
        adopt_carried_a3ms(
            fixture.runspec,
            record,
            phase_submission_id=SUBMISSION_ID,
            adopted_at="2026-08-20T13:02:00.000000Z",
        )

    if fault == "source-hash":
        assert source.read_bytes() == b">AFDB_zeta\nBBBB\n"
    else:
        assert source.read_bytes() == source_bytes
    assert target.exists() is target_published
    if fault == "existing-target":
        assert target.read_bytes() == existing_bytes
    assert not tuple(target.parent.glob(f".{target.name}.carry-*.tmp"))


def test_failed_evidence_distinguishes_pre_adoption_from_post_adoption_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before, _record, before_record_path = _carried_fixture(tmp_path / "before")
    configure_preprocessing_fakes(before, monkeypatch, search_mode="fail")
    monkeypatch.setenv("BSPP_FAKE_A3MS", "AFDB_alpha.a3m|AFDB_mu.a3m")
    skip_preprocessing_server_warmup(monkeypatch)
    before_result = invoke_preprocessing_execution(
        before,
        extra_args=(
            "--carry-forward-record",
            str(before_record_path),
            "--phase-submission-id",
            SUBMISSION_ID,
        ),
    )
    assert before_result.exit_code != 0
    before_evidence = load_failed_preprocessing_evidence(before.evidence_path)
    assert before_evidence.carry_forward_adoption is None

    after, after_record, after_record_path = _carried_fixture(tmp_path / "after")
    configure_preprocessing_fakes(after, monkeypatch)
    monkeypatch.setenv("BSPP_FAKE_A3MS", "AFDB_alpha.a3m|AFDB_mu.a3m")
    skip_preprocessing_server_warmup(monkeypatch)
    import bspp.orchestration.runtime.preprocessing.execution as execution

    real_run = execution.subprocess.run

    def fail_record_ls(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        argv = args[0]
        assert isinstance(argv, tuple)
        if len(argv) > 3 and argv[3] == "record-ls":
            return subprocess.CompletedProcess(argv, 23)
        return real_run(*args, **kwargs)  # type: ignore[arg-type,return-value]

    monkeypatch.setattr(execution.subprocess, "run", fail_record_ls)
    after_result = invoke_preprocessing_execution(
        after,
        extra_args=(
            "--carry-forward-record",
            str(after_record_path),
            "--phase-submission-id",
            SUBMISSION_ID,
        ),
    )
    assert after_result.exit_code != 0
    after_evidence = load_failed_preprocessing_evidence(after.evidence_path)
    assert after_evidence.carry_forward_adoption is not None
    assert after_evidence.carry_forward_adoption.attempt_carry_forward_id == after_record.attempt_carry_forward_id
