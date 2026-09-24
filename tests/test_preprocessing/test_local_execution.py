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

"""Process-boundary tests for one declared preprocessing Runtime Action."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from click.testing import CliRunner, Result
from tests.support.preprocessing_execution import LocalExecutionFixture
from tests.support.preprocessing_execution import (
    configure_preprocessing_fakes as _configure_fakes,
)
from tests.support.preprocessing_execution import (
    invoke_preprocessing_execution as _invoke,
)
from tests.support.preprocessing_execution import (
    load_failed_preprocessing_evidence as _load_failed,
)
from tests.support.preprocessing_execution import (
    preprocessing_chunk_bytes as _chunk_bytes,
)
from tests.support.preprocessing_execution import (
    preprocessing_execution_fixture as _fixture,
)
from tests.support.preprocessing_execution import (
    preprocessing_invocations as _invocations,
)
from tests.support.preprocessing_execution import (
    skip_preprocessing_server_warmup as _skip_server_warmup,
)

from bspp.orchestration.contract.phase import PreprocessingRuntimeAction
from bspp.orchestration.contract.preprocessing_action import (
    PREPROCESSING_COMMAND_ORDER,
    PreprocessingCommandOutcome,
    PreprocessingDatabasePlacementCommandFailureEvidence,
    preprocessing_chunk_action_evidence_from_mapping,
    preprocessing_command_outcome_from_mapping,
)
from bspp.orchestration.contract.preprocessing_runtime import PREPROCESSING_ADAPTER_VERSION
from bspp.orchestration.runtime.cli import cli
from bspp.orchestration.runtime.preprocessing.execution import (
    PRODUCTION_SCIENTIFIC_KERNEL_LAUNCHER,
    RECORD_LS_SCRIPT,
    ScientificKernelLauncher,
    ScientificKernelProcess,
    reconcile_preprocessing_chunk_action_evidence,
    reconcile_preprocessing_chunk_action_evidence_for_finalization,
)


def _launcher_after_shutdown(callback: Callable[[], None]) -> ScientificKernelLauncher:
    """Run one filesystem transition after the real kernel reaches terminal state."""

    def shutdown(
        process: ScientificKernelProcess,
        action: PreprocessingRuntimeAction,
    ) -> PreprocessingCommandOutcome:
        outcome = PRODUCTION_SCIENTIFIC_KERNEL_LAUNCHER.shutdown(process, action)
        callback()
        return outcome

    return replace(PRODUCTION_SCIENTIFIC_KERNEL_LAUNCHER, shutdown=shutdown)


def _invoke_runtime_cli(fixture: LocalExecutionFixture) -> Result:
    return CliRunner().invoke(
        cli,
        [
            "preprocessing",
            "execute-chunk",
            "--phase-runspec",
            str(fixture.runspec_path),
            "--action-id",
            fixture.action.action_id,
            "--write-evidence",
            str(fixture.evidence_path),
            "--database-placement-result",
            str(fixture.database_placement_result_path),
            "--database-placement-failure",
            str(fixture.database_placement_failure_path),
            "--placement-process-status",
            "0",
        ],
    )


@pytest.mark.parametrize("legacy_paired", [False, True], ids=["unpaired_paired", "legacy-paired"])
def test_runtime_cli_executes_exact_chunk_and_emits_reconcilable_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    legacy_paired: bool,
) -> None:
    fixture = _fixture(tmp_path, legacy_paired=legacy_paired)
    _configure_fakes(fixture, monkeypatch)
    _skip_server_warmup(monkeypatch)

    result = _invoke(fixture)

    assert result.exit_code == 0, result.output
    evidence = preprocessing_chunk_action_evidence_from_mapping(json.loads(fixture.evidence_path.read_text()))
    assert evidence.adapter_version == "preprocessing-scientific-backend-v3"
    assert evidence.adapter_version == PREPROCESSING_ADAPTER_VERSION
    assert evidence.outcome == "succeeded"
    assert tuple(item.command_kind for item in evidence.command_outcomes) == PREPROCESSING_COMMAND_ORDER
    assert evidence.phase_run_id == fixture.runspec.phase_run_id
    assert evidence.attempt_id == fixture.runspec.attempt_id
    assert evidence.phase_runspec_digest == fixture.runspec.digest
    assert evidence.action_id == fixture.action.action_id
    assert evidence.raw_search_evidence is not None
    assert evidence.raw_search_evidence.raw_search_output_directory == (
        fixture.action.payload.evidence.raw_search_output_directory
    )
    assert evidence.raw_search_evidence.searched_source_ordinals == (0, 1, 2)
    numeric = evidence.raw_search_evidence.artifacts[3:]
    assert tuple(item.role for item in evidence.raw_search_evidence.artifacts[:3]) == ("named-a3m",) * 3
    assert tuple(item.role for item in numeric) == ("numeric-placeholder",) * (3 if legacy_paired else 0)
    assert tuple(item.member_name for item in numeric) == (("3.a3m", "4.a3m", "5.a3m") if legacy_paired else ())
    assert tuple(item.modeled_chain_length for item in numeric) == ((3, 4, 2) if legacy_paired else ())
    assert {path.name for path in Path(fixture.action.payload.evidence.scratch_output_directory).iterdir()} == {
        "AFDB_zeta.a3m",
        "AFDB_alpha.a3m",
        "AFDB_mu.a3m",
    }
    assert evidence.paired_evidence.record_lines is not None
    record_names = tuple(line.rsplit("/", maxsplit=1)[-1] for line in evidence.paired_evidence.record_lines)
    assert record_names == ("AFDB_alpha.a3m", "AFDB_mu.a3m", "AFDB_zeta.a3m")
    reconcile_preprocessing_chunk_action_evidence(fixture.runspec, evidence)

    package = fixture.action.payload.package
    assert Path(package.durable_tar_path).stat().st_size > 0
    assert Path(package.durable_lz4_path).stat().st_size > 0
    assert Path(package.completed_input_path).read_bytes() == _chunk_bytes()
    assert not Path(package.completed_input_source_path).exists()
    subprocess.run(["/usr/bin/lz4", "-t", package.durable_lz4_path], check=True, capture_output=True)

    invocations = _invocations(fixture.invocation_path)
    assert [item["kind"] for item in invocations] == ["gpuserver", "search", "gpuserver-stopped"]
    assert invocations[0]["argv"] == list(fixture.action.payload.gpuserver_argv[1:])
    assert invocations[1]["argv"] == list(fixture.action.payload.search_argv[1:])
    assert invocations[0]["cuda"] == "0"
    assert invocations[1]["cuda"] == "0"
    execution_module = __import__("bspp.orchestration.runtime.preprocessing.execution", fromlist=["x"])
    assert Path(execution_module.__file__).suffix == ".py"


def test_gpuserver_is_stopped_before_record_and_packaging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    _configure_fakes(fixture, monkeypatch)
    _skip_server_warmup(monkeypatch)
    result = _invoke(fixture)

    assert result.exit_code == 0, result.output
    evidence = preprocessing_chunk_action_evidence_from_mapping(json.loads(fixture.evidence_path.read_text()))
    assert evidence.command_outcomes[0].disposition == "terminated-by-adapter"
    assert _invocations(fixture.invocation_path)[-1]["kind"] == "gpuserver-stopped"
    assert tuple(item.command_kind for item in evidence.command_outcomes) == PREPROCESSING_COMMAND_ORDER
    assert Path(fixture.action.payload.package.durable_lz4_path).is_file()


@pytest.mark.parametrize("legacy_paired", [False, True], ids=["unpaired_paired", "legacy-paired"])
def test_mixed_chain_arities_preserve_named_closure_and_source_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    legacy_paired: bool,
) -> None:
    members = tuple(f"AFDB_{name}.a3m" for name in ("mono", "hetero", "trimer", "tetramer", "homo"))
    fixture = _fixture(
        tmp_path,
        legacy_paired=legacy_paired,
        expected_a3m_members=members,
        fasta_bytes=b">mono\nAAAA\n>hetero\nCC:GGG\n>trimer\nAA:CC:GG\n>tetramer\nA:C:G:T\n>homo\nAAA:AAA\n",
    )
    _configure_fakes(fixture, monkeypatch)
    _skip_server_warmup(monkeypatch)
    result = _invoke(fixture)

    assert result.exit_code == 0, result.output
    evidence = preprocessing_chunk_action_evidence_from_mapping(json.loads(fixture.evidence_path.read_text()))
    raw = evidence.raw_search_evidence
    assert raw is not None
    assert raw.searched_source_ordinals == (0, 1, 2, 3, 4)
    assert tuple(item.member_name for item in raw.artifacts[:5]) == members
    assert tuple(item.source_ordinal for item in raw.artifacts[:5]) == (0, 1, 2, 3, 4)
    assert tuple(item.raw_query_id for item in raw.artifacts[5:]) == (tuple(range(5, 11)) if legacy_paired else ())
    if legacy_paired:
        assert Path(raw.artifacts[-1].path).read_bytes() == b"#3\t2\n"
    for artifact in raw.artifacts[:5]:
        assert (
            Path(artifact.path).read_bytes()
            == (Path(fixture.action.payload.evidence.scratch_output_directory) / artifact.member_name).read_bytes()
        )
    reconcile_preprocessing_chunk_action_evidence(fixture.runspec, evidence)
    reconcile_preprocessing_chunk_action_evidence_for_finalization(fixture.runspec, evidence)


@pytest.mark.parametrize("mutation", ["kind", "size", "mtime-ns", "alias-resolution", "extra", "missing"])
def test_direct_source_drift_after_kernel_rejects_all_output_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    fixture = _fixture(tmp_path)
    _configure_fakes(fixture, monkeypatch)
    _skip_server_warmup(monkeypatch)

    def mutate_after_shutdown() -> None:
        members = fixture.runspec.payload.database.source_manifest.members
        first = fixture.database_source_root / members[0].source_path
        second = fixture.database_source_root / members[1].source_path
        if mutation == "kind":
            first.unlink()
            first.mkdir()
        elif mutation == "size":
            first.chmod(0o600)
            first.write_bytes(b"changed-size")
            os.utime(first, ns=(members[0].mtime_ns, members[0].mtime_ns))
        elif mutation == "mtime-ns":
            os.utime(first, ns=(members[0].mtime_ns + 1, members[0].mtime_ns + 1))
        elif mutation == "alias-resolution":
            first.unlink()
            first.symlink_to(second.name)
        elif mutation == "extra":
            (fixture.database_source_root / "unexpected").write_bytes(b"extra")
        else:
            first.unlink()

    result = _invoke(
        fixture,
        scientific_kernel_launcher=_launcher_after_shutdown(mutate_after_shutdown),
    )

    assert result.exit_code != 0
    evidence = _load_failed(fixture.evidence_path)
    assert evidence.database_placement.science_started is True
    assert evidence.database_placement.post_science_observation is not None
    assert evidence.database_placement.result is not None
    assert not evidence.database_placement.post_science_observation.matches(
        evidence.database_placement.result.pre_science_observation
    )
    assert tuple(item.command_kind for item in evidence.command_outcomes) == ("gpuserver", "search")
    assert evidence.output_hashes == ()
    assert evidence.raw_search_evidence is None
    for path in (
        fixture.action.payload.evidence.durable_log_path,
        fixture.action.payload.evidence.durable_record_path,
        fixture.action.payload.package.durable_tar_path,
        fixture.action.payload.package.durable_lz4_path,
        fixture.action.payload.package.completed_input_path,
    ):
        assert not Path(path).exists()


@pytest.mark.parametrize(
    "root_mutation",
    ["missing", "wrong-kind", "symlink", "substitution", "open-denied", "traversal-denied"],
)
def test_unobservable_selected_root_after_kernel_publishes_observation_failure_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    root_mutation: str,
) -> None:
    fixture = _fixture(tmp_path)
    _configure_fakes(fixture, monkeypatch)
    _skip_server_warmup(monkeypatch)
    real_open = os.open
    real_listdir = os.listdir
    displaced_root = tmp_path / "displaced-database-source"

    if root_mutation in {"substitution", "open-denied"}:

        def intercept_root_open(
            path: os.PathLike[str] | str,
            flags: int,
            mode: int = 0o777,
            *,
            dir_fd: int | None = None,
        ) -> int:
            if Path(path) == fixture.database_source_root and dir_fd is None:
                if root_mutation == "open-denied":
                    raise PermissionError("injected selected-root open denial")
                fixture.database_source_root.rename(displaced_root)
                fixture.database_source_root.mkdir()
            return real_open(path, flags, mode, dir_fd=dir_fd)

        monkeypatch.setattr(os, "open", intercept_root_open)
    elif root_mutation == "traversal-denied":

        def deny_root_traversal(path: int | os.PathLike[str] | str) -> list[str]:
            if isinstance(path, int):
                raise PermissionError("injected selected-root traversal denial")
            return real_listdir(path)

        monkeypatch.setattr(os, "listdir", deny_root_traversal)

    def mutate_after_shutdown() -> None:
        if root_mutation in {"missing", "wrong-kind", "symlink"}:
            if root_mutation == "symlink":
                fixture.database_source_root.rename(displaced_root)
                fixture.database_source_root.symlink_to(displaced_root, target_is_directory=True)
            else:
                for member in fixture.database_source_root.iterdir():
                    member.unlink()
                fixture.database_source_root.rmdir()
                if root_mutation == "wrong-kind":
                    fixture.database_source_root.write_bytes(b"not-a-directory")

    result = _invoke(
        fixture,
        scientific_kernel_launcher=_launcher_after_shutdown(mutate_after_shutdown),
    )

    assert result.exit_code != 0
    assert fixture.evidence_path.is_file()
    evidence = _load_failed(fixture.evidence_path)
    post = evidence.database_placement.post_science_observation
    assert evidence.database_placement.science_started is True
    assert post is not None
    assert post.verification == "observation-failed"
    assert post.error and len(post.error) <= 2048
    assert tuple(item.command_kind for item in evidence.command_outcomes) == ("gpuserver", "search")
    assert evidence.output_hashes == ()
    assert evidence.raw_search_evidence is None
    for path in (
        fixture.action.payload.evidence.durable_log_path,
        fixture.action.payload.evidence.durable_record_path,
        fixture.action.payload.package.durable_tar_path,
        fixture.action.payload.package.durable_lz4_path,
        fixture.action.payload.package.completed_input_path,
    ):
        assert not Path(path).exists()


def test_execute_chunk_rejects_contradictory_result_and_failure_documents_before_science(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    _configure_fakes(fixture, monkeypatch)
    fixture.database_placement_failure_path.write_bytes(b"contradictory")

    result = _invoke(fixture)

    assert result.exit_code != 0
    assert "cannot coexist" in result.output
    evidence = _load_failed(fixture.evidence_path)
    placement = evidence.database_placement
    assert isinstance(placement, PreprocessingDatabasePlacementCommandFailureEvidence)
    assert placement.classification == "evidence-reconciliation-failed"
    assert placement.placement_process_status == 0
    assert placement.result is None
    assert placement.result_digest is None
    assert not fixture.invocation_path.exists()


@pytest.mark.parametrize("authority", ["absent", "malformed"])
def test_missing_or_malformed_placement_authority_publishes_reconciled_failed_action_before_science(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    authority: str,
) -> None:
    fixture = _fixture(tmp_path)
    _configure_fakes(fixture, monkeypatch)
    fixture.database_placement_result_path.chmod(0o600)
    if authority == "absent":
        fixture.database_placement_result_path.unlink()
    else:
        fixture.database_placement_result_path.write_text('{"unknown": {}}\n')

    result = _invoke(fixture)

    assert result.exit_code != 0
    evidence = _load_failed(fixture.evidence_path)
    assert evidence.placement_process_status == 0
    placement = evidence.database_placement
    assert isinstance(placement, PreprocessingDatabasePlacementCommandFailureEvidence)
    assert placement.classification == "evidence-reconciliation-failed"
    assert placement.placement_process_status == 0
    assert placement.result is None
    assert placement.result_digest is None
    assert evidence.command_outcomes == ()
    assert evidence.output_hashes == ()
    assert evidence.raw_search_evidence is None
    assert not fixture.invocation_path.exists()


def test_nonzero_placement_status_after_exact_result_publishes_reconciled_failed_action_before_science(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    _configure_fakes(fixture, monkeypatch)

    result = _invoke(fixture, placement_process_status=23)

    assert result.exit_code != 0
    evidence = _load_failed(fixture.evidence_path)
    assert evidence.placement_process_status == 23
    placement = evidence.database_placement
    assert isinstance(placement, PreprocessingDatabasePlacementCommandFailureEvidence)
    assert placement.classification == "nonzero-after-result"
    assert placement.placement_process_status == 23
    assert placement.result is not None
    assert placement.science_started is False
    assert evidence.command_outcomes == ()
    assert evidence.output_hashes == ()
    assert evidence.raw_search_evidence is None
    assert evidence.paired_evidence.record_lines is None
    assert evidence.paired_evidence.log_lines is None
    assert evidence.archive_evidence.tar_size_bytes is None
    assert evidence.archive_evidence.lz4_size_bytes is None
    assert evidence.archive_evidence.tar_members is None
    assert not fixture.invocation_path.exists()
    with pytest.raises(ValueError, match="requires successful action evidence"):
        reconcile_preprocessing_chunk_action_evidence_for_finalization(fixture.runspec, evidence)
    published = fixture.evidence_path.read_bytes()
    assert fixture.evidence_path.stat().st_mode & 0o777 == 0o444
    retry = _invoke(fixture, placement_process_status=24)
    assert retry.exit_code != 0
    assert fixture.evidence_path.read_bytes() == published


def test_execution_api_and_action_contract_require_exact_placement_process_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bspp.orchestration.runtime.preprocessing.execution as execution

    parameter = inspect.signature(execution.execute_preprocessing_chunk_action).parameters["placement_process_status"]
    assert parameter.default is inspect.Parameter.empty
    fixture = _fixture(tmp_path)
    _configure_fakes(fixture, monkeypatch)
    result = _invoke(fixture)
    assert result.exit_code == 0, result.output
    mapping = json.loads(fixture.evidence_path.read_text())
    assert mapping["placement_process_status"] == 0

    missing = deepcopy(mapping)
    missing.pop("placement_process_status")
    with pytest.raises(ValueError, match="placement_process_status"):
        preprocessing_chunk_action_evidence_from_mapping(missing)
    for invalid in (True, -1, 256):
        tampered = deepcopy(mapping)
        tampered["placement_process_status"] = invalid
        with pytest.raises(ValueError, match=r"placement[_ ]process[_ ]status"):
            preprocessing_chunk_action_evidence_from_mapping(tampered)


@pytest.mark.parametrize(
    "forgery",
    ["successful-outcome", "command", "output", "paired-observation", "archive-observation"],
)
def test_placement_command_failure_action_rejects_science_or_payload_forgery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    forgery: str,
) -> None:
    fixture = _fixture(tmp_path)
    _configure_fakes(fixture, monkeypatch)
    result = _invoke(fixture, placement_process_status=23)
    assert result.exit_code != 0
    mapping = json.loads(fixture.evidence_path.read_text())

    if forgery == "successful-outcome":
        mapping["outcome"] = "succeeded"
        mapping["error"] = None
    elif forgery == "command":
        mapping["command_outcomes"] = [
            {
                "schema_version": 1,
                "command_kind": "gpuserver",
                "command_digest": "a" * 64,
                "disposition": "completed",
                "return_code": 0,
            }
        ]
    elif forgery == "output":
        mapping["output_hashes"] = [
            {
                "schema_version": 1,
                "role": "log",
                "path": fixture.action.payload.evidence.durable_log_path,
                "size_bytes": 1,
                "sha256": "a" * 64,
                "member_name": None,
            }
        ]
    elif forgery == "paired-observation":
        mapping["paired_evidence"]["record_lines"] = ["observed"]
    else:
        mapping["archive_evidence"]["tar_size_bytes"] = 1

    with pytest.raises(ValueError):
        preprocessing_chunk_action_evidence_from_mapping(mapping)


@pytest.mark.parametrize(
    "mode", ["missing", "extra", "empty", "malformed", "nonfile", "symlink", "special", "unexpected-numeric"]
)
def test_invalid_or_undeclared_scientific_outputs_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    fixture = _fixture(tmp_path)
    _configure_fakes(fixture, monkeypatch, search_mode=mode)
    _skip_server_warmup(monkeypatch)

    result = _invoke(fixture)

    assert result.exit_code != 0
    assert "Traceback" not in result.output
    evidence = _load_failed(fixture.evidence_path)
    assert tuple(item.command_kind for item in evidence.command_outcomes) == ("gpuserver", "search")
    assert not Path(fixture.action.payload.package.durable_lz4_path).exists()
    assert not Path(fixture.action.payload.package.completed_input_path).exists()


@pytest.mark.parametrize(
    "mode",
    [
        "placeholder-missing",
        "placeholder-empty",
        "placeholder-malformed",
        "placeholder-length-drift",
        "placeholder-name-drift",
        "placeholder-symlink",
        "placeholder-special",
    ],
)
def test_invalid_numeric_raw_placeholders_fail_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    fixture = _fixture(tmp_path, legacy_paired=True)
    _configure_fakes(fixture, monkeypatch, search_mode=mode)
    _skip_server_warmup(monkeypatch)

    result = _invoke(fixture)

    assert result.exit_code != 0
    evidence = _load_failed(fixture.evidence_path)
    assert evidence.raw_search_evidence is None
    staging = Path(fixture.action.payload.evidence.scratch_output_directory)
    assert tuple(staging.iterdir()) == ()
    assert tuple(item.command_kind for item in evidence.command_outcomes) == ("gpuserver", "search")


def test_record_ls_uses_fixed_safe_glob_vector_and_nonzero_fails_before_packaging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    _configure_fakes(fixture, monkeypatch)
    _skip_server_warmup(monkeypatch)
    real_run = subprocess.run

    def fail_record_ls(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        argv = args[0]
        assert isinstance(argv, tuple)
        if len(argv) > 3 and argv[3] == "record-ls":
            assert argv == (
                "/bin/bash",
                "-c",
                RECORD_LS_SCRIPT,
                "record-ls",
                fixture.action.payload.evidence.scratch_output_directory,
            )
            return subprocess.CompletedProcess(argv, 23)
        return real_run(*args, **kwargs)  # type: ignore[arg-type,return-value]

    monkeypatch.setattr(subprocess, "run", fail_record_ls)

    result = _invoke(fixture)

    assert result.exit_code != 0
    assert "record-ls command failed with return code 23" in result.output
    assert "Traceback" not in result.output
    evidence = _load_failed(fixture.evidence_path)
    assert tuple(item.command_kind for item in evidence.command_outcomes) == ("gpuserver", "search", "record-ls")
    assert evidence.command_outcomes[-1].return_code == 23
    assert not Path(fixture.action.payload.package.scratch_tar_path).exists()
    assert not Path(fixture.action.payload.package.scratch_lz4_path).exists()


def test_search_failure_and_early_gpuserver_exit_preserve_valid_failed_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_fixture = _fixture(tmp_path / "search")
    _configure_fakes(search_fixture, monkeypatch, search_mode="fail")
    _skip_server_warmup(monkeypatch)
    search_result = _invoke(search_fixture)
    assert search_result.exit_code != 0
    search_evidence = _load_failed(search_fixture.evidence_path)
    assert tuple(item.command_kind for item in search_evidence.command_outcomes) == ("gpuserver", "search")
    assert search_evidence.command_outcomes[-1].return_code == 7
    expected_log = b"search failure stdout\nsearch failure stderr\n"
    scratch_log = Path(search_fixture.action.payload.evidence.scratch_log_path)
    durable_log = Path(search_fixture.action.payload.evidence.durable_log_path)
    assert scratch_log.read_bytes() == durable_log.read_bytes() == expected_log
    assert search_evidence.paired_evidence.log_lines == ("search failure stdout", "search failure stderr")
    assert len(search_evidence.output_hashes) == 1
    log_hash = search_evidence.output_hashes[0]
    assert (log_hash.role, log_hash.path, log_hash.size_bytes, log_hash.sha256) == (
        "log",
        str(durable_log),
        len(expected_log),
        hashlib.sha256(expected_log).hexdigest(),
    )
    assert search_evidence.raw_search_evidence is None
    assert not Path(search_fixture.action.payload.package.durable_tar_path).exists()
    assert not Path(search_fixture.action.payload.package.durable_lz4_path).exists()
    assert not Path(search_fixture.action.payload.package.completed_input_path).exists()
    assert _invocations(search_fixture.invocation_path)[-1]["kind"] == "gpuserver-stopped"
    scratch_log.unlink()
    reconcile_preprocessing_chunk_action_evidence(search_fixture.runspec, search_evidence)
    # Keep parsed lines identical so the exact-byte hash must catch this change.
    durable_log.write_bytes(expected_log.replace(b"\n", b"\r\n"))
    with pytest.raises(ValueError, match="output hash does not match"):
        reconcile_preprocessing_chunk_action_evidence(search_fixture.runspec, search_evidence)

    server_fixture = _fixture(tmp_path / "server")
    _configure_fakes(server_fixture, monkeypatch, gpuserver_exit=True)
    server_result = _invoke(server_fixture)
    assert server_result.exit_code != 0
    server_evidence = _load_failed(server_fixture.evidence_path)
    assert tuple(item.command_kind for item in server_evidence.command_outcomes) == ("gpuserver",)
    assert server_evidence.command_outcomes[0].disposition == "exited-unexpectedly"
    assert server_evidence.command_outcomes[0].return_code == 9
    assert "gpuserver exited before search" in server_evidence.error
    assert Path(server_fixture.action.payload.evidence.durable_log_path).read_bytes() == b""
    assert server_evidence.paired_evidence.log_lines == ()


@pytest.mark.parametrize("fault", ["collision", "parent-file", "source-symlink"])
def test_failed_log_retention_preserves_primary_error_and_never_replaces_foreign_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    fixture = _fixture(tmp_path)
    _configure_fakes(fixture, monkeypatch, search_mode="fail")
    durable_log = Path(fixture.action.payload.evidence.durable_log_path)
    scratch_log = Path(fixture.action.payload.evidence.scratch_log_path)
    foreign_file = tmp_path / "foreign-log"
    foreign_file.write_bytes(b"foreign bytes\n")

    def inject_fault() -> None:
        if fault == "collision":
            durable_log.parent.mkdir(parents=True, exist_ok=True)
            durable_log.write_bytes(foreign_file.read_bytes())
        elif fault == "parent-file":
            durable_log.parent.parent.mkdir(parents=True, exist_ok=True)
            durable_log.parent.write_bytes(foreign_file.read_bytes())
        else:
            scratch_log.unlink()
            scratch_log.symlink_to(foreign_file)

    result = _invoke(fixture, scientific_kernel_launcher=_launcher_after_shutdown(inject_fault))

    assert result.exit_code != 0
    evidence = _load_failed(fixture.evidence_path)
    assert "search command failed with return code 7" in evidence.error
    assert "failed to retain preprocessing diagnostic log" in evidence.error
    assert foreign_file.read_bytes() == b"foreign bytes\n"
    if fault == "collision":
        assert durable_log.read_bytes() == b"foreign bytes\n"
        assert "File exists" in evidence.error
    else:
        assert not durable_log.exists()
        assert evidence.paired_evidence.log_lines is None
        assert evidence.output_hashes == ()
        if fault == "parent-file":
            assert durable_log.parent.read_bytes() == b"foreign bytes\n"
        else:
            assert "regular non-symlink" in evidence.error


def test_preflight_failure_does_not_retain_a_preexisting_scratch_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    _configure_fakes(fixture, monkeypatch)
    scratch_log = Path(fixture.action.payload.evidence.scratch_log_path)
    scratch_log.parent.mkdir(parents=True)
    scratch_log.write_bytes(b"previous invocation\n")

    result = _invoke(fixture)

    assert result.exit_code != 0
    evidence = _load_failed(fixture.evidence_path)
    assert evidence.command_outcomes == ()
    assert evidence.paired_evidence.log_lines is None
    assert evidence.output_hashes == ()
    assert not Path(fixture.action.payload.evidence.durable_log_path).exists()
    assert scratch_log.read_bytes() == b"previous invocation\n"


def test_failed_cleanup_does_not_publish_a_nonterminal_kernel_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    _configure_fakes(fixture, monkeypatch, search_mode="fail")
    kernels: list[ScientificKernelProcess] = []

    def fail_shutdown(
        process: ScientificKernelProcess, _action: PreprocessingRuntimeAction
    ) -> PreprocessingCommandOutcome:
        kernels.append(process)
        raise OSError("injected shutdown failure")

    launcher = replace(PRODUCTION_SCIENTIFIC_KERNEL_LAUNCHER, shutdown=fail_shutdown)
    try:
        result = _invoke(fixture, scientific_kernel_launcher=launcher)
        assert len(kernels) == 1
        assert kernels[0].poll() is None
        assert result.exit_code != 0
        evidence = _load_failed(fixture.evidence_path)
        assert "search command failed with return code 7" in evidence.error
        assert "injected shutdown failure" in evidence.error
        assert "gpuserver is still running; diagnostic log is not final" in evidence.error
        assert evidence.paired_evidence.log_lines is None
        assert not Path(fixture.action.payload.evidence.durable_log_path).exists()
    finally:
        for kernel in kernels:
            kernel.kill()
            kernel.wait(timeout=5)


@pytest.mark.parametrize("failed_tool", ["tar", "lz4"])
def test_packaging_command_failure_is_attested_without_success_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_tool: str,
) -> None:
    fixture = _fixture(tmp_path, failed_tool=failed_tool)
    _configure_fakes(fixture, monkeypatch)
    _skip_server_warmup(monkeypatch)

    result = _invoke(fixture)

    assert result.exit_code != 0
    evidence = _load_failed(fixture.evidence_path)
    expected = ("gpuserver", "search", "record-ls", "tar")
    if failed_tool == "lz4":
        expected = (*expected, "lz4")
    assert tuple(item.command_kind for item in evidence.command_outcomes) == expected
    assert evidence.command_outcomes[-1].return_code == 19
    assert "failed to retain preprocessing diagnostic log" not in evidence.error
    assert Path(fixture.action.payload.evidence.durable_log_path).read_text() == "search complete\n"
    assert not Path(fixture.action.payload.package.durable_lz4_path).exists()
    assert not Path(fixture.action.payload.package.completed_input_path).exists()


def test_preflight_rejects_action_input_drift_and_existing_output_before_kernel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path / "wrong-action")
    _configure_fakes(fixture, monkeypatch)
    _skip_server_warmup(monkeypatch)
    result = _invoke(fixture, action_id="preprocessing-chunk-999999")
    assert result.exit_code != 0
    assert _load_failed(fixture.evidence_path).command_outcomes == ()
    assert not fixture.invocation_path.exists()

    drift = _fixture(tmp_path / "drift")
    _configure_fakes(drift, monkeypatch)
    Path(drift.action.payload.search_argv[3]).write_text(">changed\nAAAA\n")
    result = _invoke(drift)
    assert result.exit_code != 0
    assert _load_failed(drift.evidence_path).command_outcomes == ()
    assert not drift.invocation_path.exists()

    existing = _fixture(tmp_path / "existing")
    _configure_fakes(existing, monkeypatch)
    destination = Path(existing.action.payload.package.durable_lz4_path)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"owned")
    result = _invoke(existing)
    assert result.exit_code != 0
    assert destination.read_bytes() == b"owned"
    assert _load_failed(existing.evidence_path).command_outcomes == ()
    assert not existing.invocation_path.exists()


@pytest.mark.parametrize("missing_input", ["staged", "split"])
def test_preflight_rejects_missing_declared_input_before_kernel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_input: str,
) -> None:
    fixture = _fixture(tmp_path)
    _configure_fakes(fixture, monkeypatch)
    _skip_server_warmup(monkeypatch)
    path = (
        Path(fixture.action.payload.search_argv[3])
        if missing_input == "staged"
        else Path(fixture.action.payload.package.completed_input_source_path)
    )
    path.unlink()

    result = _invoke(fixture)

    assert result.exit_code != 0
    assert "must be an existing regular non-symlink file" in result.output
    assert "Traceback" not in result.output
    assert _load_failed(fixture.evidence_path).command_outcomes == ()
    assert not fixture.invocation_path.exists()


def test_preflight_rejects_preexisting_raw_search_directory_before_kernel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    _configure_fakes(fixture, monkeypatch)
    raw = Path(fixture.action.payload.evidence.raw_search_output_directory)
    raw.mkdir(parents=True)
    (raw / "owned").write_text("do not replace\n")

    result = _invoke(fixture)

    assert result.exit_code != 0
    assert "refusing to reuse existing preprocessing action path" in result.output
    assert (raw / "owned").read_text() == "do not replace\n"
    assert _load_failed(fixture.evidence_path).command_outcomes == ()
    assert not fixture.invocation_path.exists()


def test_cli_rejects_missing_or_tampered_runspec_before_kernel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path / "missing")
    _configure_fakes(fixture, monkeypatch)
    fixture.runspec_path.unlink()

    result = _invoke_runtime_cli(fixture)

    assert result.exit_code != 0
    assert "does not exist" in result.output
    assert "Traceback" not in result.output
    assert not fixture.evidence_path.exists()
    assert not fixture.invocation_path.exists()

    tampered = _fixture(tmp_path / "tampered")
    _configure_fakes(tampered, monkeypatch)
    payload = json.loads(tampered.runspec_path.read_text())
    del payload["payload"]["actions"][0]["schema_version"]
    tampered.runspec_path.write_text(json.dumps(payload))

    result = _invoke_runtime_cli(tampered)

    assert result.exit_code != 0
    assert "schema_version" in result.output
    assert "Traceback" not in result.output
    assert not tampered.evidence_path.exists()
    assert not tampered.invocation_path.exists()


def test_action_evidence_loader_rejects_schema_order_identity_and_nested_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    _configure_fakes(fixture, monkeypatch)
    _skip_server_warmup(monkeypatch)
    assert _invoke(fixture).exit_code == 0
    evidence = preprocessing_chunk_action_evidence_from_mapping(json.loads(fixture.evidence_path.read_text()))
    assert preprocessing_chunk_action_evidence_from_mapping(evidence.to_mapping()) == evidence

    explicit_null_return_code = evidence.command_outcomes[0].to_mapping()
    explicit_null_return_code["disposition"] = "failed-to-start"
    explicit_null_return_code["return_code"] = None
    assert preprocessing_command_outcome_from_mapping(explicit_null_return_code).return_code is None

    unknown = deepcopy(evidence.to_mapping())
    unknown["unexpected"] = True
    with pytest.raises(ValueError, match="Unknown PreprocessingChunkActionEvidence"):
        preprocessing_chunk_action_evidence_from_mapping(unknown)

    missing_version = deepcopy(evidence.to_mapping())
    assert isinstance(missing_version["command_outcomes"], list)
    del missing_version["command_outcomes"][0]["schema_version"]
    with pytest.raises(ValueError, match="schema_version"):
        preprocessing_chunk_action_evidence_from_mapping(missing_version)

    for path in (
        ("error",),
        ("command_outcomes", 0, "return_code"),
        ("output_hashes", 0, "member_name"),
    ):
        missing_nullable = deepcopy(evidence.to_mapping())
        target = missing_nullable
        for key in path[:-1]:
            target = target[key]
        del target[path[-1]]
        with pytest.raises(ValueError, match=rf"{path[-1]} is required"):
            preprocessing_chunk_action_evidence_from_mapping(missing_nullable)

    reordered = deepcopy(evidence.to_mapping())
    assert isinstance(reordered["command_outcomes"], list)
    reordered["command_outcomes"][1:3] = reversed(reordered["command_outcomes"][1:3])
    with pytest.raises(ValueError, match="ordered prefix"):
        preprocessing_chunk_action_evidence_from_mapping(reordered)

    forged_identity = deepcopy(evidence.to_mapping())
    forged_identity["phase_runspec_digest"] = "0" * 64
    with pytest.raises(ValueError, match=r"Database Placement evidence.*identity"):
        preprocessing_chunk_action_evidence_from_mapping(forged_identity)

    forged_hash = deepcopy(evidence.to_mapping())
    assert isinstance(forged_hash["output_hashes"], list)
    forged_hash["output_hashes"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        reconcile_preprocessing_chunk_action_evidence(
            fixture.runspec,
            preprocessing_chunk_action_evidence_from_mapping(forged_hash),
        )

    missing_raw = deepcopy(evidence.to_mapping())
    del missing_raw["raw_search_evidence"]
    with pytest.raises(ValueError, match="adapter-v3 evidence requires raw-search"):
        preprocessing_chunk_action_evidence_from_mapping(missing_raw)

    for legacy_version in (
        "preprocessing-scientific-backend-v1",
        "preprocessing-scientific-backend-v2",
    ):
        legacy = deepcopy(evidence.to_mapping())
        legacy["adapter_version"] = legacy_version
        with pytest.raises(ValueError, match="unsupported preprocessing adapter version"):
            preprocessing_chunk_action_evidence_from_mapping(legacy)

    reordered_raw = deepcopy(evidence.to_mapping())
    raw_artifacts = reordered_raw["raw_search_evidence"]["artifacts"]
    raw_artifacts[0], raw_artifacts[1] = raw_artifacts[1], raw_artifacts[0]
    with pytest.raises(ValueError, match="searched source order"):
        preprocessing_chunk_action_evidence_from_mapping(reordered_raw)

    escaped_raw = deepcopy(evidence.to_mapping())
    escaped_raw["raw_search_evidence"]["artifacts"][0]["path"] = "/outside/AFDB_zeta.a3m"
    with pytest.raises(ValueError, match="direct children"):
        preprocessing_chunk_action_evidence_from_mapping(escaped_raw)

    uppercase_raw_hash = deepcopy(evidence.to_mapping())
    uppercase_raw_hash["raw_search_evidence"]["artifacts"][0]["sha256"] = "A" * 64
    with pytest.raises(ValueError, match="lowercase"):
        preprocessing_chunk_action_evidence_from_mapping(uppercase_raw_hash)


def test_failed_evidence_rejects_undeclared_hash_path_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    _configure_fakes(fixture, monkeypatch)
    _skip_server_warmup(monkeypatch)
    assert _invoke(fixture).exit_code == 0
    forged_failed = json.loads(fixture.evidence_path.read_text())
    forged_failed["outcome"] = "failed"
    forged_failed["error"] = "injected failure"
    member_name = forged_failed["output_hashes"][0]["member_name"]
    forged_failed["output_hashes"][0]["path"] = str(tmp_path / "undeclared" / member_name)
    evidence = preprocessing_chunk_action_evidence_from_mapping(forged_failed)

    with pytest.raises(ValueError, match="subset of the exact declared outputs"):
        reconcile_preprocessing_chunk_action_evidence(fixture.runspec, evidence)
