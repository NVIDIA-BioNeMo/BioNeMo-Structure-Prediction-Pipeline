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

"""Staged execute-chunk lease lifetime and failure dispatch tests."""

from __future__ import annotations

import fcntl
import io
import json
import os
import subprocess
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from bspp.orchestration.contract.database_placement import (
    DatabaseAccessPolicy,
    DatabaseSetSelection,
    build_preprocessing_database_binding,
)
from bspp.orchestration.contract.database_replica import (
    DatabaseReplicaWarmResult,
    canonical_database_replica_warm_result_bytes,
)
from bspp.orchestration.contract.database_replica_lease import DatabaseReplicaWarmLeaseTerminalEvidence
from bspp.orchestration.contract.preprocessing_action import (
    PreprocessingDatabasePlacementCommandFailureEvidence,
    PreprocessingDatabasePlacementEvidence,
    PreprocessingStagedDatabasePlacementEvidence,
    preprocessing_chunk_action_evidence_from_mapping,
)
from bspp.orchestration.runtime.preprocessing._database_replica_errors import (
    ClassifiedDatabaseReplicaError,
)
from bspp.orchestration.runtime.preprocessing._database_replica_evidence_io import (
    load_database_replica_result,
)
from bspp.orchestration.runtime.preprocessing.execution import (
    PRODUCTION_SCIENTIFIC_KERNEL_LAUNCHER,
    ScientificKernelProcess,
)
from tests.support.database_cold_replica import _stage_required_fixture
from tests.support.database_cold_replica import cold_cache_root as cold_cache_root
from tests.support.preprocessing_execution import (
    ExecutionResult,
    LocalExecutionFixture,
    configure_preprocessing_fakes,
    invoke_preprocessing_execution,
    preprocessing_execution_fixture,
)


def _staged_execution_fixture(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    warm: bool = False,
) -> tuple[LocalExecutionFixture, Path, Path]:
    fixture, manifest_path, cache_root = _stage_required_fixture(
        tmp_path / "work",
        cold_cache_root,
        monkeypatch,
    )
    configure_preprocessing_fakes(fixture, monkeypatch)
    fixture.database_placement_result_path.chmod(0o600)
    fixture.database_placement_result_path.unlink()
    monkeypatch.setattr(os, "fstatvfs", lambda _descriptor: SimpleNamespace(f_bavail=2, f_frsize=1))
    fixture.place_database(
        fixture.runspec,
        action_id=fixture.action.action_id,
        source_manifest_path=manifest_path,
        result_path=fixture.database_placement_result_path,
        failure_path=fixture.database_placement_failure_path,
    )
    if warm:
        fixture.database_placement_result_path.unlink()
        warm_result = fixture.place_database(
            fixture.runspec,
            action_id=fixture.action.action_id,
            source_manifest_path=manifest_path,
            result_path=fixture.database_placement_result_path,
            failure_path=fixture.database_placement_failure_path,
        )
        assert isinstance(warm_result, DatabaseReplicaWarmResult)
    identity = fixture.runspec.payload.database.source_manifest_sha256
    selected_root = cache_root / "replicas" / identity
    lease_path = cache_root / ".locks" / f"{identity}.lock"
    fixture = replace(
        fixture,
        database_placement_paths=replace(
            fixture.database_placement_paths,
            selected_root=selected_root,
            lease=lease_path,
        ),
    )
    return fixture, selected_root, lease_path


@pytest.mark.parametrize("warm", [False, True], ids=["cold-result", "warm-result"])
def test_staged_nonzero_placement_status_retains_exact_result_without_lease_or_science(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    warm: bool,
) -> None:
    fixture, _selected_root, _lease_path = _staged_execution_fixture(
        tmp_path,
        cold_cache_root,
        monkeypatch,
        warm=warm,
    )
    result = invoke_preprocessing_execution(fixture, placement_process_status=19)

    assert result.exit_code != 0
    evidence = preprocessing_chunk_action_evidence_from_mapping(json.loads(fixture.evidence_path.read_text()))
    placement = evidence.database_placement
    assert isinstance(placement, PreprocessingDatabasePlacementCommandFailureEvidence)
    assert placement.classification == "nonzero-after-result"
    assert placement.placement_process_status == 19
    assert isinstance(placement.result, DatabaseReplicaWarmResult) is warm
    assert placement.science_started is False
    assert evidence.command_outcomes == ()
    assert evidence.output_hashes == ()
    assert evidence.raw_search_evidence is None
    assert not fixture.invocation_path.exists()


def test_direct_action_rejects_staged_result_family_before_science(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged, _selected_root, _lease_path = _staged_execution_fixture(
        tmp_path / "staged",
        cold_cache_root,
        monkeypatch,
    )
    direct = preprocessing_execution_fixture(tmp_path / "direct", direct_policy=True)
    configure_preprocessing_fakes(direct, monkeypatch)
    direct.database_placement_result_path.chmod(0o600)
    direct.database_placement_result_path.write_bytes(staged.database_placement_result_path.read_bytes())
    direct.database_placement_result_path.chmod(0o444)

    result = invoke_preprocessing_execution(direct)

    assert result.exit_code != 0
    evidence = preprocessing_chunk_action_evidence_from_mapping(json.loads(direct.evidence_path.read_text()))
    placement = evidence.database_placement
    assert isinstance(placement, PreprocessingDatabasePlacementCommandFailureEvidence)
    assert placement.classification == "evidence-reconciliation-failed"
    assert placement.result is None
    assert evidence.command_outcomes == ()
    assert evidence.output_hashes == ()
    assert not direct.invocation_path.exists()


@pytest.mark.parametrize("warm", [False, True], ids=["cold-result", "warm-result"])
def test_staged_handoff_mutation_writes_lease_failure_without_any_science_popen_or_output_publication(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    warm: bool,
) -> None:
    fixture, selected_root, _ = _staged_execution_fixture(
        tmp_path,
        cold_cache_root,
        monkeypatch,
        warm=warm,
    )
    member = selected_root / fixture.runspec.payload.database.source_manifest.members[0].logical_name
    member.chmod(0o644)
    start_calls: list[tuple[object, ...]] = []

    def forbidden_start(*args: object, **kwargs: object) -> ScientificKernelProcess:
        start_calls.append(args)
        raise AssertionError("handoff mutation must fail before Popen")

    launcher = replace(PRODUCTION_SCIENTIFIC_KERNEL_LAUNCHER, start=forbidden_start)
    result = invoke_preprocessing_execution(fixture, scientific_kernel_launcher=launcher)

    assert result.exit_code != 0
    assert start_calls == []
    evidence = preprocessing_chunk_action_evidence_from_mapping(json.loads(fixture.evidence_path.read_text()))
    assert isinstance(evidence.database_placement, PreprocessingStagedDatabasePlacementEvidence)
    assert evidence.database_placement.result is not None
    assert isinstance(evidence.database_placement.result, DatabaseReplicaWarmResult) is warm
    assert evidence.database_placement.failure is None
    assert evidence.database_placement.science_started is False
    assert evidence.database_placement.lease_outcome is not None
    assert evidence.database_placement.lease_outcome.lease_outcome_kind == "failure"
    assert evidence.command_outcomes == ()
    assert evidence.output_hashes == ()
    payload = fixture.action.payload
    assert not any(
        Path(path).exists()
        for path in (
            payload.evidence.durable_log_path,
            payload.evidence.durable_record_path,
            payload.package.durable_tar_path,
            payload.package.durable_lz4_path,
            payload.package.completed_input_path,
        )
    )


def test_fresh_science_lease_rejects_current_missing_identity_path_before_popen(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, manifest_path, cache_root = _stage_required_fixture(
        tmp_path / "work",
        cold_cache_root,
        monkeypatch,
    )
    configure_preprocessing_fakes(fixture, monkeypatch)
    fixture.database_placement_result_path.chmod(0o600)
    fixture.database_placement_result_path.unlink()
    monkeypatch.setattr(
        os,
        "fstatvfs",
        lambda _descriptor: SimpleNamespace(f_bavail=2, f_frsize=1),
    )
    fixture.place_database(
        fixture.runspec,
        action_id=fixture.action.action_id,
        source_manifest_path=manifest_path,
        result_path=tmp_path / "initial" / "result.json",
        failure_path=tmp_path / "initial" / "failure.json",
    )

    identity = fixture.runspec.payload.database.source_manifest_sha256
    selected_root = cache_root / "replicas" / identity
    lease_path = cache_root / ".locks" / f"{identity}.lock"
    displaced_lease_path = lease_path.with_name(f"{lease_path.name}.displaced")
    fixture = replace(
        fixture,
        database_placement_paths=replace(
            fixture.database_placement_paths,
            selected_root=selected_root,
            lease=lease_path,
        ),
    )
    system_link = os.link
    displaced = False

    def publish_then_displace_lease(source: object, destination: object, **kwargs: object) -> None:
        nonlocal displaced
        system_link(source, destination, **kwargs)
        if destination == fixture.database_placement_result_path.name:
            lease_path.rename(displaced_lease_path)
            displaced = True

    monkeypatch.setattr(os, "link", publish_then_displace_lease)
    start_calls: list[tuple[object, ...]] = []

    def forbidden_start(*args: object, **_kwargs: object) -> ScientificKernelProcess:
        start_calls.append(args)
        raise AssertionError("current missing identity path must be rejected before Popen")

    launcher = replace(PRODUCTION_SCIENTIFIC_KERNEL_LAUNCHER, start=forbidden_start)
    try:
        with pytest.raises(ClassifiedDatabaseReplicaError) as raised:
            fixture.place_database(
                fixture.runspec,
                action_id=fixture.action.action_id,
                source_manifest_path=manifest_path,
                result_path=fixture.database_placement_result_path,
                failure_path=fixture.database_placement_failure_path,
            )
        assert raised.value.classification == "lock-unavailable"
        assert isinstance(
            load_database_replica_result(fixture.database_placement_result_path),
            DatabaseReplicaWarmResult,
        )
        assert not fixture.database_placement_failure_path.exists()
        assert not lease_path.exists()

        result = invoke_preprocessing_execution(fixture, scientific_kernel_launcher=launcher)
    finally:
        if displaced:
            displaced_lease_path.rename(lease_path)

    assert result.exit_code != 0
    assert start_calls == []
    evidence = preprocessing_chunk_action_evidence_from_mapping(json.loads(fixture.evidence_path.read_text()))
    assert isinstance(evidence.database_placement, PreprocessingStagedDatabasePlacementEvidence)
    assert isinstance(evidence.database_placement.result, DatabaseReplicaWarmResult)
    assert evidence.database_placement.failure is None
    assert evidence.database_placement.science_started is False
    assert evidence.database_placement.lease_outcome is not None
    assert evidence.database_placement.lease_outcome.lease_outcome_kind == "failure"
    assert evidence.database_placement.lease_outcome.classification == "lease-open-failed"
    assert evidence.command_outcomes == ()


@pytest.mark.parametrize("warm", [False, True], ids=["cold-result", "warm-result"])
def test_staged_shared_lease_covers_search_and_gpuserver_cleanup_then_releases_before_packaging(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    warm: bool,
) -> None:
    fixture, _, lease_path = _staged_execution_fixture(
        tmp_path,
        cold_cache_root,
        monkeypatch,
        warm=warm,
    )
    real_shutdown = PRODUCTION_SCIENTIFIC_KERNEL_LAUNCHER.shutdown
    lock_checks: list[str] = []

    def assert_lease_during_cleanup(server: ScientificKernelProcess, action: object) -> object:
        descriptor = os.open(lease_path, os.O_RDWR | os.O_NOFOLLOW)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_checks.append("before-terminal-cleanup")
            outcome = real_shutdown(server, action)
            with pytest.raises(BlockingIOError):
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_checks.append("after-terminal-cleanup")
            return outcome
        finally:
            os.close(descriptor)

    launcher = replace(PRODUCTION_SCIENTIFIC_KERNEL_LAUNCHER, shutdown=assert_lease_during_cleanup)
    result = invoke_preprocessing_execution(fixture, scientific_kernel_launcher=launcher)

    assert result.exit_code == 0, result.output
    assert lock_checks == ["before-terminal-cleanup", "after-terminal-cleanup"]
    descriptor = os.open(lease_path, os.O_RDWR | os.O_NOFOLLOW)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)
    evidence = preprocessing_chunk_action_evidence_from_mapping(json.loads(fixture.evidence_path.read_text()))
    assert isinstance(evidence.database_placement, PreprocessingStagedDatabasePlacementEvidence)
    assert isinstance(evidence.database_placement.result, DatabaseReplicaWarmResult) is warm
    assert evidence.database_placement.lease_outcome is not None
    assert evidence.database_placement.lease_outcome.held_through_kernel_exit is True


def test_warm_staged_execution_reacquires_lease_and_retains_true_terminal_family(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, _, lease_path = _staged_execution_fixture(
        tmp_path,
        cold_cache_root,
        monkeypatch,
        warm=True,
    )

    result = invoke_preprocessing_execution(fixture)

    assert result.exit_code == 0, result.output
    evidence = preprocessing_chunk_action_evidence_from_mapping(json.loads(fixture.evidence_path.read_text()))
    assert isinstance(evidence.database_placement, PreprocessingStagedDatabasePlacementEvidence)
    assert isinstance(evidence.database_placement.result, DatabaseReplicaWarmResult)
    assert isinstance(evidence.database_placement.lease_outcome, DatabaseReplicaWarmLeaseTerminalEvidence)
    assert evidence.database_placement.lease_outcome.kernel_started is True
    assert evidence.database_placement.lease_outcome.held_through_kernel_exit is True
    competitor = os.open(lease_path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(competitor)


def test_two_warm_science_executions_share_lease_while_exclusive_waits_for_both_to_finish(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, _, lease_path = _staged_execution_fixture(
        tmp_path / "reader-0",
        cold_cache_root,
        monkeypatch,
        warm=True,
    )
    first_result = load_database_replica_result(first.database_placement_result_path)
    assert isinstance(first_result, DatabaseReplicaWarmResult)
    second = preprocessing_execution_fixture(tmp_path / "reader-1" / "work", direct_policy=False)
    first_database = first.runspec.payload.database
    second_action = second.action.payload
    second_binding = build_preprocessing_database_binding(
        selection=DatabaseSetSelection(
            database_set=first_database.database_set,
            requested_policy=DatabaseAccessPolicy.STAGE_REQUIRED,
        ),
        source_manifest=first_database.source_manifest,
        source_manifest_projection=second.runspec.payload.database.source_manifest_projection,
        staging=first_database.staging,
        gpuserver_argv=second_action.gpuserver_argv,
        search_argv=second_action.search_argv,
    )
    second_runspec = replace(
        second.runspec,
        payload=replace(second.runspec.payload, database=second_binding),
    )
    second = replace(
        second,
        runspec=second_runspec,
        database_placement_paths=replace(
            second.database_placement_paths,
            selected_root=first.database_placement_paths.selected_root,
            lease=first.database_placement_paths.lease,
        ),
    )
    second.runspec_path.write_text(json.dumps(second_runspec.to_mapping(), indent=2, sort_keys=True) + "\n")
    configure_preprocessing_fakes(second, monkeypatch)
    second.database_placement_result_path.chmod(0o600)
    second.database_placement_result_path.write_bytes(
        canonical_database_replica_warm_result_bytes(replace(first_result, phase_runspec_digest=second_runspec.digest))
    )
    second.database_placement_result_path.chmod(0o444)
    fixtures = (first, second)
    entered = threading.Barrier(3)
    releases = (threading.Event(), threading.Event())
    finished = (threading.Event(), threading.Event())
    ordinal_by_thread: dict[int, int] = {}
    evidence: list[object] = []
    errors: list[BaseException] = []
    real_shutdown = PRODUCTION_SCIENTIFIC_KERNEL_LAUNCHER.shutdown

    def hold_each_reader_until_released(server: ScientificKernelProcess, action: object) -> object:
        ordinal = ordinal_by_thread[threading.get_ident()]
        entered.wait(timeout=5)
        if not releases[ordinal].wait(timeout=5):
            raise AssertionError("science reader was not released")
        return real_shutdown(server, action)

    def run_reader(ordinal: int) -> None:
        ordinal_by_thread[threading.get_ident()] = ordinal
        fixture = fixtures[ordinal]
        try:
            evidence.append(fixture.execute_preprocessing_chunk_action(scientific_kernel_launcher=launcher))
        except BaseException as exc:
            errors.append(exc)
        finally:
            finished[ordinal].set()

    launcher = replace(PRODUCTION_SCIENTIFIC_KERNEL_LAUNCHER, shutdown=hold_each_reader_until_released)
    readers = tuple(threading.Thread(target=run_reader, args=(ordinal,), daemon=True) for ordinal in range(2))
    for reader in readers:
        reader.start()
    entered.wait(timeout=5)

    competitor = os.open(lease_path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        with pytest.raises(BlockingIOError):
            fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        releases[0].set()
        assert finished[0].wait(timeout=5)
        with pytest.raises(BlockingIOError):
            fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        releases[1].set()
        assert finished[1].wait(timeout=5)
        fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(competitor, fcntl.LOCK_UN)
    finally:
        releases[0].set()
        releases[1].set()
        os.close(competitor)
        for reader in readers:
            reader.join(timeout=5)

    assert all(not reader.is_alive() for reader in readers)
    assert errors == []
    assert len(evidence) == 2


def test_direct_execution_still_uses_unchanged_direct_wrapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = preprocessing_execution_fixture(tmp_path / "direct")
    configure_preprocessing_fakes(fixture, monkeypatch)

    result = invoke_preprocessing_execution(fixture)

    assert result.exit_code == 0, result.output
    evidence = preprocessing_chunk_action_evidence_from_mapping(json.loads(fixture.evidence_path.read_text()))
    assert type(evidence.database_placement) is PreprocessingDatabasePlacementEvidence


def test_staged_kernel_start_failure_retains_terminal_lease_evidence_for_the_invocation_interval(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, _, lease_path = _staged_execution_fixture(tmp_path, cold_cache_root, monkeypatch)
    lock_checks: list[str] = []

    def fail_start(*_args: object, **_kwargs: object) -> object:
        descriptor = os.open(lease_path, os.O_RDWR | os.O_NOFOLLOW)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_checks.append("blocked-during-popen")
            raise OSError("injected kernel startup failure")
        finally:
            os.close(descriptor)

    launcher = replace(PRODUCTION_SCIENTIFIC_KERNEL_LAUNCHER, start=fail_start)
    result = invoke_preprocessing_execution(fixture, scientific_kernel_launcher=launcher)

    assert result.exit_code != 0
    assert lock_checks == ["blocked-during-popen"]
    assert fixture.evidence_path.exists(), result.output
    evidence = preprocessing_chunk_action_evidence_from_mapping(json.loads(fixture.evidence_path.read_text()))
    assert isinstance(evidence.database_placement, PreprocessingStagedDatabasePlacementEvidence)
    assert evidence.database_placement.science_started is False
    assert evidence.database_placement.lease_outcome is not None
    assert evidence.database_placement.lease_outcome.lease_outcome_kind == "terminal"
    assert evidence.database_placement.lease_outcome.kernel_started is False
    assert evidence.database_placement.lease_outcome.held_through_kernel_exit is False
    assert evidence.command_outcomes[0].disposition == "failed-to-start"
    assert evidence.output_hashes == ()
    assert evidence.paired_evidence.log_lines == ()
    assert Path(fixture.action.payload.evidence.durable_log_path).read_bytes() == b""
    descriptor = os.open(lease_path, os.O_RDWR | os.O_NOFOLLOW)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


@pytest.mark.parametrize("warm", [False, True], ids=["cold-result", "warm-result"])
def test_failed_staged_search_retains_log_without_accepting_science_outputs(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    warm: bool,
) -> None:
    fixture, _, _ = _staged_execution_fixture(tmp_path, cold_cache_root, monkeypatch, warm=warm)
    monkeypatch.setenv("BSPP_FAKE_SEARCH_MODE", "fail")

    result = invoke_preprocessing_execution(fixture)

    assert result.exit_code != 0
    evidence = preprocessing_chunk_action_evidence_from_mapping(json.loads(fixture.evidence_path.read_text()))
    assert evidence.outcome == "failed"
    assert "search command failed with return code 7" in evidence.error
    assert isinstance(evidence.database_placement, PreprocessingStagedDatabasePlacementEvidence)
    assert evidence.database_placement.lease_outcome is not None
    assert evidence.database_placement.lease_outcome.held_through_kernel_exit is True
    assert evidence.output_hashes == ()
    assert evidence.raw_search_evidence is None
    assert evidence.paired_evidence.log_lines == ("search failure stdout", "search failure stderr")
    assert Path(fixture.action.payload.evidence.durable_log_path).read_bytes() == (
        b"search failure stdout\nsearch failure stderr\n"
    )
    assert not Path(fixture.action.payload.package.durable_lz4_path).exists()


def test_staged_cleanup_cannot_release_lease_or_return_while_kernel_is_nonterminal(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, _, lease_path = _staged_execution_fixture(tmp_path, cold_cache_root, monkeypatch)
    real_popen = subprocess.Popen
    cleanup_failed = threading.Event()
    terminal_retry_failed = threading.Event()
    allow_terminal = threading.Event()
    invocation_finished = threading.Event()
    invocation_results: list[ExecutionResult] = []
    kernels: list[ControlledKernel] = []

    class ControlledKernel:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.process = real_popen(*args, **kwargs)
            self.terminate_calls = 0
            self.kill_calls = 0
            self.wait_calls = 0

        def poll(self) -> int | None:
            if not allow_terminal.is_set():
                return None
            return self.process.poll()

        def terminate(self) -> None:
            self.terminate_calls += 1
            if self.terminate_calls == 1:
                cleanup_failed.set()
                raise OSError("injected cleanup failure with live kernel")
            if not allow_terminal.wait(timeout=5):
                raise AssertionError("staged terminal cleanup did not remain blocked")
            self.process.terminate()

        def kill(self) -> None:
            self.kill_calls += 1
            if self.kill_calls == 1:
                raise OSError("transient staged force-kill failure")
            if not allow_terminal.wait(timeout=5):
                raise AssertionError("staged terminal cleanup did not remain blocked")
            self.process.kill()

        def wait(self, timeout: float | None = None) -> int:
            self.wait_calls += 1
            if self.wait_calls == 1:
                terminal_retry_failed.set()
                raise OSError("transient staged terminal-wait failure")
            if not allow_terminal.is_set():
                raise subprocess.TimeoutExpired(self.process.args, timeout)
            return self.process.wait(timeout=timeout)

    def controlled_start(
        argv: tuple[str, ...],
        environment: object,
        stdout: object,
    ) -> ControlledKernel:
        kernel = ControlledKernel(
            argv,
            env=environment,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        kernels.append(kernel)
        return kernel

    launcher = replace(PRODUCTION_SCIENTIFIC_KERNEL_LAUNCHER, start=controlled_start)

    def invoke() -> None:
        try:
            invocation_results.append(invoke_preprocessing_execution(fixture, scientific_kernel_launcher=launcher))
        finally:
            invocation_finished.set()

    worker = threading.Thread(target=invoke, daemon=True)
    worker.start()
    assert cleanup_failed.wait(timeout=5), "the injected live-kernel cleanup failure was not reached"
    assert terminal_retry_failed.wait(timeout=5), "the staged terminal retry seam was not reached"

    exclusive_descriptor = os.open(lease_path, os.O_RDWR | os.O_NOFOLLOW)
    exclusive_acquired = False
    try:
        try:
            fcntl.flock(exclusive_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            exclusive_acquired = True
        except BlockingIOError:
            pass
        observed_while_nonterminal = (
            exclusive_acquired,
            invocation_finished.is_set(),
            fixture.evidence_path.exists(),
            kernels[0].poll(),
        )
    finally:
        if exclusive_acquired:
            fcntl.flock(exclusive_descriptor, fcntl.LOCK_UN)
        os.close(exclusive_descriptor)
        allow_terminal.set()
        worker.join(timeout=5)
        if kernels and kernels[0].process.poll() is None:
            kernels[0].process.kill()
            kernels[0].process.wait(timeout=5)

    assert observed_while_nonterminal == (False, False, False, None)
    assert invocation_finished.is_set()
    assert len(invocation_results) == 1
    assert invocation_results[0].exit_code != 0
    evidence = preprocessing_chunk_action_evidence_from_mapping(json.loads(fixture.evidence_path.read_text()))
    assert isinstance(evidence.database_placement, PreprocessingStagedDatabasePlacementEvidence)
    assert evidence.error is not None
    assert "transient staged force-kill failure" in evidence.error
    assert "transient staged terminal-wait failure" in evidence.error
    assert evidence.database_placement.lease_outcome is not None
    assert evidence.database_placement.lease_outcome.kernel_started is True
    assert evidence.database_placement.lease_outcome.held_through_kernel_exit is True


@pytest.mark.parametrize(
    "failure_path",
    ["primary-late-failure", "construction-rebuild", "reconciliation-rebuild"],
)
def test_failed_started_staged_runtime_never_accepts_candidate_science_evidence(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_path: str,
) -> None:
    fixture, _, _ = _staged_execution_fixture(tmp_path, cold_cache_root, monkeypatch)
    payload = fixture.action.payload
    real_run = subprocess.run

    if failure_path in {"primary-late-failure", "construction-rebuild"}:

        def mutate_after_lz4(*args: object, **kwargs: object) -> subprocess.CompletedProcess[object]:
            completed = real_run(*args, **kwargs)
            if args[0] == payload.package.lz4_argv:
                if failure_path == "primary-late-failure":
                    completed_input = Path(payload.package.completed_input_path)
                    completed_input.parent.mkdir(parents=True, exist_ok=True)
                    completed_input.write_bytes(b"late exclusive-publication contender")
                else:
                    Path(payload.evidence.durable_log_path).unlink()
            return completed

        monkeypatch.setattr(subprocess, "run", mutate_after_lz4)
    else:
        durable_record = Path(payload.evidence.durable_record_path)
        real_open = io.open
        record_reads = 0

        def drift_only_during_reconciliation(
            path: str | bytes | int | os.PathLike[str] | os.PathLike[bytes],
            mode: str = "r",
            buffering: int = -1,
            encoding: str | None = None,
            errors: str | None = None,
            newline: str | None = None,
            closefd: bool = True,
            opener: object | None = None,
        ) -> object:
            nonlocal record_reads
            if not isinstance(path, int) and Path(path) == durable_record and "r" in mode:
                record_reads += 1
                if record_reads == 2:
                    with real_open(path, mode, buffering, encoding, errors, newline, closefd, opener) as handle:
                        content = handle.read()
                    if isinstance(content, bytes):
                        return io.BytesIO(content + b"\nreconciliation-only drift")
                    return io.StringIO(content + "\nreconciliation-only drift")
            return real_open(path, mode, buffering, encoding, errors, newline, closefd, opener)

        monkeypatch.setattr(io, "open", drift_only_during_reconciliation)

    result = invoke_preprocessing_execution(fixture)

    assert result.exit_code != 0
    assert fixture.evidence_path.exists(), result.output
    evidence = preprocessing_chunk_action_evidence_from_mapping(json.loads(fixture.evidence_path.read_text()))
    assert evidence.outcome == "failed"
    assert isinstance(evidence.database_placement, PreprocessingStagedDatabasePlacementEvidence)
    assert evidence.database_placement.science_started is True
    assert evidence.database_placement.lease_outcome is not None
    assert evidence.database_placement.lease_outcome.kernel_started is True
    assert evidence.database_placement.lease_outcome.held_through_kernel_exit is True
    assert tuple(item.command_kind for item in evidence.command_outcomes) == (
        "gpuserver",
        "search",
        "record-ls",
        "tar",
        "lz4",
    )
    assert evidence.output_hashes == ()
    assert evidence.raw_search_evidence is None
    assert tuple(Path(payload.evidence.raw_search_output_directory).iterdir())


def test_pre_lease_failure_surfaces_real_error_instead_of_lease_outcome_mask(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F9: a pre-lease failure (e.g. missing staged input) must surface the real error.

    Before the fix, the cold Result with ``lease_outcome=None`` raised
    "successful cold placement requires a subsequent lease outcome" in
    ``__post_init__``, masking the actual preflight input-validation error.
    """
    fixture, _selected_root, _lease_path = _staged_execution_fixture(
        tmp_path,
        cold_cache_root,
        monkeypatch,
    )
    # Note: _staged_execution_fixture already calls configure_preprocessing_fakes
    # and creates a staged placement result. Do NOT call configure_preprocessing_fakes
    # again — it would overwrite the staged result with a DIRECT one.

    # Remove the staged search input to trigger a pre-lease preflight failure.
    search_input = Path(fixture.action.payload.search_argv[3])
    split_input = Path(fixture.action.payload.package.completed_input_source_path)
    search_input.unlink()
    split_input.unlink()

    result = invoke_preprocessing_execution(fixture)

    assert result.exit_code != 0
    assert fixture.evidence_path.exists(), result.output
    evidence = preprocessing_chunk_action_evidence_from_mapping(json.loads(fixture.evidence_path.read_text()))
    assert evidence.outcome == "failed"
    assert isinstance(evidence.database_placement, PreprocessingStagedDatabasePlacementEvidence)
    assert evidence.database_placement.result is not None
    assert evidence.database_placement.lease_outcome is None
    assert evidence.database_placement.science_started is False
    # The real error must mention the missing input, not the lease-outcome mask.
    assert evidence.error is not None
    assert "subsequent lease outcome" not in evidence.error
    assert "must be an existing regular non-symlink file" in evidence.error
