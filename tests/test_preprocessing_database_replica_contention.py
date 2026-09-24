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

"""Bounded Database Replica contender coordination and ownership tests."""

from __future__ import annotations

import fcntl
import multiprocessing
import os
import stat
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from bspp.orchestration.contract.database_replica import (
    DatabaseReplicaColdResult,
    DatabaseReplicaWarmResult,
)
from bspp.orchestration.runtime.preprocessing import _database_placement_evidence_io as evidence_io
from bspp.orchestration.runtime.preprocessing import _database_replica as replica_runtime
from bspp.orchestration.runtime.preprocessing._database_replica_errors import (
    ClassifiedDatabaseReplicaError,
)
from bspp.orchestration.runtime.preprocessing._database_replica_evidence_io import (
    load_database_replica_cold_failure,
    load_database_replica_result,
)
from bspp.orchestration.runtime.preprocessing._database_replica_lock import LockWait
from bspp.orchestration.runtime.preprocessing._database_replica_repair_evidence import (
    load_database_replica_cleanup,
    load_database_replica_invalidation,
)
from bspp.orchestration.runtime.preprocessing._database_replica_services import (
    LockWaitFactory,
)
from bspp.orchestration.runtime.preprocessing.database_placement import (
    DatabasePlacementError,
    _production_staged_database_placement_services,
)
from tests.support.database_cold_replica import (
    _replace_stage_manifest,
    _stage_required_fixture,
)
from tests.support.database_cold_replica import (
    cold_cache_root as cold_cache_root,
)


def test_cold_result_is_durably_published_before_population_locks_are_released(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, manifest_path, cache_root = _stage_required_fixture(
        tmp_path / "work",
        cold_cache_root,
        monkeypatch,
    )
    monkeypatch.setattr(
        replica_runtime.os,
        "fstatvfs",
        lambda _descriptor: SimpleNamespace(f_bavail=2, f_frsize=1),
    )
    result_path = tmp_path / "placement" / "result.json"
    system_link = os.link
    locks_blocked: list[bool] = []

    def observe_locks_then_link(source: object, destination: object, **kwargs: object) -> None:
        if destination == result_path.name:
            locks = cache_root / ".locks"
            identity = fixture.runspec.payload.database.source_manifest_sha256
            for name in ("cache.lock", f"{identity}.lock"):
                descriptor = os.open(locks / name, os.O_RDWR)
                try:
                    try:
                        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        locks_blocked.append(True)
                    else:
                        locks_blocked.append(False)
                        fcntl.flock(descriptor, fcntl.LOCK_UN)
                finally:
                    os.close(descriptor)
        system_link(source, destination, **kwargs)

    monkeypatch.setattr(evidence_io.os, "link", observe_locks_then_link)

    fixture.place_database(
        fixture.runspec,
        action_id=fixture.action.action_id,
        source_manifest_path=manifest_path,
        result_path=result_path,
        failure_path=tmp_path / "placement" / "failure.json",
    )

    assert locks_blocked == [True, True]


def test_two_same_identity_public_placements_copy_once_and_reuse_the_winner(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, manifest_path, cache_root = _stage_required_fixture(
        tmp_path / "work",
        cold_cache_root,
        monkeypatch,
    )
    assert fixture.runspec.payload.database.staging is not None
    database = replace(
        fixture.runspec.payload.database,
        staging=replace(fixture.runspec.payload.database.staging, lock_wait_seconds=2),
    )
    first_runspec = replace(fixture.runspec, payload=replace(fixture.runspec.payload, database=database))
    second_database = replace(
        database,
        source_manifest_projection="attempts/attempt-0002/database-source-manifest.json",
    )
    second_runspec = replace(
        fixture.runspec,
        phase_run_id="phase-run-fedcba9876543210fedcba9876543210",
        attempt_id="attempt-0002",
        payload=replace(fixture.runspec.payload, database=second_database),
    )
    runspecs = (first_runspec, second_runspec)
    (cache_root / "replicas").mkdir(mode=0o700)
    monkeypatch.setattr(
        replica_runtime.os,
        "fstatvfs",
        lambda _descriptor: SimpleNamespace(f_bavail=2, f_frsize=1),
    )

    context = multiprocessing.get_context("fork")
    simultaneous_miss = context.Barrier(2)
    copy_entered = context.Event()
    release_copy = context.Event()
    copy_count = context.Value("i", 0)
    services = _production_staged_database_placement_services()
    original_probe = services.warm_selector.probe
    original_populate = services.cold.population.populate

    def synchronized_missing_probe(*args: object, **kwargs: object) -> object:
        disposition = original_probe(*args, **kwargs)
        simultaneous_miss.wait(timeout=5)
        return disposition

    def held_population(*args: object, **kwargs: object) -> object:
        with copy_count.get_lock():
            copy_count.value += 1
        copy_entered.set()
        if not release_copy.wait(timeout=5):
            raise AssertionError("same-identity population test did not release the cold owner")
        return original_populate(*args, **kwargs)

    selector = replace(services.warm_selector, probe=synchronized_missing_probe)
    population = replace(services.cold.population, populate=held_population)
    services = replace(
        services,
        warm_selector=selector,
        cold=replace(
            services.cold,
            warm_selector=selector,
            population=population,
        ),
    )
    result_paths = tuple(tmp_path / "placement" / f"result-{ordinal}.json" for ordinal in range(2))
    failure_paths = tuple(tmp_path / "placement" / f"failure-{ordinal}.json" for ordinal in range(2))
    receivers = []
    processes = []

    def place(ordinal: int, sender: object) -> None:
        try:
            result = fixture.place_database(
                runspecs[ordinal],
                action_id=runspecs[ordinal].payload.actions[0].action_id,
                source_manifest_path=manifest_path,
                result_path=result_paths[ordinal],
                failure_path=failure_paths[ordinal],
                staged_services=services,
            )
            sender.send(("ok", type(result).__name__))  # type: ignore[attr-defined]
        except BaseException as exc:
            sender.send(("error", type(exc).__name__, str(exc)))  # type: ignore[attr-defined]
        finally:
            sender.close()  # type: ignore[attr-defined]

    try:
        for ordinal in range(2):
            receiver, sender = context.Pipe(duplex=False)
            process = context.Process(target=place, args=(ordinal, sender))
            receivers.append(receiver)
            processes.append(process)
            process.start()
            sender.close()
        assert copy_entered.wait(timeout=5)
        release_copy.set()
        messages = [receiver.recv() for receiver in receivers]
        for process in processes:
            process.join(timeout=5)
        assert all(not process.is_alive() and process.exitcode == 0 for process in processes)
    finally:
        release_copy.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        for receiver in receivers:
            receiver.close()

    assert messages.count(("ok", "DatabaseReplicaColdResult")) == 1
    assert messages.count(("ok", "DatabaseReplicaWarmResult")) == 1
    assert copy_count.value == 1
    results = tuple(load_database_replica_result(path) for path in result_paths)
    assert sum(isinstance(result, DatabaseReplicaColdResult) for result in results) == 1
    assert sum(isinstance(result, DatabaseReplicaWarmResult) for result in results) == 1
    for ordinal, result in enumerate(results):
        caller = runspecs[ordinal]
        assert (
            result.phase_run_id,
            result.attempt_id,
            result.phase_runspec_digest,
            result.action_id,
        ) == (
            caller.phase_run_id,
            caller.attempt_id,
            caller.digest,
            caller.payload.actions[0].action_id,
        )
    assert (
        len(
            {
                (result.phase_run_id, result.attempt_id, result.phase_runspec_digest, result.action_id)
                for result in results
            }
        )
        == 2
    )
    assert results[0].replica_manifest_sha256 == results[1].replica_manifest_sha256
    assert not any(path.exists() for path in failure_paths)
    identity = first_runspec.payload.database.source_manifest_sha256
    assert second_runspec.payload.database.source_manifest_sha256 == identity
    assert [path.name for path in (cache_root / "replicas").iterdir()] == [identity]


def test_contended_same_identity_invalid_winner_is_repaired_after_wait(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, manifest_path, cache_root = _stage_required_fixture(
        tmp_path / "work",
        cold_cache_root,
        monkeypatch,
    )
    staging = fixture.runspec.payload.database.staging
    assert staging is not None
    database = replace(fixture.runspec.payload.database, staging=replace(staging, lock_wait_seconds=3))
    runspec = replace(fixture.runspec, payload=replace(fixture.runspec.payload, database=database))
    identity = runspec.payload.database.source_manifest_sha256

    context = multiprocessing.get_context("fork")
    owner_ready = context.Event()
    contender_waiting = context.Event()
    original_sleep = time.sleep

    def observed_sleep(seconds: float) -> None:
        contender_waiting.set()
        original_sleep(seconds)

    def bounded_wait(timeout_seconds: int) -> LockWait:
        assert timeout_seconds == 3
        return LockWait(
            deadline=time.monotonic() + timeout_seconds,
            monotonic=time.monotonic,
            sleeper=observed_sleep,
            poll_quantum_seconds=0.01,
        )

    services = _production_staged_database_placement_services()
    services = replace(services, wait_factory=LockWaitFactory(build=bounded_wait))
    monkeypatch.setattr(
        replica_runtime.os,
        "fstatvfs",
        lambda _descriptor: SimpleNamespace(f_bavail=2, f_frsize=1),
    )

    invalid_candidate = cache_root / "replicas" / identity
    marker = invalid_candidate / "invalid-winner-marker"

    def publish_invalid_winner() -> None:
        locks = cache_root / ".locks"
        locks.mkdir(mode=0o700)
        descriptor = os.open(locks / f"{identity}.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            owner_ready.set()
            if not contender_waiting.wait(timeout=5):
                os._exit(25)
            invalid_candidate.parent.mkdir(mode=0o700)
            invalid_candidate.mkdir(mode=0o700)
            marker.write_bytes(b"invalid-winner")
        finally:
            os.close(descriptor)

    result_path = tmp_path / "placement" / "result.json"
    failure_path = tmp_path / "placement" / "failure.json"
    receiver, sender = context.Pipe(duplex=False)

    def place_contender() -> None:
        try:
            result = fixture.place_database(
                runspec,
                action_id=runspec.payload.actions[0].action_id,
                source_manifest_path=manifest_path,
                result_path=result_path,
                failure_path=failure_path,
                staged_services=services,
            )
            sender.send(("ok", type(result).__name__))
        except BaseException as exc:
            sender.send(("error", type(exc).__name__, str(exc)))
        finally:
            sender.close()

    owner = context.Process(target=publish_invalid_winner)
    contender = context.Process(target=place_contender)
    try:
        owner.start()
        assert owner_ready.wait(timeout=5)
        contender.start()
        sender.close()
        assert contender_waiting.wait(timeout=5)
        message = receiver.recv()
        owner.join(timeout=5)
        contender.join(timeout=5)
        assert not owner.is_alive() and owner.exitcode == 0
        assert not contender.is_alive() and contender.exitcode == 0
    finally:
        for process in (owner, contender):
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        receiver.close()
        sender.close()

    assert message == ("ok", "DatabaseReplicaColdResult")
    assert result_path.exists()
    assert not failure_path.exists()
    assert not marker.exists()
    assert stat.S_IMODE(invalid_candidate.stat().st_mode) == 0o555
    invalidation = load_database_replica_invalidation(tmp_path / "placement" / "database-replica-invalidation.json")
    assert invalidation.visible_final_absent is True


def test_cache_lock_contention_mutates_no_replica_namespace_before_ownership(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, manifest_path, cache_root = _stage_required_fixture(
        tmp_path / "work",
        cold_cache_root,
        monkeypatch,
    )
    staging = fixture.runspec.payload.database.staging
    assert staging is not None
    database = replace(fixture.runspec.payload.database, staging=replace(staging, lock_wait_seconds=1))
    runspec = replace(fixture.runspec, payload=replace(fixture.runspec.payload, database=database))
    locks = cache_root / ".locks"
    locks.mkdir(mode=0o700)
    cache_lock = locks / "cache.lock"
    descriptor = os.open(cache_lock, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    result_path = tmp_path / "placement" / "result.json"
    failure_path = tmp_path / "placement" / "failure.json"
    try:
        with pytest.raises(DatabasePlacementError, match="timed out waiting"):
            fixture.place_database(
                runspec,
                action_id=fixture.action.action_id,
                source_manifest_path=manifest_path,
                result_path=result_path,
                failure_path=failure_path,
            )
    finally:
        os.close(descriptor)

    assert load_database_replica_cold_failure(failure_path).classification == "lock-unavailable"
    assert not (cache_root / "replicas").exists()
    assert not (cache_root / ".staging").exists()
    identity = cache_root / ".locks" / f"{runspec.payload.database.source_manifest_sha256}.lock"
    identity_descriptor = os.open(identity, os.O_RDWR | os.O_NOFOLLOW)
    try:
        fcntl.flock(identity_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(identity_descriptor)


def test_same_identity_timeout_uses_the_short_declared_profile_without_downstream_work(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, manifest_path, cache_root = _stage_required_fixture(
        tmp_path / "work",
        cold_cache_root,
        monkeypatch,
    )
    staging = fixture.runspec.payload.database.staging
    assert staging is not None
    action = fixture.runspec.payload.actions[0]
    short_action = replace(action, resources=replace(action.resources, time="00:00:01"))
    database = replace(fixture.runspec.payload.database, staging=replace(staging, lock_wait_seconds=1))
    runspec = replace(
        fixture.runspec,
        payload=replace(fixture.runspec.payload, actions=(short_action,), database=database),
    )
    locks = cache_root / ".locks"
    locks.mkdir(mode=0o700)
    identity = runspec.payload.database.source_manifest_sha256
    owner = os.open(locks / f"{identity}.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)

    clock = [0.0]
    sleeps: list[float] = []
    requested_timeouts: list[int] = []

    def monotonic() -> float:
        return clock[0]

    def advance(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += seconds

    def bounded_wait(timeout_seconds: int) -> LockWait:
        requested_timeouts.append(timeout_seconds)
        return LockWait(
            deadline=monotonic() + timeout_seconds,
            monotonic=monotonic,
            sleeper=advance,
            poll_quantum_seconds=0.25,
        )

    services = _production_staged_database_placement_services()
    services = replace(services, wait_factory=LockWaitFactory(build=bounded_wait))
    result_path = tmp_path / "placement" / "result.json"
    failure_path = tmp_path / "placement" / "failure.json"
    try:
        with pytest.raises(DatabasePlacementError, match="timed out waiting"):
            fixture.place_database(
                runspec,
                action_id=short_action.action_id,
                source_manifest_path=manifest_path,
                result_path=result_path,
                failure_path=failure_path,
                staged_services=services,
            )
    finally:
        os.close(owner)

    assert requested_timeouts == [1]
    assert sleeps == [0.25, 0.25, 0.25, 0.25]
    assert not result_path.exists()
    failure = load_database_replica_cold_failure(failure_path)
    assert failure.classification == "lock-unavailable"
    assert failure.capacity_gate is None
    assert not (cache_root / "replicas").exists()
    assert not (cache_root / ".staging").exists()


@pytest.mark.parametrize("contended_lock", ["identity", "cache"])
def test_public_placement_times_out_before_retry_when_holder_releases_after_oversleep(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    contended_lock: str,
) -> None:
    fixture, manifest_path, cache_root = _stage_required_fixture(
        tmp_path / "work",
        cold_cache_root,
        monkeypatch,
    )
    staging = fixture.runspec.payload.database.staging
    assert staging is not None
    database = replace(fixture.runspec.payload.database, staging=replace(staging, lock_wait_seconds=1))
    runspec = replace(fixture.runspec, payload=replace(fixture.runspec.payload, database=database))
    locks = cache_root / ".locks"
    locks.mkdir(mode=0o700)
    identity = runspec.payload.database.source_manifest_sha256
    lock_name = f"{identity}.lock" if contended_lock == "identity" else "cache.lock"
    holder = os.open(locks / lock_name, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)

    now = 0.0
    sleeps: list[float] = []

    def monotonic() -> float:
        return now

    def oversleep_and_release(seconds: float) -> None:
        nonlocal now, holder
        sleeps.append(seconds)
        now += 1.25
        os.close(holder)
        holder = -1

    def bounded_wait(timeout_seconds: int) -> LockWait:
        assert timeout_seconds == 1
        return LockWait(
            deadline=1.0,
            monotonic=monotonic,
            sleeper=oversleep_and_release,
            poll_quantum_seconds=0.25,
        )

    services = _production_staged_database_placement_services()
    services = replace(services, wait_factory=LockWaitFactory(build=bounded_wait))
    result_path = tmp_path / "placement" / "result.json"
    failure_path = tmp_path / "placement" / "failure.json"
    try:
        with pytest.raises(DatabasePlacementError, match="timed out waiting"):
            fixture.place_database(
                runspec,
                action_id=runspec.payload.actions[0].action_id,
                source_manifest_path=manifest_path,
                result_path=result_path,
                failure_path=failure_path,
                staged_services=services,
            )
    finally:
        if holder >= 0:
            os.close(holder)

    assert sleeps == [0.25]
    assert not result_path.exists()
    failure = load_database_replica_cold_failure(failure_path)
    assert failure.classification == "lock-unavailable"
    assert failure.capacity_gate is None
    assert not (cache_root / "replicas").exists()
    assert not (cache_root / ".staging").exists()


@pytest.mark.parametrize("drifted_authority", ["cache-root", "locks-directory", "identity", "cache"])
def test_public_placement_rejects_same_inode_authority_mode_drift_after_wait(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    drifted_authority: str,
) -> None:
    fixture, manifest_path, cache_root = _stage_required_fixture(
        tmp_path / "work",
        cold_cache_root,
        monkeypatch,
    )
    staging = fixture.runspec.payload.database.staging
    assert staging is not None
    database = replace(fixture.runspec.payload.database, staging=replace(staging, lock_wait_seconds=1))
    runspec = replace(fixture.runspec, payload=replace(fixture.runspec.payload, database=database))
    locks = cache_root / ".locks"
    locks.mkdir(mode=0o700)
    identity = runspec.payload.database.source_manifest_sha256
    contended_name = "cache.lock" if drifted_authority == "cache" else f"{identity}.lock"
    holder = os.open(locks / contended_name, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
    drifted_path = _authority_path(cache_root, identity, drifted_authority)
    original_mode = stat.S_IMODE(drifted_path.stat().st_mode)
    drifted_mode = _different_mode(original_mode)
    now = 0.0

    def monotonic() -> float:
        return now

    def drift_and_release(_seconds: float) -> None:
        nonlocal now, holder
        now += 0.25
        drifted_path.chmod(drifted_mode)
        os.close(holder)
        holder = -1

    def bounded_wait(timeout_seconds: int) -> LockWait:
        assert timeout_seconds == 1
        return LockWait(
            deadline=1.0,
            monotonic=monotonic,
            sleeper=drift_and_release,
            poll_quantum_seconds=0.25,
        )

    services = _production_staged_database_placement_services()
    services = replace(services, wait_factory=LockWaitFactory(build=bounded_wait))
    result_path = tmp_path / "placement" / "result.json"
    failure_path = tmp_path / "placement" / "failure.json"
    try:
        with pytest.raises(DatabasePlacementError, match=r"authority|changed|invalid"):
            fixture.place_database(
                runspec,
                action_id=runspec.payload.actions[0].action_id,
                source_manifest_path=manifest_path,
                result_path=result_path,
                failure_path=failure_path,
                staged_services=services,
            )
    finally:
        drifted_path.chmod(original_mode)
        if holder >= 0:
            os.close(holder)

    assert not result_path.exists()
    failure = load_database_replica_cold_failure(failure_path)
    assert failure.classification == "lock-unavailable"
    assert failure.capacity_gate is None
    assert not (cache_root / "replicas").exists()
    assert not (cache_root / ".staging").exists()


@pytest.mark.parametrize("drifted_authority", ["cache-root", "locks-directory", "identity", "cache"])
def test_public_placement_rejects_live_same_inode_authority_mode_drift(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    drifted_authority: str,
) -> None:
    fixture, manifest_path, cache_root = _stage_required_fixture(
        tmp_path / "work",
        cold_cache_root,
        monkeypatch,
    )
    identity = fixture.runspec.payload.database.source_manifest_sha256
    services = _production_staged_database_placement_services()
    original_recheck = services.cold.warm_selector.select_under_identity
    rechecks = 0
    drifted_path: Path | None = None
    original_mode: int | None = None

    def drift_after_both_locks_are_live(*args: object, **kwargs: object) -> object:
        nonlocal rechecks, drifted_path, original_mode
        result = original_recheck(*args, **kwargs)
        rechecks += 1
        if rechecks == 2:
            drifted_path = _authority_path(cache_root, identity, drifted_authority)
            original_mode = stat.S_IMODE(drifted_path.stat().st_mode)
            drifted_path.chmod(_different_mode(original_mode))
        return result

    selector = replace(
        services.warm_selector,
        select_under_identity=drift_after_both_locks_are_live,
    )
    services = replace(
        services,
        cold=replace(services.cold, warm_selector=selector),
    )
    result_path = tmp_path / "placement" / "result.json"
    failure_path = tmp_path / "placement" / "failure.json"
    try:
        with pytest.raises(DatabasePlacementError, match=r"authority|changed|invalid"):
            fixture.place_database(
                fixture.runspec,
                action_id=fixture.runspec.payload.actions[0].action_id,
                source_manifest_path=manifest_path,
                result_path=result_path,
                failure_path=failure_path,
                staged_services=services,
            )
    finally:
        if drifted_path is not None and original_mode is not None:
            drifted_path.chmod(original_mode)

    assert rechecks == 2
    assert not result_path.exists()
    failure = load_database_replica_cold_failure(failure_path)
    assert failure.classification == "lock-unavailable"
    assert failure.capacity_gate is None
    assert not (cache_root / "replicas").exists()
    assert not (cache_root / ".staging").exists()


@pytest.mark.parametrize("drifted_authority", ["cache-root", "locks-directory", "identity", "cache"])
def test_public_cold_placement_revalidates_authority_after_freeze_before_publication(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    drifted_authority: str,
) -> None:
    fixture, manifest_path, cache_root = _stage_required_fixture(
        tmp_path / "work",
        cold_cache_root,
        monkeypatch,
    )
    monkeypatch.setattr(
        replica_runtime.os,
        "fstatvfs",
        lambda _descriptor: SimpleNamespace(f_bavail=2, f_frsize=1),
    )
    identity = fixture.runspec.payload.database.source_manifest_sha256
    authority_path = _authority_path(cache_root, identity, drifted_authority)
    services = _production_staged_database_placement_services()
    original_freeze = services.cold.population.freeze
    original_mode: int | None = None

    def freeze_then_drift(*args: object, **kwargs: object) -> None:
        nonlocal original_mode
        original_freeze(*args, **kwargs)
        original_mode = stat.S_IMODE(authority_path.stat().st_mode)
        authority_path.chmod(_different_mode(original_mode))

    population = replace(services.cold.population, freeze=freeze_then_drift)
    services = replace(services, cold=replace(services.cold, population=population))
    result_path = tmp_path / "placement" / "result.json"
    failure_path = tmp_path / "placement" / "failure.json"
    caught: DatabasePlacementError | None = None
    try:
        try:
            fixture.place_database(
                fixture.runspec,
                action_id=fixture.action.action_id,
                source_manifest_path=manifest_path,
                result_path=result_path,
                failure_path=failure_path,
                staged_services=services,
            )
        except DatabasePlacementError as exc:
            caught = exc
    finally:
        if original_mode is not None:
            authority_path.chmod(original_mode)

    replicas = cache_root / "replicas"
    final = replicas / identity
    assert not final.exists()
    assert caught is not None
    assert isinstance(caught, ClassifiedDatabaseReplicaError)
    assert caught.classification == "lock-unavailable"
    assert not result_path.exists()
    failure = load_database_replica_cold_failure(failure_path)
    assert failure.classification == "lock-unavailable"
    assert failure.capacity_gate is not None
    assert failure.capacity_gate.decision == "sufficient"
    assert tuple(replicas.iterdir()) == ()


@pytest.mark.parametrize("displaced_authority", ["identity-basename", "locks-directory"])
def test_public_warm_placement_revalidates_authority_after_result_publication(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    displaced_authority: str,
) -> None:
    fixture, manifest_path, cache_root = _stage_required_fixture(
        tmp_path / "work",
        cold_cache_root,
        monkeypatch,
    )
    monkeypatch.setattr(
        replica_runtime.os,
        "fstatvfs",
        lambda _descriptor: SimpleNamespace(f_bavail=2, f_frsize=1),
    )
    initial_result = fixture.place_database(
        fixture.runspec,
        action_id=fixture.action.action_id,
        source_manifest_path=manifest_path,
        result_path=tmp_path / "initial" / "result.json",
        failure_path=tmp_path / "initial" / "failure.json",
    )
    assert isinstance(initial_result, DatabaseReplicaColdResult)

    identity = fixture.runspec.payload.database.source_manifest_sha256
    locks = cache_root / ".locks"
    authority_path = locks / f"{identity}.lock" if displaced_authority == "identity-basename" else locks
    displaced_path = authority_path.with_name(f"{authority_path.name}.displaced")
    result_path = tmp_path / "warm" / "result.json"
    failure_path = tmp_path / "warm" / "failure.json"
    system_link = os.link
    displaced = False

    def publish_then_displace_authority(source: object, destination: object, **kwargs: object) -> None:
        nonlocal displaced
        system_link(source, destination, **kwargs)
        if destination == result_path.name:
            authority_path.rename(displaced_path)
            displaced = True

    monkeypatch.setattr(evidence_io.os, "link", publish_then_displace_authority)
    try:
        with pytest.raises(DatabasePlacementError) as raised:
            fixture.place_database(
                fixture.runspec,
                action_id=fixture.action.action_id,
                source_manifest_path=manifest_path,
                result_path=result_path,
                failure_path=failure_path,
            )
    finally:
        if displaced:
            displaced_path.rename(authority_path)

    assert isinstance(raised.value, ClassifiedDatabaseReplicaError)
    assert raised.value.classification == "lock-unavailable"
    assert isinstance(load_database_replica_result(result_path), DatabaseReplicaWarmResult)
    assert not failure_path.exists()


def _authority_path(cache_root: Path, identity: str, authority: str) -> Path:
    return {
        "cache-root": cache_root,
        "locks-directory": cache_root / ".locks",
        "identity": cache_root / ".locks" / f"{identity}.lock",
        "cache": cache_root / ".locks" / "cache.lock",
    }[authority]


def _different_mode(mode: int) -> int:
    return 0o700 if mode != 0o700 else 0o777


def test_cache_lock_swap_after_acquisition_fails_before_capacity_or_evidence(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, manifest_path, cache_root = _stage_required_fixture(
        tmp_path / "work",
        cold_cache_root,
        monkeypatch,
    )
    services = _production_staged_database_placement_services()
    original_recheck = services.cold.warm_selector.select_under_identity
    rechecks = 0
    displaced = cache_root / ".locks" / "cache.lock.displaced"

    def swap_after_second_recheck(*args: object, **kwargs: object) -> object:
        nonlocal rechecks
        result = original_recheck(*args, **kwargs)
        rechecks += 1
        if rechecks == 2:
            cache_lock = cache_root / ".locks" / "cache.lock"
            cache_lock.rename(displaced)
            cache_lock.touch(mode=0o600)
        return result

    selector = replace(
        services.warm_selector,
        select_under_identity=swap_after_second_recheck,
    )
    services = replace(services, cold=replace(services.cold, warm_selector=selector))
    result_path = tmp_path / "placement" / "result.json"
    try:
        with pytest.raises(DatabasePlacementError, match="cache lock changed"):
            fixture.place_database(
                fixture.runspec,
                action_id=fixture.action.action_id,
                source_manifest_path=manifest_path,
                result_path=result_path,
                failure_path=None,
                staged_services=services,
            )
    finally:
        cache_lock = cache_root / ".locks" / "cache.lock"
        if displaced.exists():
            cache_lock.unlink()
            displaced.rename(cache_lock)

    assert rechecks == 2
    assert not result_path.exists()


@pytest.mark.parametrize("second_has_capacity", [True, False])
def test_different_identities_serialize_post_wait_capacity_and_copy_under_the_cache_lock(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    second_has_capacity: bool,
) -> None:
    first_fixture, original_manifest_path, cache_root = _stage_required_fixture(
        tmp_path / "work",
        cold_cache_root,
        monkeypatch,
    )
    first_manifest_path = tmp_path / "manifest-first.json"
    first_manifest_path.write_bytes(original_manifest_path.read_bytes())
    second_manifest_path = tmp_path / "manifest-second.json"
    first_manifest = first_fixture.runspec.payload.database.source_manifest
    second_manifest = replace(
        first_manifest,
        database_set=replace(first_manifest.database_set, version="2026-09"),
    )
    second_fixture = _replace_stage_manifest(first_fixture, second_manifest, second_manifest_path)

    def bounded_runspec(fixture: object) -> object:
        runspec = fixture.runspec  # type: ignore[attr-defined]
        staging = runspec.payload.database.staging
        assert staging is not None
        database = replace(runspec.payload.database, staging=replace(staging, lock_wait_seconds=3))
        return replace(runspec, payload=replace(runspec.payload, database=database))

    runspecs = (bounded_runspec(first_fixture), bounded_runspec(second_fixture))
    identities = tuple(runspec.payload.database.source_manifest_sha256 for runspec in runspecs)  # type: ignore[attr-defined]
    assert identities[0] != identities[1]
    (cache_root / "replicas").mkdir(mode=0o700)

    context = multiprocessing.get_context("fork")
    simultaneous_miss = context.Barrier(2)
    first_copy_entered = context.Event()
    second_cache_waiting = context.Event()
    release_first_copy = context.Event()
    second_capacity_entered = context.Event()
    capacity_counts = context.Array("i", (0, 0))
    copy_counts = context.Array("i", (0, 0))
    worker_ordinal = -1
    services = _production_staged_database_placement_services()
    original_probe = services.warm_selector.probe
    original_fstatvfs = replica_runtime.os.fstatvfs
    original_populate = services.cold.population.populate

    def synchronized_probe(*args: object, **kwargs: object) -> object:
        disposition = original_probe(*args, **kwargs)
        simultaneous_miss.wait(timeout=5)
        if worker_ordinal == 1 and not first_copy_entered.wait(timeout=5):
            raise AssertionError("first identity never entered its held copy")
        return disposition

    def coordinating_sleep(seconds: float) -> None:
        if worker_ordinal == 1:
            second_cache_waiting.set()
        time.sleep(seconds)

    def bounded_wait(timeout_seconds: int) -> LockWait:
        assert timeout_seconds == 3
        return LockWait(
            deadline=time.monotonic() + timeout_seconds,
            monotonic=time.monotonic,
            sleeper=coordinating_sleep,
            poll_quantum_seconds=0.01,
        )

    def observed_capacity(descriptor: int) -> object:
        capacity_counts[worker_ordinal] += 1
        if worker_ordinal == 1:
            second_capacity_entered.set()
            if not second_has_capacity:
                return SimpleNamespace(f_bavail=0, f_frsize=1)
        return original_fstatvfs(descriptor)

    def coordinated_copy(*args: object, **kwargs: object) -> object:
        copy_counts[worker_ordinal] += 1
        if worker_ordinal == 0:
            first_copy_entered.set()
            if not second_cache_waiting.wait(timeout=5):
                raise AssertionError("second identity never waited for cache ownership")
            if not release_first_copy.wait(timeout=5):
                raise AssertionError("different-identity test did not release the first copy")
        return original_populate(*args, **kwargs)

    monkeypatch.setattr(replica_runtime.os, "fstatvfs", observed_capacity)
    selector = replace(services.warm_selector, probe=synchronized_probe)
    population = replace(services.cold.population, populate=coordinated_copy)
    services = replace(
        services,
        warm_selector=selector,
        wait_factory=LockWaitFactory(build=bounded_wait),
        cold=replace(
            services.cold,
            warm_selector=selector,
            population=population,
        ),
    )
    result_paths = tuple(tmp_path / "placement" / f"different-{ordinal}.json" for ordinal in range(2))
    failure_paths = tuple(tmp_path / "placement" / f"different-failure-{ordinal}.json" for ordinal in range(2))
    manifest_paths = (first_manifest_path, second_manifest_path)
    receivers = []
    processes = []

    def place(ordinal: int, sender: object) -> None:
        nonlocal worker_ordinal
        worker_ordinal = ordinal
        runspec = runspecs[ordinal]
        try:
            result = first_fixture.place_database(
                runspec,  # type: ignore[arg-type]
                action_id=runspec.payload.actions[0].action_id,  # type: ignore[attr-defined]
                source_manifest_path=manifest_paths[ordinal],
                result_path=result_paths[ordinal],
                failure_path=failure_paths[ordinal],
                staged_services=services,
            )
            sender.send(("ok", type(result).__name__))  # type: ignore[attr-defined]
        except BaseException as exc:
            sender.send(("error", type(exc).__name__, str(exc)))  # type: ignore[attr-defined]
        finally:
            sender.close()  # type: ignore[attr-defined]

    try:
        for ordinal in range(2):
            receiver, sender = context.Pipe(duplex=False)
            process = context.Process(target=place, args=(ordinal, sender))
            receivers.append(receiver)
            processes.append(process)
            process.start()
            sender.close()
        assert first_copy_entered.wait(timeout=5)
        assert second_cache_waiting.wait(timeout=5)
        assert capacity_counts[0] == 1
        assert capacity_counts[1] == 0
        assert not second_capacity_entered.is_set()
        second_identity = os.open(cache_root / ".locks" / f"{identities[1]}.lock", os.O_RDWR | os.O_NOFOLLOW)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(second_identity, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(second_identity)
        release_first_copy.set()
        messages = [receiver.recv() for receiver in receivers]
        for process in processes:
            process.join(timeout=5)
        assert all(not process.is_alive() and process.exitcode == 0 for process in processes)
    finally:
        release_first_copy.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        for receiver in receivers:
            receiver.close()

    expected_second_message = (
        ("ok", "DatabaseReplicaColdResult")
        if second_has_capacity
        else (
            "error",
            "DatabasePlacementError",
            "insufficient user-available cache capacity for required cold Database Replica",
        )
    )
    assert messages == [("ok", "DatabaseReplicaColdResult"), expected_second_message]
    assert tuple(capacity_counts) == (1, 1)
    assert tuple(copy_counts) == ((1, 1) if second_has_capacity else (1, 0))
    assert second_capacity_entered.is_set()
    if second_has_capacity:
        assert not any(path.exists() for path in failure_paths)
        assert {path.name for path in (cache_root / "replicas").iterdir()} == set(identities)
    else:
        assert not failure_paths[0].exists()
        failure = load_database_replica_cold_failure(failure_paths[1])
        assert failure.classification == "insufficient-capacity"
        assert failure.capacity_gate is not None
        assert failure.capacity_gate.decision == "insufficient"
        assert {path.name for path in (cache_root / "replicas").iterdir()} == {identities[0]}


def test_dead_cold_owner_waiter_hard_fails_and_only_later_fresh_action_cleans_orphan(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, manifest_path, cache_root = _stage_required_fixture(
        tmp_path / "work",
        cold_cache_root,
        monkeypatch,
    )
    staging = fixture.runspec.payload.database.staging
    assert staging is not None
    database = replace(fixture.runspec.payload.database, staging=replace(staging, lock_wait_seconds=3))
    runspec = replace(fixture.runspec, payload=replace(fixture.runspec.payload, database=database))
    (cache_root / "replicas").mkdir(mode=0o700)
    monkeypatch.setattr(
        replica_runtime.os,
        "fstatvfs",
        lambda _descriptor: SimpleNamespace(f_bavail=2, f_frsize=1),
    )

    context = multiprocessing.get_context("fork")
    owner_temp_ready = context.Event()
    contender_waiting = context.Event()
    worker_role = "fresh"
    services = _production_staged_database_placement_services()
    original_populate = services.cold.population.populate

    def coordinating_sleep(seconds: float) -> None:
        if worker_role == "contender":
            contender_waiting.set()
        time.sleep(seconds)

    def bounded_wait(timeout_seconds: int) -> LockWait:
        assert timeout_seconds == 3
        return LockWait(
            deadline=time.monotonic() + timeout_seconds,
            monotonic=time.monotonic,
            sleeper=coordinating_sleep,
            poll_quantum_seconds=0.01,
        )

    def owner_dies_in_copy(*args: object, **kwargs: object) -> object:
        if worker_role != "owner":
            return original_populate(*args, **kwargs)
        temporary_descriptor = kwargs["temporary_descriptor"]
        assert isinstance(temporary_descriptor, int)
        marker = os.open(
            "owner-death-marker",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=temporary_descriptor,
        )
        try:
            os.write(marker, b"incomplete")
        finally:
            os.close(marker)
        owner_temp_ready.set()
        if not contender_waiting.wait(timeout=5):
            os._exit(24)
        os._exit(23)

    population = replace(services.cold.population, populate=owner_dies_in_copy)
    services = replace(
        services,
        wait_factory=LockWaitFactory(build=bounded_wait),
        cold=replace(services.cold, population=population),
    )
    contender_result = tmp_path / "placement" / "contender-result.json"
    contender_failure = tmp_path / "placement" / "contender-failure.json"
    receiver, sender = context.Pipe(duplex=False)

    def place(role: str, sender_connection: object | None = None) -> None:
        nonlocal worker_role
        worker_role = role
        try:
            result = fixture.place_database(
                runspec,
                action_id=runspec.payload.actions[0].action_id,
                source_manifest_path=manifest_path,
                result_path=(
                    tmp_path / "placement" / "dead-owner-result.json" if role == "owner" else contender_result
                ),
                failure_path=(
                    tmp_path / "placement" / "dead-owner-failure.json" if role == "owner" else contender_failure
                ),
                staged_services=services,
            )
            if sender_connection is not None:
                sender_connection.send(("ok", type(result).__name__))  # type: ignore[attr-defined]
        except BaseException as exc:
            if sender_connection is not None:
                sender_connection.send(("error", type(exc).__name__, str(exc)))  # type: ignore[attr-defined]
        finally:
            if sender_connection is not None:
                sender_connection.close()  # type: ignore[attr-defined]

    owner = context.Process(target=place, args=("owner",))
    contender = context.Process(target=place, args=("contender", sender))
    try:
        owner.start()
        assert owner_temp_ready.wait(timeout=5)
        contender.start()
        sender.close()
        assert contender_waiting.wait(timeout=5)
        owner.join(timeout=5)
        assert not owner.is_alive() and owner.exitcode == 23
        message = receiver.recv()
        contender.join(timeout=5)
        assert not contender.is_alive() and contender.exitcode == 0
    finally:
        for process in (owner, contender):
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        receiver.close()
        sender.close()

    assert message[0:2] == ("error", "ClassifiedDatabaseReplicaError")
    assert "released without a valid publication" in message[2]
    assert not contender_result.exists()
    assert load_database_replica_cold_failure(contender_failure).classification == "lock-unavailable"
    identity = runspec.payload.database.source_manifest_sha256
    assert not (cache_root / "replicas" / identity).exists()
    orphans = tuple((cache_root / "replicas").glob(".population-*"))
    assert len(orphans) == 1
    marker = orphans[0] / "owner-death-marker"
    assert marker.read_bytes() == b"incomplete"
    assert not (tmp_path / "placement" / "database-replica-stale-population-cleanup.json").exists()

    worker_role = "fresh"
    fresh_result = fixture.place_database(
        runspec,
        action_id=runspec.payload.actions[0].action_id,
        source_manifest_path=manifest_path,
        result_path=tmp_path / "placement" / "fresh-result.json",
        failure_path=tmp_path / "placement" / "fresh-failure.json",
        staged_services=services,
    )
    assert isinstance(fresh_result, DatabaseReplicaColdResult)
    assert not marker.exists()
    assert not orphans[0].exists()
    assert (cache_root / "replicas" / identity).is_dir()
    assert not (tmp_path / "placement" / "fresh-failure.json").exists()
    cleanup = load_database_replica_cleanup(tmp_path / "placement" / "database-replica-stale-population-cleanup.json")
    assert cleanup.removed_basename_samples == (orphans[0].name,)
