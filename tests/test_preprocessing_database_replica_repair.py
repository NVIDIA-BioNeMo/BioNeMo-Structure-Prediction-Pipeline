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

"""Automatic invalid Database Replica repair through ordinary placement."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import multiprocessing
import os
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from bspp.orchestration.contract.database_placement_result import DatabaseSourceObservation
from bspp.orchestration.contract.database_replica import (
    DatabaseReplicaColdResult,
    DatabaseReplicaCopyEvidence,
    DatabaseReplicaMember,
    DatabaseReplicaWarmResult,
)
from bspp.orchestration.contract.phase import PhaseRunSpec
from bspp.orchestration.runtime.preprocessing import _database_replica_publication as publication_runtime
from bspp.orchestration.runtime.preprocessing import _database_replica_repair as repair_runtime
from bspp.orchestration.runtime.preprocessing._database_placement_errors import DatabasePlacementError
from bspp.orchestration.runtime.preprocessing._database_replica_errors import ClassifiedDatabaseReplicaError
from bspp.orchestration.runtime.preprocessing._database_replica_evidence_io import (
    load_database_replica_cold_failure,
)
from bspp.orchestration.runtime.preprocessing._database_replica_lease import (
    DatabaseReplicaLeasePaths,
    _database_replica_lease,
)
from bspp.orchestration.runtime.preprocessing._database_replica_lock import (
    LockWait,
    acquire_existing_shared_identity_lock,
)
from bspp.orchestration.runtime.preprocessing._database_replica_lock_types import CacheExclusiveOwnership
from bspp.orchestration.runtime.preprocessing._database_replica_repair_evidence import (
    DatabaseReplicaCleanupEvidence,
    DatabaseReplicaInvalidationEvidence,
    load_database_replica_cleanup,
    load_database_replica_invalidation,
    publish_database_replica_cleanup,
    publish_database_replica_invalidation,
)
from bspp.orchestration.runtime.preprocessing._database_replica_services import (
    LockWaitFactory,
)
from bspp.orchestration.runtime.preprocessing._owned_tree import (
    ExpectedDirectoryBinding,
    OwnedTreeRemover,
)
from bspp.orchestration.runtime.preprocessing.database_placement import (
    _production_staged_database_placement_services,
)
from tests.support.database_cold_replica import _stage_required_fixture
from tests.support.database_cold_replica import cold_cache_root as cold_cache_root
from tests.support.preprocessing_execution import LocalExecutionFixture


class _RenameAt2Function:
    def __init__(self, call: Callable[[int, bytes, int, bytes, int], int]) -> None:
        self._call = call
        self.argtypes: object = None
        self.restype: object = None

    def __call__(
        self,
        source_parent: int,
        source_name: bytes,
        destination_parent: int,
        destination_name: bytes,
        flags: int,
    ) -> int:
        return self._call(source_parent, source_name, destination_parent, destination_name, flags)


class _LibcWithRenameAt2:
    def __init__(self, library: object, renameat2: Callable[[int, bytes, int, bytes, int], int]) -> None:
        self._library = library
        self.renameat2 = _RenameAt2Function(renameat2)

    def __getattr__(self, name: str) -> object:
        return getattr(self._library, name)


def _replace_libc_renameat2(
    monkeypatch: pytest.MonkeyPatch,
    renameat2: Callable[[int, bytes, int, bytes, int], int],
) -> None:
    real_cdll = cast(Callable[..., object], ctypes.CDLL)

    def load_library(*args: object, **kwargs: object) -> _LibcWithRenameAt2:
        return _LibcWithRenameAt2(real_cdll(*args, **kwargs), renameat2)

    monkeypatch.setattr(ctypes, "CDLL", load_library)


def _prepare_invalid_replica(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[LocalExecutionFixture, Path, Path, int]:
    fixture, manifest_path, cache_root = _stage_required_fixture(
        tmp_path / "work",
        cold_cache_root,
        monkeypatch,
    )
    monkeypatch.setattr(
        os,
        "fstatvfs",
        lambda _descriptor: SimpleNamespace(f_bavail=2, f_frsize=1),
    )
    first = fixture.place_database(
        fixture.runspec,
        action_id=fixture.action.action_id,
        source_manifest_path=manifest_path,
        result_path=tmp_path / "first-action" / "database-placement-result.json",
        failure_path=tmp_path / "first-action" / "database-placement-failure.json",
    )
    assert isinstance(first, DatabaseReplicaColdResult)
    identity = fixture.runspec.payload.database.source_manifest_sha256
    candidate = cache_root / "replicas" / identity
    candidate.chmod(0o755)
    candidate_manifest = candidate / "replica-manifest.json"
    candidate_manifest.chmod(0o644)
    candidate_manifest.write_bytes(b"{}\n")
    candidate_manifest.chmod(0o444)
    candidate.chmod(0o555)
    return fixture, manifest_path, candidate, candidate.stat().st_ino


def test_pre_retirement_rebound_stat_oserror_publishes_replica_validation_failure(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, manifest_path, candidate, original_inode = _prepare_invalid_replica(
        tmp_path,
        cold_cache_root,
        monkeypatch,
    )
    identity = fixture.runspec.payload.database.source_manifest_sha256
    real_stat = os.stat
    identity_stats = 0
    injected = False

    def fail_pre_retirement_rebound_stat(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        nonlocal identity_stats, injected
        if path == identity and dir_fd is not None:
            identity_stats += 1
            if identity_stats == 6:
                injected = True
                raise PermissionError(errno.EACCES, "injected pre-retirement rebound stat failure")
        return real_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(os, "stat", fail_pre_retirement_rebound_stat)
    action_dir = tmp_path / "repair-action"
    result_path = action_dir / "database-placement-result.json"
    failure_path = action_dir / "database-placement-failure.json"

    with pytest.raises(DatabasePlacementError, match="cannot observe invalid Database Replica before retirement"):
        fixture.place_database(
            fixture.runspec,
            action_id=fixture.action.action_id,
            source_manifest_path=manifest_path,
            result_path=result_path,
            failure_path=failure_path,
        )

    assert injected
    assert not result_path.exists()
    assert candidate.stat().st_ino == original_inode
    assert (candidate / "replica-manifest.json").read_bytes() == b"{}\n"
    assert not (action_dir / "database-replica-invalidation.json").exists()
    assert not (action_dir / "database-replica-stale-population-cleanup.json").exists()
    failure = load_database_replica_cold_failure(failure_path)
    assert failure.classification == "replica-validation-failed"
    assert failure.science_started is False


def test_post_retirement_final_absence_stat_oserror_publishes_publication_failure(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, manifest_path, candidate, original_inode = _prepare_invalid_replica(
        tmp_path,
        cold_cache_root,
        monkeypatch,
    )
    identity = fixture.runspec.payload.database.source_manifest_sha256
    real_stat = os.stat
    identity_stats = 0
    injected = False

    def fail_final_absence_stat(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        nonlocal identity_stats, injected
        if path == identity and dir_fd is not None:
            identity_stats += 1
            if identity_stats == 7:
                injected = True
                raise PermissionError(errno.EACCES, "injected post-retirement final-absence stat failure")
        return real_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(os, "stat", fail_final_absence_stat)
    action_dir = tmp_path / "repair-action"
    result_path = action_dir / "database-placement-result.json"
    failure_path = action_dir / "database-placement-failure.json"

    with pytest.raises(DatabasePlacementError, match="cannot confirm retired Database Replica final absence"):
        fixture.place_database(
            fixture.runspec,
            action_id=fixture.action.action_id,
            source_manifest_path=manifest_path,
            result_path=result_path,
            failure_path=failure_path,
        )

    assert injected
    retired_candidates = tuple(candidate.parent.glob(f".population-{identity}-*"))
    assert len(retired_candidates) == 1
    retired = retired_candidates[0]
    assert not result_path.exists()
    assert not candidate.exists()
    assert retired.stat().st_ino == original_inode
    assert (retired / "replica-manifest.json").read_bytes() == b"{}\n"
    assert not (action_dir / "database-replica-invalidation.json").exists()
    assert not (action_dir / "database-replica-stale-population-cleanup.json").exists()
    failure = load_database_replica_cold_failure(failure_path)
    assert failure.classification == "publication-failed"
    assert failure.science_started is False


@pytest.mark.parametrize("sidecar_kind", ["invalidation", "cleanup"])
@pytest.mark.parametrize("fault", ["parent-fsync", "parent-rebind"])
def test_repair_sidecar_post_link_failure_is_a_publication_failure_before_repopulation(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    sidecar_kind: str,
    fault: str,
) -> None:
    original_inode: int | None = None
    candidate: Path | None = None
    stale_name: str | None = None
    if sidecar_kind == "invalidation":
        fixture, manifest_path, candidate, original_inode = _prepare_invalid_replica(
            tmp_path,
            cold_cache_root,
            monkeypatch,
        )
        action_dir = tmp_path / "repair-action"
        sidecar = action_dir / "database-replica-invalidation.json"
    else:
        fixture, manifest_path, cache_root = _stage_required_fixture(
            tmp_path / "work",
            cold_cache_root,
            monkeypatch,
        )
        monkeypatch.setattr(
            os,
            "fstatvfs",
            lambda _descriptor: SimpleNamespace(f_bavail=2, f_frsize=1),
        )
        identity = fixture.runspec.payload.database.source_manifest_sha256
        replicas = cache_root / "replicas"
        replicas.mkdir(mode=0o700)
        stale_name = f".population-{identity}-{'4' * 32}"
        (replicas / stale_name).mkdir(mode=0o700)
        action_dir = tmp_path / "fresh-action"
        sidecar = action_dir / "database-replica-stale-population-cleanup.json"

    services = _production_staged_database_placement_services()
    real_populate = services.cold.population.populate
    population_calls = 0

    def observe_population(
        *,
        source_descriptor: int,
        temporary_descriptor: int,
        observation: DatabaseSourceObservation,
        allocated_cpus: int,
        rsync_executable: Path,
    ) -> tuple[DatabaseReplicaCopyEvidence, tuple[DatabaseReplicaMember, ...]]:
        nonlocal population_calls
        population_calls += 1
        return real_populate(
            source_descriptor=source_descriptor,
            temporary_descriptor=temporary_descriptor,
            observation=observation,
            allocated_cpus=allocated_cpus,
            rsync_executable=rsync_executable,
        )

    population = replace(services.cold.population, populate=observe_population)
    services = replace(services, cold=replace(services.cold, population=population))
    injected = False
    if fault == "parent-fsync":
        real_fsync = os.fsync

        def fail_post_link_parent_fsync(descriptor: int) -> None:
            nonlocal injected
            try:
                target = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
            except OSError:
                target = None
            if not injected and target == action_dir and sidecar.exists():
                injected = True
                raise OSError("injected repair-sidecar post-link parent fsync failure")
            real_fsync(descriptor)

        monkeypatch.setattr(os, "fsync", fail_post_link_parent_fsync)
    else:
        real_lstat = Path.lstat

        def fail_post_link_parent_rebind(path: Path) -> os.stat_result:
            nonlocal injected
            if not injected and path == action_dir and sidecar.exists():
                injected = True
                raise PermissionError(errno.EACCES, "injected repair-sidecar post-link parent rebind failure")
            return real_lstat(path)

        monkeypatch.setattr(Path, "lstat", fail_post_link_parent_rebind)

    result_path = action_dir / "database-placement-result.json"
    failure_path = action_dir / "database-placement-failure.json"

    with pytest.raises(DatabasePlacementError):
        fixture.place_database(
            fixture.runspec,
            action_id=fixture.action.action_id,
            source_manifest_path=manifest_path,
            result_path=result_path,
            failure_path=failure_path,
            staged_services=services,
        )

    assert injected
    assert population_calls == 0
    assert not result_path.exists()
    failure = load_database_replica_cold_failure(failure_path)
    assert failure.classification == "publication-failed"
    assert failure.science_started is False
    identity = fixture.runspec.payload.database.source_manifest_sha256
    staging = fixture.runspec.payload.database.staging
    assert staging is not None
    replicas = Path(staging.user_cache_root) / "replicas"
    if sidecar_kind == "invalidation":
        assert candidate is not None
        assert original_inode is not None
        assert not candidate.exists()
        retired = tuple(replicas.glob(f".population-{identity}-*"))
        assert len(retired) == 1
        assert retired[0].stat().st_ino == original_inode
        assert (retired[0] / "replica-manifest.json").read_bytes() == b"{}\n"
        invalidation = load_database_replica_invalidation(sidecar)
        assert invalidation.original_replica_inode == original_inode
        assert not (action_dir / "database-replica-stale-population-cleanup.json").exists()
    else:
        assert stale_name is not None
        assert not (replicas / identity).exists()
        assert not (replicas / stale_name).exists()
        assert not tuple(replicas.glob(f".population-{identity}-*"))
        cleanup = load_database_replica_cleanup(sidecar)
        assert cleanup.removed_count == 1
        assert cleanup.removed_basename_samples == (stale_name,)
        assert not (action_dir / "database-replica-invalidation.json").exists()


@pytest.mark.parametrize("sidecar_kind", ["invalidation", "cleanup"])
def test_repair_sidecar_publisher_accepts_only_a_genuine_exact_eexist_collision(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    sidecar_kind: str,
) -> None:
    fixture, manifest_path, _, _ = _prepare_invalid_replica(
        tmp_path,
        cold_cache_root,
        monkeypatch,
    )
    action_dir = tmp_path / "repair-action"
    repaired = fixture.place_database(
        fixture.runspec,
        action_id=fixture.action.action_id,
        source_manifest_path=manifest_path,
        result_path=action_dir / "database-placement-result.json",
        failure_path=action_dir / "database-placement-failure.json",
    )
    assert isinstance(repaired, DatabaseReplicaColdResult)

    if sidecar_kind == "invalidation":
        destination = action_dir / "database-replica-invalidation.json"
        evidence = load_database_replica_invalidation(destination)
        publish_database_replica_invalidation(evidence, destination)
        assert load_database_replica_invalidation(destination) == evidence
    else:
        destination = action_dir / "database-replica-stale-population-cleanup.json"
        evidence = load_database_replica_cleanup(destination)
        publish_database_replica_cleanup(evidence, destination)
        assert load_database_replica_cleanup(destination) == evidence


@pytest.mark.parametrize("sidecar_kind", ["invalidation", "cleanup"])
@pytest.mark.parametrize("collision", ["unequal", "invalid"])
def test_repair_sidecar_publisher_rejects_nonexact_eexist_collision_as_publication_failed(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    sidecar_kind: str,
    collision: str,
) -> None:
    fixture, manifest_path, _, _ = _prepare_invalid_replica(
        tmp_path,
        cold_cache_root,
        monkeypatch,
    )
    action_dir = tmp_path / "repair-action"
    repaired = fixture.place_database(
        fixture.runspec,
        action_id=fixture.action.action_id,
        source_manifest_path=manifest_path,
        result_path=action_dir / "database-placement-result.json",
        failure_path=action_dir / "database-placement-failure.json",
    )
    assert isinstance(repaired, DatabaseReplicaColdResult)

    if sidecar_kind == "invalidation":
        destination = action_dir / "database-replica-invalidation.json"
        published = load_database_replica_invalidation(destination)
        requested: DatabaseReplicaInvalidationEvidence | DatabaseReplicaCleanupEvidence = replace(
            published,
            validation_error=f"{published.validation_error} unequal",
        )
    else:
        destination = action_dir / "database-replica-stale-population-cleanup.json"
        published = load_database_replica_cleanup(destination)
        requested = replace(published, failure="unequal collision")
    if collision == "invalid":
        destination.chmod(0o644)
        destination.write_bytes(b"invalid\n")
        destination.chmod(0o444)
        requested = published

    with pytest.raises(ClassifiedDatabaseReplicaError) as raised:
        if isinstance(requested, DatabaseReplicaInvalidationEvidence):
            publish_database_replica_invalidation(requested, destination)
        else:
            publish_database_replica_cleanup(requested, destination)

    assert raised.value.classification == "publication-failed"


def test_invalid_replica_retirement_waits_for_two_real_shared_readers(
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
    monkeypatch.setattr(
        os,
        "fstatvfs",
        lambda _descriptor: SimpleNamespace(f_bavail=2, f_frsize=1),
    )
    first = fixture.place_database(
        runspec,
        action_id=fixture.action.action_id,
        source_manifest_path=manifest_path,
        result_path=tmp_path / "first-action" / "database-placement-result.json",
        failure_path=tmp_path / "first-action" / "database-placement-failure.json",
    )
    assert isinstance(first, DatabaseReplicaColdResult)
    identity = runspec.payload.database.source_manifest_sha256
    candidate = cache_root / "replicas" / identity
    lease_path = cache_root / ".locks" / f"{identity}.lock"
    lease_paths = DatabaseReplicaLeasePaths(selected_root=candidate, lease=lease_path)

    context = multiprocessing.get_context("fork")
    science_ready = context.Event()
    reader_ready = context.Event()
    repair_waiting = context.Event()
    release_science = context.Event()
    release_reader = context.Event()
    original_sleep = time.sleep

    def bounded_wait(timeout_seconds: int) -> LockWait:
        def observed_sleep(seconds: float) -> None:
            repair_waiting.set()
            original_sleep(seconds)

        return LockWait(
            deadline=time.monotonic() + timeout_seconds,
            monotonic=time.monotonic,
            sleeper=observed_sleep,
            poll_quantum_seconds=0.01,
        )

    services = _production_staged_database_placement_services()
    services = replace(services, wait_factory=LockWaitFactory(build=bounded_wait))

    def hold_science_lease() -> None:
        with _database_replica_lease(runspec, first, paths=lease_paths):
            science_ready.set()
            if not release_science.wait(timeout=5):
                os._exit(21)

    def hold_direct_reader() -> None:
        cache_descriptor = os.open(cache_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        locks_descriptor = os.open(
            ".locks",
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=cache_descriptor,
        )
        lease_descriptor = acquire_existing_shared_identity_lock(locks_descriptor, f"{identity}.lock")
        try:
            reader_ready.set()
            if not release_reader.wait(timeout=5):
                os._exit(22)
        finally:
            os.close(lease_descriptor)
            os.close(locks_descriptor)
            os.close(cache_descriptor)

    result_path = tmp_path / "repair-action" / "database-placement-result.json"
    failure_path = tmp_path / "repair-action" / "database-placement-failure.json"
    receiver, sender = context.Pipe(duplex=False)

    def repair() -> None:
        try:
            result = fixture.place_database(
                runspec,
                action_id=fixture.action.action_id,
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

    science = context.Process(target=hold_science_lease)
    reader = context.Process(target=hold_direct_reader)
    repairer = context.Process(target=repair)
    try:
        science.start()
        reader.start()
        assert science_ready.wait(timeout=5)
        assert reader_ready.wait(timeout=5)
        original_inode = candidate.stat().st_ino
        candidate.chmod(0o755)
        replica_manifest = candidate / "replica-manifest.json"
        replica_manifest.chmod(0o644)
        replica_manifest.write_bytes(b"{}\n")
        replica_manifest.chmod(0o444)
        candidate.chmod(0o555)

        repairer.start()
        sender.close()
        assert repair_waiting.wait(timeout=5)
        assert candidate.stat().st_ino == original_inode
        assert not result_path.exists()
        assert not (tmp_path / "repair-action" / "database-replica-invalidation.json").exists()

        release_reader.set()
        reader.join(timeout=5)
        assert not reader.is_alive() and reader.exitcode == 0
        original_sleep(0.1)
        assert candidate.stat().st_ino == original_inode
        assert repairer.is_alive()
        assert not result_path.exists()

        release_science.set()
        message = receiver.recv()
        science.join(timeout=5)
        repairer.join(timeout=5)
        assert not science.is_alive() and science.exitcode == 0
        assert not repairer.is_alive() and repairer.exitcode == 0
    finally:
        release_reader.set()
        release_science.set()
        for process in (science, reader, repairer):
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        receiver.close()
        sender.close()

    assert message == ("ok", "DatabaseReplicaColdResult")
    assert candidate.stat().st_ino != original_inode
    assert result_path.exists()
    assert not failure_path.exists()
    invalidation = load_database_replica_invalidation(tmp_path / "repair-action" / "database-replica-invalidation.json")
    assert invalidation.original_replica_inode == original_inode


def test_invalid_replica_is_repaired_by_the_existing_cold_coordinator(
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
        os,
        "fstatvfs",
        lambda _descriptor: SimpleNamespace(f_bavail=2, f_frsize=1),
    )
    first = fixture.place_database(
        fixture.runspec,
        action_id=fixture.action.action_id,
        source_manifest_path=manifest_path,
        result_path=tmp_path / "first-action" / "database-placement-result.json",
        failure_path=tmp_path / "first-action" / "database-placement-failure.json",
    )
    assert isinstance(first, DatabaseReplicaColdResult)
    identity = fixture.runspec.payload.database.source_manifest_sha256
    candidate = cache_root / "replicas" / identity
    candidate.chmod(0o755)
    replica_manifest = candidate / "replica-manifest.json"
    replica_manifest.chmod(0o644)
    replica_manifest.write_bytes(b"{}\n")
    replica_manifest.chmod(0o444)
    candidate.chmod(0o555)

    repaired = fixture.place_database(
        fixture.runspec,
        action_id=fixture.action.action_id,
        source_manifest_path=manifest_path,
        result_path=tmp_path / "repair-action" / "database-placement-result.json",
        failure_path=tmp_path / "repair-action" / "database-placement-failure.json",
    )

    assert isinstance(repaired, DatabaseReplicaColdResult)
    assert repaired.outcome.value == "replica-cold"
    assert not (tmp_path / "repair-action" / "database-placement-failure.json").exists()
    invalidation = load_database_replica_invalidation(tmp_path / "repair-action" / "database-replica-invalidation.json")
    cleanup = load_database_replica_cleanup(
        tmp_path / "repair-action" / "database-replica-stale-population-cleanup.json"
    )
    assert invalidation.authority.phase_runspec_digest == fixture.runspec.digest
    assert invalidation.authority.action_id == fixture.action.action_id
    assert invalidation.visible_final_absent is True
    assert invalidation.science_started is False
    assert invalidation.original_replica_inode > 0
    assert invalidation.retired_population_basename in cleanup.removed_basename_samples
    assert cleanup.removed_count == 1
    assert cleanup.omitted_count == 0
    assert cleanup.failure is None


def test_fresh_population_cleans_only_exact_same_identity_stale_populations(
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
        os,
        "fstatvfs",
        lambda _descriptor: SimpleNamespace(f_bavail=2, f_frsize=1),
    )
    identity = fixture.runspec.payload.database.source_manifest_sha256
    replicas = cache_root / "replicas"
    replicas.mkdir(mode=0o700)
    stale_names = (
        f".population-{identity}-{'a' * 32}",
        f".population-{identity}-{'b' * 32}",
    )
    for name in stale_names:
        stale = replicas / name
        stale.mkdir(mode=0o700)
        (stale / "partial").write_bytes(b"owned")
    sibling_names = (
        f".population-{'f' * 64}-{'c' * 32}",
        f".population-{identity}-not-a-uuid",
        ".population-unrelated",
    )
    for name in sibling_names:
        sibling = replicas / name
        sibling.mkdir(mode=0o700)
        (sibling / "keep").write_bytes(name.encode())
    before = {
        name: ((replicas / name).stat().st_ino, (replicas / name / "keep").read_bytes()) for name in sibling_names
    }
    result_path = tmp_path / "fresh-action" / "database-placement-result.json"

    result = fixture.place_database(
        fixture.runspec,
        action_id=fixture.action.action_id,
        source_manifest_path=manifest_path,
        result_path=result_path,
        failure_path=tmp_path / "fresh-action" / "database-placement-failure.json",
    )

    assert isinstance(result, DatabaseReplicaColdResult)
    assert all(not (replicas / name).exists() for name in stale_names)
    assert {
        name: ((replicas / name).stat().st_ino, (replicas / name / "keep").read_bytes()) for name in sibling_names
    } == before
    assert not (tmp_path / "fresh-action" / "database-replica-invalidation.json").exists()
    cleanup = load_database_replica_cleanup(
        tmp_path / "fresh-action" / "database-replica-stale-population-cleanup.json"
    )
    assert cleanup.removed_count == 2
    assert cleanup.removed_basename_samples == tuple(sorted(stale_names))
    assert cleanup.omitted_count == 0


def test_valid_warm_placement_does_not_clean_an_exact_same_identity_stale_population(
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
        os,
        "fstatvfs",
        lambda _descriptor: SimpleNamespace(f_bavail=2, f_frsize=1),
    )
    cold = fixture.place_database(
        fixture.runspec,
        action_id=fixture.action.action_id,
        source_manifest_path=manifest_path,
        result_path=tmp_path / "cold-action" / "database-placement-result.json",
        failure_path=tmp_path / "cold-action" / "database-placement-failure.json",
    )
    assert isinstance(cold, DatabaseReplicaColdResult)
    identity = fixture.runspec.payload.database.source_manifest_sha256
    stale = cache_root / "replicas" / f".population-{identity}-{'d' * 32}"
    stale.mkdir(mode=0o700)
    marker = stale / "keep"
    marker.write_bytes(b"not-owned-by-the-warm-reader")

    warm = fixture.place_database(
        fixture.runspec,
        action_id=fixture.action.action_id,
        source_manifest_path=manifest_path,
        result_path=tmp_path / "warm-action" / "database-placement-result.json",
        failure_path=tmp_path / "warm-action" / "database-placement-failure.json",
    )

    assert isinstance(warm, DatabaseReplicaWarmResult)
    assert marker.read_bytes() == b"not-owned-by-the-warm-reader"
    assert not (tmp_path / "warm-action" / "database-replica-stale-population-cleanup.json").exists()


def test_cleanup_evidence_is_bounded_and_hashes_the_full_sorted_same_identity_inventory(
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
        os,
        "fstatvfs",
        lambda _descriptor: SimpleNamespace(f_bavail=2, f_frsize=1),
    )
    identity = fixture.runspec.payload.database.source_manifest_sha256
    replicas = cache_root / "replicas"
    replicas.mkdir(mode=0o700)
    stale_names = tuple(f".population-{identity}-{ordinal:032x}" for ordinal in range(35))
    for name in stale_names:
        (replicas / name).mkdir(mode=0o700)
    foreign_names = (
        f".population-{'f' * 64}-{'a' * 32}",
        f".population-{identity}-{'A' * 32}",
        f".population-{identity}-{'b' * 31}",
    )
    for name in foreign_names:
        directory = replicas / name
        directory.mkdir(mode=0o700)
        (directory / "keep").write_bytes(name.encode())
    foreign_before = {
        name: ((replicas / name).stat().st_ino, (replicas / name / "keep").read_bytes()) for name in foreign_names
    }

    result = fixture.place_database(
        fixture.runspec,
        action_id=fixture.action.action_id,
        source_manifest_path=manifest_path,
        result_path=tmp_path / "fresh-action" / "database-placement-result.json",
        failure_path=tmp_path / "fresh-action" / "database-placement-failure.json",
    )

    assert isinstance(result, DatabaseReplicaColdResult)
    cleanup = load_database_replica_cleanup(
        tmp_path / "fresh-action" / "database-replica-stale-population-cleanup.json"
    )
    sorted_names = tuple(sorted(stale_names))
    expected_digest = hashlib.sha256("".join(f"{name}\n" for name in sorted_names).encode()).hexdigest()
    assert cleanup.removed_count == 35
    assert cleanup.removed_basename_samples == sorted_names[:32]
    assert cleanup.omitted_count == 3
    assert cleanup.removed_basenames_sha256 == expected_digest
    assert all(not (replicas / name).exists() for name in stale_names)
    assert {
        name: ((replicas / name).stat().st_ino, (replicas / name / "keep").read_bytes()) for name in foreign_names
    } == foreign_before


@pytest.mark.parametrize("tamper", ["truncated-samples", "complete-digest-mismatch", "empty-success"])
def test_cleanup_evidence_rejects_noncanonical_inventory_claims(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    fixture, manifest_path, cache_root = _stage_required_fixture(
        tmp_path / "work",
        cold_cache_root,
        monkeypatch,
    )
    monkeypatch.setattr(
        os,
        "fstatvfs",
        lambda _descriptor: SimpleNamespace(f_bavail=2, f_frsize=1),
    )
    identity = fixture.runspec.payload.database.source_manifest_sha256
    replicas = cache_root / "replicas"
    replicas.mkdir(mode=0o700)
    stale_count = 35 if tamper == "truncated-samples" else 2
    stale_names = tuple(f".population-{identity}-{ordinal:032x}" for ordinal in range(stale_count))
    for name in stale_names:
        (replicas / name).mkdir(mode=0o700)
    action_dir = tmp_path / "fresh-action"

    result = fixture.place_database(
        fixture.runspec,
        action_id=fixture.action.action_id,
        source_manifest_path=manifest_path,
        result_path=action_dir / "database-placement-result.json",
        failure_path=action_dir / "database-placement-failure.json",
    )

    assert isinstance(result, DatabaseReplicaColdResult)
    sidecar = action_dir / "database-replica-stale-population-cleanup.json"
    payload = json.loads(sidecar.read_text())
    body = payload["database_replica_stale_population_cleanup"]
    if tamper == "truncated-samples":
        body["removed_basename_samples"] = body["removed_basename_samples"][:1]
        body["omitted_count"] = 34
    elif tamper == "complete-digest-mismatch":
        body["removed_basenames_sha256"] = "0" * 64
    else:
        body["removed_count"] = 0
        body["removed_basenames_sha256"] = hashlib.sha256(b"").hexdigest()
        body["removed_basename_samples"] = []
        body["omitted_count"] = 0
        body["failure"] = None
    sidecar.chmod(0o644)
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    sidecar.chmod(0o444)

    with pytest.raises(DatabasePlacementError, match="cleanup evidence"):
        load_database_replica_cleanup(sidecar)


def test_retirement_rebind_failure_cannot_delete_a_swapped_same_identity_name_or_publish_success(
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
        os,
        "fstatvfs",
        lambda _descriptor: SimpleNamespace(f_bavail=2, f_frsize=1),
    )
    first = fixture.place_database(
        fixture.runspec,
        action_id=fixture.action.action_id,
        source_manifest_path=manifest_path,
        result_path=tmp_path / "first-action" / "database-placement-result.json",
        failure_path=tmp_path / "first-action" / "database-placement-failure.json",
    )
    assert isinstance(first, DatabaseReplicaColdResult)
    identity = fixture.runspec.payload.database.source_manifest_sha256
    candidate = cache_root / "replicas" / identity
    candidate.chmod(0o755)
    candidate_manifest = candidate / "replica-manifest.json"
    candidate_manifest.chmod(0o644)
    candidate_manifest.write_bytes(b"{}\n")
    candidate_manifest.chmod(0o444)
    candidate.chmod(0o555)
    swapped: list[tuple[str, str]] = []
    real_stat = os.stat

    def swap_before_retired_rebind(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        retired_name = os.fsdecode(path)
        if not swapped and dir_fd is not None and retired_name.startswith(f".population-{identity}-"):
            displaced_name = f"{retired_name}.displaced"
            os.rename(
                retired_name,
                displaced_name,
                src_dir_fd=dir_fd,
                dst_dir_fd=dir_fd,
            )
            os.mkdir(retired_name, 0o700, dir_fd=dir_fd)
            replacement_directory = os.open(
                retired_name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=dir_fd,
            )
            try:
                replacement = os.open(
                    "uncontrolled-marker",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=replacement_directory,
                )
                try:
                    os.write(replacement, b"foreign")
                finally:
                    os.close(replacement)
            finally:
                os.close(replacement_directory)
            swapped.append((retired_name, displaced_name))
        return real_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(os, "stat", swap_before_retired_rebind)
    result_path = tmp_path / "repair-action" / "database-placement-result.json"
    failure_path = tmp_path / "repair-action" / "database-placement-failure.json"

    with pytest.raises(DatabasePlacementError, match="Database Replica"):
        fixture.place_database(
            fixture.runspec,
            action_id=fixture.action.action_id,
            source_manifest_path=manifest_path,
            result_path=result_path,
            failure_path=failure_path,
        )

    assert len(swapped) == 1
    retired_name, displaced_name = swapped[0]
    assert not result_path.exists()
    assert not candidate.exists()
    assert (candidate.parent / retired_name / "uncontrolled-marker").read_bytes() == b"foreign"
    assert (candidate.parent / displaced_name).is_dir()
    assert load_database_replica_cold_failure(failure_path).classification == "replica-validation-failed"
    assert not (tmp_path / "repair-action" / "database-replica-invalidation.json").exists()


def test_post_invalidation_substitution_preserves_retired_and_substitute_populations(
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
        os,
        "fstatvfs",
        lambda _descriptor: SimpleNamespace(f_bavail=2, f_frsize=1),
    )
    first = fixture.place_database(
        fixture.runspec,
        action_id=fixture.action.action_id,
        source_manifest_path=manifest_path,
        result_path=tmp_path / "first-action" / "database-placement-result.json",
        failure_path=tmp_path / "first-action" / "database-placement-failure.json",
    )
    assert isinstance(first, DatabaseReplicaColdResult)
    identity = fixture.runspec.payload.database.source_manifest_sha256
    candidate = cache_root / "replicas" / identity
    candidate.chmod(0o755)
    candidate_manifest = candidate / "replica-manifest.json"
    candidate_manifest.chmod(0o644)
    candidate_manifest.write_bytes(b"{}\n")
    candidate_manifest.chmod(0o444)
    candidate.chmod(0o555)
    swapped: list[tuple[Path, Path, int, int]] = []
    real_link = os.link

    def link_then_swap(
        source: str | bytes,
        destination: str | bytes,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        real_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )
        if os.fsdecode(destination) == "database-replica-invalidation.json":
            evidence = load_database_replica_invalidation(
                tmp_path / "repair-action" / "database-replica-invalidation.json"
            )
            retired = candidate.parent / evidence.retired_population_basename
            displaced = retired.with_name(f"{retired.name}.displaced")
            retired.rename(displaced)
            retired.mkdir(mode=0o700)
            (retired / "uncontrolled-marker").write_bytes(b"foreign")
            swapped.append((retired, displaced, retired.stat().st_ino, displaced.stat().st_ino))

    monkeypatch.setattr(os, "link", link_then_swap)
    result_path = tmp_path / "repair-action" / "database-placement-result.json"
    failure_path = tmp_path / "repair-action" / "database-placement-failure.json"

    with pytest.raises(DatabasePlacementError, match="cleanup failed"):
        fixture.place_database(
            fixture.runspec,
            action_id=fixture.action.action_id,
            source_manifest_path=manifest_path,
            result_path=result_path,
            failure_path=failure_path,
        )

    assert len(swapped) == 1
    substitute, displaced, substitute_inode, displaced_inode = swapped[0]
    assert not result_path.exists()
    assert not candidate.exists()
    assert substitute.stat().st_ino == substitute_inode
    assert (substitute / "uncontrolled-marker").read_bytes() == b"foreign"
    assert displaced.stat().st_ino == displaced_inode
    cleanup = load_database_replica_cleanup(
        tmp_path / "repair-action" / "database-replica-stale-population-cleanup.json"
    )
    assert cleanup.removed_count == 0
    assert cleanup.failure is not None
    assert load_database_replica_cold_failure(failure_path).classification == "replica-validation-failed"


def test_pre_unlink_child_substitution_preserves_original_and_substitute(
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
        os,
        "fstatvfs",
        lambda _descriptor: SimpleNamespace(f_bavail=2, f_frsize=1),
    )
    identity = fixture.runspec.payload.database.source_manifest_sha256
    replicas = cache_root / "replicas"
    replicas.mkdir(mode=0o700)
    stale_name = f".population-{identity}-{'6' * 32}"
    stale = replicas / stale_name
    stale.mkdir(mode=0o700)
    child = stale / "owned-child"
    displaced = stale / "owned-child.displaced"
    child.write_bytes(b"original")
    original_inode = child.stat().st_ino
    real_stat = os.stat
    swapped = False
    substitute_inode: int | None = None

    def swap_after_child_observation(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        nonlocal substitute_inode, swapped
        info = real_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)
        if not swapped and dir_fd is not None and os.fsdecode(path) == child.name:
            claimed_roots = tuple(replicas.glob(f".population-{identity}-*"))
            assert len(claimed_roots) == 1
            claimed_child = claimed_roots[0] / child.name
            claimed_displaced = claimed_roots[0] / displaced.name
            claimed_child.rename(claimed_displaced)
            claimed_child.write_bytes(b"substitute")
            substitute_inode = claimed_child.stat().st_ino
            swapped = True
        return info

    monkeypatch.setattr(os, "stat", swap_after_child_observation)
    result_path = tmp_path / "fresh-action" / "database-placement-result.json"
    failure_path = tmp_path / "fresh-action" / "database-placement-failure.json"

    with pytest.raises(DatabasePlacementError, match="cleanup failed"):
        fixture.place_database(
            fixture.runspec,
            action_id=fixture.action.action_id,
            source_manifest_path=manifest_path,
            result_path=result_path,
            failure_path=failure_path,
        )

    assert swapped
    assert not result_path.exists()
    assert not (replicas / identity).exists()
    assert substitute_inode is not None
    claimed_roots = tuple(replicas.glob(f".population-{identity}-*"))
    assert len(claimed_roots) == 1
    assert (claimed_roots[0] / child.name).stat().st_ino == substitute_inode
    assert (claimed_roots[0] / child.name).read_bytes() == b"substitute"
    assert (claimed_roots[0] / displaced.name).stat().st_ino == original_inode
    assert (claimed_roots[0] / displaced.name).read_bytes() == b"original"
    cleanup = load_database_replica_cleanup(
        tmp_path / "fresh-action" / "database-replica-stale-population-cleanup.json"
    )
    assert cleanup.removed_count == 0
    assert cleanup.failure is not None
    assert load_database_replica_cold_failure(failure_path).classification == "replica-validation-failed"


def test_pre_rmdir_root_substitution_preserves_original_and_substitute(
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
        os,
        "fstatvfs",
        lambda _descriptor: SimpleNamespace(f_bavail=2, f_frsize=1),
    )
    identity = fixture.runspec.payload.database.source_manifest_sha256
    replicas = cache_root / "replicas"
    replicas.mkdir(mode=0o700)
    stale_name = f".population-{identity}-{'7' * 32}"
    stale = replicas / stale_name
    displaced = replicas / f"{stale_name}.displaced"
    stale.mkdir(mode=0o700)
    original_inode = stale.stat().st_ino
    real_fstat = os.fstat
    swapped = False
    substitute_inode: int | None = None

    def swap_after_opened_root_observation(descriptor: int) -> os.stat_result:
        nonlocal substitute_inode, swapped
        info = real_fstat(descriptor)
        if not swapped and info.st_ino == original_inode:
            stale.rename(displaced)
            stale.mkdir(mode=0o700)
            substitute_inode = stale.stat().st_ino
            swapped = True
        return info

    monkeypatch.setattr(os, "fstat", swap_after_opened_root_observation)
    result_path = tmp_path / "fresh-action" / "database-placement-result.json"
    failure_path = tmp_path / "fresh-action" / "database-placement-failure.json"

    with pytest.raises(DatabasePlacementError, match="cleanup failed"):
        fixture.place_database(
            fixture.runspec,
            action_id=fixture.action.action_id,
            source_manifest_path=manifest_path,
            result_path=result_path,
            failure_path=failure_path,
        )

    assert swapped
    assert not result_path.exists()
    assert not (replicas / identity).exists()
    assert substitute_inode is not None
    assert stale.stat().st_ino == substitute_inode
    assert displaced.stat().st_ino == original_inode
    cleanup = load_database_replica_cleanup(
        tmp_path / "fresh-action" / "database-replica-stale-population-cleanup.json"
    )
    assert cleanup.removed_count == 0
    assert cleanup.failure is not None
    assert load_database_replica_cold_failure(failure_path).classification == "replica-validation-failed"


def test_atomic_child_claim_preserves_a_substitute_injected_at_the_source_rename_boundary(
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
        os,
        "fstatvfs",
        lambda _descriptor: SimpleNamespace(f_bavail=2, f_frsize=1),
    )
    identity = fixture.runspec.payload.database.source_manifest_sha256
    replicas = cache_root / "replicas"
    replicas.mkdir(mode=0o700)
    stale_name = f".population-{identity}-{'9' * 32}"
    stale = replicas / stale_name
    stale.mkdir(mode=0o700)
    child_name = "owned-child"
    displaced_name = "owned-child.displaced"
    child = stale / child_name
    child.write_bytes(b"original")
    original_inode = child.stat().st_ino
    real_unlink = os.unlink
    real_renameat2 = publication_runtime.require_renameat2()
    swapped = False
    substitute_inode: int | None = None

    def substitute(parent_descriptor: int) -> None:
        nonlocal substitute_inode, swapped
        os.rename(
            child_name,
            displaced_name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        descriptor = os.open(
            child_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_descriptor,
        )
        try:
            os.write(descriptor, b"substitute")
            substitute_inode = os.fstat(descriptor).st_ino
        finally:
            os.close(descriptor)
        swapped = True

    def substitute_at_legacy_unlink(path: str | bytes, *, dir_fd: int | None = None) -> None:
        if not swapped and os.fsdecode(path) == child_name and dir_fd is not None:
            substitute(dir_fd)
        real_unlink(path, dir_fd=dir_fd)

    def substitute_at_claim_rename(
        source_parent: int,
        source_name: bytes,
        destination_parent: int,
        destination_name: bytes,
        flags: int,
    ) -> int:
        if not swapped and os.fsdecode(source_name) == child_name:
            substitute(source_parent)
        return real_renameat2(
            source_parent,
            source_name,
            destination_parent,
            destination_name,
            flags,
        )

    monkeypatch.setattr(os, "unlink", substitute_at_legacy_unlink)
    _replace_libc_renameat2(monkeypatch, substitute_at_claim_rename)
    action_dir = tmp_path / "fresh-action"
    result_path = action_dir / "database-placement-result.json"
    failure_path = action_dir / "database-placement-failure.json"

    with pytest.raises(DatabasePlacementError, match="cleanup failed"):
        fixture.place_database(
            fixture.runspec,
            action_id=fixture.action.action_id,
            source_manifest_path=manifest_path,
            result_path=result_path,
            failure_path=failure_path,
        )

    assert swapped
    assert substitute_inode is not None
    assert not result_path.exists()
    assert not (replicas / identity).exists()
    preserved_roots = tuple(replicas.glob(f".population-{identity}-*"))
    assert len(preserved_roots) == 1
    preserved_root = preserved_roots[0]
    substitute_path = preserved_root / child_name
    displaced_path = preserved_root / displaced_name
    assert substitute_path.stat().st_ino == substitute_inode
    assert substitute_path.read_bytes() == b"substitute"
    assert displaced_path.stat().st_ino == original_inode
    assert displaced_path.read_bytes() == b"original"
    cleanup = load_database_replica_cleanup(action_dir / "database-replica-stale-population-cleanup.json")
    assert cleanup.removed_count == 0
    assert cleanup.removed_basename_samples == ()
    assert cleanup.failure is not None
    assert load_database_replica_cold_failure(failure_path).classification == "replica-validation-failed"


def test_atomic_root_claim_preserves_a_substitute_injected_at_the_source_rename_boundary(
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
        os,
        "fstatvfs",
        lambda _descriptor: SimpleNamespace(f_bavail=2, f_frsize=1),
    )
    identity = fixture.runspec.payload.database.source_manifest_sha256
    replicas = cache_root / "replicas"
    replicas.mkdir(mode=0o700)
    stale_name = f".population-{identity}-{'a' * 32}"
    displaced_name = f"{stale_name}.displaced"
    stale = replicas / stale_name
    displaced = replicas / displaced_name
    stale.mkdir(mode=0o700)
    (stale / "original-marker").write_bytes(b"original")
    os.setxattr(stale, "user.bspp-marker", b"original")
    original_inode = stale.stat().st_ino
    real_rmdir = os.rmdir
    real_renameat2 = publication_runtime.require_renameat2()
    swapped = False
    substitute_inode: int | None = None

    def substitute(parent_descriptor: int) -> None:
        nonlocal substitute_inode, swapped
        os.rename(
            stale_name,
            displaced_name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        os.mkdir(stale_name, 0o700, dir_fd=parent_descriptor)
        descriptor = os.open(
            stale_name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
        try:
            os.setxattr(descriptor, "user.bspp-marker", b"substitute")
            substitute_inode = os.fstat(descriptor).st_ino
        finally:
            os.close(descriptor)
        swapped = True

    def substitute_at_legacy_rmdir(path: str | bytes, *, dir_fd: int | None = None) -> None:
        if not swapped and os.fsdecode(path) == stale_name and dir_fd is not None:
            substitute(dir_fd)
        real_rmdir(path, dir_fd=dir_fd)

    def substitute_at_claim_rename(
        source_parent: int,
        source_name: bytes,
        destination_parent: int,
        destination_name: bytes,
        flags: int,
    ) -> int:
        if not swapped and os.fsdecode(source_name) == stale_name:
            substitute(source_parent)
        return real_renameat2(
            source_parent,
            source_name,
            destination_parent,
            destination_name,
            flags,
        )

    monkeypatch.setattr(os, "rmdir", substitute_at_legacy_rmdir)
    _replace_libc_renameat2(monkeypatch, substitute_at_claim_rename)
    action_dir = tmp_path / "fresh-action"
    result_path = action_dir / "database-placement-result.json"
    failure_path = action_dir / "database-placement-failure.json"

    with pytest.raises(DatabasePlacementError, match="cleanup failed"):
        fixture.place_database(
            fixture.runspec,
            action_id=fixture.action.action_id,
            source_manifest_path=manifest_path,
            result_path=result_path,
            failure_path=failure_path,
        )

    assert swapped
    assert substitute_inode is not None
    assert not result_path.exists()
    assert not (replicas / identity).exists()
    assert stale.stat().st_ino == substitute_inode
    assert os.getxattr(stale, "user.bspp-marker") == b"substitute"
    assert displaced.stat().st_ino == original_inode
    assert os.getxattr(displaced, "user.bspp-marker") == b"original"
    assert (displaced / "original-marker").read_bytes() == b"original"
    cleanup = load_database_replica_cleanup(action_dir / "database-replica-stale-population-cleanup.json")
    assert cleanup.removed_count == 0
    assert cleanup.removed_basename_samples == ()
    assert cleanup.failure is not None
    assert load_database_replica_cold_failure(failure_path).classification == "replica-validation-failed"


def test_partial_cleanup_publishes_bounded_failure_and_never_exposes_a_final_or_touches_siblings(
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
        os,
        "fstatvfs",
        lambda _descriptor: SimpleNamespace(f_bavail=2, f_frsize=1),
    )
    identity = fixture.runspec.payload.database.source_manifest_sha256
    replicas = cache_root / "replicas"
    replicas.mkdir(mode=0o700)
    stale_names = (
        f".population-{identity}-{'1' * 32}",
        f".population-{identity}-{'2' * 32}",
    )
    for name in stale_names:
        (replicas / name).mkdir(mode=0o700)
    sibling = replicas / f".population-{'f' * 64}-{'3' * 32}"
    sibling.mkdir(mode=0o700)
    sibling_marker = sibling / "keep"
    sibling_marker.write_bytes(b"foreign")
    original_remove = repair_runtime._remove_owned_tree
    calls = 0

    def interrupt_second(
        parent_descriptor: int,
        name: str,
        *,
        cache_device: int,
        expected_binding: ExpectedDirectoryBinding | None = None,
        root_claim_prefix: str | None = None,
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise DatabasePlacementError("injected cleanup interruption")
        original_remove(
            parent_descriptor,
            name,
            cache_device=cache_device,
            expected_binding=expected_binding,
            root_claim_prefix=root_claim_prefix,
        )

    services = _production_staged_database_placement_services()

    def cleanup_with_interruption(
        runspec: PhaseRunSpec,
        *,
        action_id: str,
        cache_descriptor: int,
        ownership: CacheExclusiveOwnership,
        result_path: Path,
    ) -> None:
        repair_runtime._cleanup_stale_populations(
            runspec,
            action_id=action_id,
            cache_descriptor=cache_descriptor,
            ownership=ownership,
            result_path=result_path,
            owned_tree_remover=OwnedTreeRemover(remove=interrupt_second),
        )

    repair = replace(services.cold.repair, cleanup_stale=cleanup_with_interruption)
    services = replace(services, cold=replace(services.cold, repair=repair))
    result_path = tmp_path / "fresh-action" / "database-placement-result.json"
    failure_path = tmp_path / "fresh-action" / "database-placement-failure.json"

    with pytest.raises(DatabasePlacementError, match="cleanup failed"):
        fixture.place_database(
            fixture.runspec,
            action_id=fixture.action.action_id,
            source_manifest_path=manifest_path,
            result_path=result_path,
            failure_path=failure_path,
            staged_services=services,
        )

    assert not result_path.exists()
    assert not (replicas / identity).exists()
    assert not (replicas / stale_names[0]).exists()
    assert (replicas / stale_names[1]).is_dir()
    assert sibling_marker.read_bytes() == b"foreign"
    cleanup = load_database_replica_cleanup(
        tmp_path / "fresh-action" / "database-replica-stale-population-cleanup.json"
    )
    assert cleanup.removed_count == 1
    assert cleanup.removed_basename_samples == (stale_names[0],)
    assert cleanup.failure == "injected cleanup interruption"
    assert load_database_replica_cold_failure(failure_path).classification == "replica-validation-failed"


def test_final_replicas_fsync_failure_publishes_bounded_private_and_public_failure_evidence(
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
        os,
        "fstatvfs",
        lambda _descriptor: SimpleNamespace(f_bavail=2, f_frsize=1),
    )
    identity = fixture.runspec.payload.database.source_manifest_sha256
    replicas = cache_root / "replicas"
    replicas.mkdir(mode=0o700)
    stale_name = f".population-{identity}-{'8' * 32}"
    (replicas / stale_name).mkdir(mode=0o700)
    real_fsync = os.fsync
    replicas_fsyncs = 0

    def fail_final_replicas_fsync(descriptor: int) -> None:
        nonlocal replicas_fsyncs
        try:
            target = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
        except OSError:
            target = None
        if target == replicas:
            replicas_fsyncs += 1
            if replicas_fsyncs == 3:
                raise OSError("injected final replicas fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_final_replicas_fsync)
    action_dir = tmp_path / "fresh-action"
    result_path = action_dir / "database-placement-result.json"
    failure_path = action_dir / "database-placement-failure.json"

    with pytest.raises(DatabasePlacementError, match="cleanup failed"):
        fixture.place_database(
            fixture.runspec,
            action_id=fixture.action.action_id,
            source_manifest_path=manifest_path,
            result_path=result_path,
            failure_path=failure_path,
        )

    assert replicas_fsyncs == 3
    assert not result_path.exists()
    assert not (replicas / identity).exists()
    assert not (replicas / stale_name).exists()
    cleanup = load_database_replica_cleanup(action_dir / "database-replica-stale-population-cleanup.json")
    assert cleanup.removed_count == 1
    assert cleanup.removed_basename_samples == (stale_name,)
    assert cleanup.failure == "cannot durably confirm same-identity stale Database Replica cleanup"
    assert load_database_replica_cold_failure(failure_path).classification == "replica-validation-failed"


def test_retirement_post_rename_fsync_failure_preserves_retired_state_and_publishes_failure(
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
        os,
        "fstatvfs",
        lambda _descriptor: SimpleNamespace(f_bavail=2, f_frsize=1),
    )
    first = fixture.place_database(
        fixture.runspec,
        action_id=fixture.action.action_id,
        source_manifest_path=manifest_path,
        result_path=tmp_path / "first-action" / "database-placement-result.json",
        failure_path=tmp_path / "first-action" / "database-placement-failure.json",
    )
    assert isinstance(first, DatabaseReplicaColdResult)
    identity = fixture.runspec.payload.database.source_manifest_sha256
    replicas = cache_root / "replicas"
    candidate = replicas / identity
    candidate.chmod(0o755)
    candidate_manifest = candidate / "replica-manifest.json"
    candidate_manifest.chmod(0o644)
    candidate_manifest.write_bytes(b"{}\n")
    candidate_manifest.chmod(0o444)
    candidate.chmod(0o555)
    original_inode = candidate.stat().st_ino
    real_renameat2 = publication_runtime.require_renameat2()
    real_fsync = os.fsync
    retired_name: str | None = None
    injected = False

    def record_retirement_rename(
        source_parent: int,
        source_name: bytes,
        destination_parent: int,
        destination_name: bytes,
        flags: int,
    ) -> int:
        nonlocal retired_name
        result = real_renameat2(
            source_parent,
            source_name,
            destination_parent,
            destination_name,
            flags,
        )
        if result == 0 and os.fsdecode(source_name) == identity:
            retired_name = os.fsdecode(destination_name)
        return result

    def fail_retirement_parent_fsync(descriptor: int) -> None:
        nonlocal injected
        target = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
        if not injected and retired_name is not None and target == replicas:
            injected = True
            raise OSError("injected retirement post-rename parent fsync failure")
        real_fsync(descriptor)

    _replace_libc_renameat2(monkeypatch, record_retirement_rename)
    monkeypatch.setattr(os, "fsync", fail_retirement_parent_fsync)
    action_dir = tmp_path / "repair-action"
    result_path = action_dir / "database-placement-result.json"
    failure_path = action_dir / "database-placement-failure.json"

    with pytest.raises(DatabasePlacementError, match="durably confirm atomic Database Replica publication"):
        fixture.place_database(
            fixture.runspec,
            action_id=fixture.action.action_id,
            source_manifest_path=manifest_path,
            result_path=result_path,
            failure_path=failure_path,
        )

    assert injected
    assert retired_name is not None
    retired = replicas / retired_name
    assert not result_path.exists()
    assert not candidate.exists()
    assert retired.stat().st_ino == original_inode
    assert (retired / "replica-manifest.json").read_bytes() == b"{}\n"
    assert not (action_dir / "database-replica-invalidation.json").exists()
    assert not (action_dir / "database-replica-stale-population-cleanup.json").exists()
    assert load_database_replica_cold_failure(failure_path).classification == "publication-failed"


def test_fresh_publication_post_rename_fsync_failure_preserves_valid_final_and_publishes_failure(
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
        os,
        "fstatvfs",
        lambda _descriptor: SimpleNamespace(f_bavail=2, f_frsize=1),
    )
    identity = fixture.runspec.payload.database.source_manifest_sha256
    replicas = cache_root / "replicas"
    real_renameat2 = publication_runtime.require_renameat2()
    real_fsync = os.fsync
    final_inode: int | None = None
    renamed = False
    injected = False

    def record_fresh_rename(
        source_parent: int,
        source_name: bytes,
        destination_parent: int,
        destination_name: bytes,
        flags: int,
    ) -> int:
        nonlocal final_inode, renamed
        result = real_renameat2(
            source_parent,
            source_name,
            destination_parent,
            destination_name,
            flags,
        )
        if result == 0 and os.fsdecode(destination_name) == identity:
            final_inode = os.stat(identity, dir_fd=destination_parent, follow_symlinks=False).st_ino
            renamed = True
        return result

    def fail_fresh_parent_fsync(descriptor: int) -> None:
        nonlocal injected
        target = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
        if not injected and renamed and target == replicas:
            injected = True
            raise OSError("injected fresh post-rename parent fsync failure")
        real_fsync(descriptor)

    _replace_libc_renameat2(monkeypatch, record_fresh_rename)
    monkeypatch.setattr(os, "fsync", fail_fresh_parent_fsync)
    action_dir = tmp_path / "fresh-action"
    result_path = action_dir / "database-placement-result.json"
    failure_path = action_dir / "database-placement-failure.json"

    with pytest.raises(DatabasePlacementError, match="durably confirm atomic Database Replica publication"):
        fixture.place_database(
            fixture.runspec,
            action_id=fixture.action.action_id,
            source_manifest_path=manifest_path,
            result_path=result_path,
            failure_path=failure_path,
        )

    assert injected
    assert final_inode is not None
    final = replicas / identity
    assert not result_path.exists()
    assert final.stat().st_ino == final_inode
    assert (final / "replica-manifest.json").is_file()
    assert not tuple(replicas.glob(f".population-{identity}-*"))
    assert load_database_replica_cold_failure(failure_path).classification == "publication-failed"


def test_foreign_invalidation_sidecar_stops_after_retirement_before_cleanup_or_repopulation(
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
        os,
        "fstatvfs",
        lambda _descriptor: SimpleNamespace(f_bavail=2, f_frsize=1),
    )
    first = fixture.place_database(
        fixture.runspec,
        action_id=fixture.action.action_id,
        source_manifest_path=manifest_path,
        result_path=tmp_path / "first-action" / "database-placement-result.json",
        failure_path=tmp_path / "first-action" / "database-placement-failure.json",
    )
    assert isinstance(first, DatabaseReplicaColdResult)
    identity = fixture.runspec.payload.database.source_manifest_sha256
    candidate = cache_root / "replicas" / identity
    candidate.chmod(0o755)
    candidate_manifest = candidate / "replica-manifest.json"
    candidate_manifest.chmod(0o644)
    candidate_manifest.write_bytes(b"{}\n")
    candidate_manifest.chmod(0o444)
    candidate.chmod(0o555)
    action_dir = tmp_path / "repair-action"
    action_dir.mkdir()
    sidecar = action_dir / "database-replica-invalidation.json"
    sidecar.write_bytes(b"foreign\n")
    sidecar.chmod(0o444)
    result_path = action_dir / "database-placement-result.json"
    failure_path = action_dir / "database-placement-failure.json"

    with pytest.raises(DatabasePlacementError, match="invalidation evidence"):
        fixture.place_database(
            fixture.runspec,
            action_id=fixture.action.action_id,
            source_manifest_path=manifest_path,
            result_path=result_path,
            failure_path=failure_path,
        )

    assert not result_path.exists()
    assert not candidate.exists()
    retired = tuple(candidate.parent.glob(f".population-{identity}-*"))
    assert len(retired) == 1
    assert sidecar.read_bytes() == b"foreign\n"
    assert not (action_dir / "database-replica-stale-population-cleanup.json").exists()
    assert load_database_replica_cold_failure(failure_path).classification == "publication-failed"
