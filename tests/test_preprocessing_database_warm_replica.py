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

"""Warm Database Replica reuse through the public placement seam."""

from __future__ import annotations

import fcntl
import os
import stat
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from bspp.orchestration.contract.database_placement import DatabasePlacementOutcomeKind
from bspp.orchestration.contract.database_replica import (
    DatabaseReplicaColdResult,
    DatabaseReplicaWarmResult,
    DatabaseWarmCacheMountFacts,
)
from bspp.orchestration.runtime.preprocessing import _database_placement_evidence_io as evidence_io
from bspp.orchestration.runtime.preprocessing import _database_replica as replica_runtime
from bspp.orchestration.runtime.preprocessing import _database_replica_warm as warm_runtime
from bspp.orchestration.runtime.preprocessing._database_placement_errors import DatabasePlacementError
from bspp.orchestration.runtime.preprocessing._database_replica_evidence_io import (
    load_database_replica_cold_failure,
    load_database_replica_result,
)
from bspp.orchestration.runtime.preprocessing._database_replica_lock import LockWait
from bspp.orchestration.runtime.preprocessing._database_replica_repair_evidence import (
    load_database_replica_invalidation,
)
from bspp.orchestration.runtime.preprocessing._database_replica_services import ColdReplicaCoordinator
from bspp.orchestration.runtime.preprocessing.database_placement import (
    _production_staged_database_placement_services,
)
from tests.support.database_cold_replica import _stage_required_fixture
from tests.support.database_cold_replica import cold_cache_root as cold_cache_root


def _cache_snapshot(root: Path) -> dict[str, tuple[int, int, bytes | None]]:
    return {
        path.relative_to(root).as_posix(): (
            path.lstat().st_ino,
            stat.S_IMODE(path.lstat().st_mode),
            path.read_bytes() if path.is_file() else None,
        )
        for path in sorted(root.rglob("*"))
    }


def _cold_populate(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[object, Path, Path, object]:
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
    cold = fixture.place_database(
        fixture.runspec,
        action_id=fixture.action.action_id,
        source_manifest_path=manifest_path,
        result_path=tmp_path / "placement" / "cold-result.json",
        failure_path=tmp_path / "placement" / "cold-failure.json",
    )
    return fixture, manifest_path, cache_root, cold


def test_second_stage_required_placement_reuses_the_complete_replica_without_cache_mutation(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, manifest_path, cache_root, cold = _cold_populate(tmp_path, cold_cache_root, monkeypatch)
    before = _cache_snapshot(cache_root)

    warm = fixture.place_database(
        fixture.runspec,
        action_id=fixture.action.action_id,
        source_manifest_path=manifest_path,
        result_path=tmp_path / "placement" / "warm-result.json",
        failure_path=tmp_path / "placement" / "warm-failure.json",
    )

    assert isinstance(warm, DatabaseReplicaWarmResult)
    assert warm.outcome == DatabasePlacementOutcomeKind.REPLICA_WARM
    assert warm.replica_manifest_sha256 == cold.replica_manifest_sha256
    assert _cache_snapshot(cache_root) == before


def test_warm_result_roundtrip_uses_read_only_mount_facts_without_writable_proof(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, manifest_path, _, _ = _cold_populate(tmp_path, cold_cache_root, monkeypatch)
    result_path = tmp_path / "placement" / "warm-result.json"

    warm = fixture.place_database(
        fixture.runspec,
        action_id=fixture.action.action_id,
        source_manifest_path=manifest_path,
        result_path=result_path,
        failure_path=tmp_path / "placement" / "warm-failure.json",
    )

    assert isinstance(warm, DatabaseReplicaWarmResult)
    assert load_database_replica_result(result_path) == warm
    assert "writable_verified" not in warm.cache_mount.to_mapping()


def test_genuine_warm_miss_closes_probe_descriptors_and_shared_lock_before_unchanged_cold_delegate(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, manifest_path, cache_root = _stage_required_fixture(
        tmp_path / "work",
        cold_cache_root,
        monkeypatch,
    )
    locks = cache_root / ".locks"
    replicas = cache_root / "replicas"
    locks.mkdir(mode=0o700)
    replicas.mkdir(mode=0o700)
    identity = fixture.runspec.payload.database.source_manifest_sha256
    lock_path = locks / f"{identity}.lock"
    lock_path.touch(mode=0o600)
    sentinel = object()
    cold_calls: list[dict[str, object]] = []

    def cold_delegate(*_args: object, **_kwargs: object) -> object:
        cold_calls.append(_kwargs)
        cache_targets = {
            entry.resolve() for entry in Path("/proc/self/fd").iterdir() if entry.name.isdecimal() and entry.exists()
        }
        assert cache_root.resolve() not in cache_targets
        assert locks.resolve() not in cache_targets
        assert replicas.resolve() not in cache_targets
        competitor = os.open(lock_path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(competitor, fcntl.LOCK_UN)
        finally:
            os.close(competitor)
        return sentinel

    services = _production_staged_database_placement_services()
    services = replace(
        services,
        cold_coordinator=ColdReplicaCoordinator(place=cold_delegate),
    )

    assert (
        fixture.place_database(
            fixture.runspec,
            action_id=fixture.action.action_id,
            source_manifest_path=manifest_path,
            result_path=tmp_path / "placement" / "result.json",
            failure_path=tmp_path / "placement" / "failure.json",
            staged_services=services,
        )
        is sentinel
    )
    assert len(cold_calls) == 1
    assert cold_calls[0]["preprobe_contended"] is False
    assert isinstance(cold_calls[0]["wait"], LockWait)


def test_warm_miss_reuses_a_valid_candidate_published_before_coordinator_entry(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, manifest_path, cache_root, cold = _cold_populate(tmp_path, cold_cache_root, monkeypatch)
    assert isinstance(cold, DatabaseReplicaColdResult)
    identity = fixture.runspec.payload.database.source_manifest_sha256
    candidate = cache_root / "replicas" / identity
    held = candidate.with_name(f"{identity}.held")
    candidate.rename(held)
    published = False

    services = _production_staged_database_placement_services()
    production_probe = services.warm_selector.probe

    def publish_after_absent_observation(*args: object, **kwargs: object) -> object:
        nonlocal published
        disposition = production_probe(*args, **kwargs)
        held.rename(candidate)
        published = True
        return disposition

    selector = replace(services.warm_selector, probe=publish_after_absent_observation)
    services = replace(
        services,
        warm_selector=selector,
        cold=replace(services.cold, warm_selector=selector),
    )
    result_path = tmp_path / "placement" / "warm-race-result.json"
    failure_path = tmp_path / "placement" / "warm-race-failure.json"

    result = fixture.place_database(
        fixture.runspec,
        action_id=fixture.action.action_id,
        source_manifest_path=manifest_path,
        result_path=result_path,
        failure_path=failure_path,
        staged_services=services,
    )

    assert published
    assert isinstance(result, DatabaseReplicaWarmResult)
    assert result_path.exists()
    assert not failure_path.exists()
    assert candidate.is_dir()
    assert not held.exists()


def test_contended_warm_probe_publishes_no_evidence_before_the_exclusive_coordinator(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, manifest_path, cache_root, _ = _cold_populate(tmp_path, cold_cache_root, monkeypatch)
    identity = fixture.runspec.payload.database.source_manifest_sha256
    result_path = tmp_path / "placement" / "contended-result.json"
    failure_path = tmp_path / "placement" / "contended-failure.json"
    held = os.open(cache_root / ".locks" / f"{identity}.lock", os.O_RDWR | os.O_NOFOLLOW)
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sentinel = object()

    def coordinator(*_args: object, **kwargs: object) -> object:
        assert kwargs["preprobe_contended"] is True
        assert isinstance(kwargs["wait"], LockWait)
        assert not result_path.exists()
        assert not failure_path.exists()
        return sentinel

    services = _production_staged_database_placement_services()
    services = replace(
        services,
        cold_coordinator=ColdReplicaCoordinator(place=coordinator),
    )
    try:
        assert (
            fixture.place_database(
                fixture.runspec,
                action_id=fixture.action.action_id,
                source_manifest_path=manifest_path,
                result_path=result_path,
                failure_path=failure_path,
                staged_services=services,
            )
            is sentinel
        )
    finally:
        os.close(held)


@pytest.mark.parametrize(
    "invalid_state",
    ["manifest", "root", "member", "metadata", "mode", "hard-link", "symlink"],
)
def test_visible_invalid_warm_replica_is_repaired_by_the_cold_coordinator(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_state: str,
) -> None:
    fixture, manifest_path, cache_root, _ = _cold_populate(tmp_path, cold_cache_root, monkeypatch)
    identity = fixture.runspec.payload.database.source_manifest_sha256
    candidate = cache_root / "replicas" / identity
    manifest = candidate / "replica-manifest.json"
    members = sorted(path for path in candidate.iterdir() if path.name != manifest.name)
    if invalid_state == "manifest":
        candidate.chmod(0o755)
        manifest.chmod(0o644)
        manifest.write_bytes(b"{}\n")
        manifest.chmod(0o444)
        candidate.chmod(0o555)
    elif invalid_state == "root":
        candidate.chmod(0o755)
    elif invalid_state == "member":
        candidate.chmod(0o755)
        members[0].unlink()
        candidate.chmod(0o555)
    elif invalid_state == "metadata":
        os.utime(members[0], ns=(7, 7))
    elif invalid_state == "mode":
        members[0].chmod(0o644)
    elif invalid_state == "hard-link":
        candidate.chmod(0o755)
        members[1].unlink()
        os.link(members[0], members[1])
        candidate.chmod(0o555)
    else:
        candidate.chmod(0o755)
        members[0].unlink()
        members[0].symlink_to(members[1].name)
        candidate.chmod(0o555)
    original_inode = candidate.stat().st_ino
    result_path = tmp_path / "placement" / "warm-result.json"
    failure_path = tmp_path / "placement" / "warm-invalid.json"

    result = fixture.place_database(
        fixture.runspec,
        action_id=fixture.action.action_id,
        source_manifest_path=manifest_path,
        result_path=result_path,
        failure_path=failure_path,
    )

    assert isinstance(result, DatabaseReplicaColdResult)
    assert candidate.stat().st_ino != original_inode
    assert not failure_path.exists()
    invalidation = load_database_replica_invalidation(tmp_path / "placement" / "database-replica-invalidation.json")
    assert invalidation.original_replica_inode == original_inode


def test_warm_reuse_repairs_a_member_with_an_external_hard_link_without_unlinking_the_external_name(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, manifest_path, cache_root, _ = _cold_populate(tmp_path, cold_cache_root, monkeypatch)
    identity = fixture.runspec.payload.database.source_manifest_sha256
    candidate = cache_root / "replicas" / identity
    member = candidate / fixture.runspec.payload.database.source_manifest.members[0].logical_name
    external = cache_root / "external-member-link"
    os.link(member, external)
    external_inode = external.stat().st_ino
    result_path = tmp_path / "placement" / "warm-result.json"
    failure_path = tmp_path / "placement" / "warm-failure.json"

    result = fixture.place_database(
        fixture.runspec,
        action_id=fixture.action.action_id,
        source_manifest_path=manifest_path,
        result_path=result_path,
        failure_path=failure_path,
    )

    assert isinstance(result, DatabaseReplicaColdResult)
    assert external.stat().st_ino == external_inode
    assert not failure_path.exists()


def test_warm_repair_fails_closed_when_retired_member_authority_is_off_device(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, manifest_path, cache_root, _ = _cold_populate(tmp_path, cold_cache_root, monkeypatch)
    identity = fixture.runspec.payload.database.source_manifest_sha256
    candidate = cache_root / "replicas" / identity
    member = candidate / fixture.runspec.payload.database.source_manifest.members[0].logical_name
    real_stat = os.stat

    def off_device_stat(
        path: int | str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result | SimpleNamespace:
        info = real_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)
        if dir_fd is not None and path == member.name and not follow_symlinks:
            parent = Path(os.readlink(f"/proc/self/fd/{dir_fd}"))
            if parent.parent == candidate.parent:
                return SimpleNamespace(
                    st_mode=info.st_mode,
                    st_size=info.st_size,
                    st_mtime_ns=info.st_mtime_ns,
                    st_dev=info.st_dev + 1,
                    st_ino=info.st_ino,
                    st_nlink=info.st_nlink,
                    st_uid=info.st_uid,
                )
        return info

    monkeypatch.setattr(os, "stat", off_device_stat)
    result_path = tmp_path / "placement" / "warm-result.json"
    failure_path = tmp_path / "placement" / "warm-failure.json"

    with pytest.raises(DatabasePlacementError, match="Database Replica"):
        fixture.place_database(
            fixture.runspec,
            action_id=fixture.action.action_id,
            source_manifest_path=manifest_path,
            result_path=result_path,
            failure_path=failure_path,
        )

    assert not result_path.exists()
    failure = load_database_replica_cold_failure(failure_path)
    assert failure.classification == "replica-validation-failed"


def test_warm_failure_after_mount_observation_retains_facts_when_shared_lock_authority_fails(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, manifest_path, cache_root, _ = _cold_populate(tmp_path, cold_cache_root, monkeypatch)
    identity = fixture.runspec.payload.database.source_manifest_sha256
    lock_path = cache_root / ".locks" / f"{identity}.lock"
    displaced = lock_path.with_suffix(".lock.displaced")
    system_flock = fcntl.flock
    rebound = False

    def lock_then_rebind(descriptor: int, operation: int) -> None:
        nonlocal rebound
        system_flock(descriptor, operation)
        if not rebound and operation & fcntl.LOCK_SH:
            lock_path.rename(displaced)
            lock_path.touch(mode=0o600)
            rebound = True

    monkeypatch.setattr(fcntl, "flock", lock_then_rebind)
    failure_path = tmp_path / "placement" / "warm-failure.json"
    try:
        with pytest.raises(DatabasePlacementError, match=r"lock|authority|changed"):
            fixture.place_database(
                fixture.runspec,
                action_id=fixture.action.action_id,
                source_manifest_path=manifest_path,
                result_path=tmp_path / "placement" / "warm-result.json",
                failure_path=failure_path,
            )
    finally:
        if rebound:
            lock_path.unlink()
            displaced.rename(lock_path)

    failure = load_database_replica_cold_failure(failure_path)
    assert failure.classification == "lock-unavailable"
    assert isinstance(failure.cache_mount, DatabaseWarmCacheMountFacts)


def test_warm_failure_before_result_visibility_retains_mount_facts(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, manifest_path, _, _ = _cold_populate(tmp_path, cold_cache_root, monkeypatch)

    result_path = tmp_path / "placement" / "warm-result.json"
    failure_path = tmp_path / "placement" / "warm-failure.json"
    system_link = os.link

    def reject_result_publication(source: object, destination: object, **kwargs: object) -> None:
        if destination == result_path.name:
            raise OSError("injected warm Result publication failure")
        system_link(source, destination, **kwargs)

    monkeypatch.setattr(evidence_io.os, "link", reject_result_publication)
    with pytest.raises(DatabasePlacementError, match=r"publish|publication"):
        fixture.place_database(
            fixture.runspec,
            action_id=fixture.action.action_id,
            source_manifest_path=manifest_path,
            result_path=result_path,
            failure_path=failure_path,
        )

    assert not result_path.exists()
    failure = load_database_replica_cold_failure(failure_path)
    assert failure.classification == "result-publication-failed"
    assert isinstance(failure.cache_mount, DatabaseWarmCacheMountFacts)


def test_warm_reuse_rejects_exact_candidate_path_swap_after_full_validation_before_result_publication(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, manifest_path, cache_root, _ = _cold_populate(tmp_path, cold_cache_root, monkeypatch)
    identity = fixture.runspec.payload.database.source_manifest_sha256
    candidate = cache_root / "replicas" / identity
    displaced = candidate.with_name(f"{identity}.displaced")
    system_stat = os.stat
    candidate_observations = 0

    def stat_then_swap(
        path: int | str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        nonlocal candidate_observations
        if dir_fd is not None and path == identity and not follow_symlinks:
            parent = Path(os.readlink(f"/proc/self/fd/{dir_fd}"))
            if parent == candidate.parent:
                candidate_observations += 1
                if candidate_observations == 3:
                    candidate.rename(displaced)
                    candidate.mkdir(mode=0o555)
        return system_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(warm_runtime.os, "stat", stat_then_swap)
    result_path = tmp_path / "placement" / "warm-result.json"
    failure_path = tmp_path / "placement" / "warm-failure.json"

    with pytest.raises(DatabasePlacementError, match="Replica"):
        fixture.place_database(
            fixture.runspec,
            action_id=fixture.action.action_id,
            source_manifest_path=manifest_path,
            result_path=result_path,
            failure_path=failure_path,
        )

    assert not result_path.exists()
    assert displaced.is_dir()
    failure = load_database_replica_cold_failure(failure_path)
    assert failure.classification == "replica-validation-failed"
    assert isinstance(failure.cache_mount, DatabaseWarmCacheMountFacts)


def test_late_warm_result_publication_error_never_adds_contradictory_failure_evidence(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, manifest_path, _, _ = _cold_populate(tmp_path, cold_cache_root, monkeypatch)
    result_path = tmp_path / "placement" / "warm-result.json"
    failure_path = tmp_path / "placement" / "warm-failure.json"
    system_link = os.link

    def publish_then_fail(source: object, destination: object, **kwargs: object) -> None:
        system_link(source, destination, **kwargs)
        if destination == result_path.name:
            raise OSError("injected failure after exact warm Result became visible")

    monkeypatch.setattr(evidence_io.os, "link", publish_then_fail)

    with pytest.raises(DatabasePlacementError, match=r"publish|visible"):
        fixture.place_database(
            fixture.runspec,
            action_id=fixture.action.action_id,
            source_manifest_path=manifest_path,
            result_path=result_path,
            failure_path=failure_path,
        )

    assert isinstance(load_database_replica_result(result_path), DatabaseReplicaWarmResult)
    assert not failure_path.exists()
    assert sum(path.exists() for path in (result_path, failure_path)) == 1


def test_warm_probe_ignores_sibling_user_namespace_without_enumeration_or_selection(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, manifest_path, cache_root = _stage_required_fixture(
        tmp_path / "work",
        cold_cache_root,
        monkeypatch,
    )
    (cache_root / ".locks").mkdir(mode=0o700)
    (cache_root / "replicas").mkdir(mode=0o700)
    identity = fixture.runspec.payload.database.source_manifest_sha256
    sibling = tmp_path / "sibling-user-cache" / "replicas" / identity
    sibling.mkdir(parents=True)
    marker = sibling / "must-not-select"
    marker.write_bytes(b"sibling-user replica")
    sentinel = object()
    cold_calls: list[str] = []

    def forbidden_enumeration(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("warm miss must not enumerate sibling namespaces")

    def cold_delegate(*_args: object, **_kwargs: object) -> object:
        cold_calls.append("cold")
        return sentinel

    monkeypatch.setattr(warm_runtime.os, "listdir", forbidden_enumeration)
    services = _production_staged_database_placement_services()
    services = replace(
        services,
        cold_coordinator=ColdReplicaCoordinator(place=cold_delegate),
    )

    assert (
        fixture.place_database(
            fixture.runspec,
            action_id=fixture.action.action_id,
            source_manifest_path=manifest_path,
            result_path=tmp_path / "placement" / "result.json",
            failure_path=tmp_path / "placement" / "failure.json",
            staged_services=services,
        )
        is sentinel
    )
    assert cold_calls == ["cold"]
    assert marker.read_bytes() == b"sibling-user replica"


@pytest.mark.parametrize("invalid_authority", ["effective-user", "owner", "ro", "device", "filesystem"])
def test_invalid_warm_cache_authority_fails_before_replica_selection(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_authority: str,
) -> None:
    fixture, manifest_path, cache_root = _stage_required_fixture(
        tmp_path / "work",
        cold_cache_root,
        monkeypatch,
        unix_user="definitely-not-the-effective-user" if invalid_authority == "effective-user" else None,
    )
    mountinfo = tmp_path / "work" / "mountinfo"
    if invalid_authority in {"ro", "device", "filesystem"}:
        raw = mountinfo.read_text()
        if invalid_authority == "ro":
            raw = raw.replace(
                " /run/bspp/database/cache rw -",
                " /run/bspp/database/cache ro -",
            )
        elif invalid_authority == "device":
            device = f"{os.major(cache_root.stat().st_dev)}:{os.minor(cache_root.stat().st_dev)}"
            raw = raw.replace(
                f"{device} / /run/bspp/database/cache",
                "999:999 / /run/bspp/database/cache",
            )
        else:
            raw = raw.replace("tmpfs shm rw", "ext4 shm rw")
        mountinfo.write_text(raw)
    if invalid_authority == "owner":
        real_fstat = os.fstat

        def wrong_owner(descriptor: int) -> os.stat_result:
            info = real_fstat(descriptor)
            try:
                target = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
            except OSError:
                return info
            if target != cache_root:
                return info
            values = list(info)
            values[4] = os.geteuid() + 1
            return os.stat_result(values)

        monkeypatch.setattr(warm_runtime.os, "fstat", wrong_owner)

    with pytest.raises(DatabasePlacementError):
        fixture.place_database(
            fixture.runspec,
            action_id=fixture.action.action_id,
            source_manifest_path=manifest_path,
            result_path=tmp_path / "placement" / "result.json",
            failure_path=tmp_path / "placement" / "failure.json",
        )


def test_two_real_warm_shared_readers_coexist_while_exclusive_lock_is_blocked(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, manifest_path, cache_root, _ = _cold_populate(tmp_path, cold_cache_root, monkeypatch)
    identity = fixture.runspec.payload.database.source_manifest_sha256
    lease_path = cache_root / ".locks" / f"{identity}.lock"
    entered = threading.Barrier(3)
    release = threading.Event()
    system_link = os.link
    errors: list[BaseException] = []

    def hold_during_publish(source: object, destination: object, **kwargs: object) -> None:
        if str(destination).startswith("warm-"):
            entered.wait(timeout=5)
            if not release.wait(timeout=5):
                raise AssertionError("shared reader test did not release publishers")
        system_link(source, destination, **kwargs)

    def reader(ordinal: int) -> None:
        try:
            fixture.place_database(
                fixture.runspec,
                action_id=fixture.action.action_id,
                source_manifest_path=manifest_path,
                result_path=tmp_path / "placement" / f"warm-{ordinal}.json",
                failure_path=tmp_path / "placement" / f"failure-{ordinal}.json",
            )
        except BaseException as exc:
            errors.append(exc)

    monkeypatch.setattr(evidence_io.os, "link", hold_during_publish)
    threads = [threading.Thread(target=reader, args=(ordinal,)) for ordinal in range(2)]
    for thread in threads:
        thread.start()
    entered.wait(timeout=5)
    competitor = os.open(lease_path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        with pytest.raises(BlockingIOError):
            fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(competitor)
        release.set()
    for thread in threads:
        thread.join(timeout=5)
    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
