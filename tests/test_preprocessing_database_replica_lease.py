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

"""Real shared-lease and staged replica handoff tests."""

from __future__ import annotations

import fcntl
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from bspp.orchestration.contract.database_replica import DatabaseReplicaColdResult, DatabaseReplicaWarmResult
from bspp.orchestration.contract.database_replica_lease import (
    DatabaseReplicaLeaseTerminalEvidence,
    DatabaseReplicaWarmLeaseTerminalEvidence,
)
from bspp.orchestration.runtime.preprocessing._database_replica_lease import (
    DatabaseReplicaLeasePaths,
    _database_replica_lease,
)
from bspp.orchestration.runtime.preprocessing.database_placement import DatabasePlacementError
from tests.support.database_cold_replica import _stage_required_fixture
from tests.support.database_cold_replica import cold_cache_root as cold_cache_root
from tests.support.preprocessing_execution import LocalExecutionFixture


def _populated_fixture(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[LocalExecutionFixture, DatabaseReplicaColdResult, DatabaseReplicaLeasePaths]:
    fixture, manifest_path, cache_root = _stage_required_fixture(
        tmp_path / "work",
        cold_cache_root,
        monkeypatch,
    )
    monkeypatch.setattr(os, "fstatvfs", lambda _descriptor: SimpleNamespace(f_bavail=2, f_frsize=1))
    result = fixture.place_database(
        fixture.runspec,
        action_id=fixture.action.action_id,
        source_manifest_path=manifest_path,
        result_path=tmp_path / "placement" / "result.json",
        failure_path=tmp_path / "placement" / "failure.json",
    )
    assert isinstance(result, DatabaseReplicaColdResult)
    identity = fixture.runspec.payload.database.source_manifest_sha256
    selected_root = cache_root / "replicas" / identity
    lease_path = cache_root / ".locks" / f"{identity}.lock"
    return fixture, result, DatabaseReplicaLeasePaths(selected_root=selected_root, lease=lease_path)


def test_shared_lease_uses_persistent_read_only_protocol_file_and_covers_terminal_evidence(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, result, paths = _populated_fixture(tmp_path, cold_cache_root, monkeypatch)
    lease_path = paths.lease
    before = lease_path.stat()
    competitor = os.open(lease_path, os.O_RDWR | os.O_NOFOLLOW)
    try:
        with _database_replica_lease(fixture.runspec, result, paths=paths) as held:
            with pytest.raises(BlockingIOError):
                fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            terminal = held.terminal_evidence(kernel_started=True, kernel_terminal=True)
            assert isinstance(terminal, DatabaseReplicaLeaseTerminalEvidence)
            assert terminal.held_through_kernel_exit is True
        fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(competitor, fcntl.LOCK_UN)
    finally:
        os.close(competitor)

    after = lease_path.stat()
    assert (after.st_dev, after.st_ino, after.st_mode) == (before.st_dev, before.st_ino, before.st_mode)


def test_warm_result_reacquires_fresh_shared_lease_and_emits_matching_terminal_family(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, _, paths = _populated_fixture(tmp_path, cold_cache_root, monkeypatch)
    lease_path = paths.lease
    warm = fixture.place_database(
        fixture.runspec,
        action_id=fixture.action.action_id,
        source_manifest_path=tmp_path / "work" / "database-source-manifest.json",
        result_path=tmp_path / "placement" / "warm-result.json",
        failure_path=tmp_path / "placement" / "warm-failure.json",
    )
    assert isinstance(warm, DatabaseReplicaWarmResult)
    competitor = os.open(lease_path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        with _database_replica_lease(fixture.runspec, warm, paths=paths) as held:
            with pytest.raises(BlockingIOError):
                fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            terminal = held.terminal_evidence(kernel_started=True, kernel_terminal=True)
            assert isinstance(terminal, DatabaseReplicaWarmLeaseTerminalEvidence)
        fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(competitor)


@pytest.mark.parametrize("mutation", ["missing", "symlink", "directory", "wrong-mode", "contended"])
def test_shared_lease_rejects_invalid_or_contended_protocol_file_without_creating_or_replacing_it(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    fixture, result, paths = _populated_fixture(tmp_path, cold_cache_root, monkeypatch)
    lease_path = paths.lease
    contention_descriptor: int | None = None
    if mutation == "missing":
        lease_path.unlink()
    elif mutation == "symlink":
        lease_path.unlink()
        target = lease_path.with_suffix(".target")
        target.write_bytes(b"")
        target.chmod(0o600)
        lease_path.symlink_to(target)
    elif mutation == "directory":
        lease_path.unlink()
        lease_path.mkdir(mode=0o600)
    elif mutation == "wrong-mode":
        lease_path.chmod(0o644)
    else:
        contention_descriptor = os.open(lease_path, os.O_RDWR | os.O_NOFOLLOW)
        fcntl.flock(contention_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(DatabasePlacementError), _database_replica_lease(fixture.runspec, result, paths=paths):
            pytest.fail("invalid lease authority must fail before yield")
    finally:
        if contention_descriptor is not None:
            os.close(contention_descriptor)

    if mutation == "missing":
        assert not lease_path.exists()
    elif mutation == "symlink":
        assert lease_path.is_symlink()
    elif mutation == "directory":
        assert lease_path.is_dir()


def test_shared_lease_rejects_result_runspec_authority_forgery_before_open(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, result, paths = _populated_fixture(tmp_path, cold_cache_root, monkeypatch)
    lease_path = paths.lease
    before = lease_path.stat()

    with (
        pytest.raises(DatabasePlacementError, match="RunSpec authority"),
        _database_replica_lease(
            fixture.runspec,
            replace(result, phase_runspec_digest="f" * 64),
            paths=paths,
        ),
    ):
        pytest.fail("forged Result must fail before lease acquisition")

    after = lease_path.stat()
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)


def test_shared_lease_revalidates_complete_replica_before_yield(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, result, paths = _populated_fixture(tmp_path, cold_cache_root, monkeypatch)
    selected_root = paths.selected_root
    member = selected_root / fixture.runspec.payload.database.source_manifest.members[0].logical_name
    member.chmod(0o644)

    with (
        pytest.raises(DatabasePlacementError, match="revalidation"),
        _database_replica_lease(fixture.runspec, result, paths=paths),
    ):
        pytest.fail("invalid handoff must fail before staged science receives a lease")


@pytest.mark.parametrize("invalid_member_authority", ["external-hardlink", "off-device"])
def test_fresh_science_lease_rejects_member_authority_outside_the_immutable_replica(
    tmp_path: Path,
    cold_cache_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_member_authority: str,
) -> None:
    fixture, result, paths = _populated_fixture(tmp_path, cold_cache_root, monkeypatch)
    selected_root = paths.selected_root
    member = selected_root / fixture.runspec.payload.database.source_manifest.members[0].logical_name
    if invalid_member_authority == "external-hardlink":
        os.link(member, selected_root.parent.parent / "external-member-link")
    else:
        real_stat = os.stat

        def off_device_stat(
            path: int | str | bytes | os.PathLike[str] | os.PathLike[bytes],
            *,
            dir_fd: int | None = None,
            follow_symlinks: bool = True,
        ) -> os.stat_result | SimpleNamespace:
            info = real_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)
            if dir_fd is not None and path == member.name and not follow_symlinks:
                return SimpleNamespace(
                    st_mode=info.st_mode,
                    st_size=info.st_size,
                    st_mtime_ns=info.st_mtime_ns,
                    st_dev=info.st_dev + 1,
                    st_ino=info.st_ino,
                    st_nlink=info.st_nlink,
                )
            return info

        monkeypatch.setattr(os, "stat", off_device_stat)

    with (
        pytest.raises(DatabasePlacementError, match="revalidation"),
        _database_replica_lease(fixture.runspec, result, paths=paths),
    ):
        pytest.fail("invalid member authority must fail before staged science receives a lease")
