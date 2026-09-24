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

"""Bounded Database Replica lock ordering and cleanup."""

from __future__ import annotations

import fcntl
import os
import stat
from pathlib import Path

import pytest

from bspp.orchestration.runtime.preprocessing._database_replica_errors import ClassifiedDatabaseReplicaError
from bspp.orchestration.runtime.preprocessing._database_replica_lock import (
    FilesystemAuthority,
    LockWait,
    hold_exclusive_cache,
    hold_exclusive_identity,
    verify_cache_ownership,
    verify_identity_ownership,
)


def _open_cache(root: Path) -> int:
    root.mkdir(mode=0o700)
    return os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)


def _hold_lock(path: Path) -> int:
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return descriptor


def _cache_authority(descriptor: int) -> FilesystemAuthority:
    info = os.fstat(descriptor)
    return FilesystemAuthority(
        file_type=stat.S_IFMT(info.st_mode),
        device=info.st_dev,
        inode=info.st_ino,
        owner_uid=info.st_uid,
        permissions=stat.S_IMODE(info.st_mode),
    )


def test_identity_and_cache_waits_consume_one_deadline_and_record_contention(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache_descriptor = _open_cache(cache)
    locks = cache / ".locks"
    locks.mkdir(mode=0o700)
    identity_path = locks / f"{'a' * 64}.lock"
    identity_holder = _hold_lock(identity_path)
    cache_holder = _hold_lock(locks / "cache.lock")
    now = 0.0
    sleeps: list[float] = []

    def monotonic() -> float:
        return now

    def sleeper(seconds: float) -> None:
        nonlocal now, identity_holder
        sleeps.append(seconds)
        now += seconds
        if identity_holder >= 0:
            os.close(identity_holder)
            identity_holder = -1

    wait = LockWait.for_timeout(
        1,
        monotonic=monotonic,
        sleeper=sleeper,
        poll_quantum_seconds=0.25,
    )
    try:
        with hold_exclusive_identity(
            cache_descriptor,
            cache_path=cache,
            cache_identity=_cache_authority(cache_descriptor),
            source_manifest_sha256="a" * 64,
            wait=wait,
        ) as identity:
            assert identity.contended is True
            with (
                pytest.raises(ClassifiedDatabaseReplicaError, match="timed out") as captured,
                hold_exclusive_cache(identity, wait=wait),
            ):
                pytest.fail("cache lock must remain held")
            assert captured.value.classification == "lock-unavailable"
        assert sleeps == [0.25, 0.25, 0.25, 0.25]
        assert now == 1.0
    finally:
        if identity_holder >= 0:
            os.close(identity_holder)
        os.close(cache_holder)
        os.close(cache_descriptor)


@pytest.mark.parametrize("raised", [OSError("sentinel body error"), KeyboardInterrupt()])
def test_lock_body_exceptions_propagate_unchanged_and_release_both_locks(
    tmp_path: Path,
    raised: BaseException,
) -> None:
    cache = tmp_path / "cache"
    cache_descriptor = _open_cache(cache)
    wait = LockWait.for_timeout(1)
    identity_name = f"{'b' * 64}.lock"
    try:
        with (
            pytest.raises(type(raised), match="sentinel" if isinstance(raised, OSError) else None),
            hold_exclusive_identity(
                cache_descriptor,
                cache_path=cache,
                cache_identity=_cache_authority(cache_descriptor),
                source_manifest_sha256="b" * 64,
                wait=wait,
            ) as identity,
            hold_exclusive_cache(identity, wait=wait) as cache_ownership,
        ):
            assert identity.contended is False
            assert cache_ownership.contended is False
            raise raised

        locks_descriptor = os.open(
            ".locks",
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=cache_descriptor,
        )
        try:
            for name in (identity_name, "cache.lock"):
                descriptor = os.open(name, os.O_RDWR | os.O_NOFOLLOW, dir_fd=locks_descriptor)
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                finally:
                    os.close(descriptor)
        finally:
            os.close(locks_descriptor)
    finally:
        os.close(cache_descriptor)


@pytest.mark.parametrize("swapped_authority", ["identity", "locks-directory"])
def test_identity_authority_basename_swap_fails_before_downstream_work(
    tmp_path: Path,
    swapped_authority: str,
) -> None:
    cache = tmp_path / "cache"
    cache_descriptor = _open_cache(cache)
    wait = LockWait.for_timeout(1)
    identity_name = f"{'c' * 64}.lock"
    downstream: list[str] = []
    try:
        with hold_exclusive_identity(
            cache_descriptor,
            cache_path=cache,
            cache_identity=_cache_authority(cache_descriptor),
            source_manifest_sha256="c" * 64,
            wait=wait,
        ) as ownership:
            locks = cache / ".locks"
            if swapped_authority == "identity":
                original = locks / identity_name
                displaced = locks / f"{identity_name}.displaced"
                original.rename(displaced)
                original.touch(mode=0o600)
            else:
                displaced = cache / ".locks.displaced"
                locks.rename(displaced)
                locks.mkdir(mode=0o700)
            try:
                with pytest.raises(ClassifiedDatabaseReplicaError, match="changed"):
                    verify_identity_ownership(cache_descriptor, ownership)
                    downstream.extend(("candidate-validation", "capacity", "evidence"))
            finally:
                if swapped_authority == "identity":
                    original.unlink()
                    displaced.rename(original)
                else:
                    locks.rmdir()
                    displaced.rename(locks)
        assert downstream == []
    finally:
        os.close(cache_descriptor)


def test_cache_lock_basename_swap_fails_before_downstream_work(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache_descriptor = _open_cache(cache)
    wait = LockWait.for_timeout(1)
    downstream: list[str] = []
    try:
        with (
            hold_exclusive_identity(
                cache_descriptor,
                cache_path=cache,
                cache_identity=_cache_authority(cache_descriptor),
                source_manifest_sha256="d" * 64,
                wait=wait,
            ) as identity,
            hold_exclusive_cache(identity, wait=wait) as ownership,
        ):
            original = cache / ".locks" / "cache.lock"
            displaced = original.with_name("cache.lock.displaced")
            original.rename(displaced)
            original.touch(mode=0o600)
            try:
                with pytest.raises(ClassifiedDatabaseReplicaError, match="changed"):
                    verify_cache_ownership(cache_descriptor, ownership)
                    downstream.extend(("candidate-validation", "capacity", "evidence"))
            finally:
                original.unlink()
                displaced.rename(original)
        assert downstream == []
    finally:
        os.close(cache_descriptor)
