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

"""Tests for bspp.orchestration.runtime.constants."""

from __future__ import annotations

import errno
import fcntl
import threading
from pathlib import Path

from bspp.orchestration.contract import model_identity
from bspp.orchestration.runtime.constants import (
    IO_BASE_DELAY,
    IO_MAX_RETRIES,
    KNOWN_SUFFIXES,
    MAX_PROTEINS_PER_SHARD,
    RETRYABLE_ERRNOS,
    locked_parquet,
)


def test_known_suffixes_is_nonempty_list_of_strings() -> None:
    assert isinstance(KNOWN_SUFFIXES, list)
    assert len(KNOWN_SUFFIXES) > 0
    assert all(isinstance(s, str) for s in KNOWN_SUFFIXES)


def test_known_suffixes_contains_canonical_suffixes() -> None:
    assert "-model_v1.pdb" in KNOWN_SUFFIXES
    assert "-meta_v1.json" in KNOWN_SUFFIXES


def test_max_proteins_per_shard() -> None:
    assert MAX_PROTEINS_PER_SHARD == 5000
    assert isinstance(MAX_PROTEINS_PER_SHARD, int)


def test_known_suffixes_identity_binding() -> None:
    assert KNOWN_SUFFIXES is model_identity.KNOWN_SUFFIXES


def test_max_proteins_per_shard_identity_binding() -> None:
    assert MAX_PROTEINS_PER_SHARD is model_identity.MAX_PROTEINS_PER_SHARD


def test_retryable_errnos_contains_expected() -> None:
    assert errno.ESTALE in RETRYABLE_ERRNOS
    assert errno.EAGAIN in RETRYABLE_ERRNOS
    assert errno.EIO in RETRYABLE_ERRNOS
    assert errno.EBUSY in RETRYABLE_ERRNOS
    assert isinstance(RETRYABLE_ERRNOS, set)


def test_io_constants() -> None:
    assert IO_MAX_RETRIES == 3
    assert IO_BASE_DELAY == 1.0


def test_locked_parquet_creates_lock_and_yields_path(tmp_path: Path) -> None:
    pq_path = tmp_path / "data.parquet"
    pq_path.write_bytes(b"")
    lock_path = tmp_path / "data.parquet.lock"

    with locked_parquet(pq_path) as p:
        assert p == pq_path
        assert lock_path.exists()


def test_locked_parquet_mutual_exclusion(tmp_path: Path) -> None:
    pq_path = tmp_path / "data.parquet"
    pq_path.write_bytes(b"")
    lock_path = tmp_path / "data.parquet.lock"

    with locked_parquet(pq_path):
        # While the lock is held, a non-blocking attempt should fail
        fd = open(lock_path, "w")  # noqa: SIM115
        try:
            acquired = False
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except OSError:
                acquired = False
            assert not acquired, "Should not be able to acquire lock while held"
        finally:
            if acquired:
                fcntl.flock(fd, fcntl.LOCK_UN)
            fd.close()


def test_locked_parquet_releases_on_exit(tmp_path: Path) -> None:
    pq_path = tmp_path / "data.parquet"
    pq_path.write_bytes(b"")
    lock_path = tmp_path / "data.parquet.lock"

    # Acquire and release via context manager
    with locked_parquet(pq_path):
        pass

    # Now we should be able to acquire the lock without blocking
    acquired = threading.Event()

    def try_lock() -> None:
        fd = open(lock_path, "w")  # noqa: SIM115
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired.set()
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            fd.close()

    t = threading.Thread(target=try_lock)
    t.start()
    t.join(timeout=2.0)
    assert acquired.is_set(), "Lock should be acquirable after context manager exits"
