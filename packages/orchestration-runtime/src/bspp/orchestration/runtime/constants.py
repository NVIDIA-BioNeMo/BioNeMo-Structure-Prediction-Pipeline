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

"""Shared constants and utilities for the BSPP pipelines."""

from __future__ import annotations

import contextlib
import errno
import fcntl
from collections.abc import Generator
from pathlib import Path

# KNOWN_SUFFIXES is a direct name binding to the contract model_identity list
# and is shared by object identity.  Do not mutate or rebind it here
# or in the contract module; treat it as read-only.
from bspp.orchestration.contract.model_identity import KNOWN_SUFFIXES, MAX_PROTEINS_PER_SHARD

__all__ = [
    "IO_BASE_DELAY",
    "IO_MAX_RETRIES",
    "KNOWN_SUFFIXES",
    "MAX_PROTEINS_PER_SHARD",
    "RETRYABLE_ERRNOS",
    "locked_parquet",
]

RETRYABLE_ERRNOS: set[int] = {errno.ESTALE, errno.EAGAIN, errno.EIO, errno.EBUSY}

IO_MAX_RETRIES: int = 3

IO_BASE_DELAY: float = 1.0


@contextlib.contextmanager
def locked_parquet(parquet_path: Path | str) -> Generator[Path, None, None]:
    """Context manager that holds an exclusive flock while the parquet is being modified.

    Usage::

        with locked_parquet(tracking_path) as p:
            table = pq.read_table(p)
            # ... modify table ...
            pq.write_table(table, p)
    """
    parquet_path = Path(parquet_path)
    lock_path = parquet_path.with_suffix(".parquet.lock")
    lock_fd = open(lock_path, "w")  # noqa: SIM115
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        yield parquet_path
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()
