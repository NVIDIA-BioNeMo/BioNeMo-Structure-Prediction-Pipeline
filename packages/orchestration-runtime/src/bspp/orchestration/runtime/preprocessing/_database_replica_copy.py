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

"""Pinned unique-source rsync and replica-contained hard-link construction."""

from __future__ import annotations

import os
import stat
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from bspp.orchestration.contract.database_placement import DATABASE_CACHE_ROOT, DATABASE_SOURCE_ROOT
from bspp.orchestration.contract.database_placement_result import DatabaseSourceObservation
from bspp.orchestration.contract.database_replica import (
    DatabaseReplicaCopyEvidence,
    DatabaseReplicaMember,
    DatabaseRsyncOutcome,
)
from bspp.orchestration.contract.database_set_provisioning import DatabaseSourceMember

from ._database_placement_errors import DatabasePlacementError
from ._database_replica_errors import ClassifiedDatabaseReplicaError

_RSYNC_OPTIONS = ("--archive", "--no-owner", "--no-group", "--no-perms", "--protect-args")


@dataclass(frozen=True)
class _SourceGroup:
    resolved_path: str
    logical_names: tuple[str, ...]
    size_bytes: int
    mtime_ns: int

    @property
    def leader(self) -> str:
        return self.logical_names[0]


def population_worker_count(*, unique_source_count: int, allocated_cpus: int) -> int:
    """Return ADR 0067's exact automatic cold-copy concurrency."""
    if unique_source_count <= 0 or allocated_cpus <= 0:
        raise DatabasePlacementError("cold replica worker inputs must be positive")
    return min(8, unique_source_count, max(1, allocated_cpus // 2))


def allocated_replica_bytes(observation: DatabaseSourceObservation) -> int:
    """Count each unique resolved payload exactly once; aliases allocate no bytes."""
    return sum(group.size_bytes for group in _source_groups(observation))


def populate_replica_payload(
    *,
    source_descriptor: int,
    temporary_descriptor: int,
    observation: DatabaseSourceObservation,
    allocated_cpus: int,
    rsync_executable: Path,
) -> tuple[DatabaseReplicaCopyEvidence, tuple[DatabaseReplicaMember, ...]]:
    """Copy every unique source once, create aliases, then validate exact topology."""
    groups = _source_groups(observation)
    workers = population_worker_count(unique_source_count=len(groups), allocated_cpus=allocated_cpus)
    started = time.monotonic_ns()
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="bspp-database-rsync") as executor:
        outcomes = tuple(
            executor.map(
                lambda group: _copy_source_group(
                    source_descriptor,
                    temporary_descriptor,
                    group,
                    rsync_executable=rsync_executable,
                ),
                groups,
            )
        )
    elapsed = max(1, time.monotonic_ns() - started)
    for group in groups:
        for alias in group.logical_names[1:]:
            try:
                os.link(
                    group.leader,
                    alias,
                    src_dir_fd=temporary_descriptor,
                    dst_dir_fd=temporary_descriptor,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise ClassifiedDatabaseReplicaError(
                    "hard-links-unsupported",
                    f"Database Replica cache does not support required hard-link alias {alias!r}",
                ) from exc
    members = _replica_members(observation)
    validate_replica_payload(temporary_descriptor, members, immutable=False)
    total = sum(item.size_bytes for item in outcomes)
    return (
        DatabaseReplicaCopyEvidence(
            source_count=len(groups),
            selected_workers=workers,
            total_copied_bytes=total,
            elapsed_nanoseconds=elapsed,
            aggregate_bytes_per_second=total * 1_000_000_000 // elapsed,
            outcomes=outcomes,
        ),
        members,
    )


def validate_replica_payload(
    root_descriptor: int,
    members: tuple[DatabaseReplicaMember, ...],
    *,
    immutable: bool,
) -> None:
    """Verify complete regular-file inventory, metadata and hard-link topology."""
    expected_names = {item.replica_path for item in members}
    try:
        observed_names = set(os.listdir(root_descriptor))
    except OSError as exc:
        raise DatabasePlacementError("cannot list Database Replica payload") from exc
    allowed = expected_names | ({"replica-manifest.json"} if immutable else set())
    if observed_names != allowed:
        raise DatabasePlacementError(
            f"Database Replica inventory is not complete; missing={sorted(allowed - observed_names)!r}; "
            f"extra={sorted(observed_names - allowed)!r}"
        )
    inode_by_group: dict[str, tuple[int, int]] = {}
    group_by_inode: dict[tuple[int, int], str] = {}
    for member in members:
        try:
            info = os.stat(member.replica_path, dir_fd=root_descriptor, follow_symlinks=False)
        except OSError as exc:
            raise DatabasePlacementError(f"Database Replica member is unavailable: {member.replica_path}") from exc
        if not stat.S_ISREG(info.st_mode):
            raise DatabasePlacementError(f"Database Replica member must be a regular file: {member.replica_path}")
        if info.st_size != member.size_bytes or info.st_mtime_ns != member.mtime_ns:
            raise DatabasePlacementError(f"Database Replica member metadata is invalid: {member.replica_path}")
        if immutable and stat.S_IMODE(info.st_mode) != 0o444:
            raise DatabasePlacementError(f"Database Replica member must be immutable 0444: {member.replica_path}")
        inode = (info.st_dev, info.st_ino)
        prior = inode_by_group.setdefault(member.hardlink_group, inode)
        if prior != inode:
            raise DatabasePlacementError(f"Database Replica aliases do not share one inode: {member.hardlink_group}")
        prior_group = group_by_inode.setdefault(inode, member.hardlink_group)
        if prior_group != member.hardlink_group:
            raise DatabasePlacementError("distinct Database Replica source groups must not share an inode")


def _source_groups(observation: DatabaseSourceObservation) -> tuple[_SourceGroup, ...]:
    grouped: dict[str, list[DatabaseSourceMember]] = {}
    for member in observation.members:
        grouped.setdefault(member.resolved_path, []).append(member)
    result: list[_SourceGroup] = []
    for resolved_path, members in sorted(grouped.items()):
        identities = {(item.size_bytes, item.mtime_ns) for item in members}
        if len(identities) != 1:
            raise DatabasePlacementError(f"logical aliases disagree about resolved source metadata: {resolved_path!r}")
        size_bytes, mtime_ns = next(iter(identities))
        result.append(
            _SourceGroup(
                resolved_path=resolved_path,
                logical_names=tuple(sorted(item.logical_name for item in members)),
                size_bytes=size_bytes,
                mtime_ns=mtime_ns,
            )
        )
    if not result:
        raise DatabasePlacementError("Database Replica population requires at least one unique source")
    return tuple(result)


def _copy_source_group(
    source_descriptor: int,
    temporary_descriptor: int,
    group: _SourceGroup,
    *,
    rsync_executable: Path,
) -> DatabaseRsyncOutcome:
    source = f"/proc/self/fd/{source_descriptor}/{group.resolved_path}"
    destination = f"/proc/self/fd/{temporary_descriptor}/{group.leader}"
    argv = (str(rsync_executable), *_RSYNC_OPTIONS, source, destination)
    try:
        completed = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            close_fds=True,
            pass_fds=(source_descriptor, temporary_descriptor),
            env={"LC_ALL": "C"},
        )
    except OSError as exc:
        raise DatabasePlacementError(f"cannot execute pinned rsync for {group.resolved_path!r}") from exc
    if completed.returncode != 0:
        error = completed.stderr.decode(errors="replace")[:1024]
        raise DatabasePlacementError(
            f"pinned rsync failed for {group.resolved_path!r} with exit {completed.returncode}: {error}"
        )
    recorded_argv = (
        "/usr/bin/rsync",
        *_RSYNC_OPTIONS,
        f"{DATABASE_SOURCE_ROOT}/{group.resolved_path}",
        f"{DATABASE_CACHE_ROOT}/replicas/.population-<private>/{group.leader}",
    )
    return DatabaseRsyncOutcome(
        resolved_source_path=group.resolved_path,
        destination_logical_name=group.leader,
        logical_names=group.logical_names,
        size_bytes=group.size_bytes,
        argv=recorded_argv,
        exit_code=0,
    )


def _replica_members(observation: DatabaseSourceObservation) -> tuple[DatabaseReplicaMember, ...]:
    return tuple(
        sorted(
            (
                DatabaseReplicaMember(
                    role=source.role,
                    database_name=source.database_name,
                    logical_name=source.logical_name,
                    resolved_source_path=source.resolved_path,
                    replica_path=source.logical_name,
                    size_bytes=source.size_bytes,
                    mtime_ns=source.mtime_ns,
                    hardlink_group=source.resolved_path,
                )
                for source in observation.members
            ),
            key=lambda item: item.logical_name,
        )
    )


__all__ = [
    "allocated_replica_bytes",
    "populate_replica_payload",
    "population_worker_count",
    "validate_replica_payload",
]
