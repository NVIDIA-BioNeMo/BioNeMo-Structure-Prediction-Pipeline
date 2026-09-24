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

"""Shared Database Replica Lease and staged handoff revalidation."""

from __future__ import annotations

import fcntl
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from bspp.orchestration.contract.database_placement import (
    DATABASE_REPLICA_LEASE_TARGET,
    SELECTED_DATABASE_ROOT,
)
from bspp.orchestration.contract.database_replica import (
    DatabaseReplicaColdResult,
    DatabaseReplicaResult,
    database_replica_cold_result_digest,
    database_replica_warm_result_digest,
)
from bspp.orchestration.contract.database_replica_lease import (
    DatabaseReplicaLeaseTerminalEvidence,
    DatabaseReplicaWarmLeaseTerminalEvidence,
    PreprocessingStagedDatabasePlacementEvidence,
)
from bspp.orchestration.contract.phase import PhaseRunSpec

from ._database_placement_errors import DatabasePlacementError
from ._database_replica_authority import require_staged_authority
from ._database_replica_validation import validate_immutable_database_replica
from ._filesystem_authority import FilesystemObjectIdentity, filesystem_object_identity

_LEASE_PATH = Path(DATABASE_REPLICA_LEASE_TARGET)
_SELECTED_ROOT_PATH = Path(SELECTED_DATABASE_ROOT)


@dataclass(frozen=True)
class DatabaseReplicaLeasePaths:
    """Physical paths bound by the private lease coordinator."""

    selected_root: Path
    lease: Path


@dataclass
class HeldDatabaseReplicaLease:
    """Capability yielded only while the shared lease descriptor is held."""

    result: DatabaseReplicaResult
    _active: bool = True

    def terminal_evidence(
        self,
        *,
        kernel_started: bool,
        kernel_terminal: bool,
    ) -> DatabaseReplicaLeaseTerminalEvidence | DatabaseReplicaWarmLeaseTerminalEvidence:
        if not self._active or kernel_started != kernel_terminal:
            raise DatabasePlacementError(
                "Database Replica Lease terminal evidence requires a held lease and an exact kernel lifetime pair"
            )
        if isinstance(self.result, DatabaseReplicaColdResult):
            return DatabaseReplicaLeaseTerminalEvidence(
                database_replica_cold_result_digest=database_replica_cold_result_digest(self.result),
                source_manifest_sha256=self.result.source_manifest_sha256,
                replica_manifest_sha256=self.result.replica_manifest_sha256,
                selected_container_root=self.result.selected_container_root,
                lease_target=DATABASE_REPLICA_LEASE_TARGET,
                verification="metadata-verified",
                acquisition="shared-nonblocking",
                kernel_started=kernel_started,
                held_through_kernel_exit=kernel_terminal,
            )
        return DatabaseReplicaWarmLeaseTerminalEvidence(
            database_replica_warm_result_digest=database_replica_warm_result_digest(self.result),
            source_manifest_sha256=self.result.source_manifest_sha256,
            replica_manifest_sha256=self.result.replica_manifest_sha256,
            selected_container_root=self.result.selected_container_root,
            lease_target=DATABASE_REPLICA_LEASE_TARGET,
            verification="metadata-verified",
            acquisition="shared-nonblocking",
            kernel_started=kernel_started,
            held_through_kernel_exit=kernel_terminal,
        )


@contextmanager
def database_replica_lease(
    runspec: PhaseRunSpec,
    result: DatabaseReplicaResult,
) -> Iterator[HeldDatabaseReplicaLease]:
    """Acquire and hold a fresh shared lease around staged science."""
    production_paths = DatabaseReplicaLeasePaths(
        selected_root=_SELECTED_ROOT_PATH,
        lease=_LEASE_PATH,
    )
    with _database_replica_lease(runspec, result, paths=production_paths) as held:
        yield held


@contextmanager
def _database_replica_lease(
    runspec: PhaseRunSpec,
    result: DatabaseReplicaResult,
    *,
    paths: DatabaseReplicaLeasePaths,
) -> Iterator[HeldDatabaseReplicaLease]:
    """Acquire the lease using an explicitly composed physical site."""
    reconcile_database_replica_result(runspec, result)
    lease_descriptor = _open_shared_lease(paths.lease)
    root_descriptor: int | None = None
    held = HeldDatabaseReplicaLease(result=result)
    try:
        root_descriptor, root_identity = _open_selected_root(paths.selected_root)
        validate_immutable_database_replica(
            root_descriptor,
            paths.selected_root,
            database_set=result.database_set,
            source_manifest_sha256=result.source_manifest_sha256,
            expected_replica_manifest_sha256=result.replica_manifest_sha256,
            expected_copy_evidence=result.copy_evidence if isinstance(result, DatabaseReplicaColdResult) else None,
            expected_identity=root_identity,
        )
        _verify_selected_root(paths.selected_root, root_descriptor, root_identity)
        yield held
    except DatabasePlacementError as exc:
        raise DatabasePlacementError(f"Database Replica handoff revalidation failed: {exc}") from exc
    except OSError as exc:
        raise DatabasePlacementError("Database Replica handoff revalidation failed") from exc
    finally:
        held._active = False
        if root_descriptor is not None:
            os.close(root_descriptor)
        os.close(lease_descriptor)


def reconcile_database_replica_cold_result(
    runspec: PhaseRunSpec,
    result: DatabaseReplicaColdResult,
) -> None:
    """Bind one cold Result to exact staged RunSpec/action authority."""
    reconcile_database_replica_result(runspec, result)


def reconcile_database_replica_result(
    runspec: PhaseRunSpec,
    result: DatabaseReplicaResult,
) -> None:
    """Bind one concrete cold-or-warm Result to exact staged authority."""
    require_staged_authority(runspec)
    action = runspec.payload.actions[0]
    binding = runspec.payload.database
    if (
        result.phase_run_id != runspec.phase_run_id
        or result.attempt_id != runspec.attempt_id
        or result.phase_runspec_digest != runspec.digest
        or result.action_id != action.action_id
        or result.database_set != binding.database_set
        or result.requested_policy != binding.requested_policy
        or result.source_manifest_sha256 != binding.source_manifest_sha256
        or result.selected_container_root != SELECTED_DATABASE_ROOT
        or result.replica_container_root != f"/run/bspp/database/cache/replicas/{binding.source_manifest_sha256}"
    ):
        raise DatabasePlacementError("Database Replica Result does not match exact RunSpec authority")


def reconcile_staged_database_placement_evidence(
    runspec: PhaseRunSpec,
    evidence: PreprocessingStagedDatabasePlacementEvidence,
    *,
    require_success: bool,
) -> None:
    """Reconcile the staged wrapper without duplicating cold parsing or authority."""
    if evidence.result is None:
        if require_success or evidence.failure is None:
            raise ValueError("successful staged reconciliation requires a cold or warm Result")
        failure = evidence.failure
        binding = runspec.payload.database
        action = runspec.payload.actions[0]
        if (
            failure.phase_run_id != runspec.phase_run_id
            or failure.attempt_id != runspec.attempt_id
            or failure.phase_runspec_digest != runspec.digest
            or failure.action_id != action.action_id
            or failure.database_set != binding.database_set
            or failure.requested_policy != binding.requested_policy
            or failure.source_manifest_sha256 != binding.source_manifest_sha256
        ):
            raise ValueError("staged cold failure does not match exact RunSpec authority")
        return
    reconcile_database_replica_result(runspec, evidence.result)
    if require_success and (
        not isinstance(
            evidence.lease_outcome,
            DatabaseReplicaLeaseTerminalEvidence | DatabaseReplicaWarmLeaseTerminalEvidence,
        )
        or not evidence.science_started
        or not evidence.lease_outcome.kernel_started
        or not evidence.lease_outcome.held_through_kernel_exit
    ):
        raise ValueError("successful staged reconciliation requires held-through-kernel-exit lease evidence")


def _open_shared_lease(path: Path) -> int:
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o600:
            raise DatabasePlacementError("Database Replica Lease protocol file authority is invalid")
        fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        return descriptor
    except BlockingIOError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise DatabasePlacementError("Database Replica Lease is contended") from exc
    except DatabasePlacementError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise DatabasePlacementError("Database Replica Lease protocol file is unavailable") from exc


def _open_selected_root(path: Path) -> tuple[int, FilesystemObjectIdentity]:
    try:
        path_info = path.stat(follow_symlinks=False)
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        raise DatabasePlacementError("Database Replica handoff revalidation cannot open selected root") from exc
    descriptor_info = os.fstat(descriptor)
    identity = filesystem_object_identity(path_info)
    if (
        not stat.S_ISDIR(path_info.st_mode)
        or path_info.st_uid != os.geteuid()
        or stat.S_IMODE(path_info.st_mode) != 0o555
        or filesystem_object_identity(descriptor_info) != identity
    ):
        os.close(descriptor)
        raise DatabasePlacementError("Database Replica handoff revalidation selected-root authority is invalid")
    return descriptor, identity


def _verify_selected_root(path: Path, descriptor: int, identity: FilesystemObjectIdentity) -> None:
    try:
        path_info = path.stat(follow_symlinks=False)
        descriptor_info = os.fstat(descriptor)
    except OSError as exc:
        raise DatabasePlacementError("Database Replica handoff revalidation lost selected-root authority") from exc
    if filesystem_object_identity(path_info) != identity or filesystem_object_identity(descriptor_info) != identity:
        raise DatabasePlacementError("Database Replica handoff revalidation selected root changed")


__all__ = [
    "HeldDatabaseReplicaLease",
    "database_replica_lease",
    "reconcile_database_replica_cold_result",
    "reconcile_database_replica_result",
    "reconcile_staged_database_placement_evidence",
]
