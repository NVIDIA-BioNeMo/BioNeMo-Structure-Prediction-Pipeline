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

"""Exclusive invalid Database Replica retirement and same-identity cleanup."""

from __future__ import annotations

import os
import re
import stat
import uuid
from pathlib import Path

from bspp.orchestration.contract.phase import PhaseRunSpec

from ._database_placement_errors import DatabasePlacementError
from ._database_replica_errors import ClassifiedDatabaseReplicaError
from ._database_replica_lock_authority import verify_cache_ownership
from ._database_replica_lock_types import CacheExclusiveOwnership
from ._database_replica_publication import (
    PopulationWorkspace,
    publish_replica_noreplace,
    require_renameat2,
)
from ._database_replica_repair_evidence import (
    DatabaseReplicaCleanupEvidence,
    DatabaseReplicaInvalidationEvidence,
    RepairActionAuthority,
    publish_database_replica_cleanup,
    publish_database_replica_invalidation,
    removed_basenames_digest,
)
from ._database_replica_validation import validate_immutable_database_replica
from ._filesystem_authority import (
    FilesystemAuthority,
    FilesystemObjectIdentity,
    filesystem_authority,
    filesystem_object_identity,
)
from ._owned_tree import ExpectedDirectoryBinding as _ExpectedDirectoryBinding
from ._owned_tree import OwnedTreeRemover
from ._owned_tree import remove_owned_tree as _shared_remove_owned_tree


def retire_invalid_replica(
    runspec: PhaseRunSpec,
    *,
    action_id: str,
    cache_descriptor: int,
    ownership: CacheExclusiveOwnership,
    result_path: Path,
) -> None:
    """Revalidate and atomically retire one invalid final under both EX locks."""
    _retire_invalid_replica(
        runspec,
        action_id=action_id,
        cache_descriptor=cache_descriptor,
        ownership=ownership,
        result_path=result_path,
        owned_tree_remover=OwnedTreeRemover(remove=_remove_owned_tree),
    )


def _retire_invalid_replica(
    runspec: PhaseRunSpec,
    *,
    action_id: str,
    cache_descriptor: int,
    ownership: CacheExclusiveOwnership,
    result_path: Path,
    owned_tree_remover: OwnedTreeRemover,
) -> None:
    """Private retirement coordinator with explicit cleanup composition."""
    identity = runspec.payload.database.source_manifest_sha256
    verify_cache_ownership(cache_descriptor, ownership)
    replicas_descriptor = _open_private_replicas(cache_descriptor)
    replica_descriptor: int | None = None
    try:
        try:
            visible = os.stat(identity, dir_fd=replicas_descriptor, follow_symlinks=False)
            replica_descriptor = os.open(
                identity,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=replicas_descriptor,
            )
        except OSError as exc:
            raise ClassifiedDatabaseReplicaError(
                "replica-validation-failed",
                "invalid Database Replica cannot be safely bound for repair",
            ) from exc
        opened = os.fstat(replica_descriptor)
        expected = _directory_authority(visible, cache_descriptor=cache_descriptor)
        if _directory_authority(opened, cache_descriptor=cache_descriptor) != expected:
            raise ClassifiedDatabaseReplicaError(
                "replica-validation-failed",
                "invalid Database Replica changed while it was opened for repair",
            )
        try:
            validate_immutable_database_replica(
                replica_descriptor,
                ownership.identity.cache_path / "replicas" / identity,
                database_set=runspec.payload.database.database_set,
                source_manifest_sha256=identity,
                expected_device=os.fstat(cache_descriptor).st_dev,
                expected_identity=FilesystemObjectIdentity(device=visible.st_dev, inode=visible.st_ino),
            )
        except DatabasePlacementError as exc:
            exclusive_validation_error = str(exc)[:2048]
        else:
            raise ClassifiedDatabaseReplicaError(
                "replica-validation-failed",
                "Database Replica became valid before repair",
            )
        verify_cache_ownership(cache_descriptor, ownership)
        try:
            rebound = os.stat(identity, dir_fd=replicas_descriptor, follow_symlinks=False)
        except OSError as exc:
            raise ClassifiedDatabaseReplicaError(
                "replica-validation-failed",
                "cannot observe invalid Database Replica before retirement",
            ) from exc
        if (
            _directory_authority(rebound, cache_descriptor=cache_descriptor) != expected
            or _directory_authority(os.fstat(replica_descriptor), cache_descriptor=cache_descriptor) != expected
        ):
            raise ClassifiedDatabaseReplicaError(
                "replica-validation-failed",
                "invalid Database Replica changed immediately before retirement",
            )
        retired_name = f".population-{identity}-{uuid.uuid4().hex}"
        publish_replica_noreplace(
            require_renameat2(),
            PopulationWorkspace(
                replicas_descriptor=replicas_descriptor,
                temporary_descriptor=replica_descriptor,
                temporary_name=identity,
            ),
            destination_name=retired_name,
        )
        verify_cache_ownership(cache_descriptor, ownership)
        try:
            retired = os.stat(retired_name, dir_fd=replicas_descriptor, follow_symlinks=False)
        except OSError as exc:
            raise ClassifiedDatabaseReplicaError(
                "replica-validation-failed",
                "retired Database Replica cannot be rebound after retirement",
            ) from exc
        if (
            _directory_authority(retired, cache_descriptor=cache_descriptor) != expected
            or _directory_authority(os.fstat(replica_descriptor), cache_descriptor=cache_descriptor) != expected
        ):
            raise ClassifiedDatabaseReplicaError(
                "replica-validation-failed",
                "retired Database Replica changed immediately after retirement",
            )
        try:
            os.stat(identity, dir_fd=replicas_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise ClassifiedDatabaseReplicaError(
                "publication-failed",
                "cannot confirm retired Database Replica final absence",
            ) from exc
        else:
            raise ClassifiedDatabaseReplicaError(
                "publication-failed",
                "retired Database Replica remained visible at its final basename",
            )
        authority = _repair_authority(runspec, action_id=action_id)
        publish_database_replica_invalidation(
            DatabaseReplicaInvalidationEvidence(
                authority=authority,
                original_replica_device=expected.device,
                original_replica_inode=expected.inode,
                validation_error=exclusive_validation_error,
                retired_population_basename=retired_name,
            ),
            result_path.parent / "database-replica-invalidation.json",
        )
        _cleanup_same_identity_populations(
            runspec,
            action_id=action_id,
            cache_descriptor=cache_descriptor,
            ownership=ownership,
            result_path=result_path,
            replicas_descriptor=replicas_descriptor,
            expected_bindings={
                retired_name: _ExpectedDirectoryBinding(
                    authority=expected,
                    descriptor=replica_descriptor,
                )
            },
            owned_tree_remover=owned_tree_remover,
        )
    finally:
        if replica_descriptor is not None:
            os.close(replica_descriptor)
        os.close(replicas_descriptor)


def cleanup_stale_populations(
    runspec: PhaseRunSpec,
    *,
    action_id: str,
    cache_descriptor: int,
    ownership: CacheExclusiveOwnership,
    result_path: Path,
) -> None:
    """Remove exact same-identity ownerless temps during a genuine fresh path."""
    _cleanup_stale_populations(
        runspec,
        action_id=action_id,
        cache_descriptor=cache_descriptor,
        ownership=ownership,
        result_path=result_path,
        owned_tree_remover=OwnedTreeRemover(remove=_remove_owned_tree),
    )


def _cleanup_stale_populations(
    runspec: PhaseRunSpec,
    *,
    action_id: str,
    cache_descriptor: int,
    ownership: CacheExclusiveOwnership,
    result_path: Path,
    owned_tree_remover: OwnedTreeRemover,
) -> None:
    """Private stale-population coordinator with explicit tree removal."""
    verify_cache_ownership(cache_descriptor, ownership)
    replicas_descriptor = _open_private_replicas(cache_descriptor, create=True)
    try:
        _cleanup_same_identity_populations(
            runspec,
            action_id=action_id,
            cache_descriptor=cache_descriptor,
            ownership=ownership,
            result_path=result_path,
            replicas_descriptor=replicas_descriptor,
            owned_tree_remover=owned_tree_remover,
        )
    finally:
        os.close(replicas_descriptor)


def _cleanup_same_identity_populations(
    runspec: PhaseRunSpec,
    *,
    action_id: str,
    cache_descriptor: int,
    ownership: CacheExclusiveOwnership,
    result_path: Path,
    replicas_descriptor: int,
    owned_tree_remover: OwnedTreeRemover,
    expected_bindings: dict[str, _ExpectedDirectoryBinding] | None = None,
) -> None:
    verify_cache_ownership(cache_descriptor, ownership)
    identity = runspec.payload.database.source_manifest_sha256
    grammar = re.compile(rf"\.population-{re.escape(identity)}-[0-9a-f]{{32}}")
    try:
        names = tuple(sorted(name for name in os.listdir(replicas_descriptor) if grammar.fullmatch(name)))
    except OSError as exc:
        raise ClassifiedDatabaseReplicaError(
            "replica-validation-failed",
            "cannot enumerate same-identity stale Database Replica populations",
        ) from exc
    if not names and not expected_bindings:
        return
    removed: list[str] = []
    failure: str | None = None
    try:
        missing_expected = set(expected_bindings or ()) - set(names)
        if missing_expected:
            raise ClassifiedDatabaseReplicaError(
                "replica-validation-failed",
                "retired Database Replica binding disappeared before cleanup",
            )
        for name in names:
            verify_cache_ownership(cache_descriptor, ownership)
            expected_binding = (expected_bindings or {}).get(name)
            if expected_binding is None:
                owned_tree_remover.remove(
                    replicas_descriptor,
                    name,
                    cache_device=os.fstat(cache_descriptor).st_dev,
                    root_claim_prefix=f".population-{identity}-",
                )
            else:
                owned_tree_remover.remove(
                    replicas_descriptor,
                    name,
                    cache_device=os.fstat(cache_descriptor).st_dev,
                    expected_binding=expected_binding,
                    root_claim_prefix=f".population-{identity}-",
                )
            removed.append(name)
        try:
            os.fsync(replicas_descriptor)
        except OSError as exc:
            raise ClassifiedDatabaseReplicaError(
                "replica-validation-failed",
                "cannot durably confirm same-identity stale Database Replica cleanup",
            ) from exc
    except DatabasePlacementError as exc:
        failure = str(exc)[:2048]
    removed_tuple = tuple(removed)
    publish_database_replica_cleanup(
        DatabaseReplicaCleanupEvidence(
            authority=_repair_authority(runspec, action_id=action_id),
            removed_count=len(removed_tuple),
            removed_basenames_sha256=removed_basenames_digest(removed_tuple),
            removed_basename_samples=removed_tuple[:32],
            omitted_count=max(0, len(removed_tuple) - 32),
            failure=failure,
        ),
        result_path.parent / "database-replica-stale-population-cleanup.json",
    )
    if failure is not None:
        raise ClassifiedDatabaseReplicaError(
            "replica-validation-failed",
            f"same-identity stale Database Replica cleanup failed: {failure}",
        )


def _remove_owned_tree(
    parent_descriptor: int,
    name: str,
    *,
    cache_device: int,
    expected_binding: _ExpectedDirectoryBinding | None = None,
    root_claim_prefix: str | None = None,
) -> None:
    _shared_remove_owned_tree(
        parent_descriptor,
        name,
        cache_device=cache_device,
        expected_binding=expected_binding,
        root_claim_prefix=root_claim_prefix,
        identity_validator=_owned_entry_identity,
        renameat2_factory=require_renameat2,
    )


def _open_private_replicas(cache_descriptor: int, *, create: bool = False) -> int:
    if create:
        try:
            os.mkdir("replicas", 0o700, dir_fd=cache_descriptor)
        except FileExistsError:
            pass
        except OSError as exc:
            raise ClassifiedDatabaseReplicaError(
                "cache-authority-invalid",
                "cannot prepare private Database Replica directory for stale cleanup",
            ) from exc
    try:
        info = os.stat("replicas", dir_fd=cache_descriptor, follow_symlinks=False)
        descriptor = os.open(
            "replicas",
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=cache_descriptor,
        )
    except OSError as exc:
        raise ClassifiedDatabaseReplicaError(
            "cache-authority-invalid",
            "private Database Replica directory is unavailable during repair",
        ) from exc
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) != 0o700
        or filesystem_object_identity(info) != filesystem_object_identity(opened)
        or info.st_dev != os.fstat(cache_descriptor).st_dev
    ):
        os.close(descriptor)
        raise ClassifiedDatabaseReplicaError(
            "cache-authority-invalid",
            "private Database Replica directory authority is invalid during repair",
        )
    return descriptor


def _directory_authority(info: os.stat_result, *, cache_descriptor: int) -> FilesystemAuthority:
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or info.st_dev != os.fstat(cache_descriptor).st_dev
    ):
        raise ClassifiedDatabaseReplicaError(
            "replica-validation-failed",
            "invalid Database Replica authority is unsafe for repair",
        )
    return filesystem_authority(info)


def _owned_entry_identity(
    info: os.stat_result,
    *,
    cache_device: int,
    require_directory: bool,
    name: str,
) -> FilesystemAuthority:
    is_expected_type = stat.S_ISDIR(info.st_mode) if require_directory else not stat.S_ISDIR(info.st_mode)
    if not is_expected_type or info.st_uid != os.geteuid() or info.st_dev != cache_device:
        raise ClassifiedDatabaseReplicaError(
            "replica-validation-failed",
            f"same-identity stale population contains unsafe authority: {name!r}",
        )
    return filesystem_authority(info)


def _repair_authority(runspec: PhaseRunSpec, *, action_id: str) -> RepairActionAuthority:
    database_set = runspec.payload.database.database_set
    return RepairActionAuthority(
        phase_run_id=runspec.phase_run_id,
        attempt_id=runspec.attempt_id,
        phase_runspec_digest=runspec.digest,
        action_id=action_id,
        database_set_identifier=database_set.identifier,
        database_set_version=database_set.version,
        source_manifest_sha256=runspec.payload.database.source_manifest_sha256,
    )


__all__ = ["cleanup_stale_populations", "retire_invalid_replica"]
