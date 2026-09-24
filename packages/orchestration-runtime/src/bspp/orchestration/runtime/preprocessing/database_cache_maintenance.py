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

"""Compatibility facade and coordinator for Database Acceptance Cache maintenance."""

from __future__ import annotations

import os
import pwd
import stat
from pathlib import Path

from bspp.orchestration.contract.database_cache_maintenance import (
    DatabaseAcceptanceCacheProfile,
    DatabaseCacheIdentityLockObservation,
    DatabaseCacheIdentityLockSummary,
    DatabaseCacheMaintenanceDescriptor,
    DatabaseCacheMaintenanceEvidence,
    DatabaseCacheRemovedEntry,
    database_cache_identity_lock_observations_digest,
    empty_database_cache_entries_digest,
    removed_database_cache_entries_digest,
)

from ._database_cache_authority import (
    _authorize_evidence_publication,
    _freeze_deletion_target_authority,
    _freeze_maintenance_lock_mount_authority,
    _verify_deletion_target_authority,
    _verify_maintenance_lock_mount_authority,
)
from ._database_cache_evidence import (
    _close_evidence_destination,
    _finish_evidence_destination,
    _reserve_evidence_destination,
    _verify_evidence_destination,
    publish_database_cache_maintenance_evidence,
)
from ._database_cache_maintenance_types import (
    DatabaseCacheEvidenceStore,
    DatabaseCacheMaintenanceError,
    DatabaseCacheMaintenancePaths,
    OwnedTreeRemover,
    _DeletionTargetAuthority,
    _Entry,
    _EvidenceDestination,
    _PhysicalCacheRootBinding,
    _Scope,
)
from ._database_cache_scope import (
    _bind_configured_sibling_roots,
    _entry_from_stat,
    _open_optional_locks,
    _open_optional_replicas,
    _open_scope,
    _require_filesystem_type,
    _scan_replicas,
    _verify_scope,
    load_database_acceptance_cache_authority,
    load_database_acceptance_cache_profile,
)
from ._database_replica_errors import ClassifiedDatabaseReplicaError
from ._database_replica_lock_authority import (
    verify_maintenance_cache_ownership,
    verify_maintenance_identity_ownership,
)
from ._database_replica_lock_coordination import (
    hold_exclusive_cache_for_maintenance,
    hold_exclusive_identities_for_maintenance,
)
from ._database_replica_lock_types import LockWait, MaintenanceIdentityContendedError
from ._filesystem_authority import (
    FilesystemAuthority,
    FilesystemObjectIdentity,
)
from ._owned_tree import ExpectedDirectoryBinding, remove_owned_tree


def clear_database_acceptance_cache(
    profile: DatabaseAcceptanceCacheProfile,
    *,
    evidence_path: Path,
    configured_sibling_cache_roots: tuple[Path, ...] = (),
) -> DatabaseCacheMaintenanceEvidence:
    """Clear exactly one effective-user acceptance namespace under held locks."""
    return _clear_database_acceptance_cache(
        profile,
        evidence_path=evidence_path,
        configured_sibling_cache_roots=configured_sibling_cache_roots,
        owned_tree_remover=OwnedTreeRemover(remove_owned_tree),
        paths=DatabaseCacheMaintenancePaths(mountinfo=Path("/proc/self/mountinfo")),
        evidence_store=DatabaseCacheEvidenceStore(
            reserve=_reserve_evidence_destination,
            authorize=_authorize_evidence_publication,
            verify=_verify_evidence_destination,
            finish=_finish_evidence_destination,
            close=_close_evidence_destination,
        ),
    )


def _clear_database_acceptance_cache(
    profile: DatabaseAcceptanceCacheProfile,
    *,
    evidence_path: Path,
    configured_sibling_cache_roots: tuple[Path, ...] = (),
    owned_tree_remover: OwnedTreeRemover,
    paths: DatabaseCacheMaintenancePaths,
    evidence_store: DatabaseCacheEvidenceStore,
) -> DatabaseCacheMaintenanceEvidence:
    """Run the state machine with an explicitly composed removal capability."""
    effective_uid = os.geteuid()
    try:
        effective_user = pwd.getpwuid(effective_uid).pw_name
    except KeyError as exc:
        raise DatabaseCacheMaintenanceError("effective UID has no exact Unix user identity") from exc
    user_namespace = Path(profile.cache_root) / "users" / effective_user
    replicas_path = user_namespace / "replicas"
    descriptors: tuple[DatabaseCacheMaintenanceDescriptor, ...] = ()
    cache_lock_acquired = False
    cache_lock_contended = False
    identity_observations: tuple[DatabaseCacheIdentityLockObservation, ...] = ()
    removed: list[DatabaseCacheRemovedEntry] = []
    scope: _Scope | None = None
    locks_descriptor: int | None = None
    replicas_descriptor: int | None = None
    sibling_root_bindings: tuple[_PhysicalCacheRootBinding, ...] = ()
    deletion_target_authority: _DeletionTargetAuthority | None = None
    evidence_destination: _EvidenceDestination | None = None
    mutation_started = False
    try:
        if _is_within(evidence_path, user_namespace):
            raise DatabaseCacheMaintenanceError("maintenance evidence destination must be outside the managed cache")
        scope = _open_scope(profile, effective_user=effective_user, effective_uid=effective_uid)
        descriptors = _scope_descriptors(scope)
        sibling_root_bindings = _bind_configured_sibling_roots(
            configured_sibling_cache_roots,
            mountinfo_path=paths.mountinfo,
        )
        locks_descriptor = _open_optional_locks(scope)
        replicas_descriptor = _open_optional_replicas(scope)
        if replicas_descriptor is not None:
            descriptors += (_descriptor("replicas", os.fstat(replicas_descriptor)),)
        deletion_target_authority = _freeze_deletion_target_authority(
            scope,
            sibling_root_bindings=sibling_root_bindings,
            locks_descriptor=locks_descriptor,
            replicas_descriptor=replicas_descriptor,
            mountinfo_path=paths.mountinfo,
        )
        evidence_destination = evidence_store.reserve(evidence_path)
        evidence_store.authorize(
            evidence_destination,
            deletion_target_authority,
        )
        _verify_deletion_target_authority(deletion_target_authority)
        if profile.unix_user != effective_user:
            raise DatabaseCacheMaintenanceError(
                "configured database cache Unix user does not match the effective Unix user"
            )
        _require_filesystem_type(
            scope.cache_root,
            scope.cache_root_descriptor,
            profile.expected_filesystem_type,
            mountinfo_path=paths.mountinfo,
        )
        if replicas_descriptor is None:
            _verify_scope(scope)
            _verify_deletion_target_authority(deletion_target_authority)
            evidence = _evidence(
                scope,
                descriptors=descriptors,
                cache_lock_acquired=False,
                cache_lock_contended=False,
                identity_observations=(),
                removed=(),
                terminal_result="already-empty",
                diagnostic=None,
            )
            evidence_store.finish(evidence_destination, evidence)
            return evidence
        initial = _scan_replicas(replicas_descriptor, cache_device=scope.user_identity.device)
        if not initial:
            _verify_scope(scope, replicas_descriptor=replicas_descriptor)
            _verify_deletion_target_authority(deletion_target_authority)
            evidence = _evidence(
                scope,
                descriptors=descriptors,
                cache_lock_acquired=False,
                cache_lock_contended=False,
                identity_observations=(),
                removed=(),
                terminal_result="already-empty",
                diagnostic=None,
            )
            evidence_store.finish(evidence_destination, evidence)
            return evidence
        if evidence_destination.existing is not None:
            raise DatabaseCacheMaintenanceError("existing maintenance evidence does not authorize a new cache mutation")
        if locks_descriptor is None:
            raise DatabaseCacheMaintenanceError(
                "nonempty acceptance database replicas require an existing lock directory"
            )
        wait = LockWait.for_timeout(profile.lock_wait_seconds)
        try:
            with hold_exclusive_cache_for_maintenance(
                scope.user_descriptor,
                cache_path=scope.user_namespace,
                cache_identity=FilesystemAuthority(
                    file_type=stat.S_IFDIR,
                    device=scope.user_identity.device,
                    inode=scope.user_identity.inode,
                    owner_uid=scope.user_identity.owner_uid,
                    permissions=scope.user_identity.permissions,
                ),
                wait=wait,
            ) as cache_ownership:
                cache_lock_acquired = True
                cache_lock_contended = cache_ownership.contended
                descriptors += (_descriptor("locks", os.fstat(cache_ownership.locks_descriptor)),)
                _verify_deletion_target_authority(deletion_target_authority)
                lock_mount_authority = _freeze_maintenance_lock_mount_authority(
                    deletion_target_authority,
                    cache_lock_descriptor=cache_ownership.descriptor,
                )
                locked_scan = _scan_replicas(replicas_descriptor, cache_device=scope.user_identity.device)
                if locked_scan != initial:
                    raise DatabaseCacheMaintenanceError("replica inventory changed while cache ownership was acquired")
                identities = tuple(sorted({item.identity for item in locked_scan}))
                try:
                    with hold_exclusive_identities_for_maintenance(
                        scope.user_descriptor, cache_ownership, identities
                    ) as identity_ownership:
                        identity_observations = tuple(
                            DatabaseCacheIdentityLockObservation(item, "acquired") for item in identities
                        )
                        lock_mount_authority = _freeze_maintenance_lock_mount_authority(
                            deletion_target_authority,
                            cache_lock_descriptor=cache_ownership.descriptor,
                            identities=identity_ownership.identities,
                            identity_lock_descriptors=identity_ownership.descriptors,
                        )
                        stable = _scan_replicas(replicas_descriptor, cache_device=scope.user_identity.device)
                        if stable != locked_scan:
                            raise DatabaseCacheMaintenanceError(
                                "replica inventory changed while identity ownership was acquired"
                            )
                        _verify_scope(scope, replicas_descriptor=replicas_descriptor)
                        verify_maintenance_identity_ownership(scope.user_descriptor, identity_ownership)
                        for item in stable:
                            evidence_store.verify(evidence_destination, require_empty=True)
                            _verify_deletion_target_authority(deletion_target_authority)
                            verify_maintenance_cache_ownership(scope.user_descriptor, cache_ownership)
                            verify_maintenance_identity_ownership(scope.user_descriptor, identity_ownership)
                            _verify_scope(scope, replicas_descriptor=replicas_descriptor)
                            rebound = _entry_from_stat(
                                item.basename,
                                os.stat(item.basename, dir_fd=replicas_descriptor, follow_symlinks=False),
                                cache_device=scope.user_identity.device,
                            )
                            if rebound != item:
                                raise DatabaseCacheMaintenanceError(
                                    f"replica entry changed immediately before removal: {item.basename!r}"
                                )
                            _verify_maintenance_lock_mount_authority(
                                deletion_target_authority,
                                lock_mount_authority,
                            )
                            mutation_started = True
                            try:
                                owned_tree_remover.remove(
                                    replicas_descriptor,
                                    item.basename,
                                    cache_device=scope.user_identity.device,
                                    expected_binding=ExpectedDirectoryBinding(item.authority),
                                    root_claim_prefix=f".population-{item.identity}-",
                                )
                            except BaseException as exc:
                                if _entry_absent_from_stable_inventory(
                                    replicas_descriptor,
                                    item,
                                    cache_device=scope.user_identity.device,
                                ):
                                    removed.append(item.evidence())
                                if isinstance(exc, Exception):
                                    raise DatabaseCacheMaintenanceError("database cache entry removal failed") from exc
                                raise
                            removed.append(item.evidence())
                        os.fsync(replicas_descriptor)
                        if _scan_replicas(replicas_descriptor, cache_device=scope.user_identity.device):
                            raise DatabaseCacheMaintenanceError("replica inventory remained after maintenance")
                        _verify_scope(scope, replicas_descriptor=replicas_descriptor)
                        verify_maintenance_identity_ownership(scope.user_descriptor, identity_ownership)
                        _verify_maintenance_lock_mount_authority(
                            deletion_target_authority,
                            lock_mount_authority,
                        )
                except MaintenanceIdentityContendedError as exc:
                    identity_observations = (
                        *(DatabaseCacheIdentityLockObservation(item, "acquired") for item in exc.acquired_identities),
                        DatabaseCacheIdentityLockObservation(exc.source_manifest_sha256, "contended"),
                    )
                    raise DatabaseCacheMaintenanceError(str(exc)) from exc
        except ClassifiedDatabaseReplicaError as exc:
            if "timed out" in str(exc):
                cache_lock_contended = True
            raise DatabaseCacheMaintenanceError(str(exc)) from exc
        evidence = _evidence(
            scope,
            descriptors=descriptors,
            cache_lock_acquired=cache_lock_acquired,
            cache_lock_contended=cache_lock_contended,
            identity_observations=identity_observations,
            removed=tuple(removed),
            terminal_result="cleared",
            diagnostic=None,
        )
        evidence_store.finish(evidence_destination, evidence)
        return evidence
    except BaseException as exc:
        diagnostic = type(exc).__name__ if not isinstance(exc, Exception) else str(exc)[:2048] or type(exc).__name__
        terminal = "failed" if mutation_started else "refused"
        if (
            scope is not None
            and evidence_destination is not None
            and evidence_destination.descriptor is not None
            and evidence_destination.publication_authority is not None
            and not evidence_destination.finalization_attempted
        ):
            failure = _evidence(
                scope,
                descriptors=descriptors,
                cache_lock_acquired=cache_lock_acquired,
                cache_lock_contended=cache_lock_contended,
                identity_observations=identity_observations,
                removed=tuple(removed),
                terminal_result=terminal,
                diagnostic=diagnostic,
            )
            evidence_store.finish(evidence_destination, failure)
        elif (
            evidence_destination is not None
            and evidence_destination.descriptor is not None
            and evidence_destination.publication_authority is not None
            and not evidence_destination.finalization_attempted
            and not _is_within(evidence_path, user_namespace)
        ):
            refusal = DatabaseCacheMaintenanceEvidence(
                profile_name=profile.profile_name,
                profile_digest=profile.digest,
                effective_unix_user=effective_user,
                effective_uid=effective_uid,
                cache_root=profile.cache_root,
                user_namespace=str(user_namespace),
                replicas_scope=str(replicas_path),
                descriptors=(),
                cache_lock_acquired=False,
                cache_lock_contended=False,
                identity_locks=_identity_lock_summary(()),
                removed_count=0,
                removed_entries_sha256=empty_database_cache_entries_digest(),
                removed_entry_samples=(),
                omitted_count=0,
                terminal_result="refused",
                diagnostic=diagnostic,
            )
            evidence_store.finish(evidence_destination, refusal)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        if isinstance(exc, DatabaseCacheMaintenanceError):
            raise
        raise DatabaseCacheMaintenanceError(diagnostic) from exc
    finally:
        if locks_descriptor is not None:
            os.close(locks_descriptor)
        if replicas_descriptor is not None:
            os.close(replicas_descriptor)
        for binding in sibling_root_bindings:
            if binding.descriptor is not None:
                os.close(binding.descriptor)
        if scope is not None:
            os.close(scope.user_descriptor)
            os.close(scope.users_descriptor)
            os.close(scope.cache_root_descriptor)
        if evidence_destination is not None:
            evidence_store.close(evidence_destination)


def _scope_descriptors(scope: _Scope) -> tuple[DatabaseCacheMaintenanceDescriptor, ...]:
    return (
        _descriptor("cache-root", os.fstat(scope.cache_root_descriptor)),
        _descriptor("users-root", os.fstat(scope.users_descriptor)),
        _descriptor("user-namespace", os.fstat(scope.user_descriptor)),
    )


def _descriptor(role: str, info: os.stat_result) -> DatabaseCacheMaintenanceDescriptor:
    return DatabaseCacheMaintenanceDescriptor(
        role=role,  # type: ignore[arg-type]
        device=info.st_dev,
        inode=info.st_ino,
        uid=info.st_uid,
        mode=stat.S_IMODE(info.st_mode),
    )


def _evidence(
    scope: _Scope,
    *,
    descriptors: tuple[DatabaseCacheMaintenanceDescriptor, ...],
    cache_lock_acquired: bool,
    cache_lock_contended: bool,
    identity_observations: tuple[DatabaseCacheIdentityLockObservation, ...],
    removed: tuple[DatabaseCacheRemovedEntry, ...],
    terminal_result: str,
    diagnostic: str | None,
) -> DatabaseCacheMaintenanceEvidence:
    return DatabaseCacheMaintenanceEvidence(
        profile_name=scope.profile.profile_name,
        profile_digest=scope.profile.digest,
        effective_unix_user=scope.effective_user,
        effective_uid=scope.effective_uid,
        cache_root=str(scope.cache_root),
        user_namespace=str(scope.user_namespace),
        replicas_scope=str(scope.replicas_path),
        descriptors=descriptors,
        cache_lock_acquired=cache_lock_acquired,
        cache_lock_contended=cache_lock_contended,
        identity_locks=_identity_lock_summary(identity_observations),
        removed_count=len(removed),
        removed_entries_sha256=removed_database_cache_entries_digest(removed),
        removed_entry_samples=removed[:32],
        omitted_count=max(0, len(removed) - 32),
        terminal_result=terminal_result,  # type: ignore[arg-type]
        diagnostic=diagnostic,
    )


def _identity_lock_summary(
    observations: tuple[DatabaseCacheIdentityLockObservation, ...],
) -> DatabaseCacheIdentityLockSummary:
    acquired_count = sum(item.observation == "acquired" for item in observations)
    contended_count = sum(item.observation == "contended" for item in observations)
    return DatabaseCacheIdentityLockSummary(
        total_count=len(observations),
        acquired_count=acquired_count,
        contended_count=contended_count,
        observations_sha256=database_cache_identity_lock_observations_digest(observations),
        samples=observations[:32],
        omitted_count=max(0, len(observations) - 32),
    )


def _entry_absent_from_stable_inventory(
    replicas_descriptor: int,
    item: _Entry,
    *,
    cache_device: int,
) -> bool:
    """Prove one pre-removal inode absent from two exact accepted inventories."""
    try:
        first = _scan_replicas(replicas_descriptor, cache_device=cache_device)
        second = _scan_replicas(replicas_descriptor, cache_device=cache_device)
    except DatabaseCacheMaintenanceError:
        return False
    if first != second:
        return False
    identity = FilesystemObjectIdentity(device=item.authority.device, inode=item.authority.inode)
    return all(
        FilesystemObjectIdentity(device=candidate.authority.device, inode=candidate.authority.inode) != identity
        for candidate in first
    )


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.absolute().relative_to(parent.absolute())
    except ValueError:
        return False
    return True


__all__ = [
    "DatabaseCacheMaintenanceError",
    "clear_database_acceptance_cache",
    "load_database_acceptance_cache_authority",
    "load_database_acceptance_cache_profile",
    "publish_database_cache_maintenance_evidence",
]
