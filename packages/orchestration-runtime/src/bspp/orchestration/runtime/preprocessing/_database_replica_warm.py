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

"""Read-only warm Database Replica placement under a shared identity lease."""

from __future__ import annotations

import os
import stat
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from bspp.orchestration.contract.database_placement import (
    DATABASE_CACHE_ROOT,
    SELECTED_DATABASE_ROOT,
    DatabasePlacementOutcomeKind,
)
from bspp.orchestration.contract.database_replica import (
    DatabaseReplicaColdFailureClassification,
    DatabaseReplicaColdFailureEvidence,
    DatabaseReplicaWarmResult,
    DatabaseWarmCacheMountFacts,
)
from bspp.orchestration.contract.phase import PhaseRunSpec

from ._database_placement_errors import DatabasePlacementError
from ._database_placement_evidence_io import load_staged_manifest
from ._database_placement_paths import (
    PRODUCTION_DATABASE_PLACEMENT_PATHS,
    DatabasePlacementPaths,
)
from ._database_replica_authority import require_staged_authority
from ._database_replica_errors import ClassifiedDatabaseReplicaError
from ._database_replica_evidence_io import (
    load_database_replica_warm_result,
    publish_database_replica_cold_failure,
    publish_database_replica_warm_result,
)
from ._database_replica_lock_authority import (
    acquire_existing_shared_identity_lock,
    verify_identity_ownership,
    verify_shared_identity_ownership,
)
from ._database_replica_lock_types import FilesystemAuthority, IdentityExclusiveOwnership, LockContendedError
from ._database_replica_publication import open_cache_root, verify_cache_root_binding
from ._database_replica_validation import validate_immutable_database_replica
from ._database_source_mount import observe_warm_cache_mount
from ._filesystem_authority import FilesystemObjectIdentity, filesystem_object_identity


@dataclass(frozen=True)
class WarmProbeHit:
    result: DatabaseReplicaWarmResult


@dataclass(frozen=True)
class WarmProbeMiss:
    pass


@dataclass(frozen=True)
class WarmProbeInvalid:
    validation_error: str


@dataclass(frozen=True)
class WarmIdentityContended:
    pass


WarmProbeDisposition = WarmProbeHit | WarmProbeMiss | WarmProbeInvalid | WarmIdentityContended


@dataclass(frozen=True)
class WarmCandidateValid:
    result: DatabaseReplicaWarmResult


@dataclass(frozen=True)
class WarmCandidateAbsent:
    pass


@dataclass(frozen=True)
class WarmCandidateInvalid:
    validation_error: str


WarmCandidateDisposition = WarmCandidateValid | WarmCandidateAbsent | WarmCandidateInvalid


class _InvalidReplicaCandidateError(DatabasePlacementError):
    """Private signal for invalid candidate content under a held identity lease."""


class WarmResultAlreadyVisibleError(ClassifiedDatabaseReplicaError):
    """Carry an exact visible Result through a late publication ambiguity."""

    def __init__(
        self,
        result: DatabaseReplicaWarmResult,
        message: str,
        *,
        classification: DatabaseReplicaColdFailureClassification = "result-publication-failed",
    ) -> None:
        super().__init__(classification, message)
        self.result = result


def try_place_warm_stage_required_database(
    runspec: PhaseRunSpec,
    *,
    action_id: str,
    source_manifest_path: Path,
    result_path: Path,
    failure_path: Path | None = None,
) -> WarmProbeDisposition:
    """Return one closed hit, miss, or identity-contention probe disposition."""
    return _try_place_warm_stage_required_database(
        runspec,
        action_id=action_id,
        source_manifest_path=source_manifest_path,
        result_path=result_path,
        failure_path=failure_path,
        paths=PRODUCTION_DATABASE_PLACEMENT_PATHS,
    )


def _try_place_warm_stage_required_database(
    runspec: PhaseRunSpec,
    *,
    action_id: str,
    source_manifest_path: Path,
    result_path: Path,
    failure_path: Path | None,
    paths: DatabasePlacementPaths,
) -> WarmProbeDisposition:
    """Probe warm placement against an explicitly composed physical site."""
    action = runspec.payload.actions[0]
    classification: DatabaseReplicaColdFailureClassification = "authority-invalid"
    failure_published = False
    result: DatabaseReplicaWarmResult | None = None
    cache_mount: DatabaseWarmCacheMountFacts | None = None
    try:
        if action_id != action.action_id:
            raise DatabasePlacementError(
                f"action id {action_id!r} does not match the sole declared action {action.action_id!r}"
            )
        staging = require_staged_authority(runspec)
        classification = "source-manifest-invalid"
        load_staged_manifest(runspec, source_manifest_path)
        classification = "cache-authority-invalid"
        with ExitStack() as stack:
            cache_descriptor, cache_identity = open_cache_root(paths.cache_root)
            stack.callback(os.close, cache_descriptor)
            if os.fstat(cache_descriptor).st_uid != os.geteuid():
                raise DatabasePlacementError("protected Database Replica cache root has the wrong owner")
            cache_mount = observe_warm_cache_mount(
                cache_descriptor,
                paths.mountinfo,
                expected_filesystem_type=staging.expected_filesystem_type,
            )
            verify_cache_root_binding(paths.cache_root, cache_descriptor, cache_identity)
            replicas_descriptor = _open_existing_private_directory(cache_descriptor, "replicas", missing_ok=True)
            if replicas_descriptor is None:
                return WarmProbeMiss()
            stack.callback(os.close, replicas_descriptor)
            identity = runspec.payload.database.source_manifest_sha256
            if not _candidate_exists(replicas_descriptor, identity):
                return WarmProbeMiss()
            classification = "replica-validation-failed"
            try:
                locks_descriptor = _open_existing_private_directory(cache_descriptor, ".locks", missing_ok=False)
                assert locks_descriptor is not None
                stack.callback(os.close, locks_descriptor)
                try:
                    lease_descriptor = acquire_existing_shared_identity_lock(locks_descriptor, f"{identity}.lock")
                except LockContendedError:
                    return WarmIdentityContended()
                stack.callback(os.close, lease_descriptor)
                lock_name = f"{identity}.lock"
                verify_shared_identity_ownership(
                    paths.cache_root,
                    cache_descriptor,
                    cache_identity,
                    locks_descriptor,
                    lock_name,
                    lease_descriptor,
                )
                try:
                    candidate_info = _stat_candidate(replicas_descriptor, identity)
                    if candidate_info is None:
                        return WarmProbeMiss()
                    result = _validate_and_publish_candidate(
                        runspec,
                        action_id=action_id,
                        cache_descriptor=cache_descriptor,
                        cache_identity=cache_identity,
                        cache_mount=cache_mount,
                        replicas_descriptor=replicas_descriptor,
                        candidate_info=candidate_info,
                        result_path=result_path,
                        paths=paths,
                        lock_revalidator=lambda: verify_shared_identity_ownership(
                            paths.cache_root,
                            cache_descriptor,
                            cache_identity,
                            locks_descriptor,
                            lock_name,
                            lease_descriptor,
                        ),
                    )
                except _InvalidReplicaCandidateError as exc:
                    return WarmProbeInvalid(str(exc)[:2048])
                return WarmProbeHit(result)
            except DatabasePlacementError as exc:
                if isinstance(exc, WarmResultAlreadyVisibleError):
                    result = exc.result
                    failure_published = True
                if isinstance(exc, ClassifiedDatabaseReplicaError):
                    classification = exc.classification
                if failure_path is not None:
                    failure_published = True
                    if result is None or not _visible_warm_result_matches(result_path, result):
                        _publish_invalid_warm_failure(
                            runspec,
                            action_id=action_id,
                            failure_path=failure_path,
                            classification=classification,
                            error=str(exc),
                            cache_mount=cache_mount,
                        )
                raise
    except DatabasePlacementError as exc:
        if failure_path is not None and not failure_published:
            _publish_invalid_warm_failure(
                runspec,
                action_id=action_id,
                failure_path=failure_path,
                classification=classification,
                error=str(exc),
                cache_mount=cache_mount,
            )
        raise


def place_existing_warm_candidate_under_identity(
    runspec: PhaseRunSpec,
    *,
    ownership: IdentityExclusiveOwnership,
    action_id: str,
    cache_descriptor: int,
    cache_identity: FilesystemAuthority,
    cache_mount: DatabaseWarmCacheMountFacts,
    result_path: Path,
) -> WarmCandidateDisposition:
    """Return a closed valid, absent, or invalid disposition under identity EX."""
    return _place_existing_warm_candidate_under_identity(
        runspec,
        ownership=ownership,
        action_id=action_id,
        cache_descriptor=cache_descriptor,
        cache_identity=cache_identity,
        cache_mount=cache_mount,
        result_path=result_path,
        paths=PRODUCTION_DATABASE_PLACEMENT_PATHS,
    )


def _place_existing_warm_candidate_under_identity(
    runspec: PhaseRunSpec,
    *,
    ownership: IdentityExclusiveOwnership,
    action_id: str,
    cache_descriptor: int,
    cache_identity: FilesystemAuthority,
    cache_mount: DatabaseWarmCacheMountFacts,
    result_path: Path,
    paths: DatabasePlacementPaths,
) -> WarmCandidateDisposition:
    """Select one warm candidate under identity EX at an explicit site."""
    identity = runspec.payload.database.source_manifest_sha256
    if ownership.source_manifest_sha256 != identity:
        raise ClassifiedDatabaseReplicaError(
            "lock-unavailable",
            "Database Replica identity ownership does not match RunSpec authority",
        )
    with ExitStack() as stack:
        if os.fstat(cache_descriptor).st_uid != os.geteuid():
            raise DatabasePlacementError("protected Database Replica cache root has the wrong owner")
        verify_identity_ownership(cache_descriptor, ownership)
        verify_cache_root_binding(paths.cache_root, cache_descriptor, cache_identity)
        replicas_descriptor = _open_existing_private_directory(cache_descriptor, "replicas", missing_ok=True)
        if replicas_descriptor is None:
            return WarmCandidateAbsent()
        stack.callback(os.close, replicas_descriptor)
        try:
            candidate_info = _stat_candidate(replicas_descriptor, identity)
            if candidate_info is None:
                return WarmCandidateAbsent()
            os.fstat(ownership.descriptor)
            result = _validate_and_publish_candidate(
                runspec,
                action_id=action_id,
                cache_descriptor=cache_descriptor,
                cache_identity=cache_identity,
                cache_mount=cache_mount,
                replicas_descriptor=replicas_descriptor,
                candidate_info=candidate_info,
                result_path=result_path,
                paths=paths,
                lock_revalidator=lambda: verify_identity_ownership(cache_descriptor, ownership),
            )
        except _InvalidReplicaCandidateError as exc:
            return WarmCandidateInvalid(str(exc)[:2048])
        return WarmCandidateValid(result)


def _validate_and_publish_candidate(
    runspec: PhaseRunSpec,
    *,
    action_id: str,
    cache_descriptor: int,
    cache_identity: FilesystemAuthority,
    cache_mount: DatabaseWarmCacheMountFacts,
    replicas_descriptor: int,
    candidate_info: os.stat_result,
    result_path: Path,
    paths: DatabasePlacementPaths,
    lock_revalidator: Callable[[], None],
) -> DatabaseReplicaWarmResult:
    identity = runspec.payload.database.source_manifest_sha256
    lock_revalidator()
    replica_descriptor = _open_candidate(replicas_descriptor, identity, candidate_info)
    try:
        try:
            _, replica_manifest_sha256 = validate_immutable_database_replica(
                replica_descriptor,
                paths.cache_root / "replicas" / identity,
                database_set=runspec.payload.database.database_set,
                source_manifest_sha256=identity,
                expected_device=os.fstat(cache_descriptor).st_dev,
                expected_identity=filesystem_object_identity(candidate_info),
            )
        except DatabasePlacementError as exc:
            raise _InvalidReplicaCandidateError(str(exc)) from exc
        _verify_candidate_binding(
            replicas_descriptor,
            identity,
            replica_descriptor,
            expected_identity=filesystem_object_identity(candidate_info),
        )
        lock_revalidator()
        verify_cache_root_binding(paths.cache_root, cache_descriptor, cache_identity)
        staging = require_staged_authority(runspec)
        if (
            observe_warm_cache_mount(
                cache_descriptor,
                paths.mountinfo,
                expected_filesystem_type=staging.expected_filesystem_type,
            )
            != cache_mount
        ):
            raise DatabasePlacementError("protected Database Replica cache mount changed during warm reuse")
        result = _warm_result(
            runspec,
            action_id=action_id,
            cache_mount=cache_mount,
            replica_manifest_sha256=replica_manifest_sha256,
        )
        lock_revalidator()
        try:
            publish_database_replica_warm_result(result, result_path)
        except DatabasePlacementError as exc:
            if _visible_warm_result_matches(result_path, result):
                raise WarmResultAlreadyVisibleError(result, str(exc)) from exc
            raise ClassifiedDatabaseReplicaError("result-publication-failed", str(exc)) from exc
        try:
            lock_revalidator()
        except DatabasePlacementError as exc:
            if _visible_warm_result_matches(result_path, result):
                classification = (
                    exc.classification if isinstance(exc, ClassifiedDatabaseReplicaError) else "lock-unavailable"
                )
                raise WarmResultAlreadyVisibleError(
                    result,
                    str(exc),
                    classification=classification,
                ) from exc
            raise
        return result
    finally:
        os.close(replica_descriptor)


def _warm_result(
    runspec: PhaseRunSpec,
    *,
    action_id: str,
    cache_mount: DatabaseWarmCacheMountFacts,
    replica_manifest_sha256: str,
) -> DatabaseReplicaWarmResult:
    digest = runspec.payload.database.source_manifest_sha256
    return DatabaseReplicaWarmResult(
        phase_run_id=runspec.phase_run_id,
        attempt_id=runspec.attempt_id,
        phase_runspec_digest=runspec.digest,
        action_id=action_id,
        database_set=runspec.payload.database.database_set,
        requested_policy=runspec.payload.database.requested_policy,
        source_manifest_sha256=digest,
        branch_kind="staged",
        outcome=DatabasePlacementOutcomeKind.REPLICA_WARM,
        selected_container_root=SELECTED_DATABASE_ROOT,
        replica_container_root=f"{DATABASE_CACHE_ROOT}/replicas/{digest}",
        verification="metadata-verified",
        cache_mount=cache_mount,
        replica_manifest_sha256=replica_manifest_sha256,
    )


def _open_existing_private_directory(parent_descriptor: int, name: str, *, missing_ok: bool) -> int | None:
    try:
        info = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        if missing_ok:
            return None
        raise DatabasePlacementError(f"private cache directory is unavailable: {name!r}") from None
    except OSError as exc:
        raise DatabasePlacementError(f"private cache directory is unavailable: {name!r}") from exc
    descriptor: int | None = None
    succeeded = False
    try:
        descriptor = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_descriptor)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o700
            or filesystem_object_identity(info) != filesystem_object_identity(opened)
            or info.st_dev != os.fstat(parent_descriptor).st_dev
        ):
            raise DatabasePlacementError(f"private cache directory authority is invalid: {name!r}")
        succeeded = True
        return descriptor
    except OSError as exc:
        raise DatabasePlacementError(f"private cache directory is unavailable: {name!r}") from exc
    except DatabasePlacementError:
        raise
    finally:
        if descriptor is not None and not succeeded:
            os.close(descriptor)


def _stat_candidate(parent_descriptor: int, name: str) -> os.stat_result | None:
    try:
        info = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise DatabasePlacementError("manifest-addressed Database Replica is unavailable") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise _InvalidReplicaCandidateError("manifest-addressed Database Replica must be a real directory")
    return info


def _candidate_exists(parent_descriptor: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise DatabasePlacementError("manifest-addressed Database Replica is unavailable") from exc
    return True


def _open_candidate(parent_descriptor: int, name: str, expected: os.stat_result) -> int:
    try:
        descriptor = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_descriptor)
    except OSError as exc:
        raise DatabasePlacementError("manifest-addressed Database Replica is unavailable") from exc
    opened = os.fstat(descriptor)
    if filesystem_object_identity(expected) != filesystem_object_identity(opened):
        os.close(descriptor)
        raise DatabasePlacementError("manifest-addressed Database Replica changed while it was opened")
    return descriptor


def _verify_candidate_binding(
    parent_descriptor: int,
    name: str,
    candidate_descriptor: int,
    *,
    expected_identity: FilesystemObjectIdentity,
) -> None:
    try:
        visible = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        opened = os.fstat(candidate_descriptor)
    except OSError as exc:
        raise DatabasePlacementError("manifest-addressed Database Replica changed during warm validation") from exc
    if (
        not stat.S_ISDIR(visible.st_mode)
        or filesystem_object_identity(visible) != expected_identity
        or filesystem_object_identity(opened) != expected_identity
    ):
        raise DatabasePlacementError("manifest-addressed Database Replica changed during warm validation")


def _visible_warm_result_matches(path: Path, expected: DatabaseReplicaWarmResult) -> bool:
    try:
        return load_database_replica_warm_result(path) == expected
    except DatabasePlacementError:
        return False


def _publish_invalid_warm_failure(
    runspec: PhaseRunSpec,
    *,
    action_id: str,
    failure_path: Path,
    classification: DatabaseReplicaColdFailureClassification,
    error: str,
    cache_mount: DatabaseWarmCacheMountFacts | None,
) -> None:
    publish_database_replica_cold_failure(
        DatabaseReplicaColdFailureEvidence(
            phase_run_id=runspec.phase_run_id,
            attempt_id=runspec.attempt_id,
            phase_runspec_digest=runspec.digest,
            action_id=action_id,
            database_set=runspec.payload.database.database_set,
            requested_policy=runspec.payload.database.requested_policy,
            source_manifest_sha256=runspec.payload.database.source_manifest_sha256,
            source_mount=None,
            cache_mount=cache_mount,
            capacity_gate=None,
            science_started=False,
            classification=classification,
            error=error[:2048],
        ),
        failure_path,
    )


__all__ = [
    "WarmCandidateAbsent",
    "WarmCandidateDisposition",
    "WarmCandidateInvalid",
    "WarmCandidateValid",
    "WarmIdentityContended",
    "WarmProbeDisposition",
    "WarmProbeHit",
    "WarmProbeInvalid",
    "WarmProbeMiss",
    "WarmResultAlreadyVisibleError",
    "place_existing_warm_candidate_under_identity",
    "try_place_warm_stage_required_database",
]
