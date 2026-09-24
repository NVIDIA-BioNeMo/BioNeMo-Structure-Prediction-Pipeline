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

"""Cold stage-required Database Replica population orchestration."""

from __future__ import annotations

import os
from pathlib import Path

from bspp.orchestration.contract.database_capacity_fallback_result import DatabaseCapacityFallbackResult
from bspp.orchestration.contract.database_placement import (
    DATABASE_CACHE_ROOT,
    SELECTED_DATABASE_ROOT,
    DatabaseAccessPolicy,
    DatabasePlacementOutcomeKind,
)
from bspp.orchestration.contract.database_placement_result import DatabaseSourceMountFacts
from bspp.orchestration.contract.database_replica import (
    DatabaseCacheMountFacts,
    DatabaseCapacityGate,
    DatabaseReplicaColdFailureClassification,
    DatabaseReplicaColdFailureEvidence,
    DatabaseReplicaColdResult,
    DatabaseReplicaManifest,
    DatabaseReplicaResult,
    database_replica_manifest_digest,
)
from bspp.orchestration.contract.phase import PhaseRunSpec

from ._database_placement_errors import DatabasePlacementError
from ._database_placement_evidence_io import (
    load_database_direct_result,
    load_staged_manifest,
    publish_database_direct_result,
)
from ._database_placement_paths import (
    PRODUCTION_DATABASE_PLACEMENT_PATHS,
    DatabasePlacementPaths,
)
from ._database_replica_authority import require_staged_authority
from ._database_replica_copy import allocated_replica_bytes, populate_replica_payload
from ._database_replica_errors import ClassifiedDatabaseReplicaError
from ._database_replica_evidence_io import (
    load_database_replica_result,
    publish_database_replica_cold_failure,
    publish_database_replica_cold_result,
)
from ._database_replica_lock_authority import verify_cache_ownership
from ._database_replica_lock_coordination import hold_exclusive_cache, hold_exclusive_identity
from ._database_replica_lock_types import LockWait
from ._database_replica_publication import (
    exclusive_population_workspace,
    make_replica_immutable,
    open_cache_root,
    publish_replica_noreplace,
    require_renameat2,
    verify_cache_root_binding,
    verify_published_replica,
)
from ._database_replica_repair import cleanup_stale_populations, retire_invalid_replica
from ._database_replica_services import (
    ColdReplicaServices,
    DatabaseCapacityObserver,
    ReplicaPopulationService,
    ReplicaRepairService,
    WarmReplicaSelector,
)
from ._database_replica_warm import (
    WarmCandidateAbsent,
    WarmCandidateInvalid,
    WarmCandidateValid,
    WarmResultAlreadyVisibleError,
    _place_existing_warm_candidate_under_identity,
    _try_place_warm_stage_required_database,
)
from ._database_source_mount import observe_cache_mount, observe_source_mount, observe_warm_cache_mount
from ._database_source_observation import (
    observe_source_inventory,
    open_source_root,
    verify_source_root_binding,
)


def place_cold_stage_required_database(
    runspec: PhaseRunSpec,
    *,
    action_id: str,
    source_manifest_path: Path,
    result_path: Path,
    failure_path: Path | None = None,
    wait: LockWait,
    preprobe_contended: bool,
) -> DatabaseReplicaResult | DatabaseCapacityFallbackResult:
    """Coordinate winner reuse or one cold stage-required population."""
    return _place_cold_stage_required_database(
        runspec,
        action_id=action_id,
        source_manifest_path=source_manifest_path,
        result_path=result_path,
        failure_path=failure_path,
        wait=wait,
        preprobe_contended=preprobe_contended,
        paths=PRODUCTION_DATABASE_PLACEMENT_PATHS,
        services=_production_cold_replica_services(),
    )


def _place_cold_stage_required_database(
    runspec: PhaseRunSpec,
    *,
    action_id: str,
    source_manifest_path: Path,
    result_path: Path,
    failure_path: Path | None,
    wait: LockWait,
    preprobe_contended: bool,
    paths: DatabasePlacementPaths,
    services: ColdReplicaServices,
) -> DatabaseReplicaResult | DatabaseCapacityFallbackResult:
    """Coordinate cold placement against an explicitly composed physical site."""
    action = runspec.payload.actions[0]
    source_mount: DatabaseSourceMountFacts | None = None
    cache_mount: DatabaseCacheMountFacts | None = None
    capacity: DatabaseCapacityGate | None = None
    result: DatabaseReplicaResult | DatabaseCapacityFallbackResult | None = None
    source_descriptor: int | None = None
    cache_descriptor: int | None = None
    classification: DatabaseReplicaColdFailureClassification = "authority-invalid"
    try:
        if action_id != action.action_id:
            raise DatabasePlacementError(
                f"action id {action_id!r} does not match the sole declared action {action.action_id!r}"
            )
        staging = require_staged_authority(runspec)
        allocated_cpus = _allocated_cpus(action.resources.cpus_per_task)
        classification = "source-manifest-invalid"
        manifest = load_staged_manifest(runspec, source_manifest_path)
        classification = "source-mount-invalid"
        source_descriptor, source_identity = open_source_root(paths.source_root)
        source_mount = observe_source_mount(source_descriptor, paths.mountinfo)
        classification = "source-inventory-invalid"
        pre_copy = observe_source_inventory(paths.source_root, source_descriptor, manifest)
        verify_source_root_binding(paths.source_root, source_descriptor, source_identity)
        classification = "cache-authority-invalid"
        cache_descriptor, cache_identity = open_cache_root(paths.cache_root)
        cache_mount = observe_cache_mount(
            cache_descriptor,
            paths.mountinfo,
            expected_filesystem_type=staging.expected_filesystem_type,
        )
        warm_cache_mount = observe_warm_cache_mount(
            cache_descriptor,
            paths.mountinfo,
            expected_filesystem_type=staging.expected_filesystem_type,
        )
        _require_distinct_mounts(source_mount, cache_mount)
        classification = "lock-unavailable"
        identity_value = runspec.payload.database.source_manifest_sha256
        with hold_exclusive_identity(
            cache_descriptor,
            cache_path=paths.cache_root,
            cache_identity=cache_identity,
            source_manifest_sha256=identity_value,
            wait=wait,
        ) as identity:
            classification = "replica-validation-failed"
            candidate = services.warm_selector.select_under_identity(
                runspec,
                ownership=identity,
                action_id=action_id,
                cache_descriptor=cache_descriptor,
                cache_identity=cache_identity,
                cache_mount=warm_cache_mount,
                result_path=result_path,
                paths=paths,
            )
            if isinstance(candidate, WarmCandidateValid):
                result = candidate.result
                return candidate.result
            initially_invalid = isinstance(candidate, WarmCandidateInvalid)
            if isinstance(candidate, WarmCandidateAbsent) and (preprobe_contended or identity.contended):
                raise ClassifiedDatabaseReplicaError(
                    "lock-unavailable",
                    "Database Replica identity contender released without a valid publication",
                )
            classification = "lock-unavailable"
            with hold_exclusive_cache(identity, wait=wait) as cache_ownership:
                classification = "replica-validation-failed"
                candidate = services.warm_selector.select_under_identity(
                    runspec,
                    ownership=identity,
                    action_id=action_id,
                    cache_descriptor=cache_descriptor,
                    cache_identity=cache_identity,
                    cache_mount=warm_cache_mount,
                    result_path=result_path,
                    paths=paths,
                )
                if isinstance(candidate, WarmCandidateValid):
                    result = candidate.result
                    return candidate.result
                if isinstance(candidate, WarmCandidateInvalid):
                    classification = "replica-validation-failed"
                    services.repair.retire_invalid(
                        runspec,
                        action_id=action_id,
                        cache_descriptor=cache_descriptor,
                        ownership=cache_ownership,
                        result_path=result_path,
                    )
                elif isinstance(candidate, WarmCandidateAbsent):
                    if initially_invalid:
                        raise ClassifiedDatabaseReplicaError(
                            "replica-validation-failed",
                            "invalid Database Replica disappeared before exclusive repair",
                        )
                    services.repair.cleanup_stale(
                        runspec,
                        action_id=action_id,
                        cache_descriptor=cache_descriptor,
                        ownership=cache_ownership,
                        result_path=result_path,
                    )
                else:
                    raise AssertionError("unhandled Database Replica candidate disposition")
                verify_cache_ownership(cache_descriptor, cache_ownership)
                classification = "publication-failed"
                renameat2 = require_renameat2()
                classification = "capacity-observation-failed"
                capacity = services.capacity.observe(
                    cache_descriptor,
                    allocated_bytes=allocated_replica_bytes(pre_copy),
                    reserved_bytes=staging.reserve_bytes,
                )
                classification = "insufficient-capacity"
                if capacity.decision != "sufficient":
                    if runspec.payload.database.requested_policy == DatabaseAccessPolicy.STAGE_PREFERRED:
                        classification = "source-inventory-invalid"
                        fallback_observation = observe_source_inventory(
                            paths.source_root,
                            source_descriptor,
                            manifest,
                        )
                        verify_source_root_binding(paths.source_root, source_descriptor, source_identity)
                        if fallback_observation != pre_copy:
                            raise DatabasePlacementError("Database Source changed before capacity fallback publication")
                        classification = "source-mount-invalid"
                        if observe_source_mount(source_descriptor, paths.mountinfo) != source_mount:
                            raise DatabasePlacementError(
                                "protected Database Source mount changed before capacity fallback"
                            )
                        classification = "cache-authority-invalid"
                        if (
                            observe_cache_mount(
                                cache_descriptor,
                                paths.mountinfo,
                                expected_filesystem_type=staging.expected_filesystem_type,
                            )
                            != cache_mount
                        ):
                            raise DatabasePlacementError(
                                "protected Database Replica cache mount changed before capacity fallback"
                            )
                        verify_cache_root_binding(paths.cache_root, cache_descriptor, cache_identity)
                        classification = "lock-unavailable"
                        verify_cache_ownership(cache_descriptor, cache_ownership)
                        result = DatabaseCapacityFallbackResult(
                            phase_run_id=runspec.phase_run_id,
                            attempt_id=runspec.attempt_id,
                            phase_runspec_digest=runspec.digest,
                            action_id=action_id,
                            database_set=runspec.payload.database.database_set,
                            requested_policy=DatabaseAccessPolicy.STAGE_PREFERRED,
                            source_manifest_sha256=identity_value,
                            branch_kind="direct-capacity-fallback",
                            outcome=DatabasePlacementOutcomeKind.DIRECT_CAPACITY_FALLBACK,
                            selected_container_root=SELECTED_DATABASE_ROOT,
                            verification="metadata-verified",
                            source_mount=source_mount,
                            cache_mount=cache_mount,
                            capacity_gate=capacity,
                            pre_science_observation=fallback_observation,
                        )
                        classification = "result-publication-failed"
                        publish_database_direct_result(result, result_path)
                        classification = "lock-unavailable"
                        verify_cache_ownership(cache_descriptor, cache_ownership)
                        classification = "source-inventory-invalid"
                        verify_source_root_binding(paths.source_root, source_descriptor, source_identity)
                        classification = "cache-authority-invalid"
                        verify_cache_root_binding(paths.cache_root, cache_descriptor, cache_identity)
                        return result
                    raise DatabasePlacementError(
                        "insufficient user-available cache capacity for required cold Database Replica"
                    )
                classification = "lock-unavailable"
                with exclusive_population_workspace(
                    cache_descriptor,
                    ownership=cache_ownership,
                    source_manifest_sha256=identity_value,
                ) as workspace:
                    classification = "copy-failed"
                    copy_evidence, replica_members = services.population.populate(
                        source_descriptor=source_descriptor,
                        temporary_descriptor=workspace.temporary_descriptor,
                        observation=pre_copy,
                        allocated_cpus=allocated_cpus,
                        rsync_executable=paths.rsync,
                    )
                    classification = "source-inventory-invalid"
                    post_copy = observe_source_inventory(paths.source_root, source_descriptor, manifest)
                    verify_source_root_binding(paths.source_root, source_descriptor, source_identity)
                    if post_copy != pre_copy:
                        raise DatabasePlacementError("Database Source changed during cold replica population")
                    classification = "source-mount-invalid"
                    if observe_source_mount(source_descriptor, paths.mountinfo) != source_mount:
                        raise DatabasePlacementError("protected Database Source mount changed during population")
                    classification = "cache-authority-invalid"
                    if (
                        observe_cache_mount(
                            cache_descriptor,
                            paths.mountinfo,
                            expected_filesystem_type=staging.expected_filesystem_type,
                        )
                        != cache_mount
                    ):
                        raise DatabasePlacementError("protected Database Replica cache mount changed during population")
                    classification = "lock-unavailable"
                    verify_cache_ownership(cache_descriptor, cache_ownership)
                    classification = "cache-authority-invalid"
                    verify_cache_root_binding(paths.cache_root, cache_descriptor, cache_identity)
                    classification = "replica-validation-failed"
                    replica_manifest = DatabaseReplicaManifest(
                        database_set=runspec.payload.database.database_set,
                        source_manifest_sha256=identity_value,
                        members=replica_members,
                        pre_copy_source_observation=pre_copy,
                        post_copy_source_observation=post_copy,
                        copy_evidence=copy_evidence,
                    )
                    services.population.freeze(
                        workspace.temporary_descriptor,
                        manifest=replica_manifest,
                        members=replica_members,
                    )
                    classification = "lock-unavailable"
                    verify_cache_ownership(cache_descriptor, cache_ownership)
                    classification = "publication-failed"
                    publish_replica_noreplace(
                        renameat2,
                        workspace,
                        destination_name=identity_value,
                    )
                    verify_published_replica(
                        workspace,
                        destination_name=identity_value,
                        manifest=replica_manifest,
                        members=replica_members,
                    )
                    classification = "lock-unavailable"
                    verify_cache_ownership(cache_descriptor, cache_ownership)
                    classification = "cache-authority-invalid"
                    verify_cache_root_binding(paths.cache_root, cache_descriptor, cache_identity)
                    result = DatabaseReplicaColdResult(
                        phase_run_id=runspec.phase_run_id,
                        attempt_id=runspec.attempt_id,
                        phase_runspec_digest=runspec.digest,
                        action_id=action_id,
                        database_set=runspec.payload.database.database_set,
                        requested_policy=runspec.payload.database.requested_policy,
                        source_manifest_sha256=identity_value,
                        branch_kind="staged",
                        outcome=DatabasePlacementOutcomeKind.REPLICA_COLD,
                        selected_container_root=SELECTED_DATABASE_ROOT,
                        replica_container_root=f"{DATABASE_CACHE_ROOT}/replicas/{identity_value}",
                        verification="metadata-verified",
                        source_mount=source_mount,
                        cache_mount=cache_mount,
                        capacity_gate=capacity,
                        replica_manifest_sha256=database_replica_manifest_digest(replica_manifest),
                        copy_evidence=copy_evidence,
                    )
                    classification = "result-publication-failed"
                    publish_database_replica_cold_result(result, result_path)
                    classification = "lock-unavailable"
                    verify_cache_ownership(cache_descriptor, cache_ownership)
        return result
    except DatabasePlacementError as exc:
        message = str(exc)
        if isinstance(exc, WarmResultAlreadyVisibleError):
            result = exc.result
        if isinstance(exc, ClassifiedDatabaseReplicaError):
            classification = exc.classification
        if failure_path is not None and (result is None or not _visible_replica_result_matches(result_path, result)):
            publish_database_replica_cold_failure(
                DatabaseReplicaColdFailureEvidence(
                    phase_run_id=runspec.phase_run_id,
                    attempt_id=runspec.attempt_id,
                    phase_runspec_digest=runspec.digest,
                    action_id=action_id,
                    database_set=runspec.payload.database.database_set,
                    requested_policy=runspec.payload.database.requested_policy,
                    source_manifest_sha256=runspec.payload.database.source_manifest_sha256,
                    source_mount=source_mount,
                    cache_mount=cache_mount,
                    capacity_gate=capacity,
                    science_started=False,
                    classification=classification,
                    error=message[:2048],
                ),
                failure_path,
            )
        raise
    finally:
        if cache_descriptor is not None:
            os.close(cache_descriptor)
        if source_descriptor is not None:
            os.close(source_descriptor)


def _allocated_cpus(expected: int) -> int:
    value = os.environ.get("SLURM_CPUS_PER_TASK")
    try:
        observed = int(value) if value is not None else 0
    except ValueError as exc:
        raise DatabasePlacementError("SLURM_CPUS_PER_TASK must be a positive exact integer") from exc
    if observed <= 0 or observed != expected:
        raise DatabasePlacementError("SLURM_CPUS_PER_TASK does not match the sole Runtime Action allocation")
    return observed


def _production_cold_replica_services() -> ColdReplicaServices:
    """Compose the exact production collaborators for cold placement."""
    selector = WarmReplicaSelector(
        probe=_try_place_warm_stage_required_database,
        select_under_identity=_place_existing_warm_candidate_under_identity,
    )
    return ColdReplicaServices(
        warm_selector=selector,
        capacity=DatabaseCapacityObserver(observe=_capacity_gate),
        population=ReplicaPopulationService(
            populate=populate_replica_payload,
            freeze=make_replica_immutable,
        ),
        repair=ReplicaRepairService(
            retire_invalid=retire_invalid_replica,
            cleanup_stale=cleanup_stale_populations,
        ),
    )


def _capacity_gate(cache_descriptor: int, *, allocated_bytes: int, reserved_bytes: int) -> DatabaseCapacityGate:
    try:
        filesystem = os.fstatvfs(cache_descriptor)
    except OSError as exc:
        raise DatabasePlacementError("cannot observe user-available Database Replica cache capacity") from exc
    available = filesystem.f_bavail * filesystem.f_frsize
    required = allocated_bytes + reserved_bytes
    return DatabaseCapacityGate(
        available_user_bytes=available,
        allocated_replica_bytes=allocated_bytes,
        reserved_bytes=reserved_bytes,
        required_bytes=required,
        decision="sufficient" if available >= required else "insufficient",
    )


def _require_distinct_mounts(source: DatabaseSourceMountFacts, cache: DatabaseCacheMountFacts) -> None:
    if source.mount_id == cache.mount_id or (source.device_major, source.device_minor) == (
        cache.device_major,
        cache.device_minor,
    ):
        raise DatabasePlacementError("Database Source and Replica cache must be different mounted filesystems")


def _visible_replica_result_matches(
    path: Path,
    expected: DatabaseReplicaResult | DatabaseCapacityFallbackResult,
) -> bool:
    try:
        if isinstance(expected, DatabaseCapacityFallbackResult):
            return load_database_direct_result(path) == expected
        return load_database_replica_result(path) == expected
    except DatabasePlacementError:
        return False


__all__ = ["place_cold_stage_required_database"]
