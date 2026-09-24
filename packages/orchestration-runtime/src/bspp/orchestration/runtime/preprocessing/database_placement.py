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

"""Policy-dispatched Database Placement lifecycle orchestration."""

from __future__ import annotations

import os
from pathlib import Path

from bspp.orchestration.contract.database_capacity_fallback_result import DatabaseCapacityFallbackResult
from bspp.orchestration.contract.database_direct_result import DatabaseDirectResult
from bspp.orchestration.contract.database_placement import (
    DATABASE_SOURCE_ROOT,
    SELECTED_DATABASE_ROOT,
    DatabaseAccessPolicy,
    DatabasePlacementOutcomeKind,
)
from bspp.orchestration.contract.database_placement_result import (
    DatabasePlacementFailureClassification,
    DatabasePlacementFailureEvidence,
    DatabasePlacementResult,
    DatabasePostScienceEvidence,
    DatabasePostScienceObservationFailure,
    DatabaseSourceMountFacts,
)
from bspp.orchestration.contract.database_replica import DatabaseReplicaResult
from bspp.orchestration.contract.phase import PhaseRunSpec

from ._database_placement_errors import (
    DatabasePlacementError,
    PostScienceSourceObservationError,
)
from ._database_placement_evidence_io import (
    load_database_direct_result,
    load_database_placement_failure,
    load_database_placement_result,
    load_staged_manifest,
    publish_database_placement_failure,
    publish_database_placement_result,
)
from ._database_placement_paths import (
    PRODUCTION_DATABASE_PLACEMENT_PATHS,
    DatabasePlacementPaths,
)
from ._database_replica_authority import require_staged_authority
from ._database_replica_services import StagedDatabasePlacementServices
from ._database_source_mount import observe_source_mount
from ._database_source_observation import (
    observe_post_science_source,
    observe_source_inventory,
    open_source_root,
    verify_source_root_binding,
)

DatabasePlacementCommandResult = DatabaseDirectResult | DatabaseReplicaResult

_SOURCE_ROOT_PATH = Path(DATABASE_SOURCE_ROOT)
_SELECTED_ROOT_PATH = Path(SELECTED_DATABASE_ROOT)
_MOUNTINFO_PATH = Path("/proc/self/mountinfo")


class _ClassifiedPlacementError(DatabasePlacementError):
    def __init__(self, classification: DatabasePlacementFailureClassification, message: str) -> None:
        super().__init__(message)
        self.classification = classification


def place_database(
    runspec: PhaseRunSpec,
    *,
    action_id: str,
    source_manifest_path: Path,
    result_path: Path,
    failure_path: Path | None = None,
) -> DatabasePlacementCommandResult:
    """Dispatch only the currently implemented policy-specific placement command."""
    return _place_database(
        runspec,
        action_id=action_id,
        source_manifest_path=source_manifest_path,
        result_path=result_path,
        failure_path=failure_path,
        paths=PRODUCTION_DATABASE_PLACEMENT_PATHS,
        staged_services=_production_staged_database_placement_services(),
    )


def _place_database(
    runspec: PhaseRunSpec,
    *,
    action_id: str,
    source_manifest_path: Path,
    result_path: Path,
    failure_path: Path | None,
    paths: DatabasePlacementPaths,
    staged_services: StagedDatabasePlacementServices,
) -> DatabasePlacementCommandResult:
    """Dispatch placement against an explicitly composed physical site."""
    policy = runspec.payload.database.requested_policy
    if policy == DatabaseAccessPolicy.DIRECT:
        return _place_direct_requested_database(
            runspec,
            action_id=action_id,
            source_manifest_path=source_manifest_path,
            result_path=result_path,
            failure_path=failure_path,
            paths=paths,
        )
    if policy == DatabaseAccessPolicy.STAGE_REQUIRED:
        return _place_stage_required_database(
            runspec,
            action_id=action_id,
            source_manifest_path=source_manifest_path,
            result_path=result_path,
            failure_path=failure_path,
            paths=paths,
            services=staged_services,
        )
    if policy == DatabaseAccessPolicy.STAGE_PREFERRED:
        return _place_stage_preferred_database(
            runspec,
            action_id=action_id,
            source_manifest_path=source_manifest_path,
            result_path=result_path,
            failure_path=failure_path,
            paths=paths,
            services=staged_services,
        )
    raise DatabasePlacementError(f"unsupported Database Access Policy: {policy!r}")


def place_stage_required_database(
    runspec: PhaseRunSpec,
    *,
    action_id: str,
    source_manifest_path: Path,
    result_path: Path,
    failure_path: Path | None = None,
) -> DatabaseReplicaResult:
    """Reuse an exact warm replica or enter the sole cold coordinator."""
    return _place_stage_required_database(
        runspec,
        action_id=action_id,
        source_manifest_path=source_manifest_path,
        result_path=result_path,
        failure_path=failure_path,
        paths=PRODUCTION_DATABASE_PLACEMENT_PATHS,
        services=_production_staged_database_placement_services(),
    )


def _place_stage_required_database(
    runspec: PhaseRunSpec,
    *,
    action_id: str,
    source_manifest_path: Path,
    result_path: Path,
    failure_path: Path | None,
    paths: DatabasePlacementPaths,
    services: StagedDatabasePlacementServices,
) -> DatabaseReplicaResult:
    """Coordinate stage-required placement at an explicitly composed site."""
    result = _place_staged_database(
        runspec,
        action_id=action_id,
        source_manifest_path=source_manifest_path,
        result_path=result_path,
        failure_path=failure_path,
        paths=paths,
        services=services,
    )
    if isinstance(result, DatabaseCapacityFallbackResult):
        raise AssertionError("stage-required placement cannot select capacity fallback")
    return result


def place_stage_preferred_database(
    runspec: PhaseRunSpec,
    *,
    action_id: str,
    source_manifest_path: Path,
    result_path: Path,
    failure_path: Path | None = None,
) -> DatabaseReplicaResult | DatabaseCapacityFallbackResult:
    """Reuse or populate a replica, falling back only on a positive insufficient gate."""
    return _place_stage_preferred_database(
        runspec,
        action_id=action_id,
        source_manifest_path=source_manifest_path,
        result_path=result_path,
        failure_path=failure_path,
        paths=PRODUCTION_DATABASE_PLACEMENT_PATHS,
        services=_production_staged_database_placement_services(),
    )


def _place_stage_preferred_database(
    runspec: PhaseRunSpec,
    *,
    action_id: str,
    source_manifest_path: Path,
    result_path: Path,
    failure_path: Path | None,
    paths: DatabasePlacementPaths,
    services: StagedDatabasePlacementServices,
) -> DatabaseReplicaResult | DatabaseCapacityFallbackResult:
    """Coordinate stage-preferred placement at an explicitly composed site."""
    return _place_staged_database(
        runspec,
        action_id=action_id,
        source_manifest_path=source_manifest_path,
        result_path=result_path,
        failure_path=failure_path,
        paths=paths,
        services=services,
    )


def _place_staged_database(
    runspec: PhaseRunSpec,
    *,
    action_id: str,
    source_manifest_path: Path,
    result_path: Path,
    failure_path: Path | None,
    paths: DatabasePlacementPaths,
    services: StagedDatabasePlacementServices,
) -> DatabaseReplicaResult | DatabaseCapacityFallbackResult:
    """Share the exact warm-probe and exclusive cold coordinator flow."""
    from ._database_replica_warm import WarmIdentityContended, WarmProbeHit

    warm = services.warm_selector.probe(
        runspec,
        action_id=action_id,
        source_manifest_path=source_manifest_path,
        result_path=result_path,
        failure_path=failure_path,
        paths=paths,
    )
    if isinstance(warm, WarmProbeHit):
        return warm.result
    staging = runspec.payload.database.staging
    if staging is None:
        raise DatabasePlacementError("staged Database Placement requires staging authority")
    return services.cold_coordinator.place(
        runspec,
        action_id=action_id,
        source_manifest_path=source_manifest_path,
        result_path=result_path,
        failure_path=failure_path,
        wait=services.wait_factory.build(staging.lock_wait_seconds),
        preprobe_contended=isinstance(warm, WarmIdentityContended),
        paths=paths,
        services=services.cold,
    )


def _production_staged_database_placement_services() -> StagedDatabasePlacementServices:
    """Compose exact fixed production collaborators for staged placement."""
    from ._database_replica import _place_cold_stage_required_database, _production_cold_replica_services
    from ._database_replica_lock_types import LockWait
    from ._database_replica_services import ColdReplicaCoordinator, LockWaitFactory
    from ._database_replica_warm import (
        _place_existing_warm_candidate_under_identity,
        _try_place_warm_stage_required_database,
    )

    cold = _production_cold_replica_services()
    selector = type(cold.warm_selector)(
        probe=_try_place_warm_stage_required_database,
        select_under_identity=_place_existing_warm_candidate_under_identity,
    )
    return StagedDatabasePlacementServices(
        warm_selector=selector,
        cold_coordinator=ColdReplicaCoordinator(place=_place_cold_stage_required_database),
        wait_factory=LockWaitFactory(build=LockWait.for_timeout),
        cold=type(cold)(
            warm_selector=selector,
            capacity=cold.capacity,
            population=cold.population,
            repair=cold.repair,
        ),
    )


def place_direct_requested_database(
    runspec: PhaseRunSpec,
    *,
    action_id: str,
    source_manifest_path: Path,
    result_path: Path,
    failure_path: Path | None = None,
) -> DatabasePlacementResult:
    """Observe and exclusively publish one direct-requested placement result."""
    return _place_direct_requested_database(
        runspec,
        action_id=action_id,
        source_manifest_path=source_manifest_path,
        result_path=result_path,
        failure_path=failure_path,
        paths=PRODUCTION_DATABASE_PLACEMENT_PATHS,
    )


def _place_direct_requested_database(
    runspec: PhaseRunSpec,
    *,
    action_id: str,
    source_manifest_path: Path,
    result_path: Path,
    failure_path: Path | None = None,
    paths: DatabasePlacementPaths,
) -> DatabasePlacementResult:
    """Run direct placement against an explicitly composed physical site."""
    action = runspec.payload.actions[0]
    mount_facts: DatabaseSourceMountFacts | None = None
    try:
        if action_id != action.action_id:
            raise _ClassifiedPlacementError(
                "authority-invalid",
                f"action id {action_id!r} does not match the sole declared action {action.action_id!r}",
            )
        try:
            _require_direct_authority(runspec)
        except DatabasePlacementError as exc:
            raise _ClassifiedPlacementError("authority-invalid", str(exc)) from exc
        try:
            manifest = load_staged_manifest(runspec, source_manifest_path)
        except DatabasePlacementError as exc:
            raise _ClassifiedPlacementError("source-manifest-invalid", str(exc)) from exc
        try:
            root_descriptor, root_identity = open_source_root(paths.source_root)
        except DatabasePlacementError as exc:
            raise _ClassifiedPlacementError("source-mount-invalid", str(exc)) from exc
        try:
            try:
                mount_facts = observe_source_mount(root_descriptor, paths.mountinfo)
            except DatabasePlacementError as exc:
                raise _ClassifiedPlacementError("source-mount-invalid", str(exc)) from exc
            try:
                observation = observe_source_inventory(paths.source_root, root_descriptor, manifest)
                verify_source_root_binding(paths.source_root, root_descriptor, root_identity)
            except DatabasePlacementError as exc:
                raise _ClassifiedPlacementError("source-inventory-invalid", str(exc)) from exc
            try:
                final_mount_facts = observe_source_mount(root_descriptor, paths.mountinfo)
                if final_mount_facts != mount_facts:
                    raise DatabasePlacementError("protected Database Source mount changed during observation")
            except DatabasePlacementError as exc:
                raise _ClassifiedPlacementError("source-mount-invalid", str(exc)) from exc
            try:
                verify_source_root_binding(paths.source_root, root_descriptor, root_identity)
            except DatabasePlacementError as exc:
                raise _ClassifiedPlacementError("source-inventory-invalid", str(exc)) from exc
        finally:
            os.close(root_descriptor)
        binding = runspec.payload.database
        result = DatabasePlacementResult(
            phase_run_id=runspec.phase_run_id,
            attempt_id=runspec.attempt_id,
            phase_runspec_digest=runspec.digest,
            action_id=action_id,
            database_set=binding.database_set,
            requested_policy=DatabaseAccessPolicy.DIRECT,
            source_manifest_sha256=binding.source_manifest_sha256,
            branch_kind="direct-requested",
            outcome=DatabasePlacementOutcomeKind.DIRECT_REQUESTED,
            selected_container_root=SELECTED_DATABASE_ROOT,
            verification="metadata-verified",
            source_mount=mount_facts,
            pre_science_observation=observation,
        )
        try:
            publish_database_placement_result(result, result_path)
        except DatabasePlacementError as exc:
            if _visible_result_matches(result_path, result):
                raise
            raise _ClassifiedPlacementError("result-publication-failed", str(exc)) from exc
        return result
    except _ClassifiedPlacementError as exc:
        if failure_path is not None:
            publish_database_placement_failure(
                _build_failure_evidence(
                    runspec,
                    action_id=action_id,
                    classification=exc.classification,
                    error=str(exc),
                    source_mount=mount_facts,
                ),
                failure_path,
            )
        raise DatabasePlacementError(str(exc)) from exc


def observe_direct_database_source(
    runspec: PhaseRunSpec,
    result: DatabaseDirectResult,
) -> DatabasePostScienceEvidence:
    """Observe the complete direct source after science or retain bounded observation failure."""
    return _observe_direct_database_source(
        runspec,
        result,
        paths=PRODUCTION_DATABASE_PLACEMENT_PATHS,
    )


def _observe_direct_database_source(
    runspec: PhaseRunSpec,
    result: DatabaseDirectResult,
    *,
    paths: DatabasePlacementPaths,
) -> DatabasePostScienceEvidence:
    """Observe the post-science source against an explicitly composed site."""
    reconcile_database_direct_result(runspec, result, action_id=result.action_id)
    try:
        return observe_post_science_source(
            paths.selected_root,
            runspec.payload.database.source_manifest,
            source_manifest_sha256=result.source_manifest_sha256,
        )
    except PostScienceSourceObservationError as exc:
        return DatabasePostScienceObservationFailure(
            source_container_root=DATABASE_SOURCE_ROOT,
            source_manifest_sha256=result.source_manifest_sha256,
            error=str(exc)[:2048],
        )


def reconcile_database_placement_result(
    runspec: PhaseRunSpec,
    result: DatabasePlacementResult,
    *,
    action_id: str,
) -> None:
    """Bind a strict Result to the exact direct RunSpec action authority."""
    binding = runspec.payload.database
    if (
        result.phase_run_id != runspec.phase_run_id
        or result.attempt_id != runspec.attempt_id
        or result.phase_runspec_digest != runspec.digest
        or result.action_id != action_id
        or result.database_set != binding.database_set
        or result.requested_policy != binding.requested_policy
        or result.source_manifest_sha256 != binding.source_manifest_sha256
        or result.pre_science_observation.members != binding.source_manifest.members
    ):
        raise DatabasePlacementError("Database Placement Result does not match RunSpec authority")
    _require_direct_authority(runspec)


def reconcile_database_direct_result(
    runspec: PhaseRunSpec,
    result: DatabaseDirectResult,
    *,
    action_id: str,
) -> None:
    """Bind one direct-requested or capacity-fallback Result to RunSpec authority."""
    if isinstance(result, DatabasePlacementResult):
        reconcile_database_placement_result(runspec, result, action_id=action_id)
        return
    binding = runspec.payload.database
    staging = require_staged_authority(runspec)
    if (
        result.phase_run_id != runspec.phase_run_id
        or result.attempt_id != runspec.attempt_id
        or result.phase_runspec_digest != runspec.digest
        or result.action_id != action_id
        or result.database_set != binding.database_set
        or result.requested_policy != DatabaseAccessPolicy.STAGE_PREFERRED
        or result.requested_policy != binding.requested_policy
        or result.source_manifest_sha256 != binding.source_manifest_sha256
        or result.pre_science_observation.members != binding.source_manifest.members
        or result.capacity_gate.reserved_bytes != staging.reserve_bytes
        or result.cache_mount.filesystem_type != staging.expected_filesystem_type
        or result.branch_kind != "direct-capacity-fallback"
        or result.outcome != DatabasePlacementOutcomeKind.DIRECT_CAPACITY_FALLBACK
    ):
        raise DatabasePlacementError("capacity fallback Result does not match RunSpec authority")


def _require_direct_authority(runspec: PhaseRunSpec) -> None:
    binding = runspec.payload.database
    if binding.requested_policy != DatabaseAccessPolicy.DIRECT:
        raise DatabasePlacementError("Database Placement requires requested policy 'direct'")
    if len(binding.branches) != 1:
        raise DatabasePlacementError("direct Database Placement requires exactly one declared branch")
    branch = binding.branches[0]
    if branch.branch_kind != "direct-requested" or branch.authorized_outcomes != (
        DatabasePlacementOutcomeKind.DIRECT_REQUESTED,
    ):
        raise DatabasePlacementError("Database Placement requires the sole direct-requested branch and outcome")
    if len(branch.placement_mounts) != 1:
        raise DatabasePlacementError("direct Database Placement requires one protected source mount")
    placement_mount = branch.placement_mounts[0]
    if (
        placement_mount.target != DATABASE_SOURCE_ROOT
        or placement_mount.purpose != "source"
        or not placement_mount.read_only
    ):
        raise DatabasePlacementError("direct Database Placement source mount authority is invalid")
    if not branch.scientific_mounts:
        raise DatabasePlacementError("direct Database Placement requires the complete source member mounts")
    selected_root_path = Path(SELECTED_DATABASE_ROOT)
    for mount in branch.scientific_mounts:
        if (
            mount.purpose != "selected-source"
            or not mount.read_only
            or mount.source_kind != "file"
            or Path(mount.target).parent != selected_root_path
        ):
            raise DatabasePlacementError("direct Database Placement scientific mount authority is invalid")
    if branch.finalization_mounts:
        raise DatabasePlacementError("direct Database Placement forbids finalization payload mounts")


def _build_failure_evidence(
    runspec: PhaseRunSpec,
    *,
    action_id: str,
    classification: DatabasePlacementFailureClassification,
    error: str,
    source_mount: DatabaseSourceMountFacts | None,
) -> DatabasePlacementFailureEvidence:
    binding = runspec.payload.database
    return DatabasePlacementFailureEvidence(
        phase_run_id=runspec.phase_run_id,
        attempt_id=runspec.attempt_id,
        phase_runspec_digest=runspec.digest,
        action_id=action_id,
        database_set=binding.database_set,
        requested_policy=binding.requested_policy,
        source_manifest_sha256=binding.source_manifest_sha256,
        source_mount=source_mount,
        science_started=False,
        classification=classification,
        error=error[:2048],
    )


def _visible_result_matches(path: Path, expected: DatabasePlacementResult) -> bool:
    try:
        return load_database_placement_result(path) == expected
    except DatabasePlacementError:
        return False


__all__ = [
    "DatabasePlacementCommandResult",
    "DatabasePlacementError",
    "load_database_direct_result",
    "load_database_placement_failure",
    "load_database_placement_result",
    "observe_direct_database_source",
    "place_database",
    "place_direct_requested_database",
    "place_stage_preferred_database",
    "place_stage_required_database",
    "reconcile_database_direct_result",
    "reconcile_database_placement_result",
]
