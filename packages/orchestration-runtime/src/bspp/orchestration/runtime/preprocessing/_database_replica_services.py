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

"""Typed collaborators for private Database Replica coordinators."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from bspp.orchestration.contract.database_capacity_fallback_result import DatabaseCapacityFallbackResult
from bspp.orchestration.contract.database_placement_result import DatabaseSourceObservation
from bspp.orchestration.contract.database_replica import (
    DatabaseCapacityGate,
    DatabaseReplicaCopyEvidence,
    DatabaseReplicaManifest,
    DatabaseReplicaMember,
    DatabaseReplicaResult,
    DatabaseWarmCacheMountFacts,
)
from bspp.orchestration.contract.phase import PhaseRunSpec

from ._database_placement_paths import DatabasePlacementPaths
from ._database_replica_lock_types import (
    CacheExclusiveOwnership,
    FilesystemAuthority,
    IdentityExclusiveOwnership,
    LockWait,
)

if TYPE_CHECKING:
    from ._database_replica_warm import WarmCandidateDisposition, WarmProbeDisposition


class _BuildLockWait(Protocol):
    def __call__(self, timeout_seconds: int) -> LockWait: ...


@dataclass(frozen=True)
class LockWaitFactory:
    """Construct one bounded lock wait from declared staging authority."""

    build: _BuildLockWait


class _ProbeWarmReplica(Protocol):
    def __call__(
        self,
        runspec: PhaseRunSpec,
        *,
        action_id: str,
        source_manifest_path: Path,
        result_path: Path,
        failure_path: Path | None,
        paths: DatabasePlacementPaths,
    ) -> WarmProbeDisposition: ...


class _SelectWarmReplicaUnderIdentity(Protocol):
    def __call__(
        self,
        runspec: PhaseRunSpec,
        *,
        ownership: IdentityExclusiveOwnership,
        action_id: str,
        cache_descriptor: int,
        cache_identity: FilesystemAuthority,
        cache_mount: DatabaseWarmCacheMountFacts,
        result_path: Path,
        paths: DatabasePlacementPaths,
    ) -> WarmCandidateDisposition: ...


@dataclass(frozen=True)
class WarmReplicaSelector:
    """Read-only warm probe and held-identity candidate selection."""

    probe: _ProbeWarmReplica
    select_under_identity: _SelectWarmReplicaUnderIdentity


class _ObserveDatabaseCapacity(Protocol):
    def __call__(
        self,
        cache_descriptor: int,
        *,
        allocated_bytes: int,
        reserved_bytes: int,
    ) -> DatabaseCapacityGate: ...


@dataclass(frozen=True)
class DatabaseCapacityObserver:
    """Observe user-available capacity beneath an already-bound cache root."""

    observe: _ObserveDatabaseCapacity


class _PopulateReplicaPayload(Protocol):
    def __call__(
        self,
        *,
        source_descriptor: int,
        temporary_descriptor: int,
        observation: DatabaseSourceObservation,
        allocated_cpus: int,
        rsync_executable: Path,
    ) -> tuple[DatabaseReplicaCopyEvidence, tuple[DatabaseReplicaMember, ...]]: ...


class _FreezeReplicaPayload(Protocol):
    def __call__(
        self,
        temporary_descriptor: int,
        *,
        manifest: DatabaseReplicaManifest,
        members: tuple[DatabaseReplicaMember, ...],
    ) -> None: ...


@dataclass(frozen=True)
class ReplicaPopulationService:
    """Copy and freeze the payload inside a privately owned workspace."""

    populate: _PopulateReplicaPayload
    freeze: _FreezeReplicaPayload


class _RetireInvalidReplica(Protocol):
    def __call__(
        self,
        runspec: PhaseRunSpec,
        *,
        action_id: str,
        cache_descriptor: int,
        ownership: CacheExclusiveOwnership,
        result_path: Path,
    ) -> None: ...


class _CleanupStalePopulations(Protocol):
    def __call__(
        self,
        runspec: PhaseRunSpec,
        *,
        action_id: str,
        cache_descriptor: int,
        ownership: CacheExclusiveOwnership,
        result_path: Path,
    ) -> None: ...


@dataclass(frozen=True)
class ReplicaRepairService:
    """Repair invalid finals and ownerless same-identity workspaces."""

    retire_invalid: _RetireInvalidReplica
    cleanup_stale: _CleanupStalePopulations


@dataclass(frozen=True)
class ColdReplicaServices:
    """Cohesive collaborators used only by the private cold coordinator."""

    warm_selector: WarmReplicaSelector
    capacity: DatabaseCapacityObserver
    population: ReplicaPopulationService
    repair: ReplicaRepairService


class _PlaceColdReplica(Protocol):
    def __call__(
        self,
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
    ) -> DatabaseReplicaResult | DatabaseCapacityFallbackResult: ...


@dataclass(frozen=True)
class ColdReplicaCoordinator:
    """Enter the sole exclusive cold placement state machine."""

    place: _PlaceColdReplica


@dataclass(frozen=True)
class StagedDatabasePlacementServices:
    """Private staged-dispatch composition with fixed production defaults."""

    warm_selector: WarmReplicaSelector
    cold_coordinator: ColdReplicaCoordinator
    wait_factory: LockWaitFactory
    cold: ColdReplicaServices


__all__ = [
    "ColdReplicaCoordinator",
    "ColdReplicaServices",
    "DatabaseCapacityObserver",
    "LockWaitFactory",
    "ReplicaPopulationService",
    "ReplicaRepairService",
    "StagedDatabasePlacementServices",
    "WarmReplicaSelector",
]
