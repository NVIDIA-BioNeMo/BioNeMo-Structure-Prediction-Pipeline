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

"""Compatibility locks for the focused preprocessing database facades."""

from __future__ import annotations

import importlib
import inspect

import pytest

from bspp.orchestration.runtime.preprocessing import _database_cache_evidence as cache_evidence
from bspp.orchestration.runtime.preprocessing import _database_cache_maintenance_types as cache_types
from bspp.orchestration.runtime.preprocessing import _database_cache_scope as cache_scope
from bspp.orchestration.runtime.preprocessing import _database_placement_errors as placement_errors
from bspp.orchestration.runtime.preprocessing import _database_placement_evidence_io as placement_evidence
from bspp.orchestration.runtime.preprocessing import _database_replica_lease as replica_lease
from bspp.orchestration.runtime.preprocessing import _database_replica_lock as lock_facade
from bspp.orchestration.runtime.preprocessing import _database_replica_lock_authority as lock_authority
from bspp.orchestration.runtime.preprocessing import _database_replica_lock_coordination as lock_coordination
from bspp.orchestration.runtime.preprocessing import _database_replica_lock_types as lock_types
from bspp.orchestration.runtime.preprocessing import database_cache_maintenance as cache_facade
from bspp.orchestration.runtime.preprocessing import database_placement as placement_facade
from bspp.orchestration.runtime.preprocessing import execution

_LOCK_EXPORTS = [
    "CacheExclusiveOwnership",
    "FilesystemAuthority",
    "IdentityExclusiveOwnership",
    "LockContendedError",
    "LockWait",
    "MaintenanceCacheExclusiveOwnership",
    "MaintenanceIdentityContendedError",
    "MaintenanceIdentityExclusiveOwnership",
    "acquire_existing_shared_identity_lock",
    "acquire_or_create_exclusive_population_lock",
    "hold_exclusive_cache",
    "hold_exclusive_cache_for_maintenance",
    "hold_exclusive_identities_for_maintenance",
    "hold_exclusive_identity",
    "verify_cache_ownership",
    "verify_existing_shared_identity_lock",
    "verify_identity_ownership",
    "verify_maintenance_cache_ownership",
    "verify_maintenance_identity_ownership",
    "verify_shared_identity_ownership",
]

_PLACEMENT_EXPORTS = [
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

_PLACEMENT_SIGNATURES = {
    "load_database_direct_result": "(path: 'Path') -> 'DatabaseDirectResult'",
    "load_database_placement_failure": "(path: 'Path') -> 'DatabasePlacementFailureEvidence'",
    "load_database_placement_result": "(path: 'Path') -> 'DatabasePlacementResult'",
    "observe_direct_database_source": (
        "(runspec: 'PhaseRunSpec', result: 'DatabaseDirectResult') -> 'DatabasePostScienceEvidence'"
    ),
    "place_database": (
        "(runspec: 'PhaseRunSpec', *, action_id: 'str', source_manifest_path: 'Path', "
        "result_path: 'Path', failure_path: 'Path | None' = None) -> 'DatabasePlacementCommandResult'"
    ),
    "place_direct_requested_database": (
        "(runspec: 'PhaseRunSpec', *, action_id: 'str', source_manifest_path: 'Path', "
        "result_path: 'Path', failure_path: 'Path | None' = None) -> 'DatabasePlacementResult'"
    ),
    "place_stage_preferred_database": (
        "(runspec: 'PhaseRunSpec', *, action_id: 'str', source_manifest_path: 'Path', "
        "result_path: 'Path', failure_path: 'Path | None' = None) -> "
        "'DatabaseReplicaResult | DatabaseCapacityFallbackResult'"
    ),
    "place_stage_required_database": (
        "(runspec: 'PhaseRunSpec', *, action_id: 'str', source_manifest_path: 'Path', "
        "result_path: 'Path', failure_path: 'Path | None' = None) -> 'DatabaseReplicaResult'"
    ),
    "reconcile_database_direct_result": (
        "(runspec: 'PhaseRunSpec', result: 'DatabaseDirectResult', *, action_id: 'str') -> 'None'"
    ),
    "reconcile_database_placement_result": (
        "(runspec: 'PhaseRunSpec', result: 'DatabasePlacementResult', *, action_id: 'str') -> 'None'"
    ),
}

_REPLICA_LEASE_EXPORTS = [
    "HeldDatabaseReplicaLease",
    "database_replica_lease",
    "reconcile_database_replica_cold_result",
    "reconcile_database_replica_result",
    "reconcile_staged_database_placement_evidence",
]

_REPLICA_LEASE_SIGNATURES = {
    "HeldDatabaseReplicaLease": "(result: 'DatabaseReplicaResult', _active: 'bool' = True) -> None",
    "database_replica_lease": (
        "(runspec: 'PhaseRunSpec', result: 'DatabaseReplicaResult') -> 'Iterator[HeldDatabaseReplicaLease]'"
    ),
    "reconcile_database_replica_cold_result": (
        "(runspec: 'PhaseRunSpec', result: 'DatabaseReplicaColdResult') -> 'None'"
    ),
    "reconcile_database_replica_result": ("(runspec: 'PhaseRunSpec', result: 'DatabaseReplicaResult') -> 'None'"),
    "reconcile_staged_database_placement_evidence": (
        "(runspec: 'PhaseRunSpec', evidence: 'PreprocessingStagedDatabasePlacementEvidence', "
        "*, require_success: 'bool') -> 'None'"
    ),
}

_EXECUTION_SIGNATURES = {
    "execute_preprocessing_chunk_action": (
        "(runspec: 'PhaseRunSpec', *, action_id: 'str', evidence_path: 'Path', "
        "database_placement_result_path: 'Path', placement_process_status: 'int', "
        "database_placement_failure_path: 'Path | None' = None, "
        "carry_forward_record: 'AttemptCarryForwardRecord | None' = None, "
        "phase_submission_id: 'str | None' = None, clock: 'Clock | None' = None, "
        "sleeper: 'Sleeper | None' = None) -> 'PreprocessingChunkActionEvidence'"
    ),
    "load_preprocessing_phase_runspec": "(path: 'Path') -> 'PhaseRunSpec'",
    "reconcile_preprocessing_chunk_action_evidence": (
        "(runspec: 'PhaseRunSpec', evidence: 'PreprocessingChunkActionEvidence', *, "
        "carry_forward_record: 'AttemptCarryForwardRecord | None' = None) -> 'None'"
    ),
    "reconcile_preprocessing_chunk_action_evidence_for_finalization": (
        "(runspec: 'PhaseRunSpec', evidence: 'PreprocessingChunkActionEvidence') -> 'None'"
    ),
}


def test_database_replica_lock_facade_preserves_exact_exports_and_object_identity() -> None:
    assert type(lock_facade.__all__) is list
    assert lock_facade.__all__ == _LOCK_EXPORTS
    implementation_modules = (lock_types, lock_authority, lock_coordination)
    for name in _LOCK_EXPORTS:
        dynamic = getattr(importlib.import_module(lock_facade.__name__), name)
        focused = next(getattr(module, name) for module in implementation_modules if hasattr(module, name))
        assert dynamic is getattr(lock_facade, name)
        assert dynamic is focused


def test_database_cache_maintenance_facade_preserves_exact_exports_and_object_identity() -> None:
    expected = [
        "DatabaseCacheMaintenanceError",
        "clear_database_acceptance_cache",
        "load_database_acceptance_cache_authority",
        "load_database_acceptance_cache_profile",
        "publish_database_cache_maintenance_evidence",
    ]
    assert type(cache_facade.__all__) is list
    assert cache_facade.__all__ == expected
    implementations = {
        "DatabaseCacheMaintenanceError": cache_types.DatabaseCacheMaintenanceError,
        "clear_database_acceptance_cache": cache_facade.clear_database_acceptance_cache,
        "load_database_acceptance_cache_authority": cache_scope.load_database_acceptance_cache_authority,
        "load_database_acceptance_cache_profile": cache_scope.load_database_acceptance_cache_profile,
        "publish_database_cache_maintenance_evidence": cache_evidence.publish_database_cache_maintenance_evidence,
    }
    assert {name: getattr(cache_facade, name) for name in expected} == implementations


def test_database_placement_facade_preserves_exact_exports_and_import_identity() -> None:
    assert type(placement_facade.__all__) is list
    assert placement_facade.__all__ == _PLACEMENT_EXPORTS
    imported = importlib.import_module(placement_facade.__name__)
    for name in _PLACEMENT_EXPORTS:
        assert getattr(imported, name) is getattr(placement_facade, name)

    assert placement_facade.DatabasePlacementError is placement_errors.DatabasePlacementError
    assert placement_facade.load_database_direct_result is placement_evidence.load_database_direct_result
    assert placement_facade.load_database_placement_failure is placement_evidence.load_database_placement_failure
    assert placement_facade.load_database_placement_result is placement_evidence.load_database_placement_result


def test_database_placement_facade_preserves_exact_callable_signatures() -> None:
    assert {name for name in _PLACEMENT_EXPORTS if callable(getattr(placement_facade, name))} == {
        "DatabasePlacementError",
        *_PLACEMENT_SIGNATURES,
    }
    with pytest.raises(ValueError):
        inspect.signature(placement_facade.DatabasePlacementError)
    assert {
        name: str(inspect.signature(getattr(placement_facade, name))) for name in _PLACEMENT_SIGNATURES
    } == _PLACEMENT_SIGNATURES


def test_database_replica_lease_preserves_exact_exports_import_identity_and_signatures() -> None:
    assert type(replica_lease.__all__) is list
    assert replica_lease.__all__ == _REPLICA_LEASE_EXPORTS
    imported = importlib.import_module(replica_lease.__name__)
    for name in _REPLICA_LEASE_EXPORTS:
        assert getattr(imported, name) is getattr(replica_lease, name)
    assert {name for name in _REPLICA_LEASE_EXPORTS if callable(getattr(replica_lease, name))} == set(
        _REPLICA_LEASE_SIGNATURES
    )
    assert {
        name: str(inspect.signature(getattr(replica_lease, name))) for name in _REPLICA_LEASE_SIGNATURES
    } == _REPLICA_LEASE_SIGNATURES


def test_preprocessing_execution_exports_and_public_signature_remain_frozen() -> None:
    assert type(execution.__all__) is list
    assert execution.__all__ == [
        "GPUSERVER_WARMUP_SECONDS",
        "PREPROCESSING_ADAPTER_VERSION",
        "PreprocessingExecutionError",
        "execute_preprocessing_chunk_action",
        "load_preprocessing_phase_runspec",
        "reconcile_preprocessing_chunk_action_evidence",
        "reconcile_preprocessing_chunk_action_evidence_for_finalization",
    ]
    assert {name for name in execution.__all__ if callable(getattr(execution, name))} == {
        "PreprocessingExecutionError",
        *_EXECUTION_SIGNATURES,
    }
    with pytest.raises(ValueError):
        inspect.signature(execution.PreprocessingExecutionError)
    assert {
        name: str(inspect.signature(getattr(execution, name))) for name in _EXECUTION_SIGNATURES
    } == _EXECUTION_SIGNATURES
