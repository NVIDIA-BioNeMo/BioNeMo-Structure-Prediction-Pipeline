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

"""Public Database Replica contract facade."""

from __future__ import annotations

from collections.abc import Mapping

from bspp.orchestration.contract.database_replica_facts import (
    DatabaseCacheMountFacts,
    DatabaseCapacityDecision,
    DatabaseCapacityGate,
    DatabaseReplicaCopyEvidence,
    DatabaseRsyncOutcome,
    DatabaseWarmCacheMountFacts,
    database_warm_cache_mount_facts_from_mapping,
)
from bspp.orchestration.contract.database_replica_manifest import (
    DatabaseReplicaManifest,
    DatabaseReplicaMember,
    canonical_database_replica_manifest_bytes,
    database_replica_manifest_digest,
    database_replica_manifest_from_mapping,
)
from bspp.orchestration.contract.database_replica_result import (
    DatabaseReplicaColdFailureClassification,
    DatabaseReplicaColdFailureEvidence,
    DatabaseReplicaColdResult,
    canonical_database_replica_cold_failure_evidence_bytes,
    canonical_database_replica_cold_result_bytes,
    database_replica_cold_failure_evidence_digest,
    database_replica_cold_failure_evidence_from_mapping,
    database_replica_cold_result_digest,
    database_replica_cold_result_from_mapping,
)
from bspp.orchestration.contract.database_replica_warm_result import (
    DatabaseReplicaWarmResult,
    canonical_database_replica_warm_result_bytes,
    database_replica_warm_result_digest,
    database_replica_warm_result_from_mapping,
)

DatabaseReplicaResult = DatabaseReplicaColdResult | DatabaseReplicaWarmResult


def database_replica_result_from_mapping(payload: Mapping[str, object]) -> DatabaseReplicaResult:
    """Dispatch one exact closed Result envelope."""
    if set(payload) == {"database_replica_cold_result"}:
        return database_replica_cold_result_from_mapping(payload)
    if set(payload) == {"database_replica_warm_result"}:
        return database_replica_warm_result_from_mapping(payload)
    raise ValueError("Database Replica Result requires one known exact envelope")


def canonical_database_replica_result_bytes(result: DatabaseReplicaResult) -> bytes:
    if isinstance(result, DatabaseReplicaColdResult):
        return canonical_database_replica_cold_result_bytes(result)
    if isinstance(result, DatabaseReplicaWarmResult):
        return canonical_database_replica_warm_result_bytes(result)
    raise TypeError("unsupported Database Replica Result type")


def database_replica_result_digest(result: DatabaseReplicaResult) -> str:
    if isinstance(result, DatabaseReplicaColdResult):
        return database_replica_cold_result_digest(result)
    if isinstance(result, DatabaseReplicaWarmResult):
        return database_replica_warm_result_digest(result)
    raise TypeError("unsupported Database Replica Result type")


__all__ = [
    "DatabaseCacheMountFacts",
    "DatabaseCapacityDecision",
    "DatabaseCapacityGate",
    "DatabaseReplicaColdFailureClassification",
    "DatabaseReplicaColdFailureEvidence",
    "DatabaseReplicaColdResult",
    "DatabaseReplicaCopyEvidence",
    "DatabaseReplicaManifest",
    "DatabaseReplicaMember",
    "DatabaseReplicaResult",
    "DatabaseReplicaWarmResult",
    "DatabaseRsyncOutcome",
    "DatabaseWarmCacheMountFacts",
    "canonical_database_replica_cold_failure_evidence_bytes",
    "canonical_database_replica_cold_result_bytes",
    "canonical_database_replica_manifest_bytes",
    "canonical_database_replica_result_bytes",
    "canonical_database_replica_warm_result_bytes",
    "database_replica_cold_failure_evidence_digest",
    "database_replica_cold_failure_evidence_from_mapping",
    "database_replica_cold_result_digest",
    "database_replica_cold_result_from_mapping",
    "database_replica_manifest_digest",
    "database_replica_manifest_from_mapping",
    "database_replica_result_digest",
    "database_replica_result_from_mapping",
    "database_replica_warm_result_digest",
    "database_replica_warm_result_from_mapping",
    "database_warm_cache_mount_facts_from_mapping",
]
